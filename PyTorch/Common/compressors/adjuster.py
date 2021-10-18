try:
    # from bayes_optim import BO, OrdinalSpace
    # from bayes_optim.Surrogate import GaussianProcess, RandomForest
    from scipy import optimize
    from scipy.cluster.vq import whiten, kmeans2, vq
except Exception:
    print("Scipy not installed")
    pass
import numpy as np
import os

class BitsAdjuster:
    def __init__(self, quantizer):
        self.quantizer = quantizer
        self.bits_default = quantizer.bits
        self.uncompressed_value_bits = 32
        # parameters of objective compression_ratio^alpha * error_metric^beta
        self.alpha = float(os.getenv("ADJUST_ALPHA", 2.0))
        self.excluded_layers = set()
        self.metric_vals_cache = {}
        self.static_mapping = {"word_emb.emb_layers.0.weight": self.bits_default}

    def _reset(self):
        self.excluded_layers = set()
        self.metric_vals_cache = {}

    def get_param_metric(self, p, state, bits):
        if p in self.excluded_layers:
            return 0.0
        if p in self.metric_vals_cache:
            if self.metric_vals_cache[p][bits] is not None:
                return self.metric_vals_cache[p][bits]
        else:
            self.metric_vals_cache[p] = [None] * 9
        state["bits"] = bits
        metric = self.quantizer.get_metric(p, True)
        if metric == float("inf") or metric != metric:
            self.excluded_layers.add(p)
            state["bits"] = self.bits_default
            metric = 0.0
        self.metric_vals_cache[p][bits] = metric
        return metric

    def prepare_bits(self, bits, states):
        for i, p in enumerate(states.keys()):
            if p in self.excluded_layers:
                bits[i] = self.bits_default

    # sets precision of original component value
    def set_original_bits(self, bits):
        self.uncompressed_value_bits = bits

    def fit_predict(self, states, bits_min, bits_max):
        raise NotImplementedError("fit_predict is not implemented")

    def get_compression_metric(self, states):
        # numerator = 0.
        # denominator = 0.
        # for p, state in states.items():
        #     numerator += float(state["bits"] * p.numel())
        #     denominator += float(self.uncompressed_value_bits * p.numel())
        # return numerator / denominator
        levels = 0
        for p, state in states.items():
            levels += 1 << state["bits"]
        return levels

    def get_objective(self, states, param_bits):
        metric_sum = 0.0
        for i, (p, state) in enumerate(states.items()):
            metric_sum += self.get_param_metric(p, state, param_bits[i])
            state["bits"] = param_bits[i]
        comp_metric = self.get_compression_metric(states)
        alpha = np.log2(self.alpha)
        return (comp_metric ** alpha) * metric_sum

    def grid_search(self, states, bits_min, bits_max):
        # lists = [
        #     list(range(bits_min, bits_max + 1)) for p in states.keys()
        # ]
        # best_bits = None
        # best_objective = 1e10
        # for bits in itertools.product(*lists):
        #     obj = self.get_objective(states, bits)
        #     if obj < best_objective:
        #         best_objective = obj
        #         best_bits = bits
        # return best_objective, list(best_bits)
        ranges = tuple([(bits_min, bits_max) for i in range(len(states))])
        def my_f(x):
            x = [int(x_i) for x_i in x]
            # print(x, self.get_objective(states, param_bits=x))
            return self.get_objective(states, param_bits=x)
        resbrute = optimize.brute(my_f, ranges, Ns=bits_max - bits_min + 1, finish=None, full_output=True)
        return resbrute[1], list(resbrute[0])


class LinearAdjuster(BitsAdjuster):
    def __init__(self, quantizer):
        super(LinearAdjuster, self).__init__(quantizer)

    def fit_predict(self, states, bits_min, bits_max):
        self._reset()
        max_value = 0.0
        result_bits = []
        for p in states.keys():
            value = self.quantizer.get_metric(p)
            if value == float("inf") or value != value or value < 1e-10:
                # don't take this value into account
                continue
            max_value = max(value, max_value)
        if max_value < 1e-10:
            for p, state in states.items():
                state["bits"] = self.bits_default
                result_bits.append(self.bits_default)
            return result_bits
        unit = (bits_max - bits_min) / max_value
        for p, state in states.items():
            value = self.quantizer.get_metric(p, compute=False)
            if value < 1e-10:
                # if wasn't compressed
                bits = 32
            elif value == float("inf") or value != value:
                # if gradient explodes or metric equal to NaN
                self.excluded_layers.add(p)
                bits = self.bits_default
            else:
                # bits = max(int(bits_min + unit * np.log2(value)), bits_min)
                bits = max(int(np.ceil(bits_max + np.log2(value / max_value))), bits_min)
            state["bits"] = bits
            result_bits.append(bits)
        return result_bits

class KMeanAdjuster(BitsAdjuster):
    def __init__(self, quantizer):
        super(KMeanAdjuster, self).__init__(quantizer)

    def fit_predict(self, states, bits_min, bits_max):
        obs = []
        excluded = set()
        excluded.add("word_emb.emb_layers.0.weight")
        for p in states.keys():
            # obs.append((p.numel(), self.quantizer.get_metric(p)))
            if states[p]["name"] in excluded:
                continue
            value = self.quantizer.get_metric(p)
            if value == float("inf") or value != value:
                excluded.add(states[p]["name"])
                continue
            obs.append((p.numel(), self.quantizer.get_metric(p)))
            # obs.append((value,))
        obs = np.array(obs)
        wh_obs = whiten(obs)
        num_clusters = bits_max - bits_min + 1
        centroids, _ = kmeans2(wh_obs, num_clusters, iter=100, minit='++')
        clus_ids, _ = vq(wh_obs, centroids)
        centroids_metric = centroids[:, 1] #- centroids[:, 0]
        codes = list(range(0, num_clusters))
        codes = [y for x, y in sorted(zip(centroids_metric, codes))]
        mapping = {code: bits_min + i for i, code in enumerate(codes)}
        result_bits = []
        count = 0
        for p in states.keys():
            if states[p]["name"] in excluded:
                result_bits.append(states[p]["bits"])
            else:
                result_bits.append(mapping[clus_ids[count]])
                count += 1
        return result_bits


class BayesianAdjuster(BitsAdjuster):
    def __init__(self, quantizer):
        super(BayesianAdjuster, self).__init__(quantizer)

    def get_bits_from_states(self, states):
        bits = []
        for state in states.values():
            bits.append(state["bits"])
        return bits

    def fit_predict(self, states, bits_min, bits_max):
        prev_bits = self.get_bits_from_states(states)
        self._reset()
        # best_obj, grid_bits = self.grid_search(states, bits_min, bits_max)
        # print("Grid found bits:{}, best obj: {}".format(grid_bits, best_obj))

        def obj_fun(x):
            return self.get_objective(states, x)
        I = OrdinalSpace([bits_min, bits_max])
        search_space = I * len(states)
        model = RandomForest(levels=search_space.levels)
        # thetaL = 1e-10 * (bits_max - bits_min) * np.ones(len(states))
        # thetaU = 10 * (bits_max - bits_min) * np.ones(len(states))
        # model = GaussianProcess(                # create the GPR model
        #     thetaL=thetaL, thetaU=thetaU
        # )
        max_FEs = len(states) * (bits_max - bits_min + 1) / 2
        print("Max FE", max_FEs)
        obj = obj_fun(prev_bits)
        X = [prev_bits]
        Y = [obj]
        # X = []
        # Y = []
        for q in range(bits_min, bits_max):
            x = [q]*len(states)
            y = obj_fun(x)
            X.append(x)
            Y.append(y)
        opt = BO(search_space, model=model, obj_fun=obj_fun, warm_data=(X, Y), max_FEs=max_FEs, verbose=False,
                 acquisition_fun='EI', minimize=True)
        xopt, fopt, stop_dict = opt.run()
        # print("Bayesian found bits: {}, best objs: {}".format(xopt, fopt))
        return xopt

class BestPerLayer(BitsAdjuster):
    def __init__(self, quantizer):
        super(BestPerLayer, self).__init__(quantizer)

    def fit_predict(self, states, bits_min, bits_max):
        opt_bits = []
        for p, state in states.items():
            do_print = False #state["name"] == "layers.5.dec_attn.o_net.weight" #state["name"] == 'conv1.weight' or state["name"] == 'layer2.0.downsample.0.weight'
            best_bits = None
            best_obj = 1e10
            if state["name"] in self.static_mapping:
                state["bits"] = self.static_mapping[state["name"]]
                opt_bits.append(state["bits"])
                continue
            # if state["name"] not in self.static_mapping:
            #     state["bits"] = self.bits_default
            #     opt_bits.append(state["bits"])
            #     continue
            state["bits"] = 8
            error = self.quantizer.get_metric(p, compute=True)
            ratio = 8 / self.uncompressed_value_bits
            alpha = 1 / np.log2(self.alpha)
            if do_print:
                print(state["name"])
                # print("Alpha: ", self.alpha)
            for b in range(bits_min, bits_max + 1):
                ratio = b / self.uncompressed_value_bits
                state["bits"] = b
                error = self.quantizer.get_metric(p, compute=True)
                if error == float("inf") or error != error:
                    best_bits = self.bits_default
                    break
                reg = ratio - (4 / self.uncompressed_value_bits)
                # obj = (ratio**self.alpha) * error # + (np.abs(reg))**(2*self.alpha) * np.sign(reg)
                obj = b + np.log2(error) * alpha
                if do_print:
                    print("Bits: {}, error: {}, log err: {} ,obj: {}".format(b, error, np.log2(error) * alpha, obj))
                if obj < best_obj:
                    best_obj = obj
                    best_bits = b
            state["bits"] = best_bits
            opt_bits.append(best_bits)
        return opt_bits
