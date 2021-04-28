from bayes_optim import BO, OrdinalSpace
from bayes_optim.Surrogate import GaussianProcess, RandomForest
import itertools
import numpy as np
from scipy import optimize

class BitsAdjuster:
    def __init__(self, quantizer):
        self.quantizer = quantizer
        self.bits_default = quantizer.bits
        self.uncompressed_value_bits = 32
        # parameters of objective compression_ratio^alpha * error_metric^beta
        self.alpha = 0.75
        self.beta = 0.25
        self.excluded_layers = set()
        self.metric_vals_cache = {}

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

    def get_compression_ratio(self, states):
        numerator = 0.
        denominator = 0.
        for p, state in states.items():
            numerator += float(state["bits"] * p.numel())
            denominator += float(self.uncompressed_value_bits * p.numel())
        return numerator / denominator

    def get_objective(self, states, param_bits):
        metric_sum = 0.0
        for i, (p, state) in enumerate(states.items()):
            metric_sum += self.get_param_metric(p, state, param_bits[i])
            state["bits"] = param_bits[i]
        ratio = self.get_compression_ratio(states)
        return (ratio ** self.alpha) * (metric_sum ** self.beta)

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
        for p in states.keys():
            value = self.quantizer.get_metric(p)
            if value == float("inf") or value != value or value < 1e-10:
                # don't take this value into account
                continue
            max_value = max(value, max_value)
        unit = (bits_max - bits_min) / max_value
        result_bits = []
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
                bits = int(bits_min + unit * value)
            state["bits"] = bits
            result_bits.append(bits)
        return result_bits


class BayesianAdjuster(BitsAdjuster):
    def __init__(self, quantizer):
        super(BayesianAdjuster, self).__init__(quantizer)
        self.metric_vals = {}
        self.excluded_layers = set()

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
        max_FEs = 2 * len(states) * (bits_max - bits_min + 1)

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
        print("Bayesian found bits: {}, best objs: {}".format(xopt, fopt))
        return xopt