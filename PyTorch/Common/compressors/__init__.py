from .quantizers import Quantizer, MaxMinQuantizer, ExponentialQuantizer, OneBitQuantizer, SanityQuantizer, NormUniformQuantizer, TernGrad, QuantileQuantizer
from .topk_compressor import TopKCompressor, TopKRMSProp, TopKCompressorBig
from .CompressedSGDFull import CompressedSGDBig
compression_types = ["none", "sanity", "maxmin", "exponential", "1bit", "norm_uniform", "terngrad", "topk", "topk_rmsprop", "svd", "stats", "quantile"]

import os
import horovod.torch as hvd

def get_compressor(args, named_parameters):
    if args.compression_type == 'none':
        return hvd.Compression.none
        # return hvd.Compression.fp16() if args.fp16_allreduce else hvd.Compression.none
    else:
        if getattr(args, "efx_bits", None) is not None or getattr(args, "efx_randk", None) is not None:
            ef_q = {"bits": args.efx_bits, "bucket_size": args.bucket_size, "rand_k": args.efx_randk}
            error_feedback = False
        else:
            ef_q = None
            error_feedback = args.error_feedback
        if args.compression_type == "maxmin":
            return MaxMinQuantizer(args.quantization_bits, args.bucket_size, args.error_feedback, named_parameters)
        elif args.compression_type == "exponential":
            return ExponentialQuantizer(args.quantization_bits, args.bucket_size, args.error_feedback)
        elif args.compression_type == "norm_uniform":
            return NormUniformQuantizer(args.quantization_bits, args.bucket_size, args.error_feedback)
        if args.compression_type == "terngrad":
            return TernGrad(args.bucket_size, args.error_feedback)
        elif args.compression_type == "sanity":
            return SanityQuantizer(args.quantization_bits, args.bucket_size, args.error_feedback)
        elif args.compression_type == "1bit":
            return OneBitQuantizer(args.bucket_size, args.error_feedback)
        elif args.compression_type == "quantile":
            return QuantileQuantizer(args.quantization_bits, args.bucket_size, args.error_feedback)

        if not args.big_grad:
            if "topk" in args.compression_type:
                if args.topk_ratio > 1.0 or args.topk_ratio < 0.0:
                    raise ValueError("No topk possible with topk_ratio = ", args.topk_ratio)

                if args.compression_type == "topk":
                    return TopKCompressor(k_perc=args.topk_ratio, bucket_size=args.bucket_size,
                                          enable_error_correction=error_feedback,
                                          warmup_steps=args.compressor_warmup_steps, ef_q=ef_q)
                    # return compressors.TopKCompressor(k_perc=args.topk_ratio, bucket_size=args.bucket_size,
                    #                                      enable_error_correction=args.error_feedback)
                elif args.compression_type == "topk_rmsprop":
                    return TopKRMSProp(k_perc=args.topk_ratio, bucket_size=None,
                                       enable_error_correction=args.error_feedback,
                                       enable_stats=args.topk_ada_stats)
            elif args.compression_type == "svd":
                return SVD_compressor(args.svd_rank, True)
            else:
                if args.quantization_bits > 8:
                    raise ValueError("No quantization possible with quantization bits = ", args.quantization_bits)
        elif args.compression_type == "stats":
            freq = 50
            if not os.path.exists(args.stats_save_dir):
                os.makedirs(args.stats_save_dir)
            filename = os.path.join(args.stats_save_dir, "quantiles.txt")
            return StatisticsSGDBig(filename, freq)
        elif "topk" in args.compression_type:
            return TopKCompressorBig(k_perc=args.topk_ratio,
                                     enable_error_correction=error_feedback,
                                     warmup_steps=args.compressor_warmup_steps, ef_q=ef_q)
