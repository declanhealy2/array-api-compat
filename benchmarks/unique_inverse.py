from argparse import ArgumentParser
from contextlib import nullcontext
import json
import resource
import sys
from statistics import mean, stdev
from time import perf_counter

import array_api_compat.numpy as np

parser = ArgumentParser()
parser.add_argument("backend", choices=("numpy", "mlx-cpu", "mlx-gpu"))
parser.add_argument("--size", type=int, default=4096)
args = parser.parse_args()
if args.backend == "numpy":
    xp, scope = np, nullcontext()
    evaluate = lambda *values: None
    dtype = np.float64
else:
    import array_api_compat.mlx as xp

    device = xp.cpu if args.backend == "mlx-cpu" else xp.gpu
    scope = xp.stream(device)
    evaluate = xp.eval
    dtype = xp.float64 if args.backend == "mlx-cpu" else xp.float32

with scope:
    values = (xp.arange(args.size, dtype=dtype) * 1021) % 257
    evaluate(values)
    measurements = []
    for _ in range(21):
        start = perf_counter()
        result = xp.unique_inverse(values)
        evaluate(*result)
        measurements.append(perf_counter() - start)
    np.testing.assert_array_equal(result.values[result.inverse_indices], values)
print(
    json.dumps(
        {
            "backend": args.backend,
            "size": args.size,
            "unique_values": result.values.size,
            "discarded_warmup_s": measurements[0],
            "seconds_mean": mean(measurements[1:]),
            "seconds_sample_deviation": stdev(measurements[1:]),
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
        }
    )
)
