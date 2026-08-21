# Multi-CDVT Quick Runtime Summary

Freshly initialized weights are used because parameter count, memory shape, and operator runtime do not depend on trained values. Each row is normal-only end-to-end inference under dynamic-random sampling.

| Dataset | Model | Parameters | Peak GPU memory (GiB) | Seconds / batch (mean +/- std) | Targets / second |
|---|---|---:|---:|---:|---:|
| Large-HI | multi_account_only | 243,561 | 11.557 | 3.3555 +/- 0.0269 | 168.70 |
| Large-HI | multi_cdvt | 362,139 | 11.584 | 4.0239 +/- 0.0418 | 140.86 |

| Dataset | Parameter ratio | Memory ratio | Latency ratio | Latency overhead | Throughput ratio |
|---|---:|---:|---:|---:|---:|
| Large-HI | 1.487x | 1.002x | 1.199x | +19.9% | 0.835x |
