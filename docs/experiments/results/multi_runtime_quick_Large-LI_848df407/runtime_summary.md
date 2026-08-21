# Multi-CDVT Quick Runtime Summary

Freshly initialized weights are used because parameter count, memory shape, and operator runtime do not depend on trained values. Each row is normal-only end-to-end inference under dynamic-random sampling.

| Dataset | Model | Parameters | Peak GPU memory (GiB) | Seconds / batch (mean +/- std) | Targets / second |
|---|---|---:|---:|---:|---:|
| Large-LI | multi_account_only | 243,561 | 11.416 | 3.8387 +/- 0.5172 | 149.04 |
| Large-LI | multi_cdvt | 362,139 | 11.514 | 3.9849 +/- 0.0929 | 141.95 |

| Dataset | Parameter ratio | Memory ratio | Latency ratio | Latency overhead | Throughput ratio |
|---|---:|---:|---:|---:|---:|
| Large-LI | 1.487x | 1.009x | 1.038x | +3.8% | 0.952x |
