# Multi-CDVT Quick Runtime Summary

Freshly initialized weights are used because parameter count, memory shape, and operator runtime do not depend on trained values. Each row is normal-only end-to-end inference under dynamic-random sampling.

| Dataset | Model | Parameters | Peak GPU memory (GiB) | Seconds / batch (mean +/- std) | Targets / second |
|---|---|---:|---:|---:|---:|
| Small-LI | multi_account_only | 243,561 | 4.824 | 0.5673 +/- 0.0072 | 3577.90 |
| Small-LI | multi_cdvt | 362,139 | 4.841 | 2.8130 +/- 0.0156 | 721.65 |

| Dataset | Parameter ratio | Memory ratio | Latency ratio | Latency overhead | Throughput ratio |
|---|---:|---:|---:|---:|---:|
| Small-LI | 1.487x | 1.003x | 4.958x | +395.8% | 0.202x |
