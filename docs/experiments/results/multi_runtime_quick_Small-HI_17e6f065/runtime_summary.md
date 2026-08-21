# Multi-CDVT Quick Runtime Summary

Freshly initialized weights are used because parameter count, memory shape, and operator runtime do not depend on trained values. Each row is normal-only end-to-end inference under dynamic-random sampling.

| Dataset | Model | Parameters | Peak GPU memory (GiB) | Seconds / batch (mean +/- std) | Targets / second |
|---|---|---:|---:|---:|---:|
| Small-HI | multi_account_only | 243,561 | 4.740 | 0.5749 +/- 0.0020 | 3530.72 |
| Small-HI | multi_cdvt | 362,139 | 4.734 | 2.6939 +/- 0.0250 | 753.43 |

| Dataset | Parameter ratio | Memory ratio | Latency ratio | Latency overhead | Throughput ratio |
|---|---:|---:|---:|---:|---:|
| Small-HI | 1.487x | 0.999x | 4.686x | +368.6% | 0.213x |
