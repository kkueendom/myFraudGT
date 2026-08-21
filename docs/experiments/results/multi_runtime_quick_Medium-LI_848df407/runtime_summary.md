# Multi-CDVT Quick Runtime Summary

Freshly initialized weights are used because parameter count, memory shape, and operator runtime do not depend on trained values. Each row is normal-only end-to-end inference under dynamic-random sampling.

| Dataset | Model | Parameters | Peak GPU memory (GiB) | Seconds / batch (mean +/- std) | Targets / second |
|---|---|---:|---:|---:|---:|
| Medium-LI | multi_account_only | 243,561 | 7.153 | 1.4343 +/- 0.0051 | 1313.60 |
| Medium-LI | multi_cdvt | 362,139 | 7.220 | 3.4516 +/- 0.0602 | 546.55 |

| Dataset | Parameter ratio | Memory ratio | Latency ratio | Latency overhead | Throughput ratio |
|---|---:|---:|---:|---:|---:|
| Medium-LI | 1.487x | 1.009x | 2.406x | +140.6% | 0.416x |
