# Multi-CDVT Quick Runtime Summary

Freshly initialized weights are used because parameter count, memory shape, and operator runtime do not depend on trained values. Each row is normal-only end-to-end inference under dynamic-random sampling.

| Dataset | Model | Parameters | Peak GPU memory (GiB) | Seconds / batch (mean +/- std) | Targets / second |
|---|---|---:|---:|---:|---:|
| Medium-HI | multi_account_only | 243,561 | 7.184 | 1.3584 +/- 0.0024 | 1388.84 |
| Medium-HI | multi_cdvt | 362,139 | 7.214 | 3.6255 +/- 0.2662 | 522.85 |

| Dataset | Parameter ratio | Memory ratio | Latency ratio | Latency overhead | Throughput ratio |
|---|---:|---:|---:|---:|---:|
| Medium-HI | 1.487x | 1.004x | 2.669x | +166.9% | 0.376x |
