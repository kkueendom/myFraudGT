"""Laundering-motif structural descriptors for the Motif-Aware Graph Transformer.

Money laundering is defined by two structural patterns the FraudGT paper names as
the core difficulty (and that message-passing GTs provably struggle to detect):

  * scatter-gather / smurfing  -> fan-out then fan-in (degree imbalance), and
  * round-tripping              -> short cycles (at the shortest scale, reciprocity).

`compute_motif_edge_features` turns a (sampled) directed multigraph into a small,
bounded, parameter-free per-edge descriptor capturing these. It is computed on the
homogeneous ``edge_index`` of the current batch (a 2-hop neighbourhood), so it is
cheap (degree scatter + a reciprocity lookup, O(E log E)) and adds negligible
latency. A downstream ``nn.Linear`` maps it to a per-head attention bias.

Only local structure (degrees, reciprocity) is used in this version; amount-weighted
motifs and longer cycles (which need a larger receptive field than the 2-hop sampler)
are documented extensions.
"""

import torch

# Number of per-edge motif features produced by `compute_motif_edge_features`.
MOTIF_EDGE_DIM = 7


def compute_motif_edge_features(edge_index, num_nodes):
    """Return a ``[num_edges, MOTIF_EDGE_DIM]`` bounded motif descriptor.

    Features per edge (src -> dst):
      0-3: log1p of src out/in-degree and dst out/in-degree (fan-in/out magnitude);
      4:   src scatter tendency  (out-in)/(out+in+1)  in [-1, 1];
      5:   dst gather tendency   (in-out)/(in+out+1)   in [-1, 1];
      6:   reciprocity flag      1 if the reverse edge dst->src also exists.
    """
    src = edge_index[0].long()
    dst = edge_index[1].long()
    device = edge_index.device
    ones = torch.ones(src.numel(), device=device)

    out_deg = torch.zeros(num_nodes, device=device).scatter_add_(0, src, ones)
    in_deg = torch.zeros(num_nodes, device=device).scatter_add_(0, dst, ones)

    s_out, s_in = out_deg[src], in_deg[src]
    d_out, d_in = out_deg[dst], in_deg[dst]
    src_scatter = (s_out - s_in) / (s_out + s_in + 1.0)
    dst_gather = (d_in - d_out) / (d_in + d_out + 1.0)

    # Reciprocity: does the reverse edge exist? Encode directed edges as integer
    # keys and test membership of the reversed keys. num_nodes is the sampled
    # subgraph size, so keys stay well within int64 range.
    keys = src * num_nodes + dst
    rev_keys = dst * num_nodes + src
    recip = torch.isin(rev_keys, keys).to(ones.dtype)

    feats = torch.stack(
        [torch.log1p(s_out), torch.log1p(s_in),
         torch.log1p(d_out), torch.log1p(d_in),
         src_scatter, dst_gather, recip],
        dim=-1,
    )
    return torch.nan_to_num(feats)
