import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import mask_to_index, scatter, softmax as pyg_softmax
from torch_scatter import scatter_max

from fraudGT.graphgym.register import register_head
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.models.layer import MLP


@register_head('hetero_edge')
class HeteroGNNEdgeHead(nn.Module):
    '''Head of Hetero GNN, edge prediction'''
    def __init__(self, dim_in, dim_out, dataset):
        super().__init__()
        self.is_hetero = isinstance(dataset[0], HeteroData)
        self.edge_decoding = cfg.model.edge_decoding
        # Scale-agnostic, uncertainty-gated evidence decoder (see
        # `_evidence_gate_head`). A single, dataset-size-independent route.
        # `evidence_gate_proto` is the M1 ablation: base decoder + bounded
        # prototype residual ONLY (no structural branch, no gate) -- it isolates
        # the prototype core so we can tell whether the (currently always-open)
        # structural branch helps at all.
        # `dmprd` keeps only a distribution-aware multi-prototype residual. It
        # has no structural expert, sample gate, auxiliary loss, or gate budget.
        # `evidence_gate_v3` is the convex-fusion gate experiment.
        # `evidence_gate_v4_residual` fixes its dead-expert failure mode with a
        # bounded residual structural expert, a nonzero fixed-open warm-up, and
        # a target-budget router. `evidence_gate_v4_nogate` is the controlled
        # diagnostic that applies the same residual expert without a router.
        # `evidence_gate_v4_noproto` removes both the prototype residual and
        # every prototype-derived router input.
        self.use_evidence_gate = self.edge_decoding in {
            'evidence_gate', 'evidence_gate_proto', 'dmprd',
            'evidence_gate_v3',
            'evidence_gate_v4_residual', 'evidence_gate_v4_nogate',
            'evidence_gate_v4_noproto'}
        self.use_dmprd = (self.edge_decoding == 'dmprd')
        self.eg_proto_only = self.edge_decoding in {
            'evidence_gate_proto', 'dmprd'}
        self.eg_gate_v3 = (self.edge_decoding == 'evidence_gate_v3')
        self.eg_gate_v4 = self.edge_decoding in {
            'evidence_gate_v4_residual', 'evidence_gate_v4_nogate',
            'evidence_gate_v4_noproto'}
        self.eg_gate_v4_nogate = (
            self.edge_decoding == 'evidence_gate_v4_nogate')
        self.eg_gate_v4_noproto = (
            self.edge_decoding == 'evidence_gate_v4_noproto')
        self.use_pair_chain_head = self.edge_decoding in {
            'pair_chain',
            'pair_chain_contextresid',
            'pair_chain_contextseqresid',
            'pair_chain_contextseqpairseqbridgebank',
            'pair_chain_contextseqpairseqbridgebankmotiflite',
            'pair_chain_contextseqpairseqbridgebankwindow',
            'pair_chain_contextseqpairseqbridgebankwindowseqselect',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusion',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflow',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylag',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmix',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsis',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassproto',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisboundresidclassproto',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixproto',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundresid',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotorouteboundresid',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisdualprotoboundresid',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisdualprotoboundresid',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisproto',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixproto',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixdot',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmix',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisproto',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflow',
        }
        self.use_chain_context_residual = self.edge_decoding in {
            'pair_chain_contextresid',
            'pair_chain_contextseqresid',
            'pair_chain_contextseqpairseqbridgebank',
            'pair_chain_contextseqpairseqbridgebankmotiflite',
            'pair_chain_contextseqpairseqbridgebankwindow',
            'pair_chain_contextseqpairseqbridgebankwindowseqselect',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusion',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflow',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylag',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmix',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsis',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassproto',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisboundresidclassproto',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixproto',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundresid',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotorouteboundresid',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisdualprotoboundresid',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisdualprotoboundresid',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisproto',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixproto',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixdot',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmix',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisproto',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflow',
        }
        self.use_sequence_context_residual = self.edge_decoding in {
            'pair_chain_contextseqresid',
            'pair_chain_contextseqpairseqbridgebank',
            'pair_chain_contextseqpairseqbridgebankmotiflite',
            'pair_chain_contextseqpairseqbridgebankwindow',
            'pair_chain_contextseqpairseqbridgebankwindowseqselect',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusion',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflow',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylag',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmix',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsis',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassproto',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisboundresidclassproto',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixproto',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundresid',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotorouteboundresid',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisdualprotoboundresid',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisdualprotoboundresid',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisproto',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixproto',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixdot',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmix',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisproto',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflow',
        }
        self.use_pair_internal_sequence = (
            self.edge_decoding in {
                'pair_chain_contextseqpairseqbridgebank',
                'pair_chain_contextseqpairseqbridgebankmotiflite',
                'pair_chain_contextseqpairseqbridgebankwindow',
                'pair_chain_contextseqpairseqbridgebankwindowseqselect',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusion',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflow',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylag',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmix',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsis',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisboundresidclassproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotorouteboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisdualprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisdualprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixdot',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmix',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflow',
            }
        )
        self.use_sequence_bridge_bank = self.edge_decoding in {
            'pair_chain_contextseqpairseqbridgebank',
            'pair_chain_contextseqpairseqbridgebankmotiflite',
            'pair_chain_contextseqpairseqbridgebankwindow',
            'pair_chain_contextseqpairseqbridgebankwindowseqselect',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusion',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflow',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylag',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmix',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsis',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassproto',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisboundresidclassproto',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixproto',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundresid',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotorouteboundresid',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisdualprotoboundresid',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisdualprotoboundresid',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisproto',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixproto',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixdot',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmix',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisproto',
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflow',
        }
        self.use_sequence_bridge_motif_lite = (
            self.edge_decoding == 'pair_chain_contextseqpairseqbridgebankmotiflite'
        )
        self.use_target_sequence_select = (
            self.edge_decoding in {
                'pair_chain_contextseqpairseqbridgebankwindowseqselect',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusion',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflow',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylag',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmix',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsis',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisboundresidclassproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotorouteboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisdualprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisdualprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixdot',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmix',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflow',
            }
        )
        self.use_difference_fusion = (
            self.edge_decoding in {
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusion',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmix',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisdualprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflow',
            }
        )
        self.use_terminal_role_flow = (
            self.edge_decoding in {
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflow',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylag',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmix',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsis',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisboundresidclassproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotorouteboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisdualprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisdualprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixdot',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmix',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflow',
            }
        )
        self.use_boundary_lag_flow = (
            self.edge_decoding in {
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylag',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmix',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsis',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisboundresidclassproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotorouteboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisdualprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisdualprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixdot',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmix',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisproto',
            }
        )
        self.use_support_conditioned_mixture = (
            self.edge_decoding in {
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmix',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsis',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisboundresidclassproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisdualprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisdualprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixdot',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmix',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisproto',
            }
        )
        self.use_sequence_consistency_filter = (
            self.edge_decoding in {
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsis',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisboundresidclassproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisdualprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisdualprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisproto',
            }
        )
        self.use_support_class_prototype_expert = (
            self.edge_decoding in {
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisboundresidclassproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisdualprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisdualprotoboundresid',
            }
        )
        self.use_support_class_mixture_prototype_expert = (
            self.edge_decoding in {
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisdualprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisdualprotoboundresid',
            }
        )
        self.use_bounded_support_residuals = (
            self.edge_decoding in {
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisboundresidclassproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisdualprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisdualprotoboundresid',
            }
        )
        self.use_support_prototype_expert = (
            self.edge_decoding in {
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisdualprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisdualprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisproto',
            }
        )
        self.use_support_proto_consensus_expert = (
            self.edge_decoding ==
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisdualprotoconsensusboundresid'
        )
        self.use_support_proto_disagreement_expert = (
            self.edge_decoding ==
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisdualprotoconsensusdisagreeboundresid'
        )
        self.use_support_confidence_dual_expert = (
            self.edge_decoding ==
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisdualprotoconsensusdisagreeconfboundresid'
        )
        self.use_support_easyhard_dual_expert = (
            self.edge_decoding ==
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisdualprotoconsensusdisagreeconfhardboundresid'
        )
        self.use_support_scale_route_expert = (
            self.edge_decoding ==
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisdualprotoconsensusdisagreeconfhardscaleboundresid'
        )
        self.use_support_class_route_expert = (
            self.edge_decoding ==
            'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisdualprotoconsensusdisagreeconfhardscaleclassrouteboundresid'
        )
        self.use_support_subgraph_route_expert = (
            self.edge_decoding ==
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotosubgraphrouteboundresid'
        )
        self.use_support_class_split_subgraph_dual_mix_route_expert = (
            self.edge_decoding ==
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphroutedualmixrouteboundresid'
        )
        self.use_support_class_split_subgraph_dual_mix_route_w4_expert = (
            self.edge_decoding ==
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphroutedualmixroutew4boundresid'
        )
        self.use_support_class_split_subgraph_dual_mix_pred_gate_expert = (
            self.edge_decoding ==
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphdualpredgateboundresid'
        )
        self.use_support_class_split_subgraph_dual_mix_uncert_gate_expert = (
            self.edge_decoding ==
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphdualuncertgateboundresid'
        )
        self.use_support_class_split_subgraph_dual_mix_uncert_weak_expert = (
            self.edge_decoding ==
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphdualuncertweakboundresid'
        )
        self.use_support_class_split_subgraph_dual_mix_uncert_late_expert = (
            self.edge_decoding ==
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphdualuncertlateboundresid'
        )
        self.use_support_class_split_subgraph_dual_mix_uncert_tight_expert = (
            self.edge_decoding ==
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphdualuncerttightboundresid'
        )
        self.use_support_class_split_subgraph_dual_mix_uncert_mid_expert = (
            self.edge_decoding ==
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphdualuncertmidboundresid'
        )
        self.use_support_class_split_subgraph_dual_mix_disagree_gate_expert = (
            self.edge_decoding ==
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphdualdisagreegateboundresid'
        )
        self.use_support_class_split_subgraph_dual_resmix_expert = (
            self.edge_decoding ==
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphdualresmixboundresid'
        )
        self.use_support_proto_route_expert = (
            self.edge_decoding ==
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotorouteboundresid'
        )
        self.use_support_class_split_expert = (
            self.edge_decoding ==
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitresid'
        )
        self.use_support_class_split_subgraph_route_expert = (
            self.edge_decoding in {
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphrouteboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphroutecalibboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphroutepredcalibboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphrouteasymcalibboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphrouteprotocalibboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphrouteflowsketchboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphroutedualmixrouteboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphroutedualmixroutew4boundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphdualpredgateboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphdualuncertgateboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphdualuncertweakboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphdualuncertlateboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphdualuncerttightboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphdualuncertmidboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphdualdisagreegateboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphdualresmixboundresid',
            }
        )
        self.use_support_class_split_subgraph_margin_calibration = (
            self.edge_decoding in {
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphroutecalibboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphroutepredcalibboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphrouteprotocalibboundresid',
            }
        )
        self.use_support_class_split_subgraph_margin_pred_calibration = (
            self.edge_decoding ==
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphroutepredcalibboundresid'
        )
        self.use_support_class_split_subgraph_asym_calibration = (
            self.edge_decoding ==
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphrouteasymcalibboundresid'
        )
        self.use_support_class_split_subgraph_proto_expert = (
            self.edge_decoding in {
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphrouteprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphrouteprotocalibboundresid',
            }
        )
        self.use_flow_sketch_score_calibration = (
            self.edge_decoding ==
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisflowsketchscorecalibboundresid'
        )
        self.use_flow_sketch_expert = (
            self.edge_decoding in {
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisflowsketchboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisflowsketchscorecalibboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundclasssplitsubgraphrouteflowsketchboundresid',
            }
        )
        self.use_dot_fallback_support_mixture = (
            self.edge_decoding ==
            'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixdot'
        )
        self.use_sequence_bridge_bank_window = (
            self.edge_decoding in {
                'pair_chain_contextseqpairseqbridgebankwindow',
                'pair_chain_contextseqpairseqbridgebankwindowseqselect',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusion',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflow',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylag',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmix',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsis',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisboundresidclassproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisclassmixprotorouteboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisdualprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisdualprotoboundresid',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixconsisproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectroleflowboundarylagsupportmixdot',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmix',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflowboundarylagsupportmixconsisproto',
                'pair_chain_contextseqpairseqbridgebankwindowseqselectdeltafusionroleflow',
            }
        )
        if self.use_support_proto_consensus_expert:
            self.use_pair_chain_head = True
            self.use_chain_context_residual = True
            self.use_sequence_context_residual = True
            self.use_pair_internal_sequence = True
            self.use_sequence_bridge_bank = True
            self.use_target_sequence_select = True
            self.use_difference_fusion = True
            self.use_terminal_role_flow = True
            self.use_boundary_lag_flow = True
            self.use_support_conditioned_mixture = True
            self.use_sequence_consistency_filter = True
            self.use_support_class_prototype_expert = True
            self.use_support_class_mixture_prototype_expert = True
            self.use_bounded_support_residuals = True
            self.use_support_prototype_expert = True
            self.use_sequence_bridge_bank_window = True
        if self.use_support_proto_disagreement_expert:
            self.use_support_proto_consensus_expert = True
            self.use_pair_chain_head = True
            self.use_chain_context_residual = True
            self.use_sequence_context_residual = True
            self.use_pair_internal_sequence = True
            self.use_sequence_bridge_bank = True
            self.use_target_sequence_select = True
            self.use_difference_fusion = True
            self.use_terminal_role_flow = True
            self.use_boundary_lag_flow = True
            self.use_support_conditioned_mixture = True
            self.use_sequence_consistency_filter = True
            self.use_support_class_prototype_expert = True
            self.use_support_class_mixture_prototype_expert = True
            self.use_bounded_support_residuals = True
            self.use_support_prototype_expert = True
            self.use_sequence_bridge_bank_window = True
        if self.use_support_confidence_dual_expert:
            self.use_support_proto_disagreement_expert = True
            self.use_support_proto_consensus_expert = True
            self.use_pair_chain_head = True
            self.use_chain_context_residual = True
            self.use_sequence_context_residual = True
            self.use_pair_internal_sequence = True
            self.use_sequence_bridge_bank = True
            self.use_target_sequence_select = True
            self.use_difference_fusion = True
            self.use_terminal_role_flow = True
            self.use_boundary_lag_flow = True
            self.use_support_conditioned_mixture = True
            self.use_sequence_consistency_filter = True
            self.use_support_class_prototype_expert = True
            self.use_support_class_mixture_prototype_expert = True
            self.use_bounded_support_residuals = True
            self.use_support_prototype_expert = True
            self.use_sequence_bridge_bank_window = True
        if self.use_support_easyhard_dual_expert:
            self.use_support_confidence_dual_expert = True
            self.use_support_proto_disagreement_expert = True
            self.use_support_proto_consensus_expert = True
            self.use_pair_chain_head = True
            self.use_chain_context_residual = True
            self.use_sequence_context_residual = True
            self.use_pair_internal_sequence = True
            self.use_sequence_bridge_bank = True
            self.use_target_sequence_select = True
            self.use_difference_fusion = True
            self.use_terminal_role_flow = True
            self.use_boundary_lag_flow = True
            self.use_support_conditioned_mixture = True
            self.use_sequence_consistency_filter = True
            self.use_support_class_prototype_expert = True
            self.use_support_class_mixture_prototype_expert = True
            self.use_bounded_support_residuals = True
            self.use_support_prototype_expert = True
            self.use_sequence_bridge_bank_window = True
        if self.use_support_scale_route_expert:
            self.use_support_easyhard_dual_expert = True
            self.use_support_confidence_dual_expert = True
            self.use_support_proto_disagreement_expert = True
            self.use_support_proto_consensus_expert = True
            self.use_pair_chain_head = True
            self.use_chain_context_residual = True
            self.use_sequence_context_residual = True
            self.use_pair_internal_sequence = True
            self.use_sequence_bridge_bank = True
            self.use_target_sequence_select = True
            self.use_difference_fusion = True
            self.use_terminal_role_flow = True
            self.use_boundary_lag_flow = True
            self.use_support_conditioned_mixture = True
            self.use_sequence_consistency_filter = True
            self.use_support_class_prototype_expert = True
            self.use_support_class_mixture_prototype_expert = True
            self.use_bounded_support_residuals = True
            self.use_support_prototype_expert = True
            self.use_sequence_bridge_bank_window = True
        if self.use_support_class_route_expert:
            self.use_support_scale_route_expert = True
            self.use_support_easyhard_dual_expert = True
            self.use_support_confidence_dual_expert = True
            self.use_support_proto_disagreement_expert = True
            self.use_support_proto_consensus_expert = True
            self.use_pair_chain_head = True
            self.use_chain_context_residual = True
            self.use_sequence_context_residual = True
            self.use_pair_internal_sequence = True
            self.use_sequence_bridge_bank = True
            self.use_target_sequence_select = True
            self.use_difference_fusion = True
            self.use_terminal_role_flow = True
            self.use_boundary_lag_flow = True
            self.use_support_conditioned_mixture = True
            self.use_sequence_consistency_filter = True
            self.use_support_class_prototype_expert = True
            self.use_support_class_mixture_prototype_expert = True
            self.use_bounded_support_residuals = True
            self.use_support_prototype_expert = True
            self.use_sequence_bridge_bank_window = True
        if self.use_support_subgraph_route_expert:
            self.use_pair_chain_head = True
            self.use_chain_context_residual = True
            self.use_sequence_context_residual = True
            self.use_pair_internal_sequence = True
            self.use_sequence_bridge_bank = True
            self.use_target_sequence_select = True
            self.use_terminal_role_flow = True
            self.use_boundary_lag_flow = True
            self.use_support_conditioned_mixture = True
            self.use_sequence_consistency_filter = True
            self.use_support_class_prototype_expert = True
            self.use_support_class_mixture_prototype_expert = True
            self.use_bounded_support_residuals = True
            self.use_sequence_bridge_bank_window = True
        if self.use_support_proto_route_expert:
            self.use_pair_chain_head = True
            self.use_chain_context_residual = True
            self.use_sequence_context_residual = True
            self.use_pair_internal_sequence = True
            self.use_sequence_bridge_bank = True
            self.use_target_sequence_select = True
            self.use_terminal_role_flow = True
            self.use_boundary_lag_flow = True
            self.use_support_conditioned_mixture = True
            self.use_sequence_consistency_filter = True
            self.use_support_class_prototype_expert = True
            self.use_support_class_mixture_prototype_expert = True
            self.use_bounded_support_residuals = True
            self.use_support_prototype_expert = True
            self.use_sequence_bridge_bank_window = True
        if self.use_support_class_split_expert:
            self.use_pair_chain_head = True
            self.use_chain_context_residual = True
            self.use_sequence_context_residual = True
            self.use_pair_internal_sequence = True
            self.use_sequence_bridge_bank = True
            self.use_target_sequence_select = True
            self.use_terminal_role_flow = True
            self.use_boundary_lag_flow = True
            self.use_support_conditioned_mixture = True
            self.use_sequence_consistency_filter = True
            self.use_support_class_prototype_expert = True
            self.use_support_class_mixture_prototype_expert = True
            self.use_bounded_support_residuals = True
            self.use_sequence_bridge_bank_window = True
        if self.use_support_class_split_subgraph_route_expert:
            self.use_support_class_split_expert = True
            self.use_support_subgraph_route_expert = True
            self.use_pair_chain_head = True
            self.use_chain_context_residual = True
            self.use_sequence_context_residual = True
            self.use_pair_internal_sequence = True
            self.use_sequence_bridge_bank = True
            self.use_target_sequence_select = True
            self.use_terminal_role_flow = True
            self.use_boundary_lag_flow = True
            self.use_support_conditioned_mixture = True
            self.use_sequence_consistency_filter = True
            self.use_support_class_prototype_expert = True
            self.use_support_class_mixture_prototype_expert = True
            self.use_bounded_support_residuals = True
            self.use_sequence_bridge_bank_window = True
        if self.use_support_class_split_subgraph_proto_expert:
            self.use_support_class_split_subgraph_route_expert = True
            self.use_support_class_split_expert = True
            self.use_support_subgraph_route_expert = True
            self.use_pair_chain_head = True
            self.use_chain_context_residual = True
            self.use_sequence_context_residual = True
            self.use_pair_internal_sequence = True
            self.use_sequence_bridge_bank = True
            self.use_target_sequence_select = True
            self.use_terminal_role_flow = True
            self.use_boundary_lag_flow = True
            self.use_support_conditioned_mixture = True
            self.use_sequence_consistency_filter = True
            self.use_support_class_prototype_expert = True
            self.use_support_class_mixture_prototype_expert = True
            self.use_bounded_support_residuals = True
            self.use_support_prototype_expert = True
            self.use_sequence_bridge_bank_window = True
        if self.use_flow_sketch_expert:
            self.use_pair_chain_head = True
            self.use_chain_context_residual = True
            self.use_sequence_context_residual = True
            self.use_pair_internal_sequence = True
            self.use_sequence_bridge_bank = True
            self.use_target_sequence_select = True
            self.use_terminal_role_flow = True
            self.use_boundary_lag_flow = True
            self.use_support_conditioned_mixture = True
            self.use_sequence_consistency_filter = True
            self.use_bounded_support_residuals = True
            self.use_sequence_bridge_bank_window = True
        self.head_layers = max(cfg.gnn.layers_post_mp, cfg.gt.layers_post_gt)
        self.train_inds = mask_to_index(dataset['train'][cfg.dataset.task_entity].split_mask).to(cfg.device)
        self.val_inds = mask_to_index(dataset['val'][cfg.dataset.task_entity].split_mask).to(cfg.device)
        self.test_inds = mask_to_index(dataset['test'][cfg.dataset.task_entity].split_mask).to(cfg.device)

        if self.use_pair_chain_head:
            self.edge_proj = MLP(dim_in * 3, dim_in,
                                 num_layers=self.head_layers,
                                 bias=True)
            if self.use_pair_internal_sequence:
                self.pair_sequence_len = 4
                self.pair_edge_sequence_encoder = nn.GRU(
                    input_size=dim_in,
                    hidden_size=dim_in,
                    batch_first=True,
                )
                self.pair_proj = MLP(dim_in * 4, dim_in,
                                     num_layers=self.head_layers,
                                     bias=True)
            else:
                self.pair_proj = MLP(dim_in * 3, dim_in,
                                     num_layers=self.head_layers,
                                     bias=True)
            if self.use_sequence_bridge_bank_window:
                self.pair_window_gate = nn.Linear(dim_in * 3, dim_in)
                self.pair_window_alpha = nn.Parameter(
                    torch.full((1,), math.log(0.10 / 0.90))
                )
            self.chain_update = MLP(dim_in * 3, dim_in,
                                    num_layers=self.head_layers,
                                    bias=True)
            self.chain_gate = nn.Linear(dim_in * 3, dim_in)
            self.pair_residual_alpha = nn.Parameter(
                torch.full((1,), math.log(0.15 / 0.85))
            )
            self.chain_residual_alpha = nn.Parameter(
                torch.full((1,), math.log(0.10 / 0.90))
            )
            self.layer_post_mp = MLP(dim_in * 3, dim_out,
                                     num_layers=self.head_layers,
                                     bias=True)
            if self.use_chain_context_residual:
                context_fusion_mult = 4 if self.use_difference_fusion else 3
                self.context_proj = MLP(dim_in * context_fusion_mult, dim_in,
                                        num_layers=self.head_layers,
                                        bias=True)
                self.context_head = MLP(dim_in, dim_out,
                                        num_layers=self.head_layers,
                                        bias=True)
                self.context_residual_alpha = nn.Parameter(
                    torch.full((1,), math.log(0.10 / 0.90))
                )
            if self.use_sequence_context_residual:
                self.sequence_len = 4
                self.outgoing_sequence_encoder = nn.GRU(
                    input_size=dim_in,
                    hidden_size=dim_in,
                    batch_first=True,
                )
                self.incoming_sequence_encoder = nn.GRU(
                    input_size=dim_in,
                    hidden_size=dim_in,
                    batch_first=True,
                )
                sequence_fusion_mult = 4 if self.use_difference_fusion else 3
                self.sequence_proj = MLP(dim_in * sequence_fusion_mult, dim_in,
                                         num_layers=self.head_layers,
                                         bias=True)
                self.sequence_head = MLP(dim_in, dim_out,
                                         num_layers=self.head_layers,
                                         bias=True)
                self.sequence_residual_alpha = nn.Parameter(
                    torch.full((1,), math.log(0.10 / 0.90))
                )
                if self.use_target_sequence_select:
                    self.outgoing_sequence_select_score = MLP(
                        dim_in * 3, 1,
                        num_layers=self.head_layers,
                        bias=True,
                    )
                    self.incoming_sequence_select_score = MLP(
                        dim_in * 3, 1,
                        num_layers=self.head_layers,
                        bias=True,
                    )
                    self.sequence_select_alpha = nn.Parameter(
                        torch.full((1,), math.log(0.30 / 0.70))
                    )
                self.sequence_time_scale = nn.Parameter(torch.tensor(86400.0))
                if self.use_sequence_bridge_bank_window:
                    self.fast_time_scale_log = nn.Parameter(torch.tensor(math.log(0.35)))
                    self.slow_time_scale_log = nn.Parameter(torch.tensor(math.log(3.0)))
                    self.sequence_window_gate = nn.Linear(dim_in * 3, dim_in)
                    self.sequence_window_alpha = nn.Parameter(
                        torch.full((1,), math.log(0.10 / 0.90))
                    )
                if self.use_sequence_bridge_bank:
                    self.bridge_partner_proj = MLP(dim_in * 2 + 2, dim_in,
                                                   num_layers=self.head_layers,
                                                   bias=True)
                    bridge_fusion_mult = 4 if self.use_difference_fusion else 3
                    self.bridge_bank_proj = MLP(dim_in * bridge_fusion_mult, dim_in,
                                                num_layers=self.head_layers,
                                                bias=True)
                    self.bridge_bank_gate = nn.Linear(dim_in * 3, dim_in)
                    self.bridge_bank_update = MLP(dim_in * 3, dim_in,
                                                  num_layers=self.head_layers,
                                                  bias=True)
                    self.bridge_bank_alpha = nn.Parameter(
                        torch.full((1,), math.log(0.10 / 0.90))
                    )
                    if self.use_terminal_role_flow:
                        self.terminal_role_proj = MLP(
                            dim_in * bridge_fusion_mult, dim_in,
                            num_layers=self.head_layers,
                            bias=True,
                        )
                        self.terminal_flow_proj = MLP(
                            dim_in * bridge_fusion_mult + 6, dim_in,
                            num_layers=self.head_layers,
                            bias=True,
                        )
                        self.terminal_flow_head = MLP(
                            dim_in, dim_out,
                            num_layers=self.head_layers,
                            bias=True,
                        )
                        self.terminal_flow_residual_alpha = nn.Parameter(
                            torch.full((1,), math.log(0.08 / 0.92))
                        )
                        if self.use_boundary_lag_flow:
                            self.boundary_lag_slot_proj = MLP(
                                dim_in * bridge_fusion_mult + 2, dim_in,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.boundary_lag_proj = MLP(
                                dim_in * 2 + 2, dim_in,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.boundary_lag_head = MLP(
                                dim_in, dim_out,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.boundary_lag_residual_alpha = nn.Parameter(
                                torch.full((1,), math.log(0.05 / 0.95))
                            )
                    if self.use_support_conditioned_mixture:
                        self.support_feature_dim = (
                            18 if self.use_sequence_consistency_filter else 14
                        )
                        if self.use_dot_fallback_support_mixture:
                            self.dot_fallback_proj = MLP(
                                dim_in, dim_in,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                        else:
                            self.edge_fallback_head = MLP(
                                dim_in, dim_out,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                        self.structure_mix_gate = MLP(
                            dim_in + self.support_feature_dim, 1,
                            num_layers=self.head_layers,
                            bias=True,
                        )
                        self.structure_mix_bias = nn.Parameter(
                            torch.tensor(math.log(0.20 / 0.80))
                        )
                        self.sequence_support_gate = MLP(
                            dim_in + self.support_feature_dim, 1,
                            num_layers=self.head_layers,
                            bias=True,
                        )
                        self.sequence_support_bias = nn.Parameter(
                            torch.tensor(math.log(0.20 / 0.80))
                        )
                        self.terminal_support_gate = MLP(
                            dim_in + self.support_feature_dim, 1,
                            num_layers=self.head_layers,
                            bias=True,
                        )
                        self.terminal_support_bias = nn.Parameter(
                            torch.tensor(math.log(0.15 / 0.85))
                        )
                        self.boundary_support_gate = MLP(
                            dim_in + self.support_feature_dim, 1,
                            num_layers=self.head_layers,
                            bias=True,
                        )
                        self.boundary_support_bias = nn.Parameter(
                            torch.tensor(math.log(0.10 / 0.90))
                        )
                        if self.use_flow_sketch_expert:
                            self.num_flow_sketch_slots = max(
                                int(cfg.gt.flow_sketch_slots), 1
                            )
                            self.support_flow_sketch_slots = nn.Parameter(
                                torch.randn(
                                    self.num_flow_sketch_slots, dim_in
                                ) / math.sqrt(dim_in)
                            )
                            self.support_flow_sketch_assign = MLP(
                                dim_in + self.support_feature_dim,
                                self.num_flow_sketch_slots,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_flow_sketch_candidate_gate = MLP(
                                dim_in + self.support_feature_dim,
                                4,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_flow_sketch_fuse = MLP(
                                dim_in * 4 + self.support_feature_dim, dim_in,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_flow_sketch_gate = MLP(
                                dim_in + self.support_feature_dim + 5, 1,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_flow_sketch_bias = nn.Parameter(
                                torch.tensor(math.log(0.08 / 0.92))
                            )
                            self.support_flow_sketch_head = MLP(
                                dim_in, dim_out,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_flow_sketch_alpha = nn.Parameter(
                                torch.full((1,), math.log(0.06 / 0.94))
                            )
                            if self.use_flow_sketch_score_calibration:
                                self.support_flow_scorecalib_fuse = MLP(
                                    dim_in * 4 + self.support_feature_dim + 10, dim_in,
                                    num_layers=self.head_layers,
                                    bias=True,
                                )
                                self.support_flow_scorecalib_gate = MLP(
                                    dim_in + self.support_feature_dim + 10, 1,
                                    num_layers=self.head_layers,
                                    bias=True,
                                )
                                self.support_flow_scorecalib_bias = nn.Parameter(
                                    torch.tensor(math.log(0.10 / 0.90))
                                )
                                self.support_flow_scorecalib_head = MLP(
                                    dim_in, 2,
                                    num_layers=self.head_layers,
                                    bias=True,
                                )
                                self.support_flow_scorecalib_alpha = nn.Parameter(
                                    torch.full((1,), math.log(0.04 / 0.96))
                                )
                        if self.use_sequence_consistency_filter:
                            consistency_input_dim = dim_in * 3 + 4
                            self.outgoing_consistency_gate = MLP(
                                consistency_input_dim, 1,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.incoming_consistency_gate = MLP(
                                consistency_input_dim, 1,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.sequence_consistency_alpha = nn.Parameter(
                                torch.tensor(math.log(0.35 / 0.65))
                            )
                        if self.use_support_prototype_expert:
                            self.num_support_prototypes = 8
                            self.support_proto_tokens = nn.Parameter(
                                torch.randn(self.num_support_prototypes, dim_in) /
                                math.sqrt(dim_in)
                            )
                            self.support_proto_assign = MLP(
                                dim_in + self.support_feature_dim,
                                self.num_support_prototypes,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_proto_fuse = MLP(
                                dim_in * 3 + self.support_feature_dim, dim_in,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_proto_gate = MLP(
                                dim_in + self.support_feature_dim + 2, 1,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_proto_bias = nn.Parameter(
                                torch.tensor(math.log(0.12 / 0.88))
                            )
                            self.support_proto_head = MLP(
                                dim_in, dim_out,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_proto_alpha = nn.Parameter(
                                torch.full((1,), math.log(0.08 / 0.92))
                            )
                        if self.use_support_class_prototype_expert:
                            self.num_class_prototypes = max(dim_out, 2)
                            self.num_class_proto_slots = (
                                4 if self.use_support_class_mixture_prototype_expert else 1
                            )
                            self.register_buffer(
                                'support_class_proto_bank',
                                torch.zeros(
                                    self.num_class_prototypes,
                                    self.num_class_proto_slots,
                                    dim_in,
                                ),
                            )
                            self.register_buffer(
                                'support_class_proto_ready',
                                torch.zeros(
                                    self.num_class_prototypes,
                                    self.num_class_proto_slots,
                                ),
                            )
                            self.support_class_proto_momentum = nn.Parameter(
                                torch.tensor(math.log(0.95 / 0.05))
                            )
                            self.support_class_proto_temperature = nn.Parameter(
                                torch.tensor(0.0)
                            )
                            self.support_class_proto_fuse = MLP(
                                dim_in * 3 + self.support_feature_dim + 8, dim_in,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_class_proto_gate = MLP(
                                dim_in + self.support_feature_dim + 8, 1,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_class_proto_bias = nn.Parameter(
                                torch.tensor(math.log(0.10 / 0.90))
                            )
                            self.support_class_proto_head = MLP(
                                dim_in, dim_out,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_class_proto_alpha = nn.Parameter(
                                torch.full((1,), math.log(0.08 / 0.92))
                            )
                        if self.use_support_class_split_expert:
                            self.support_pos_class_split_fuse = MLP(
                                dim_in * 4 + self.support_feature_dim + 8, dim_in,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_neg_class_split_fuse = MLP(
                                dim_in * 4 + self.support_feature_dim + 8, dim_in,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_class_split_gate = MLP(
                                dim_in + self.support_feature_dim + 10, 1,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_class_split_bias = nn.Parameter(
                                torch.tensor(0.0)
                            )
                            self.support_class_split_head = MLP(
                                dim_in, dim_out,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_class_split_alpha = nn.Parameter(
                                torch.full((1,), math.log(0.04 / 0.96))
                            )
                        if self.use_support_class_split_subgraph_route_expert:
                            self.support_class_split_subgraph_route_fuse = MLP(
                                dim_in * 4 + self.support_feature_dim + 19, dim_in,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_class_split_subgraph_route_gate = MLP(
                                dim_in + self.support_feature_dim + 19, 1,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_class_split_subgraph_route_bias = nn.Parameter(
                                torch.tensor(math.log(0.58 / 0.42))
                            )
                            self.support_class_split_subgraph_route_head = MLP(
                                dim_in, dim_out,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_class_split_subgraph_route_alpha = nn.Parameter(
                                torch.full((1,), math.log(0.04 / 0.96))
                            )
                            if self.use_support_class_split_subgraph_margin_calibration:
                                self.support_class_split_subgraph_margin_calib = MLP(
                                    dim_in + self.support_feature_dim + 6 + (
                                        4 if self.use_support_class_split_subgraph_margin_pred_calibration else 0
                                    ), 2,
                                    num_layers=self.head_layers,
                                    bias=True,
                                )
                                self.support_class_split_subgraph_margin_alpha = nn.Parameter(
                                    torch.full((1,), math.log(
                                        (0.08 if self.use_support_class_split_subgraph_margin_pred_calibration else 0.05) /
                                        (0.92 if self.use_support_class_split_subgraph_margin_pred_calibration else 0.95)
                                    ))
                                )
                            if self.use_support_class_split_subgraph_asym_calibration:
                                self.support_class_split_subgraph_asym_calib = MLP(
                                    dim_in + self.support_feature_dim + 10, 4,
                                    num_layers=self.head_layers,
                                    bias=True,
                                )
                                self.support_class_split_subgraph_asym_alpha = nn.Parameter(
                                    torch.full((1,), math.log(0.06 / 0.94))
                                )
                        if self.use_support_proto_consensus_expert:
                            self.support_proto_consensus_fuse = MLP(
                                dim_in * 4 + self.support_feature_dim + 14, dim_in,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_proto_consensus_gate = MLP(
                                dim_in + self.support_feature_dim + 14, 1,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_proto_consensus_bias = nn.Parameter(
                                torch.tensor(math.log(0.08 / 0.92))
                            )
                            self.support_proto_consensus_head = MLP(
                                dim_in, dim_out,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_proto_consensus_alpha = nn.Parameter(
                                torch.full((1,), math.log(0.06 / 0.94))
                            )
                        if self.use_support_proto_disagreement_expert:
                            self.support_proto_disagreement_fuse = MLP(
                                dim_in * 4 + self.support_feature_dim + 14, dim_in,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_proto_disagreement_gate = MLP(
                                dim_in + self.support_feature_dim + 14, 1,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_proto_disagreement_bias = nn.Parameter(
                                torch.tensor(math.log(0.06 / 0.94))
                            )
                            self.support_proto_disagreement_head = MLP(
                                dim_in, dim_out,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_proto_disagreement_alpha = nn.Parameter(
                                torch.full((1,), math.log(0.04 / 0.96))
                            )
                        if self.use_support_confidence_dual_expert:
                            self.support_confidence_dual_fuse = MLP(
                                dim_in * 5 + self.support_feature_dim + 12, dim_in,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_confidence_dual_gate = MLP(
                                dim_in + self.support_feature_dim + 12, 1,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_confidence_dual_bias = nn.Parameter(
                                torch.tensor(math.log(0.05 / 0.95))
                            )
                            self.support_confidence_dual_head = MLP(
                                dim_in, dim_out,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_confidence_dual_alpha = nn.Parameter(
                                torch.full((1,), math.log(0.04 / 0.96))
                            )
                        if self.use_support_easyhard_dual_expert:
                            self.support_easy_dual_fuse = MLP(
                                dim_in * 5 + self.support_feature_dim + 12, dim_in,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_hard_dual_fuse = MLP(
                                dim_in * 5 + self.support_feature_dim + 12, dim_in,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_easyhard_dual_gate = MLP(
                                dim_in + self.support_feature_dim + 12, 1,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_easyhard_dual_bias = nn.Parameter(
                                torch.tensor(math.log(0.60 / 0.40))
                            )
                            self.support_easyhard_dual_head = MLP(
                                dim_in, dim_out,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_easyhard_dual_alpha = nn.Parameter(
                                torch.full((1,), math.log(0.04 / 0.96))
                            )
                        if self.use_support_scale_route_expert:
                            self.support_scale_route_fuse = MLP(
                                dim_in * 4 + self.support_feature_dim + 10, dim_in,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_scale_route_gate = MLP(
                                dim_in + self.support_feature_dim + 10, 1,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_scale_route_bias = nn.Parameter(
                                torch.tensor(math.log(0.55 / 0.45))
                            )
                            self.support_scale_route_head = MLP(
                                dim_in, dim_out,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_scale_route_alpha = nn.Parameter(
                                torch.full((1,), math.log(0.04 / 0.96))
                            )
                        if self.use_support_class_route_expert:
                            self.support_pos_class_route_fuse = MLP(
                                dim_in * 4 + self.support_feature_dim + 16, dim_in,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_neg_class_route_fuse = MLP(
                                dim_in * 4 + self.support_feature_dim + 16, dim_in,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_class_route_fuse = MLP(
                                dim_in * 4 + self.support_feature_dim + 18, dim_in,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_class_route_gate = MLP(
                                dim_in + self.support_feature_dim + 18, 1,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_class_route_bias = nn.Parameter(
                                torch.tensor(math.log(0.58 / 0.42))
                            )
                            self.support_class_route_head = MLP(
                                dim_in, dim_out,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_class_route_alpha = nn.Parameter(
                                torch.full((1,), math.log(0.04 / 0.96))
                            )
                        if self.use_support_subgraph_route_expert:
                            self.support_subgraph_route_fuse = MLP(
                                dim_in * 4 + self.support_feature_dim + 16, dim_in,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_subgraph_route_gate = MLP(
                                dim_in + self.support_feature_dim + 16, 1,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_subgraph_route_bias = nn.Parameter(
                                torch.tensor(math.log(0.45 / 0.55))
                            )
                            self.support_subgraph_route_head = MLP(
                                dim_in, dim_out,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            subgraph_route_alpha = (
                                0.02
                                if self.use_support_class_split_subgraph_dual_mix_route_expert
                                or self.use_support_class_split_subgraph_dual_mix_uncert_weak_expert
                                or self.use_support_class_split_subgraph_dual_mix_uncert_late_expert
                                else (
                                    0.06
                                    if self.use_support_class_split_subgraph_dual_mix_pred_gate_expert
                                    or self.use_support_class_split_subgraph_dual_mix_uncert_gate_expert
                                    or self.use_support_class_split_subgraph_dual_mix_uncert_tight_expert
                                    or self.use_support_class_split_subgraph_dual_mix_disagree_gate_expert
                                    or self.use_support_class_split_subgraph_dual_resmix_expert
                                    else 0.04
                                )
                            )
                            self.support_subgraph_route_alpha = nn.Parameter(
                                torch.full(
                                    (1,),
                                    math.log(
                                        subgraph_route_alpha /
                                        (1.0 - subgraph_route_alpha)
                                    ),
                                )
                            )
                            if self.use_support_class_split_subgraph_dual_mix_pred_gate_expert:
                                self.support_subgraph_dual_pred_gate_center = (
                                    nn.Parameter(torch.tensor(0.18))
                                )
                                self.support_subgraph_dual_pred_gate_scale = (
                                    nn.Parameter(torch.tensor(6.0))
                                )
                            if (
                                self.use_support_class_split_subgraph_dual_mix_uncert_gate_expert
                                or self.use_support_class_split_subgraph_dual_mix_uncert_weak_expert
                                or self.use_support_class_split_subgraph_dual_mix_uncert_late_expert
                                or self.use_support_class_split_subgraph_dual_mix_uncert_tight_expert
                                or self.use_support_class_split_subgraph_dual_mix_uncert_mid_expert
                            ):
                                self.support_subgraph_dual_uncert_gate_center = (
                                    nn.Parameter(
                                        torch.tensor(
                                            0.65
                                            if self.use_support_class_split_subgraph_dual_mix_uncert_late_expert
                                            else 0.50
                                            if self.use_support_class_split_subgraph_dual_mix_uncert_tight_expert
                                            else 0.55
                                        )
                                    )
                                )
                                self.support_subgraph_dual_uncert_gate_scale = (
                                    nn.Parameter(torch.tensor(8.0))
                                )
                            if self.use_support_class_split_subgraph_dual_mix_disagree_gate_expert:
                                self.support_subgraph_dual_disagree_gate_center = (
                                    nn.Parameter(torch.tensor(0.08))
                                )
                                self.support_subgraph_dual_disagree_gate_scale = (
                                    nn.Parameter(torch.tensor(8.0))
                                )
                            if self.use_support_class_split_subgraph_dual_resmix_expert:
                                self.support_class_split_subgraph_dual_resmix_gate = (
                                    MLP(
                                        dim_in * 2 + self.support_feature_dim + 10,
                                        1,
                                        num_layers=self.head_layers,
                                        bias=True,
                                    )
                                )
                        if self.use_support_proto_route_expert:
                            self.support_proto_route_fuse = MLP(
                                dim_in * 4 + self.support_feature_dim + 20, dim_in,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_proto_route_gate = MLP(
                                dim_in + self.support_feature_dim + 20, 1,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_proto_route_bias = nn.Parameter(
                                torch.tensor(0.0)
                            )
                            self.support_proto_route_head = MLP(
                                dim_in, dim_out,
                                num_layers=self.head_layers,
                                bias=True,
                            )
                            self.support_proto_route_alpha = nn.Parameter(
                                torch.full((1,), math.log(0.04 / 0.96))
                            )
                    if self.use_sequence_bridge_motif_lite:
                        self.bridge_leg_pair_proj = MLP(dim_in * 3 + 1, dim_in,
                                                        num_layers=self.head_layers,
                                                        bias=True)
                        self.bridge_leg_bank_proj = MLP(dim_in * 3, dim_in,
                                                        num_layers=self.head_layers,
                                                        bias=True)
                        self.bridge_leg_bank_gate = nn.Linear(dim_in * 3, dim_in)
                        self.bridge_leg_bank_update = MLP(dim_in * 3, dim_in,
                                                          num_layers=self.head_layers,
                                                          bias=True)
                        self.bridge_leg_bank_alpha = nn.Parameter(
                            torch.full((1,), math.log(0.08 / 0.92))
                        )
        else:
            self.layer_post_mp = MLP(dim_in * 3, dim_out,
                                     num_layers=self.head_layers,
                                     bias=True)
            if self.use_evidence_gate:
                self._build_evidence_gate(dim_in, dim_out)

    def _build_evidence_gate(self, dim_in, dim_out):
        '''Modules for the scale-agnostic, uncertainty-gated evidence decoder.

        A single forward path is used for every dataset (no dataset-size route
        switching). The decoder fuses three sources of evidence:
          (1) z_base  : the original FraudGT MLP edge decoder (`layer_post_mp`);
          (2) z_proto : a bounded class-prototype residual -- the stable
                        "core evidence" identified by the ablation study;
          (3) z_struct: higher-order 1-hop structural evidence whose
                        contribution is gated by the *per-sample uncertainty*
                        of the core decision, NOT by dataset size. On confident
                        samples the gate closes, which reproduces the
                        "fall back on large graphs" behaviour automatically and
                        per-example instead of via a hard size rule.
        '''
        # Shared edge representation used for prototypes / structure / gate.
        self.eg_repr = MLP(dim_in * 3, dim_in,
                           num_layers=self.head_layers, bias=True)

        # --- class-prototype core (reuses the proven prototype machinery) ---
        self.use_support_class_prototype_expert = True
        self.num_class_prototypes = max(dim_out, 2)
        self.num_class_proto_slots = (
            int(getattr(cfg.model, 'dmprd_num_slots', 4))
            if self.use_dmprd else 4)
        if self.num_class_proto_slots < 1:
            raise ValueError("dmprd_num_slots must be at least 1")
        self.register_buffer(
            'support_class_proto_bank',
            torch.zeros(self.num_class_prototypes,
                        self.num_class_proto_slots, dim_in))
        self.register_buffer(
            'support_class_proto_ready',
            torch.zeros(self.num_class_prototypes,
                        self.num_class_proto_slots))
        self.support_class_proto_momentum = nn.Parameter(
            torch.tensor(math.log(0.95 / 0.05)))
        self.support_class_proto_temperature = nn.Parameter(torch.tensor(0.0))
        # 8 scalar prototype-evidence features -> class-logit correction.
        self.eg_proto_head = MLP(8, dim_out,
                                 num_layers=self.head_layers, bias=True)
        self.eg_proto_alpha = nn.Parameter(
            torch.full((1,), math.log(0.10 / 0.90)))

        if self.use_dmprd:
            self.dmprd_use_distribution_stats = bool(getattr(
                cfg.model, 'dmprd_use_distribution_stats', True))
            self.dmprd_delta_max = float(getattr(
                cfg.model, 'dmprd_delta_max', 1.0))
            self.dmprd_beta_max = float(getattr(
                cfg.model, 'dmprd_beta_max', 1.0))
            if self.dmprd_delta_max <= 0.0:
                raise ValueError("dmprd_delta_max must be positive")
            if self.dmprd_beta_max <= 0.0:
                raise ValueError("dmprd_beta_max must be positive")
            self._dmprd_log_step = 0
            return

        # --- higher-order 1-hop structural evidence branch ---
        self.eg_struct_head = MLP(dim_in * 3, dim_out,
                                  num_layers=self.head_layers, bias=True)
        self.eg_struct_alpha = nn.Parameter(
            torch.full((1,), math.log(0.05 / 0.95)))

        # --- uncertainty gate that routes structural evidence per sample ---
        # Inputs: edge repr + [core uncertainty, |proto margin|, proto ready,
        # tanh(base margin)]. No dataset-size signal is used anywhere.
        self.eg_gate = MLP(dim_in + 4, 1,
                           num_layers=self.head_layers, bias=True)

        if self.eg_gate_v3:
            # v3 fixed gate. Small MLP fed ONLY routing scalars (no `h`) so it
            # cannot memorise a constant; a dedicated bias starts it shut; the
            # structural branch is fused convexly (no redundant global scale).
            # Routing signals: [uncertainty, |proto_margin|, ready, struct_support,
            #                   |tanh(base_margin)|] -> 5 scalars.
            self.eg_gate_v3_signals = 5
            self.eg_gate_v3_head = MLP(self.eg_gate_v3_signals, 1,
                                       num_layers=self.head_layers, bias=True)
            self.eg_gate_v3_bias = nn.Parameter(
                torch.tensor(math.log(0.05 / 0.95)))
            self.eg_gate_l1 = float(getattr(cfg.model, 'eg_gate_l1', 1e-3))
            self.eg_gate_warmup_epochs = int(
                getattr(cfg.model, 'eg_gate_warmup_epochs', 20))
            # Set from the train loop each epoch (default large so eval is never
            # treated as warm-up).
            self._eg_cur_epoch = 10 ** 9
            self._eg_gate_penalty = None

        if self.eg_gate_v4:
            # v4 routes a bounded structural *correction*, rather than replacing
            # the reliable core logits. The two additional routing signals tell
            # the gate how large the proposed correction is and whether it
            # agrees with the current core decision.
            self.eg_gate_v4_signals = 7
            self.eg_gate_v4_head = MLP(self.eg_gate_v4_signals, 1,
                                       num_layers=self.head_layers, bias=True)
            self.eg_gate_init_open = float(
                getattr(cfg.model, 'eg_gate_init_open', 0.10))
            self.eg_gate_budget_target = float(
                getattr(cfg.model, 'eg_gate_budget_target', 0.10))
            self.eg_gate_budget_weight = float(
                getattr(cfg.model, 'eg_gate_budget_weight', 1e-2))
            self.eg_struct_aux_weight = float(
                getattr(cfg.model, 'eg_struct_aux_weight', 0.25))
            self.eg_struct_aux_epochs = int(
                getattr(cfg.model, 'eg_struct_aux_epochs', 60))
            self.eg_struct_residual_scale = float(
                getattr(cfg.model, 'eg_struct_residual_scale', 1.0))
            self.eg_gate_warmup_epochs = int(
                getattr(cfg.model, 'eg_gate_warmup_epochs', 20))

            # Initialise the router as a constant, modestly-open gate without a
            # second additive bias parameter. This removes the v3 duplicate-bias
            # ambiguity and gives the structural expert a gradient from step 1.
            final_linear = None
            for module in self.eg_gate_v4_head.modules():
                if isinstance(module, nn.Linear):
                    final_linear = module
            if final_linear is None:
                raise RuntimeError("evidence_gate_v4 router has no linear output")
            nn.init.zeros_(final_linear.weight)
            nn.init.constant_(
                final_linear.bias,
                math.log(self.eg_gate_init_open /
                         (1.0 - self.eg_gate_init_open)))

            self._eg_cur_epoch = 10 ** 9
            self._eg_gate_penalty = None
            self._eg_struct_aux_logits = None
            self._eg_struct_aux_labels = None
            self._eg_struct_aux_scale = 0.0

    def _edge_mask(self, batch):
        task = cfg.dataset.task_entity
        return torch.isin(batch[task].e_id,
                          getattr(self, f'{batch.split}_inds')[batch[task].input_id])

    def _edge_inputs(self, batch):
        task = cfg.dataset.task_entity
        edge_index = batch[task].edge_index
        return torch.cat((batch[task[0]].x[edge_index[0]],
                          batch[task[2]].x[edge_index[1]],
                          batch[task].edge_attr), dim=-1), edge_index

    def _build_recent_sequence_bank(self, pair_repr, pair_nodes, pair_timestamps,
                                    num_nodes, latest_timestamps, time_scale=None):
        seq_bank = pair_repr.new_zeros((num_nodes, self.sequence_len, pair_repr.size(-1)))
        remaining_scores = pair_timestamps.clone()
        score_floor = torch.finfo(remaining_scores.dtype).min
        if time_scale is None:
            time_scale = self.sequence_time_scale.abs().clamp(min=1.0)
        remaining_mask = torch.ones_like(pair_timestamps, dtype=torch.bool)

        for slot in range(self.sequence_len):
            _, slot_indices = scatter_max(
                remaining_scores, pair_nodes, dim=0, dim_size=num_nodes
            )
            valid_nodes = scatter(
                remaining_mask.float(), pair_nodes, dim=0, dim_size=num_nodes, reduce='sum'
            ) > 0
            if not valid_nodes.any():
                break
            chosen_indices = slot_indices[valid_nodes]
            chosen_times = pair_timestamps[chosen_indices]
            chosen_repr = pair_repr[chosen_indices]
            recency = torch.exp(
                -(
                    latest_timestamps[valid_nodes] - chosen_times
                ).clamp(min=0) / time_scale
            ).unsqueeze(-1)
            seq_bank[valid_nodes, slot] = chosen_repr * recency
            remaining_scores[chosen_indices] = score_floor
            remaining_mask[chosen_indices] = False

        return seq_bank

    def _build_recent_pair_sequence_bank(self, edge_repr, pair_inv, edge_timestamps,
                                         num_pairs, latest_timestamps, time_scale=None):
        seq_bank = edge_repr.new_zeros((num_pairs, self.pair_sequence_len, edge_repr.size(-1)))
        remaining_scores = edge_timestamps.clone()
        score_floor = torch.finfo(remaining_scores.dtype).min
        if time_scale is None:
            time_scale = self.sequence_time_scale.abs().clamp(min=1.0)
        remaining_mask = torch.ones_like(edge_timestamps, dtype=torch.bool)

        for slot in range(self.pair_sequence_len):
            _, slot_indices = scatter_max(
                remaining_scores, pair_inv, dim=0, dim_size=num_pairs
            )
            valid_pairs = scatter(
                remaining_mask.float(), pair_inv, dim=0, dim_size=num_pairs, reduce='sum'
            ) > 0
            if not valid_pairs.any():
                break
            chosen_indices = slot_indices[valid_pairs]
            chosen_times = edge_timestamps[chosen_indices]
            chosen_repr = edge_repr[chosen_indices]
            recency = torch.exp(
                -(
                    latest_timestamps[valid_pairs] - chosen_times
                ).clamp(min=0) / time_scale
            ).unsqueeze(-1)
            seq_bank[valid_pairs, slot] = chosen_repr * recency
            remaining_scores[chosen_indices] = score_floor
            remaining_mask[chosen_indices] = False

        return seq_bank

    def _build_recent_partner_bank(self, pair_nodes, partner_nodes, pair_timestamps,
                                   num_nodes):
        partner_bank = pair_nodes.new_full((num_nodes, self.sequence_len), -1)
        remaining_scores = pair_timestamps.clone()
        score_floor = torch.finfo(remaining_scores.dtype).min
        remaining_mask = torch.ones_like(pair_timestamps, dtype=torch.bool)

        for slot in range(self.sequence_len):
            _, slot_indices = scatter_max(
                remaining_scores, pair_nodes, dim=0, dim_size=num_nodes
            )
            valid_nodes = scatter(
                remaining_mask.float(), pair_nodes, dim=0, dim_size=num_nodes, reduce='sum'
            ) > 0
            if not valid_nodes.any():
                break
            chosen_indices = slot_indices[valid_nodes]
            partner_bank[valid_nodes, slot] = partner_nodes[chosen_indices]
            remaining_scores[chosen_indices] = score_floor
            remaining_mask[chosen_indices] = False

        return partner_bank

    def _build_recent_timestamp_bank(self, pair_nodes, pair_timestamps, num_nodes):
        time_bank = pair_timestamps.new_zeros((num_nodes, self.sequence_len))
        remaining_scores = pair_timestamps.clone()
        score_floor = torch.finfo(remaining_scores.dtype).min
        remaining_mask = torch.ones_like(pair_timestamps, dtype=torch.bool)

        for slot in range(self.sequence_len):
            _, slot_indices = scatter_max(
                remaining_scores, pair_nodes, dim=0, dim_size=num_nodes
            )
            valid_nodes = scatter(
                remaining_mask.float(), pair_nodes, dim=0, dim_size=num_nodes, reduce='sum'
            ) > 0
            if not valid_nodes.any():
                break
            chosen_indices = slot_indices[valid_nodes]
            time_bank[valid_nodes, slot] = pair_timestamps[chosen_indices]
            remaining_scores[chosen_indices] = score_floor
            remaining_mask[chosen_indices] = False

        return time_bank

    def _filter_sequence_bank(self, sequence_bank, query_repr, scorer):
        query_bank = query_repr.unsqueeze(1).expand(-1, sequence_bank.size(1), -1)
        gate_input = torch.cat(
            (
                query_bank,
                sequence_bank,
                query_bank * sequence_bank,
            ),
            dim=-1,
        )
        gate = torch.sigmoid(
            scorer(gate_input.reshape(-1, gate_input.size(-1)))
        ).view(sequence_bank.size(0), sequence_bank.size(1), 1)
        alpha = torch.sigmoid(self.sequence_select_alpha)
        scale = 1.0 - alpha + alpha * gate
        valid = (sequence_bank.abs().sum(dim=-1, keepdim=True) > 0).float()
        return sequence_bank * valid * scale

    def _apply_signed_margin_calibration(self, pred, scale, bias):
        if pred.size(-1) == 1:
            return scale * pred + bias
        if pred.size(-1) != 2:
            return pred
        pred_center = pred.mean(dim=-1, keepdim=True)
        pred_margin = pred[:, 1:2] - pred[:, 0:1]
        pred_margin = scale * pred_margin + bias
        return pred_center + torch.cat(
            (-0.5 * pred_margin, 0.5 * pred_margin), dim=-1
        )

    def _consistency_filter_sequence_bank(self, sequence_bank, opposite_bank, query_repr, scorer):
        valid = (sequence_bank.abs().sum(dim=-1, keepdim=True) > 0).float()
        if opposite_bank is None:
            opposite_bank = torch.zeros_like(sequence_bank)
        opposite_valid = (opposite_bank.abs().sum(dim=-1, keepdim=True) > 0).float()

        bank_norm = F.normalize(sequence_bank, dim=-1, eps=1e-6)
        opposite_norm = F.normalize(opposite_bank, dim=-1, eps=1e-6)
        self_sim = 0.5 * (
            torch.matmul(bank_norm, bank_norm.transpose(1, 2)) + 1.0
        )
        cross_sim = 0.5 * (
            torch.matmul(bank_norm, opposite_norm.transpose(1, 2)) + 1.0
        )

        self_mask = valid * valid.transpose(1, 2)
        diag_mask = torch.eye(
            sequence_bank.size(1),
            device=sequence_bank.device,
            dtype=torch.bool,
        ).unsqueeze(0)
        self_mask = self_mask.masked_fill(diag_mask, 0.0)
        self_denom = self_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        self_consistency = (self_sim * self_mask).sum(dim=-1, keepdim=True) / self_denom

        cross_mask = valid * opposite_valid.transpose(1, 2)
        cross_score = cross_sim.masked_fill(cross_mask == 0, 0.0)
        cross_consistency = cross_score.max(dim=-1, keepdim=True).values

        query_bank = query_repr.unsqueeze(1).expand(-1, sequence_bank.size(1), -1)
        query_consistency = 0.5 * (
            F.cosine_similarity(query_bank, sequence_bank, dim=-1, eps=1e-6).unsqueeze(-1) + 1.0
        )
        slot_index = torch.arange(
            sequence_bank.size(1),
            device=sequence_bank.device,
            dtype=sequence_bank.dtype,
        )
        slot_prior = (1.0 / (1.0 + slot_index)).view(1, -1, 1)
        gate_input = torch.cat(
            (
                query_bank,
                sequence_bank,
                query_bank * sequence_bank,
                self_consistency,
                cross_consistency,
                query_consistency,
                slot_prior.expand(sequence_bank.size(0), -1, -1),
            ),
            dim=-1,
        )
        gate = torch.sigmoid(
            scorer(gate_input.reshape(-1, gate_input.size(-1)))
        ).view(sequence_bank.size(0), sequence_bank.size(1), 1)
        alpha = torch.sigmoid(self.sequence_consistency_alpha)
        scale = 1.0 - alpha + alpha * gate
        filtered_bank = sequence_bank * valid * scale
        mean_gate = (gate * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return filtered_bank, mean_gate

    def _update_support_class_prototypes(self, proto_repr, labels):
        if not self.use_support_class_prototype_expert:
            return
        if proto_repr is None or labels is None or proto_repr.numel() == 0:
            return
        with torch.no_grad():
            flat_labels = labels.view(-1).long()
            valid_mask = (flat_labels >= 0) & (flat_labels < self.num_class_prototypes)
            if not valid_mask.any():
                return
            normalized_repr = F.normalize(proto_repr.detach()[valid_mask], dim=-1, eps=1e-6)
            flat_labels = flat_labels[valid_mask]
            momentum = torch.sigmoid(self.support_class_proto_momentum.detach())
            for class_idx in range(self.num_class_prototypes):
                class_mask = flat_labels == class_idx
                if not class_mask.any():
                    continue
                class_repr = normalized_repr[class_mask]
                ready_mask = self.support_class_proto_ready[class_idx] > 0
                if not ready_mask.any():
                    fill_count = min(self.num_class_proto_slots, class_repr.size(0))
                    self.support_class_proto_bank[class_idx, :fill_count].copy_(
                        class_repr[:fill_count]
                    )
                    self.support_class_proto_ready[class_idx, :fill_count] = 1.0
                    class_repr = class_repr[fill_count:]
                    ready_mask = self.support_class_proto_ready[class_idx] > 0
                elif ready_mask.sum() < self.num_class_proto_slots:
                    ready_bank = F.normalize(
                        self.support_class_proto_bank[class_idx, ready_mask],
                        dim=-1,
                        eps=1e-6,
                    )
                    diversity_score = 1.0 - (
                        class_repr @ ready_bank.transpose(0, 1)
                    ).max(dim=-1).values
                    free_slots = (~ready_mask).nonzero(as_tuple=False).view(-1)
                    fill_order = diversity_score.argsort(descending=True)
                    used_mask = torch.zeros(
                        class_repr.size(0),
                        dtype=torch.bool,
                        device=class_repr.device,
                    )
                    for slot_idx, sample_idx in zip(free_slots.tolist(), fill_order.tolist()):
                        self.support_class_proto_bank[class_idx, slot_idx].copy_(
                            class_repr[sample_idx]
                        )
                        self.support_class_proto_ready[class_idx, slot_idx] = 1.0
                        used_mask[sample_idx] = True
                    class_repr = class_repr[~used_mask]
                    ready_mask = self.support_class_proto_ready[class_idx] > 0
                if class_repr.numel() == 0:
                    continue
                ready_slots = ready_mask.nonzero(as_tuple=False).view(-1)
                class_bank = F.normalize(
                    self.support_class_proto_bank[class_idx, ready_slots],
                    dim=-1,
                    eps=1e-6,
                )
                assign = (class_repr @ class_bank.transpose(0, 1)).max(dim=-1).indices
                for local_slot, slot_idx in enumerate(ready_slots.tolist()):
                    slot_mask = assign == local_slot
                    if not slot_mask.any():
                        continue
                    slot_mean = F.normalize(
                        class_repr[slot_mask].mean(dim=0),
                        dim=0,
                        eps=1e-6,
                    )
                    updated = (
                        momentum * self.support_class_proto_bank[class_idx, slot_idx] +
                        (1.0 - momentum) * slot_mean
                    )
                    updated = F.normalize(updated, dim=0, eps=1e-6)
                    self.support_class_proto_bank[class_idx, slot_idx].copy_(updated)

    def _support_class_proto_summary(self, query_repr, class_idx):
        zero_proto = torch.zeros_like(query_repr)
        zero_score = query_repr.new_zeros((query_repr.size(0), 1))
        ready_mask = self.support_class_proto_ready[class_idx] > 0
        ready_count = int(ready_mask.sum().item())
        if ready_count == 0:
            return zero_proto, zero_score, zero_score, zero_score, zero_score

        class_bank = F.normalize(
            self.support_class_proto_bank[class_idx, ready_mask],
            dim=-1,
            eps=1e-6,
        )
        query_norm = F.normalize(query_repr, dim=-1, eps=1e-6)
        temperature = self.support_class_proto_temperature.exp().clamp(min=0.25, max=4.0)
        logits = query_norm @ class_bank.transpose(0, 1)
        weights = F.softmax(logits / temperature, dim=-1)
        proto = F.normalize(weights @ class_bank, dim=-1, eps=1e-6)
        sim = 0.5 * (
            F.cosine_similarity(query_repr, proto, dim=-1, eps=1e-6).unsqueeze(-1) + 1.0
        )
        peak = weights.max(dim=-1, keepdim=True).values
        if ready_count > 1:
            entropy = -(
                weights * weights.clamp(min=1e-6).log()
            ).sum(dim=-1, keepdim=True) / math.log(float(ready_count))
            spread = 1.0 - entropy
        else:
            spread = zero_score + 1.0
        ready_score = zero_score + (
            float(ready_count) / float(self.num_class_proto_slots)
        )
        return proto, sim, peak, spread, ready_score

    def _support_class_proto_context(self, query_repr):
        zero_proto = torch.zeros_like(query_repr)
        zero_score = query_repr.new_zeros((query_repr.size(0), 1))
        if not self.use_support_class_prototype_expert:
            return (
                zero_proto, zero_proto, zero_score, zero_score, zero_score,
                zero_score, zero_score, zero_score, zero_score, zero_score,
            )
        if self.num_class_prototypes < 2:
            return (
                zero_proto, zero_proto, zero_score, zero_score, zero_score,
                zero_score, zero_score, zero_score, zero_score, zero_score,
            )

        pos_proto, pos_sim, pos_peak, pos_spread, pos_ready = (
            self._support_class_proto_summary(query_repr, 1)
        )
        neg_proto, neg_sim, neg_peak, neg_spread, neg_ready = (
            self._support_class_proto_summary(query_repr, 0)
        )
        ready = torch.minimum(pos_ready, neg_ready)
        if ready.max().item() == 0.0:
            return (
                zero_proto, zero_proto, zero_score, zero_score, zero_score,
                zero_score, zero_score, zero_score, zero_score, zero_score,
            )
        proto_margin = pos_sim - neg_sim
        return (
            pos_proto,
            neg_proto,
            pos_sim,
            neg_sim,
            proto_margin,
            ready,
            pos_peak,
            neg_peak,
            pos_spread,
            neg_spread,
        )

    def _cosine_feature(self, left_repr, right_repr):
        if left_repr is None or right_repr is None:
            return None
        left_norm = left_repr.norm(dim=-1, keepdim=True)
        right_norm = right_repr.norm(dim=-1, keepdim=True)
        valid = (left_norm > 0) & (right_norm > 0)
        cosine = F.cosine_similarity(left_repr, right_repr, dim=-1, eps=1e-6).unsqueeze(-1)
        cosine = 0.5 * (cosine + 1.0)
        return torch.where(valid, cosine, torch.zeros_like(cosine))

    def _partner_overlap_ratio(self, bank_a, bank_b):
        valid_a = bank_a >= 0
        valid_b = bank_b >= 0
        eq = (
            bank_a.unsqueeze(2) == bank_b.unsqueeze(1)
        ) & valid_a.unsqueeze(2) & valid_b.unsqueeze(1)
        match_any = eq.any(dim=2)
        overlap = (match_any.float() * valid_a.float()).sum(dim=1, keepdim=True)
        denom = valid_a.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        return overlap / denom

    def _boundary_support_features(self, src_time, dst_time):
        valid = (src_time > 0) & (dst_time > 0)
        valid_count = valid.float().sum(dim=1, keepdim=True)
        valid_ratio = valid_count / float(self.sequence_len)
        lag = dst_time - src_time
        after_ratio = (
            ((lag >= 0) & valid).float().sum(dim=1, keepdim=True) /
            valid_count.clamp(min=1.0)
        )
        time_scale = self.sequence_time_scale.abs().clamp(min=1.0)
        lag_align = torch.exp(-lag.abs() / time_scale) * valid.float()
        lag_align = lag_align.sum(dim=1, keepdim=True) / valid_count.clamp(min=1.0)
        return valid_ratio, after_ratio, lag_align

    def _recent_overlap_partner_repr(self, bank_a, bank_b, node_x):
        valid_a = bank_a >= 0
        valid_b = bank_b >= 0
        eq = (
            bank_a.unsqueeze(2) == bank_b.unsqueeze(1)
        ) & valid_a.unsqueeze(2) & valid_b.unsqueeze(1)
        match_any = eq.any(dim=2)
        overlap_count = (match_any.float() * valid_a.float()).sum(dim=1)
        slot_index = torch.arange(
            bank_a.size(1), device=bank_a.device, dtype=torch.float32
        )
        slot_weight = 1.0 / (1.0 + slot_index)
        weighted_overlap = (
            match_any.float() * slot_weight.unsqueeze(0)
        ).sum(dim=1)
        gather_ids = bank_a.clamp(min=0)
        matched_emb = node_x[gather_ids] * match_any.unsqueeze(-1).float()
        mean_emb = matched_emb.sum(dim=1) / overlap_count.unsqueeze(-1).clamp(min=1.0)
        max_emb = matched_emb.masked_fill(~match_any.unsqueeze(-1), -1e9).max(dim=1).values
        max_emb = torch.where(
            overlap_count.unsqueeze(-1) > 0,
            max_emb,
            torch.zeros_like(max_emb),
        )
        return self.bridge_partner_proj(torch.cat(
            (
                mean_emb,
                max_emb,
                overlap_count.unsqueeze(-1),
                weighted_overlap.unsqueeze(-1),
            ),
            dim=-1,
        ))

    def _recent_overlap_leg_repr(self, bank_a, bank_b, seq_a, seq_b, time_a, time_b):
        batch_size = bank_a.size(0)
        dim = seq_a.size(-1)
        device = seq_a.device
        acc_sum = seq_a.new_zeros((batch_size, dim))
        acc_weight = seq_a.new_zeros((batch_size, 1))
        max_token = seq_a.new_full((batch_size, dim), -1e9)
        has_match = torch.zeros(batch_size, dtype=torch.bool, device=device)
        time_scale = self.sequence_time_scale.abs().clamp(min=1.0)

        for i in range(self.sequence_len):
            a_ids = bank_a[:, i]
            a_valid = a_ids >= 0
            a_seq = seq_a[:, i]
            a_time = time_a[:, i]
            for j in range(self.sequence_len):
                b_ids = bank_b[:, j]
                match = a_valid & (b_ids >= 0) & (a_ids == b_ids)
                if not match.any():
                    continue
                b_seq = seq_b[:, j]
                b_time = time_b[:, j]
                align = torch.exp(
                    -(a_time - b_time).abs() / time_scale
                ).unsqueeze(-1)
                token = self.bridge_leg_pair_proj(torch.cat(
                    (
                        a_seq,
                        b_seq,
                        a_seq * b_seq,
                        align,
                    ),
                    dim=-1,
                ))
                masked_align = align * match.unsqueeze(-1).float()
                acc_sum = acc_sum + token * masked_align
                acc_weight = acc_weight + masked_align
                max_token = torch.where(
                    match.unsqueeze(-1),
                    torch.maximum(max_token, token),
                    max_token,
                )
                has_match = has_match | match

        mean_token = acc_sum / acc_weight.clamp(min=1e-6)
        max_token = torch.where(
            has_match.unsqueeze(-1),
            max_token,
            torch.zeros_like(max_token),
        )
        return self.bridge_leg_bank_proj(torch.cat(
            (
                mean_token,
                max_token,
                mean_token * max_token,
            ),
            dim=-1,
        ))

    def _recent_boundary_lag_repr(self, src_seq, dst_seq, src_time, dst_time):
        valid = (src_time > 0) & (dst_time > 0)
        time_scale = self.sequence_time_scale.abs().clamp(min=1.0)
        lag = (dst_time - src_time).unsqueeze(-1)
        lag_align = torch.exp(-lag.abs() / time_scale)
        after = (lag >= 0).float()
        slot_input = torch.cat(
            (
                self._pairwise_fusion_inputs(src_seq, dst_seq),
                lag_align,
                after,
            ),
            dim=-1,
        )
        slot_repr = self.boundary_lag_slot_proj(
            slot_input.reshape(-1, slot_input.size(-1))
        ).view(src_seq.size(0), src_seq.size(1), -1)
        slot_repr = slot_repr * valid.unsqueeze(-1).float()
        count = valid.float().sum(dim=1, keepdim=True)
        mean_repr = slot_repr.sum(dim=1) / count.clamp(min=1.0)
        max_repr = slot_repr.masked_fill(~valid.unsqueeze(-1), -1e9).max(dim=1).values
        max_repr = torch.where(
            count > 0,
            max_repr,
            torch.zeros_like(max_repr),
        )
        after_ratio = (
            after.squeeze(-1) * valid.float()
        ).sum(dim=1, keepdim=True) / count.clamp(min=1.0)
        return self.boundary_lag_proj(torch.cat(
            (
                mean_repr,
                max_repr,
                count / float(self.sequence_len),
                after_ratio,
            ),
            dim=-1,
        ))

    def _pairwise_fusion_inputs(self, left_repr, right_repr):
        if self.use_difference_fusion:
            return torch.cat(
                (
                    left_repr,
                    right_repr,
                    left_repr * right_repr,
                    torch.abs(left_repr - right_repr),
                ),
                dim=-1,
            )
        return torch.cat(
            (
                left_repr,
                right_repr,
                left_repr * right_repr,
            ),
            dim=-1,
        )

    def _pair_chain_head(self, batch):
        task = cfg.dataset.task_entity
        mask = self._edge_mask(batch)
        edge_inputs, edge_index = self._edge_inputs(batch)
        src_nodes, dst_nodes = edge_index
        edge_repr = self.edge_proj(edge_inputs)

        num_dst_nodes = batch[task[2]].x.size(0)
        pair_key = src_nodes.to(torch.long) * num_dst_nodes + dst_nodes.to(torch.long)
        pair_keys, pair_inv = torch.unique(pair_key, sorted=True, return_inverse=True)
        num_pairs = pair_keys.numel()
        pair_count = scatter(
            torch.ones_like(pair_inv, dtype=edge_repr.dtype),
            pair_inv,
            dim=0,
            dim_size=num_pairs,
            reduce='sum',
        ).unsqueeze(-1)

        pair_mean = scatter(edge_repr, pair_inv, dim=0, dim_size=num_pairs, reduce='mean')
        pair_max, _ = scatter_max(edge_repr, pair_inv, dim=0, dim_size=num_pairs)
        pair_max = torch.where(torch.isfinite(pair_max), pair_max, torch.zeros_like(pair_max))
        edge_local_repr = edge_repr
        pair_seq_state = None
        fast_pair_seq_state = None
        slow_pair_seq_state = None
        if self.use_pair_internal_sequence and hasattr(batch[task], 'timestamps'):
            edge_timestamps = batch[task].timestamps.to(edge_repr.device).float().view(-1)
            pair_latest, _ = scatter_max(
                edge_timestamps, pair_inv, dim=0, dim_size=num_pairs
            )
            pair_latest = torch.where(
                torch.isfinite(pair_latest),
                pair_latest,
                torch.zeros_like(pair_latest),
            )
            if self.use_sequence_bridge_bank_window:
                base_time_scale = self.sequence_time_scale.abs().clamp(min=1.0)
                fast_time_scale = base_time_scale * self.fast_time_scale_log.exp().clamp(min=0.1, max=10.0)
                slow_time_scale = base_time_scale * self.slow_time_scale_log.exp().clamp(min=0.25, max=20.0)
                fast_pair_sequence_bank = self._build_recent_pair_sequence_bank(
                    edge_repr, pair_inv, edge_timestamps, num_pairs, pair_latest, fast_time_scale
                )
                slow_pair_sequence_bank = self._build_recent_pair_sequence_bank(
                    edge_repr, pair_inv, edge_timestamps, num_pairs, pair_latest, slow_time_scale
                )
                fast_pair_seq_state = self.pair_edge_sequence_encoder(
                    fast_pair_sequence_bank
                )[1].squeeze(0)
                slow_pair_seq_state = self.pair_edge_sequence_encoder(
                    slow_pair_sequence_bank
                )[1].squeeze(0)
                pair_window_input = torch.cat(
                    (
                        fast_pair_seq_state,
                        slow_pair_seq_state,
                        fast_pair_seq_state * slow_pair_seq_state,
                    ),
                    dim=-1,
                )
                pair_window_gate = torch.sigmoid(self.pair_window_gate(pair_window_input))
                pair_seq_state = fast_pair_seq_state + (
                    torch.sigmoid(self.pair_window_alpha) *
                    pair_window_gate *
                    (slow_pair_seq_state - fast_pair_seq_state)
                )
            else:
                pair_sequence_bank = self._build_recent_pair_sequence_bank(
                    edge_repr, pair_inv, edge_timestamps, num_pairs, pair_latest
                )
                pair_seq_state = self.pair_edge_sequence_encoder(
                    pair_sequence_bank
                )[1].squeeze(0)
        if self.use_pair_internal_sequence:
            if pair_seq_state is None:
                pair_seq_state = torch.zeros_like(pair_mean)
            pair_repr = self.pair_proj(torch.cat(
                (pair_mean, pair_max, pair_max - pair_mean, pair_seq_state), dim=-1
            ))
        else:
            pair_repr = self.pair_proj(torch.cat(
                (pair_mean, pair_max, pair_max - pair_mean), dim=-1
            ))
        pair_context_repr = None
        pair_sequence_repr = None
        fast_sequence_repr = None
        slow_sequence_repr = None
        pair_terminal_role_repr = None
        pair_boundary_lag_repr = None
        pair_structure_mix = None
        pair_sequence_support = None
        pair_terminal_support = None
        pair_boundary_support = None
        pair_flow_sketch_repr = None
        pair_flow_sketch_support = None
        pair_flow_scorecalib_repr = None
        pair_flow_scorecalib_support = None
        pair_proto_repr = None
        pair_proto_support = None
        pair_class_proto_repr = None
        pair_class_proto_support = None
        pair_class_split_repr = None
        pair_class_split_support = None
        pair_proto_consensus_repr = None
        pair_proto_consensus_support = None
        pair_proto_disagreement_repr = None
        pair_proto_disagreement_support = None
        pair_confidence_dual_repr = None
        pair_confidence_dual_support = None
        pair_easyhard_dual_repr = None
        pair_easyhard_dual_support = None
        pair_scale_route_repr = None
        pair_scale_route_support = None
        pair_class_route_repr = None
        pair_class_route_support = None
        pair_subgraph_route_repr = None
        pair_subgraph_route_support = None
        pair_proto_route_repr = None
        pair_proto_route_support = None
        pair_class_split_subgraph_route_repr = None
        pair_class_split_subgraph_route_support = None
        pair_support_features = None

        if task[0] == task[2]:
            num_nodes = batch[task[0]].x.size(0)
            pair_src = torch.div(pair_keys, num_dst_nodes, rounding_mode='floor')
            pair_dst = torch.remainder(pair_keys, num_dst_nodes)
            zero_support = pair_repr.new_zeros((num_pairs, 1))
            src_out_cov = zero_support
            dst_in_cov = zero_support
            src_in_cov = zero_support
            dst_out_cov = zero_support
            outgoing_consistency = zero_support
            incoming_consistency = zero_support
            src_role_consistency = zero_support
            dst_role_consistency = zero_support
            forward_overlap = zero_support
            cycle_overlap = zero_support
            boundary_valid_ratio = zero_support
            boundary_after_ratio = zero_support
            proto_confidence = zero_support
            proto_match = zero_support
            pos_sim = zero_support
            neg_sim = zero_support
            proto_margin = zero_support
            proto_ready = zero_support
            pos_peak = zero_support
            neg_peak = zero_support
            pos_spread = zero_support
            neg_spread = zero_support
            proto_consensus = zero_support
            proto_confidence_gap = zero_support
            predecessor_bank = scatter(pair_repr, pair_dst, dim=0, dim_size=num_nodes, reduce='mean')
            successor_bank = scatter(pair_repr, pair_src, dim=0, dim_size=num_nodes, reduce='mean')
            prev_context = predecessor_bank[pair_src]
            next_context = successor_bank[pair_dst]
            chain_input = torch.cat((prev_context, pair_repr, next_context), dim=-1)
            chain_gate = torch.sigmoid(self.chain_gate(chain_input))
            pair_repr = pair_repr + (
                torch.sigmoid(self.chain_residual_alpha) *
                chain_gate *
                self.chain_update(chain_input)
            )
            if self.use_chain_context_residual:
                pair_scores = pair_repr.norm(dim=-1)
                predecessor_focus_weights = pyg_softmax(
                    pair_scores, pair_dst, num_nodes=num_nodes
                )
                successor_focus_weights = pyg_softmax(
                    pair_scores, pair_src, num_nodes=num_nodes
                )
                predecessor_focus_bank = scatter(
                    pair_repr * predecessor_focus_weights.unsqueeze(-1),
                    pair_dst,
                    dim=0,
                    dim_size=num_nodes,
                    reduce='sum'
                )
                successor_focus_bank = scatter(
                    pair_repr * successor_focus_weights.unsqueeze(-1),
                    pair_src,
                    dim=0,
                    dim_size=num_nodes,
                    reduce='sum'
                )
                pair_context_repr = self.context_proj(
                    self._pairwise_fusion_inputs(
                        predecessor_focus_bank[pair_src],
                        successor_focus_bank[pair_dst],
                    )
                )
            if self.use_sequence_context_residual and hasattr(batch[task], 'timestamps'):
                edge_timestamps = batch[task].timestamps.to(edge_repr.device).float().view(-1)
                pair_timestamps, _ = scatter_max(
                    edge_timestamps, pair_inv, dim=0, dim_size=num_pairs
                )
                pair_timestamps = torch.where(
                    torch.isfinite(pair_timestamps),
                    pair_timestamps,
                    torch.zeros_like(pair_timestamps),
                )
                outgoing_latest, _ = scatter_max(
                    pair_timestamps, pair_src, dim=0, dim_size=num_nodes
                )
                outgoing_latest = torch.where(
                    torch.isfinite(outgoing_latest),
                    outgoing_latest,
                    torch.zeros_like(outgoing_latest),
                )
                incoming_latest, _ = scatter_max(
                    pair_timestamps, pair_dst, dim=0, dim_size=num_nodes
                )
                incoming_latest = torch.where(
                    torch.isfinite(incoming_latest),
                    incoming_latest,
                    torch.zeros_like(incoming_latest),
                )
                if self.use_sequence_bridge_bank_window:
                    base_time_scale = self.sequence_time_scale.abs().clamp(min=1.0)
                    fast_time_scale = base_time_scale * self.fast_time_scale_log.exp().clamp(min=0.1, max=10.0)
                    slow_time_scale = base_time_scale * self.slow_time_scale_log.exp().clamp(min=0.25, max=20.0)
                    outgoing_sequence_bank = self._build_recent_sequence_bank(
                        pair_repr, pair_src, pair_timestamps, num_nodes, outgoing_latest, fast_time_scale
                    )
                    incoming_sequence_bank = self._build_recent_sequence_bank(
                        pair_repr, pair_dst, pair_timestamps, num_nodes, incoming_latest, fast_time_scale
                    )
                    slow_outgoing_sequence_bank = self._build_recent_sequence_bank(
                        pair_repr, pair_src, pair_timestamps, num_nodes, outgoing_latest, slow_time_scale
                    )
                    slow_incoming_sequence_bank = self._build_recent_sequence_bank(
                        pair_repr, pair_dst, pair_timestamps, num_nodes, incoming_latest, slow_time_scale
                    )
                    pair_outgoing_sequence_bank = outgoing_sequence_bank[pair_src]
                    pair_incoming_sequence_bank = incoming_sequence_bank[pair_dst]
                    pair_slow_outgoing_sequence_bank = slow_outgoing_sequence_bank[pair_src]
                    pair_slow_incoming_sequence_bank = slow_incoming_sequence_bank[pair_dst]
                    if self.use_target_sequence_select:
                        pair_outgoing_sequence_bank = self._filter_sequence_bank(
                            pair_outgoing_sequence_bank,
                            pair_repr,
                            self.outgoing_sequence_select_score,
                        )
                        pair_incoming_sequence_bank = self._filter_sequence_bank(
                            pair_incoming_sequence_bank,
                            pair_repr,
                            self.incoming_sequence_select_score,
                        )
                        pair_slow_outgoing_sequence_bank = self._filter_sequence_bank(
                            pair_slow_outgoing_sequence_bank,
                            pair_repr,
                            self.outgoing_sequence_select_score,
                        )
                        pair_slow_incoming_sequence_bank = self._filter_sequence_bank(
                            pair_slow_incoming_sequence_bank,
                            pair_repr,
                            self.incoming_sequence_select_score,
                        )
                    if self.use_sequence_consistency_filter:
                        pair_outgoing_sequence_bank, outgoing_consistency = (
                            self._consistency_filter_sequence_bank(
                                pair_outgoing_sequence_bank,
                                pair_incoming_sequence_bank,
                                pair_repr,
                                self.outgoing_consistency_gate,
                            )
                        )
                        pair_incoming_sequence_bank, incoming_consistency = (
                            self._consistency_filter_sequence_bank(
                                pair_incoming_sequence_bank,
                                pair_outgoing_sequence_bank,
                                pair_repr,
                                self.incoming_consistency_gate,
                            )
                        )
                        pair_slow_outgoing_sequence_bank, slow_outgoing_consistency = (
                            self._consistency_filter_sequence_bank(
                                pair_slow_outgoing_sequence_bank,
                                pair_slow_incoming_sequence_bank,
                                pair_repr,
                                self.outgoing_consistency_gate,
                            )
                        )
                        pair_slow_incoming_sequence_bank, slow_incoming_consistency = (
                            self._consistency_filter_sequence_bank(
                                pair_slow_incoming_sequence_bank,
                                pair_slow_outgoing_sequence_bank,
                                pair_repr,
                                self.incoming_consistency_gate,
                            )
                        )
                        outgoing_consistency = 0.5 * (
                            outgoing_consistency + slow_outgoing_consistency
                        )
                        incoming_consistency = 0.5 * (
                            incoming_consistency + slow_incoming_consistency
                        )
                    outgoing_state = self.outgoing_sequence_encoder(
                        pair_outgoing_sequence_bank
                    )[1].squeeze(0)
                    incoming_state = self.incoming_sequence_encoder(
                        pair_incoming_sequence_bank
                    )[1].squeeze(0)
                    slow_outgoing_state = self.outgoing_sequence_encoder(
                        pair_slow_outgoing_sequence_bank
                    )[1].squeeze(0)
                    slow_incoming_state = self.incoming_sequence_encoder(
                        pair_slow_incoming_sequence_bank
                    )[1].squeeze(0)
                    fast_sequence_repr = self.sequence_proj(
                        self._pairwise_fusion_inputs(
                            outgoing_state,
                            incoming_state,
                        )
                    )
                    slow_sequence_repr = self.sequence_proj(
                        self._pairwise_fusion_inputs(
                            slow_outgoing_state,
                            slow_incoming_state,
                        )
                    )
                    pair_sequence_repr = fast_sequence_repr
                else:
                    outgoing_sequence_bank = self._build_recent_sequence_bank(
                        pair_repr, pair_src, pair_timestamps, num_nodes, outgoing_latest
                    )
                    incoming_sequence_bank = self._build_recent_sequence_bank(
                        pair_repr, pair_dst, pair_timestamps, num_nodes, incoming_latest
                    )
                    pair_outgoing_sequence_bank = outgoing_sequence_bank[pair_src]
                    pair_incoming_sequence_bank = incoming_sequence_bank[pair_dst]
                    if self.use_target_sequence_select:
                        pair_outgoing_sequence_bank = self._filter_sequence_bank(
                            pair_outgoing_sequence_bank,
                            pair_repr,
                            self.outgoing_sequence_select_score,
                        )
                        pair_incoming_sequence_bank = self._filter_sequence_bank(
                            pair_incoming_sequence_bank,
                            pair_repr,
                            self.incoming_sequence_select_score,
                        )
                    if self.use_sequence_consistency_filter:
                        pair_outgoing_sequence_bank, outgoing_consistency = (
                            self._consistency_filter_sequence_bank(
                                pair_outgoing_sequence_bank,
                                pair_incoming_sequence_bank,
                                pair_repr,
                                self.outgoing_consistency_gate,
                            )
                        )
                        pair_incoming_sequence_bank, incoming_consistency = (
                            self._consistency_filter_sequence_bank(
                                pair_incoming_sequence_bank,
                                pair_outgoing_sequence_bank,
                                pair_repr,
                                self.incoming_consistency_gate,
                            )
                        )
                    outgoing_state = self.outgoing_sequence_encoder(
                        pair_outgoing_sequence_bank
                    )[1].squeeze(0)
                    incoming_state = self.incoming_sequence_encoder(
                        pair_incoming_sequence_bank
                    )[1].squeeze(0)
                    pair_sequence_repr = self.sequence_proj(
                        self._pairwise_fusion_inputs(
                            outgoing_state,
                            incoming_state,
                        )
                    )
                if self.use_sequence_bridge_bank:
                    outgoing_partner_bank = self._build_recent_partner_bank(
                        pair_src, pair_dst, pair_timestamps, num_nodes
                    )
                    incoming_partner_bank = self._build_recent_partner_bank(
                        pair_dst, pair_src, pair_timestamps, num_nodes
                    )
                    src_out_cov = (
                        (outgoing_partner_bank[pair_src] >= 0).float().sum(dim=1, keepdim=True) /
                        float(self.sequence_len)
                    )
                    dst_in_cov = (
                        (incoming_partner_bank[pair_dst] >= 0).float().sum(dim=1, keepdim=True) /
                        float(self.sequence_len)
                    )
                    src_in_cov = (
                        (incoming_partner_bank[pair_src] >= 0).float().sum(dim=1, keepdim=True) /
                        float(self.sequence_len)
                    )
                    dst_out_cov = (
                        (outgoing_partner_bank[pair_dst] >= 0).float().sum(dim=1, keepdim=True) /
                        float(self.sequence_len)
                    )
                    forward_overlap = self._partner_overlap_ratio(
                        outgoing_partner_bank[pair_src],
                        incoming_partner_bank[pair_dst],
                    )
                    cycle_overlap = self._partner_overlap_ratio(
                        incoming_partner_bank[pair_src],
                        outgoing_partner_bank[pair_dst],
                    )
                    node_x = batch[task[0]].x
                    forward_bridge = self._recent_overlap_partner_repr(
                        outgoing_partner_bank[pair_src],
                        incoming_partner_bank[pair_dst],
                        node_x,
                    )
                    cycle_bridge = self._recent_overlap_partner_repr(
                        incoming_partner_bank[pair_src],
                        outgoing_partner_bank[pair_dst],
                        node_x,
                    )
                    bridge_context = self.bridge_bank_proj(
                        self._pairwise_fusion_inputs(
                            forward_bridge,
                            cycle_bridge,
                        )
                    )
                    if self.use_sequence_bridge_bank_window:
                        window_input = torch.cat(
                            (
                                fast_sequence_repr,
                                slow_sequence_repr,
                                bridge_context,
                            ),
                            dim=-1,
                        )
                        window_gate = torch.sigmoid(self.sequence_window_gate(window_input))
                        pair_sequence_repr = fast_sequence_repr + (
                            torch.sigmoid(self.sequence_window_alpha) *
                            window_gate *
                            (slow_sequence_repr - fast_sequence_repr)
                        )
                    bridge_input = torch.cat(
                        (
                            bridge_context,
                            pair_sequence_repr,
                            bridge_context * pair_sequence_repr,
                        ),
                        dim=-1,
                    )
                    bridge_gate = torch.sigmoid(self.bridge_bank_gate(bridge_input))
                    pair_sequence_repr = pair_sequence_repr + (
                        torch.sigmoid(self.bridge_bank_alpha) *
                        bridge_gate *
                        self.bridge_bank_update(bridge_input)
                    )
                    if self.use_terminal_role_flow:
                        src_incoming_sequence_bank = incoming_sequence_bank[pair_src]
                        dst_outgoing_sequence_bank = outgoing_sequence_bank[pair_dst]
                        if self.use_target_sequence_select:
                            src_incoming_sequence_bank = self._filter_sequence_bank(
                                src_incoming_sequence_bank,
                                pair_repr,
                                self.incoming_sequence_select_score,
                            )
                            dst_outgoing_sequence_bank = self._filter_sequence_bank(
                                dst_outgoing_sequence_bank,
                                pair_repr,
                                self.outgoing_sequence_select_score,
                            )
                        if self.use_sequence_consistency_filter:
                            src_incoming_sequence_bank, src_role_consistency = (
                                self._consistency_filter_sequence_bank(
                                    src_incoming_sequence_bank,
                                    dst_outgoing_sequence_bank,
                                    pair_repr,
                                    self.incoming_consistency_gate,
                                )
                            )
                            dst_outgoing_sequence_bank, dst_role_consistency = (
                                self._consistency_filter_sequence_bank(
                                    dst_outgoing_sequence_bank,
                                    src_incoming_sequence_bank,
                                    pair_repr,
                                    self.outgoing_consistency_gate,
                                )
                            )
                        src_incoming_state = self.incoming_sequence_encoder(
                            src_incoming_sequence_bank
                        )[1].squeeze(0)
                        dst_outgoing_state = self.outgoing_sequence_encoder(
                            dst_outgoing_sequence_bank
                        )[1].squeeze(0)
                        src_role_repr = self.terminal_role_proj(
                            self._pairwise_fusion_inputs(
                                outgoing_state,
                                src_incoming_state,
                            )
                        )
                        dst_role_repr = self.terminal_role_proj(
                            self._pairwise_fusion_inputs(
                                incoming_state,
                                dst_outgoing_state,
                            )
                        )
                        src_out_count = (
                            outgoing_partner_bank[pair_src] >= 0
                        ).float().sum(dim=1, keepdim=True) / float(self.sequence_len)
                        src_in_count = (
                            incoming_partner_bank[pair_src] >= 0
                        ).float().sum(dim=1, keepdim=True) / float(self.sequence_len)
                        dst_out_count = (
                            outgoing_partner_bank[pair_dst] >= 0
                        ).float().sum(dim=1, keepdim=True) / float(self.sequence_len)
                        dst_in_count = (
                            incoming_partner_bank[pair_dst] >= 0
                        ).float().sum(dim=1, keepdim=True) / float(self.sequence_len)
                        role_stats = torch.cat(
                            (
                                src_out_count,
                                src_in_count,
                                dst_out_count,
                                dst_in_count,
                                src_out_count - src_in_count,
                                dst_in_count - dst_out_count,
                            ),
                            dim=-1,
                        )
                        pair_terminal_role_repr = self.terminal_flow_proj(
                            torch.cat(
                                (
                                    self._pairwise_fusion_inputs(
                                        src_role_repr,
                                        dst_role_repr,
                                    ),
                                    role_stats,
                                ),
                                dim=-1,
                            )
                        )
                        if self.use_boundary_lag_flow:
                            outgoing_time_bank = self._build_recent_timestamp_bank(
                                pair_src, pair_timestamps, num_nodes
                            )
                            incoming_time_bank = self._build_recent_timestamp_bank(
                                pair_dst, pair_timestamps, num_nodes
                            )
                            pair_boundary_lag_repr = self._recent_boundary_lag_repr(
                                src_incoming_sequence_bank,
                                dst_outgoing_sequence_bank,
                                incoming_time_bank[pair_src],
                                outgoing_time_bank[pair_dst],
                            )
                            boundary_valid_ratio, boundary_after_ratio, _ = (
                                self._boundary_support_features(
                                    incoming_time_bank[pair_src],
                                    outgoing_time_bank[pair_dst],
                                )
                            )
                    if self.use_sequence_bridge_motif_lite:
                        outgoing_time_bank = self._build_recent_timestamp_bank(
                            pair_src, pair_timestamps, num_nodes
                        )
                        incoming_time_bank = self._build_recent_timestamp_bank(
                            pair_dst, pair_timestamps, num_nodes
                        )
                        forward_leg = self._recent_overlap_leg_repr(
                            outgoing_partner_bank[pair_src],
                            incoming_partner_bank[pair_dst],
                            outgoing_sequence_bank[pair_src],
                            incoming_sequence_bank[pair_dst],
                            outgoing_time_bank[pair_src],
                            incoming_time_bank[pair_dst],
                        )
                        cycle_leg = self._recent_overlap_leg_repr(
                            incoming_partner_bank[pair_src],
                            outgoing_partner_bank[pair_dst],
                            incoming_sequence_bank[pair_src],
                            outgoing_sequence_bank[pair_dst],
                            incoming_time_bank[pair_src],
                            outgoing_time_bank[pair_dst],
                        )
                        leg_input = torch.cat(
                            (
                                forward_leg,
                                cycle_leg,
                                forward_leg * cycle_leg,
                            ),
                            dim=-1,
                        )
                        leg_gate = torch.sigmoid(self.bridge_leg_bank_gate(leg_input))
                        pair_sequence_repr = pair_sequence_repr + (
                            torch.sigmoid(self.bridge_leg_bank_alpha) *
                            leg_gate *
                            self.bridge_leg_bank_update(leg_input)
                        )
            if self.use_support_conditioned_mixture:
                pair_fill = torch.clamp(
                    pair_count / float(self.pair_sequence_len),
                    min=0.0,
                    max=1.0,
                )
                pair_log_count = torch.log1p(pair_count) / math.log(33.0)
                fast_slow_align = self._cosine_feature(
                    fast_sequence_repr, slow_sequence_repr
                )
                seq_pair_align = self._cosine_feature(pair_sequence_repr, pair_repr)
                context_seq_align = self._cosine_feature(
                    pair_context_repr, pair_sequence_repr
                )
                role_seq_align = self._cosine_feature(
                    pair_terminal_role_repr, pair_sequence_repr
                )
                if fast_slow_align is None:
                    fast_slow_align = zero_support
                if seq_pair_align is None:
                    seq_pair_align = zero_support
                if context_seq_align is None:
                    context_seq_align = zero_support
                if role_seq_align is None:
                    role_seq_align = zero_support
                support_feature_blocks = [
                    pair_fill,
                    pair_log_count,
                    src_out_cov,
                    dst_in_cov,
                    src_in_cov,
                    dst_out_cov,
                    forward_overlap,
                    cycle_overlap,
                    boundary_valid_ratio,
                    boundary_after_ratio,
                    fast_slow_align,
                    seq_pair_align,
                    context_seq_align,
                    role_seq_align,
                ]
                if self.use_sequence_consistency_filter:
                    support_feature_blocks.extend([
                        outgoing_consistency,
                        incoming_consistency,
                        src_role_consistency,
                        dst_role_consistency,
                    ])
                pair_support_features = torch.cat(support_feature_blocks, dim=-1)
                pair_structure_mix = torch.sigmoid(
                    self.structure_mix_bias +
                    self.structure_mix_gate(torch.cat((pair_repr, pair_support_features), dim=-1))
                )
                pair_sequence_support = torch.sigmoid(
                    self.sequence_support_bias +
                    self.sequence_support_gate(
                        torch.cat(
                            (
                                pair_sequence_repr
                                if pair_sequence_repr is not None
                                else torch.zeros_like(pair_repr),
                                pair_support_features,
                            ),
                            dim=-1,
                        )
                    )
                )
                pair_terminal_support = torch.sigmoid(
                    self.terminal_support_bias +
                    self.terminal_support_gate(
                        torch.cat(
                            (
                                pair_terminal_role_repr
                                if pair_terminal_role_repr is not None
                                else torch.zeros_like(pair_repr),
                                pair_support_features,
                            ),
                            dim=-1,
                        )
                    )
                )
                pair_boundary_support = torch.sigmoid(
                    self.boundary_support_bias +
                    self.boundary_support_gate(
                        torch.cat(
                            (
                                pair_boundary_lag_repr
                                if pair_boundary_lag_repr is not None
                                else torch.zeros_like(pair_repr),
                                pair_support_features,
                            ),
                            dim=-1,
                        )
                    )
                )
                if self.use_flow_sketch_expert:
                    flow_sketch_query = torch.cat(
                        (pair_repr, pair_support_features), dim=-1
                    )
                    flow_slot_weights = F.softmax(
                        self.support_flow_sketch_assign(flow_sketch_query),
                        dim=-1,
                    )
                    flow_slot_context = (
                        flow_slot_weights @ self.support_flow_sketch_slots
                    )
                    flow_candidate_blocks = torch.stack(
                        (
                            pair_sequence_repr
                            if pair_sequence_repr is not None
                            else torch.zeros_like(pair_repr),
                            pair_boundary_lag_repr
                            if pair_boundary_lag_repr is not None
                            else torch.zeros_like(pair_repr),
                            pair_terminal_role_repr
                            if pair_terminal_role_repr is not None
                            else torch.zeros_like(pair_repr),
                            pair_context_repr
                            if pair_context_repr is not None
                            else torch.zeros_like(pair_repr),
                        ),
                        dim=1,
                    )
                    flow_candidate_weights = F.softmax(
                        self.support_flow_sketch_candidate_gate(
                            flow_sketch_query
                        ),
                        dim=-1,
                    )
                    flow_candidate_context = (
                        flow_candidate_weights.unsqueeze(-1) *
                        flow_candidate_blocks
                    ).sum(dim=1)
                    pair_flow_sketch_repr = self.support_flow_sketch_fuse(
                        torch.cat(
                            (
                                pair_repr,
                                flow_slot_context,
                                flow_candidate_context,
                                flow_slot_context * flow_candidate_context,
                                pair_support_features,
                            ),
                            dim=-1,
                        )
                    )
                    flow_slot_confidence = flow_slot_weights.max(
                        dim=-1, keepdim=True
                    ).values
                    flow_candidate_confidence = flow_candidate_weights.max(
                        dim=-1, keepdim=True
                    ).values
                    flow_slot_match = self._cosine_feature(
                        pair_repr, flow_slot_context
                    )
                    flow_candidate_match = self._cosine_feature(
                        pair_repr, flow_candidate_context
                    )
                    flow_cross_match = self._cosine_feature(
                        flow_slot_context, flow_candidate_context
                    )
                    if flow_slot_match is None:
                        flow_slot_match = zero_support
                    if flow_candidate_match is None:
                        flow_candidate_match = zero_support
                    if flow_cross_match is None:
                        flow_cross_match = zero_support
                    pair_flow_sketch_support = torch.sigmoid(
                        self.support_flow_sketch_bias +
                        self.support_flow_sketch_gate(
                            torch.cat(
                                (
                                    pair_flow_sketch_repr,
                                    pair_support_features,
                                    flow_slot_confidence,
                                    flow_candidate_confidence,
                                    flow_slot_match,
                                    flow_candidate_match,
                                    flow_cross_match,
                                ),
                                dim=-1,
                            )
                        )
                    )
                    if self.use_flow_sketch_score_calibration:
                        flow_seq_align = self._cosine_feature(
                            pair_flow_sketch_repr,
                            pair_sequence_repr,
                        )
                        flow_terminal_align = self._cosine_feature(
                            pair_flow_sketch_repr,
                            pair_terminal_role_repr,
                        )
                        flow_boundary_align = self._cosine_feature(
                            pair_flow_sketch_repr,
                            pair_boundary_lag_repr,
                        )
                        flow_context_align = self._cosine_feature(
                            pair_flow_sketch_repr,
                            pair_context_repr,
                        )
                        flow_repr_align = self._cosine_feature(
                            pair_flow_sketch_repr,
                            pair_repr,
                        )
                        if flow_seq_align is None:
                            flow_seq_align = zero_support
                        if flow_terminal_align is None:
                            flow_terminal_align = zero_support
                        if flow_boundary_align is None:
                            flow_boundary_align = zero_support
                        if flow_context_align is None:
                            flow_context_align = zero_support
                        if flow_repr_align is None:
                            flow_repr_align = zero_support
                        calib_sequence_repr = (
                            pair_sequence_repr
                            if pair_sequence_repr is not None
                            else torch.zeros_like(pair_repr)
                        )
                        calib_terminal_repr = (
                            pair_terminal_role_repr
                            if pair_terminal_role_repr is not None
                            else torch.zeros_like(pair_repr)
                        )
                        calib_boundary_repr = (
                            pair_boundary_lag_repr
                            if pair_boundary_lag_repr is not None
                            else torch.zeros_like(pair_repr)
                        )
                        calib_context_repr = (
                            pair_context_repr
                            if pair_context_repr is not None
                            else torch.zeros_like(pair_repr)
                        )
                        calib_weight_sum = (
                            pair_sequence_support +
                            pair_terminal_support +
                            pair_boundary_support +
                            pair_structure_mix
                        ).clamp(min=1e-6)
                        calib_hybrid_context = (
                            pair_sequence_support * calib_sequence_repr +
                            pair_terminal_support * calib_terminal_repr +
                            pair_boundary_support * calib_boundary_repr +
                            pair_structure_mix * calib_context_repr
                        ) / calib_weight_sum
                        flow_scorecalib_stats = torch.cat(
                            (
                                pair_structure_mix,
                                pair_sequence_support,
                                pair_terminal_support,
                                pair_boundary_support,
                                pair_flow_sketch_support,
                                flow_repr_align,
                                flow_seq_align,
                                flow_terminal_align,
                                flow_boundary_align,
                                flow_context_align,
                            ),
                            dim=-1,
                        )
                        pair_flow_scorecalib_repr = self.support_flow_scorecalib_fuse(
                            torch.cat(
                                (
                                    pair_repr,
                                    pair_flow_sketch_repr,
                                    calib_hybrid_context,
                                    pair_flow_sketch_repr * calib_hybrid_context,
                                    pair_support_features,
                                    flow_scorecalib_stats,
                                ),
                                dim=-1,
                            )
                        )
                        pair_flow_scorecalib_support = torch.sigmoid(
                            self.support_flow_scorecalib_bias +
                            self.support_flow_scorecalib_gate(
                                torch.cat(
                                    (
                                        pair_flow_scorecalib_repr,
                                        pair_support_features,
                                        flow_scorecalib_stats,
                                    ),
                                    dim=-1,
                                )
                            )
                        )
                if self.use_support_prototype_expert:
                    proto_query = pair_sequence_repr
                    if proto_query is None:
                        proto_query = pair_terminal_role_repr
                    if proto_query is None:
                        proto_query = pair_boundary_lag_repr
                    if proto_query is None:
                        proto_query = pair_repr
                    proto_weights = F.softmax(
                        self.support_proto_assign(
                            torch.cat((proto_query, pair_support_features), dim=-1)
                        ),
                        dim=-1,
                    )
                    proto_context = proto_weights @ self.support_proto_tokens
                    pair_proto_repr = self.support_proto_fuse(
                        torch.cat(
                            (
                                pair_repr,
                                proto_context,
                                pair_repr * proto_context,
                                pair_support_features,
                            ),
                            dim=-1,
                        )
                    )
                    proto_confidence = proto_weights.max(dim=-1, keepdim=True).values
                    proto_match = self._cosine_feature(pair_repr, proto_context)
                    if proto_match is None:
                        proto_match = zero_support
                    pair_proto_support = torch.sigmoid(
                        self.support_proto_bias +
                        self.support_proto_gate(
                            torch.cat(
                                (
                                    pair_proto_repr,
                                    pair_support_features,
                                    proto_confidence,
                                    proto_match,
                                ),
                                dim=-1,
                            )
                        )
                    )
                if self.use_support_class_prototype_expert:
                    class_proto_query = pair_sequence_repr
                    if class_proto_query is None:
                        class_proto_query = pair_terminal_role_repr
                    if class_proto_query is None:
                        class_proto_query = pair_boundary_lag_repr
                    if class_proto_query is None:
                        class_proto_query = pair_repr
                    if self.training:
                        self._update_support_class_prototypes(
                            class_proto_query[pair_inv][mask],
                            batch[task].y[mask],
                        )
                    (
                        pos_proto,
                        neg_proto,
                        pos_sim,
                        neg_sim,
                        proto_margin,
                        proto_ready,
                        pos_peak,
                        neg_peak,
                        pos_spread,
                        neg_spread,
                    ) = self._support_class_proto_context(class_proto_query)
                    pair_class_proto_repr = self.support_class_proto_fuse(
                        torch.cat(
                            (
                                pair_repr,
                                pos_proto,
                                neg_proto,
                                pair_support_features,
                                pos_sim,
                                neg_sim,
                                proto_margin,
                                proto_ready,
                                pos_peak,
                                neg_peak,
                                pos_spread,
                                neg_spread,
                            ),
                            dim=-1,
                        )
                    )
                    pair_class_proto_support = torch.sigmoid(
                        self.support_class_proto_bias +
                        self.support_class_proto_gate(
                            torch.cat(
                                (
                                    pair_class_proto_repr,
                                    pair_support_features,
                                    pos_sim,
                                    neg_sim,
                                    proto_margin,
                                    proto_ready,
                                    pos_peak,
                                    neg_peak,
                                    pos_spread,
                                    neg_spread,
                                ),
                                dim=-1,
                            )
                        )
                    )
                    if self.use_support_class_split_expert:
                        pos_context = pair_sequence_repr
                        if pos_context is None:
                            pos_context = pair_boundary_lag_repr
                        if pos_context is None:
                            pos_context = pair_class_proto_repr
                        if pos_context is None:
                            pos_context = pair_repr
                        neg_context = pair_context_repr
                        if neg_context is None:
                            neg_context = pair_terminal_role_repr
                        if neg_context is None:
                            neg_context = pair_class_proto_repr
                        if neg_context is None:
                            neg_context = pair_repr
                        class_split_stats = torch.cat(
                            (
                                pos_sim,
                                neg_sim,
                                proto_margin,
                                proto_ready,
                                pos_peak,
                                neg_peak,
                                pos_spread,
                                neg_spread,
                            ),
                            dim=-1,
                        )
                        pair_pos_class_split_repr = self.support_pos_class_split_fuse(
                            torch.cat(
                                (
                                    pair_repr,
                                    pos_proto,
                                    pair_class_proto_repr,
                                    pos_context,
                                    pair_support_features,
                                    class_split_stats,
                                ),
                                dim=-1,
                            )
                        )
                        pair_neg_class_split_repr = self.support_neg_class_split_fuse(
                            torch.cat(
                                (
                                    pair_repr,
                                    neg_proto,
                                    pair_class_proto_repr,
                                    neg_context,
                                    pair_support_features,
                                    class_split_stats,
                                ),
                                dim=-1,
                            )
                        )
                        class_split_align = self._cosine_feature(
                            pair_pos_class_split_repr,
                            pair_neg_class_split_repr,
                        )
                        if class_split_align is None:
                            class_split_align = zero_support
                        class_split_gate = torch.sigmoid(
                            self.support_class_split_bias +
                            self.support_class_split_gate(
                                torch.cat(
                                    (
                                        pair_class_proto_repr,
                                        pair_support_features,
                                        class_split_stats,
                                        pair_class_proto_support,
                                        class_split_align,
                                    ),
                                    dim=-1,
                                )
                            )
                        )
                        pair_class_split_repr = pair_neg_class_split_repr + class_split_gate * (
                            pair_pos_class_split_repr - pair_neg_class_split_repr
                        )
                        context_support = pair_sequence_support
                        if context_support is None:
                            context_support = pair_structure_mix
                        if context_support is None:
                            context_support = pair_boundary_support
                        if context_support is None:
                            context_support = pair_terminal_support
                        if context_support is None:
                            context_support = pair_class_proto_support
                        pair_class_split_support = (
                            class_split_gate * pair_class_proto_support +
                            (1.0 - class_split_gate) * context_support
                        )
                    if self.use_support_proto_route_expert:
                        proto_branch_align = self._cosine_feature(
                            pair_proto_repr,
                            pair_class_proto_repr,
                        )
                        if proto_branch_align is None:
                            proto_branch_align = zero_support
                        proto_confidence_gap = (proto_confidence - pos_peak).abs()
                        proto_route_stats = torch.cat(
                            (
                                proto_confidence,
                                proto_match,
                                pos_sim,
                                neg_sim,
                                proto_margin,
                                proto_ready,
                                pos_peak,
                                neg_peak,
                                pos_spread,
                                neg_spread,
                                proto_branch_align,
                                proto_confidence_gap,
                                pair_fill,
                                pair_log_count,
                                forward_overlap,
                                cycle_overlap,
                                boundary_valid_ratio,
                                boundary_after_ratio,
                                pair_proto_support,
                                pair_class_proto_support,
                            ),
                            dim=-1,
                        )
                        proto_route_query = pair_class_proto_repr - pair_proto_repr
                        proto_route = torch.sigmoid(
                            self.support_proto_route_bias +
                            self.support_proto_route_gate(
                                torch.cat(
                                    (
                                        proto_route_query,
                                        pair_support_features,
                                        proto_route_stats,
                                    ),
                                    dim=-1,
                                )
                            )
                        )
                        proto_hybrid = pair_proto_repr + proto_route * (
                            pair_class_proto_repr - pair_proto_repr
                        )
                        pair_proto_route_repr = self.support_proto_route_fuse(
                            torch.cat(
                                (
                                    pair_repr,
                                    pair_proto_repr,
                                    pair_class_proto_repr,
                                    proto_hybrid,
                                    pair_support_features,
                                    proto_route_stats,
                                ),
                                dim=-1,
                            )
                        )
                        pair_proto_route_support = (
                            proto_route * pair_class_proto_support +
                            (1.0 - proto_route) * pair_proto_support
                        )
                    if self.use_support_subgraph_route_expert:
                        community_repr = pair_sequence_repr
                        if community_repr is None:
                            community_repr = pair_context_repr
                        if community_repr is None:
                            community_repr = pair_boundary_lag_repr
                        if community_repr is None:
                            community_repr = pair_terminal_role_repr
                        if community_repr is None:
                            community_repr = pair_repr
                        pair_subgraph_align = self._cosine_feature(
                            pair_repr,
                            community_repr,
                        )
                        if pair_subgraph_align is None:
                            pair_subgraph_align = zero_support
                        class_subgraph_align = self._cosine_feature(
                            pair_class_proto_repr,
                            community_repr,
                        )
                        if class_subgraph_align is None:
                            class_subgraph_align = zero_support
                        subgraph_stats = torch.cat(
                            (
                                pos_sim,
                                neg_sim,
                                proto_margin,
                                proto_ready,
                                pos_peak,
                                neg_peak,
                                pos_spread,
                                neg_spread,
                                pair_fill,
                                pair_log_count,
                                forward_overlap,
                                cycle_overlap,
                                boundary_valid_ratio,
                                boundary_after_ratio,
                                pair_subgraph_align,
                                class_subgraph_align,
                            ),
                            dim=-1,
                        )
                        subgraph_route = torch.sigmoid(
                            self.support_subgraph_route_bias +
                            self.support_subgraph_route_gate(
                                torch.cat(
                                    (
                                        community_repr,
                                        pair_support_features,
                                        subgraph_stats,
                                    ),
                                    dim=-1,
                                )
                            )
                        )
                        abnormal_hybrid = community_repr + subgraph_route * (
                            pair_class_proto_repr - community_repr
                        )
                        pair_subgraph_route_repr = self.support_subgraph_route_fuse(
                            torch.cat(
                                (
                                    pair_repr,
                                    pair_class_proto_repr,
                                    community_repr,
                                    abnormal_hybrid,
                                    pair_support_features,
                                    subgraph_stats,
                                ),
                                dim=-1,
                            )
                        )
                        community_support = pair_sequence_support
                        if community_support is None:
                            community_support = pair_boundary_support
                        if community_support is None:
                            community_support = pair_terminal_support
                        if community_support is None:
                            community_support = pair_class_proto_support
                        pair_subgraph_route_support = (
                            subgraph_route * pair_class_proto_support +
                            (1.0 - subgraph_route) * community_support
                        )
                    if self.use_support_class_split_subgraph_route_expert:
                        split_route_repr = pair_class_split_repr
                        if split_route_repr is None:
                            split_route_repr = pair_class_proto_repr
                        if split_route_repr is None:
                            split_route_repr = pair_repr
                        subgraph_route_repr = pair_subgraph_route_repr
                        if subgraph_route_repr is None:
                            subgraph_route_repr = pair_sequence_repr
                        if subgraph_route_repr is None:
                            subgraph_route_repr = pair_context_repr
                        if subgraph_route_repr is None:
                            subgraph_route_repr = pair_boundary_lag_repr
                        if subgraph_route_repr is None:
                            subgraph_route_repr = pair_repr
                        split_route_support = pair_class_split_support
                        if split_route_support is None:
                            split_route_support = pair_class_proto_support
                        if split_route_support is None:
                            split_route_support = zero_support
                        subgraph_route_support = pair_subgraph_route_support
                        if subgraph_route_support is None:
                            subgraph_route_support = pair_sequence_support
                        if subgraph_route_support is None:
                            subgraph_route_support = pair_boundary_support
                        if subgraph_route_support is None:
                            subgraph_route_support = pair_terminal_support
                        if subgraph_route_support is None:
                            subgraph_route_support = pair_class_proto_support
                        if subgraph_route_support is None:
                            subgraph_route_support = zero_support
                        split_subgraph_align = self._cosine_feature(
                            split_route_repr,
                            subgraph_route_repr,
                        )
                        if split_subgraph_align is None:
                            split_subgraph_align = zero_support
                        split_proto_align = self._cosine_feature(
                            split_route_repr,
                            pair_class_proto_repr,
                        )
                        if split_proto_align is None:
                            split_proto_align = zero_support
                        subgraph_proto_align = self._cosine_feature(
                            subgraph_route_repr,
                            pair_class_proto_repr,
                        )
                        if subgraph_proto_align is None:
                            subgraph_proto_align = zero_support
                        class_split_subgraph_stats = torch.cat(
                            (
                                pos_sim,
                                neg_sim,
                                proto_margin,
                                proto_ready,
                                pos_peak,
                                neg_peak,
                                pos_spread,
                                neg_spread,
                                pair_fill,
                                pair_log_count,
                                forward_overlap,
                                cycle_overlap,
                                boundary_valid_ratio,
                                boundary_after_ratio,
                                split_subgraph_align,
                                split_proto_align,
                                subgraph_proto_align,
                                split_route_support,
                                subgraph_route_support,
                            ),
                            dim=-1,
                        )
                        class_split_subgraph_route = torch.sigmoid(
                            self.support_class_split_subgraph_route_bias +
                            self.support_class_split_subgraph_route_gate(
                                torch.cat(
                                    (
                                        split_route_repr - subgraph_route_repr,
                                        pair_support_features,
                                        class_split_subgraph_stats,
                                    ),
                                    dim=-1,
                                )
                            )
                        )
                        class_split_subgraph_hybrid = subgraph_route_repr + (
                            class_split_subgraph_route *
                            (split_route_repr - subgraph_route_repr)
                        )
                        pair_class_split_subgraph_route_repr = (
                            self.support_class_split_subgraph_route_fuse(
                                torch.cat(
                                    (
                                        pair_repr,
                                        split_route_repr,
                                        subgraph_route_repr,
                                        class_split_subgraph_hybrid,
                                        pair_support_features,
                                        class_split_subgraph_stats,
                                    ),
                                    dim=-1,
                                )
                            )
                        )
                        pair_class_split_subgraph_route_support = (
                            class_split_subgraph_route * split_route_support +
                            (1.0 - class_split_subgraph_route) *
                            subgraph_route_support
                        )
                if self.use_support_proto_consensus_expert:
                    proto_consensus = self._cosine_feature(
                        pair_proto_repr,
                        pair_class_proto_repr,
                    )
                    if proto_consensus is None:
                        proto_consensus = zero_support
                    proto_confidence_gap = (proto_confidence - pos_peak).abs()
                    consensus_stats = torch.cat(
                        (
                            proto_confidence,
                            proto_match,
                            pos_sim,
                            neg_sim,
                            proto_margin,
                            proto_ready,
                            pos_peak,
                            neg_peak,
                            pos_spread,
                            neg_spread,
                            proto_consensus,
                            proto_confidence_gap,
                            pair_proto_support,
                            pair_class_proto_support,
                        ),
                        dim=-1,
                    )
                    pair_proto_consensus_repr = self.support_proto_consensus_fuse(
                        torch.cat(
                            (
                                pair_repr,
                                pair_proto_repr,
                                pair_class_proto_repr,
                                pair_proto_repr * pair_class_proto_repr,
                                pair_support_features,
                                consensus_stats,
                            ),
                            dim=-1,
                        )
                    )
                    pair_proto_consensus_support = torch.sigmoid(
                        self.support_proto_consensus_bias +
                        self.support_proto_consensus_gate(
                            torch.cat(
                                (
                                    pair_proto_consensus_repr,
                                    pair_support_features,
                                    consensus_stats,
                                ),
                                dim=-1,
                            )
                        )
                    )
                    if self.use_support_proto_disagreement_expert:
                        proto_delta = pair_proto_repr - pair_class_proto_repr
                        pair_proto_disagreement_repr = self.support_proto_disagreement_fuse(
                            torch.cat(
                                (
                                    pair_repr,
                                    proto_delta,
                                    proto_delta.abs(),
                                    pair_proto_repr * pair_class_proto_repr,
                                    pair_support_features,
                                    consensus_stats,
                                ),
                                dim=-1,
                            )
                        )
                        pair_proto_disagreement_support = torch.sigmoid(
                            self.support_proto_disagreement_bias +
                            self.support_proto_disagreement_gate(
                                torch.cat(
                                    (
                                        pair_proto_disagreement_repr,
                                        pair_support_features,
                                        consensus_stats,
                                    ),
                                    dim=-1,
                                )
                            )
                        )
                    if self.use_support_confidence_dual_expert:
                        global_confidence = 0.5 * (proto_match + pos_sim)
                        inverse_confidence = 1.0 - global_confidence
                        confidence_hybrid = (
                            global_confidence * pair_proto_consensus_repr +
                            inverse_confidence * pair_proto_disagreement_repr
                        )
                        confidence_stats = torch.cat(
                            (
                                global_confidence,
                                inverse_confidence,
                                proto_consensus,
                                proto_confidence_gap,
                                pair_proto_support,
                                pair_class_proto_support,
                                pair_proto_consensus_support,
                                pair_proto_disagreement_support,
                                pos_sim,
                                neg_sim,
                                proto_margin,
                                proto_ready,
                            ),
                            dim=-1,
                        )
                        pair_confidence_dual_repr = self.support_confidence_dual_fuse(
                            torch.cat(
                                (
                                    pair_repr,
                                    pair_proto_repr,
                                    pair_class_proto_repr,
                                    confidence_hybrid,
                                    pair_proto_consensus_repr - pair_proto_disagreement_repr,
                                    pair_support_features,
                                    confidence_stats,
                                ),
                                dim=-1,
                            )
                        )
                        pair_confidence_dual_support = torch.sigmoid(
                            self.support_confidence_dual_bias +
                            self.support_confidence_dual_gate(
                                torch.cat(
                                    (
                                        pair_confidence_dual_repr,
                                        pair_support_features,
                                        confidence_stats,
                                    ),
                                    dim=-1,
                                )
                            )
                        )
                        if self.use_support_easyhard_dual_expert:
                            easy_stats = torch.cat(
                                (
                                    global_confidence,
                                    inverse_confidence,
                                    proto_consensus,
                                    proto_confidence_gap,
                                    pair_proto_support,
                                    pair_class_proto_support,
                                    pair_proto_consensus_support,
                                    pair_proto_disagreement_support,
                                    pos_sim,
                                    neg_sim,
                                    proto_margin,
                                    proto_ready,
                                ),
                                dim=-1,
                            )
                            easy_repr = self.support_easy_dual_fuse(
                                torch.cat(
                                    (
                                        pair_repr,
                                        pair_proto_repr,
                                        pair_class_proto_repr,
                                        confidence_hybrid,
                                        pair_proto_consensus_repr,
                                        pair_support_features,
                                        easy_stats,
                                    ),
                                    dim=-1,
                                )
                            )
                            hard_repr = self.support_hard_dual_fuse(
                                torch.cat(
                                    (
                                        pair_repr,
                                        pair_proto_repr,
                                        pair_class_proto_repr,
                                        pair_proto_disagreement_repr,
                                        pair_confidence_dual_repr,
                                        pair_support_features,
                                        easy_stats,
                                    ),
                                    dim=-1,
                                )
                            )
                            easy_route = torch.sigmoid(
                                self.support_easyhard_dual_bias +
                                self.support_easyhard_dual_gate(
                                    torch.cat(
                                        (
                                            pair_confidence_dual_repr,
                                            pair_support_features,
                                            easy_stats,
                                        ),
                                        dim=-1,
                                    )
                                )
                            )
                            pair_easyhard_dual_repr = hard_repr + easy_route * (
                                easy_repr - hard_repr
                            )
                            pair_easyhard_dual_support = (
                                easy_route * pair_proto_consensus_support +
                                (1.0 - easy_route) * pair_proto_disagreement_support
                            )
                            if self.use_support_scale_route_expert:
                                scale_stats = torch.cat(
                                    (
                                        pair_fill,
                                        pair_log_count,
                                        src_out_cov,
                                        dst_in_cov,
                                        src_in_cov,
                                        dst_out_cov,
                                        forward_overlap,
                                        cycle_overlap,
                                        boundary_valid_ratio,
                                        boundary_after_ratio,
                                    ),
                                    dim=-1,
                                )
                                scale_route = torch.sigmoid(
                                    self.support_scale_route_bias +
                                    self.support_scale_route_gate(
                                        torch.cat(
                                            (
                                                pair_easyhard_dual_repr,
                                                pair_support_features,
                                                scale_stats,
                                            ),
                                            dim=-1,
                                        )
                                    )
                                )
                                scale_hybrid = pair_confidence_dual_repr + scale_route * (
                                    pair_easyhard_dual_repr - pair_confidence_dual_repr
                                )
                                pair_scale_route_repr = self.support_scale_route_fuse(
                                    torch.cat(
                                        (
                                            pair_repr,
                                            pair_confidence_dual_repr,
                                            pair_easyhard_dual_repr,
                                            scale_hybrid,
                                            pair_support_features,
                                            scale_stats,
                                        ),
                                        dim=-1,
                                    )
                                )
                                pair_scale_route_support = (
                                    scale_route * pair_easyhard_dual_support +
                                    (1.0 - scale_route) * pair_confidence_dual_support
                                )
                                if self.use_support_class_route_expert:
                                    class_route_stats_base = torch.cat(
                                        (
                                            pos_sim,
                                            neg_sim,
                                            proto_margin,
                                            proto_ready,
                                            pos_peak,
                                            neg_peak,
                                            proto_consensus,
                                            proto_confidence_gap,
                                            global_confidence,
                                            inverse_confidence,
                                            pair_fill,
                                            pair_log_count,
                                            forward_overlap,
                                            cycle_overlap,
                                            boundary_valid_ratio,
                                            boundary_after_ratio,
                                        ),
                                        dim=-1,
                                    )
                                    pos_class_route_repr = self.support_pos_class_route_fuse(
                                        torch.cat(
                                            (
                                                pair_repr,
                                                pos_proto,
                                                pair_proto_consensus_repr,
                                                pair_scale_route_repr,
                                                pair_support_features,
                                                class_route_stats_base,
                                            ),
                                            dim=-1,
                                        )
                                    )
                                    neg_class_route_repr = self.support_neg_class_route_fuse(
                                        torch.cat(
                                            (
                                                pair_repr,
                                                neg_proto,
                                                pair_proto_disagreement_repr,
                                                pair_confidence_dual_repr,
                                                pair_support_features,
                                                class_route_stats_base,
                                            ),
                                            dim=-1,
                                        )
                                    )
                                    class_branch_align = self._cosine_feature(
                                        pos_class_route_repr,
                                        neg_class_route_repr,
                                    )
                                    if class_branch_align is None:
                                        class_branch_align = zero_support
                                    class_route_stats = torch.cat(
                                        (
                                            class_route_stats_base,
                                            pair_scale_route_support,
                                            class_branch_align,
                                        ),
                                        dim=-1,
                                    )
                                    class_route = torch.sigmoid(
                                        self.support_class_route_bias +
                                        self.support_class_route_gate(
                                            torch.cat(
                                                (
                                                    pair_scale_route_repr,
                                                    pair_support_features,
                                                    class_route_stats,
                                                ),
                                                dim=-1,
                                            )
                                        )
                                    )
                                    class_hybrid = neg_class_route_repr + class_route * (
                                        pos_class_route_repr - neg_class_route_repr
                                    )
                                    pair_class_route_repr = self.support_class_route_fuse(
                                        torch.cat(
                                            (
                                                pair_repr,
                                                pos_class_route_repr,
                                                neg_class_route_repr,
                                                class_hybrid,
                                                pair_support_features,
                                                class_route_stats,
                                            ),
                                            dim=-1,
                                        )
                                    )
                                    pos_branch_support = 0.5 * (
                                        pair_proto_consensus_support +
                                        pair_scale_route_support
                                    )
                                    neg_branch_support = 0.5 * (
                                        pair_proto_disagreement_support +
                                        pair_confidence_dual_support
                                    )
                                    pair_class_route_support = (
                                        class_route * pos_branch_support +
                                        (1.0 - class_route) * neg_branch_support
                                    )

        pair_edge_repr = pair_repr[pair_inv]
        edge_repr = edge_repr + torch.sigmoid(self.pair_residual_alpha) * pair_edge_repr
        pred = self.layer_post_mp(torch.cat(
            (edge_repr[mask], pair_edge_repr[mask], edge_repr[mask] * pair_edge_repr[mask]),
            dim=-1
        ))
        if self.use_chain_context_residual:
            if pair_context_repr is None:
                pair_context_repr = torch.zeros_like(pair_repr)
            context_logits = self.context_head(pair_context_repr[pair_inv][mask])
            if self.use_support_conditioned_mixture and pair_structure_mix is not None:
                context_logits = pair_structure_mix[pair_inv][mask] * context_logits
            if self.use_bounded_support_residuals:
                context_logits = torch.tanh(context_logits)
            pred = pred + torch.sigmoid(self.context_residual_alpha) * context_logits
        if self.use_sequence_context_residual:
            if pair_sequence_repr is None:
                pair_sequence_repr = torch.zeros_like(pair_repr)
            sequence_logits = self.sequence_head(pair_sequence_repr[pair_inv][mask])
            if self.use_support_conditioned_mixture and pair_sequence_support is not None:
                sequence_logits = pair_sequence_support[pair_inv][mask] * sequence_logits
            if self.use_bounded_support_residuals:
                sequence_logits = torch.tanh(sequence_logits)
            pred = pred + torch.sigmoid(self.sequence_residual_alpha) * sequence_logits
        if self.use_terminal_role_flow:
            if pair_terminal_role_repr is None:
                pair_terminal_role_repr = torch.zeros_like(pair_repr)
            terminal_role_logits = self.terminal_flow_head(
                pair_terminal_role_repr[pair_inv][mask]
            )
            if self.use_support_conditioned_mixture and pair_terminal_support is not None:
                terminal_role_logits = (
                    pair_terminal_support[pair_inv][mask] * terminal_role_logits
                )
            if self.use_bounded_support_residuals:
                terminal_role_logits = torch.tanh(terminal_role_logits)
            pred = pred + (
                torch.sigmoid(self.terminal_flow_residual_alpha) *
                terminal_role_logits
            )
        if self.use_boundary_lag_flow:
            if pair_boundary_lag_repr is None:
                pair_boundary_lag_repr = torch.zeros_like(pair_repr)
            boundary_lag_logits = self.boundary_lag_head(
                pair_boundary_lag_repr[pair_inv][mask]
            )
            if self.use_support_conditioned_mixture and pair_boundary_support is not None:
                boundary_lag_logits = (
                    pair_boundary_support[pair_inv][mask] * boundary_lag_logits
                )
            if self.use_bounded_support_residuals:
                boundary_lag_logits = torch.tanh(boundary_lag_logits)
            pred = pred + (
                torch.sigmoid(self.boundary_lag_residual_alpha) *
                boundary_lag_logits
            )
        if self.use_support_conditioned_mixture and pair_structure_mix is not None:
            if self.use_dot_fallback_support_mixture:
                src_node_repr = self.dot_fallback_proj(
                    batch[task[0]].x[src_nodes[mask]]
                )
                dst_node_repr = self.dot_fallback_proj(
                    batch[task[2]].x[dst_nodes[mask]]
                )
                dot_score = (
                    (src_node_repr * dst_node_repr).sum(dim=-1, keepdim=True) /
                    math.sqrt(src_node_repr.size(-1))
                )
                if pred.size(-1) == 1:
                    fallback_pred = dot_score
                else:
                    fallback_pred = torch.cat((-dot_score, dot_score), dim=-1)
            else:
                fallback_pred = self.edge_fallback_head(edge_local_repr[mask])
            pred = fallback_pred + pair_structure_mix[pair_inv][mask] * (
                pred - fallback_pred
            )
        if self.use_flow_sketch_expert:
            if pair_flow_sketch_repr is None:
                pair_flow_sketch_repr = torch.zeros_like(pair_repr)
            flow_sketch_logits = self.support_flow_sketch_head(
                pair_flow_sketch_repr[pair_inv][mask]
            )
            if pair_flow_sketch_support is not None:
                flow_sketch_logits = (
                    pair_flow_sketch_support[pair_inv][mask] *
                    flow_sketch_logits
                )
            if self.use_bounded_support_residuals:
                flow_sketch_logits = torch.tanh(flow_sketch_logits)
            pred = pred + (
                torch.sigmoid(self.support_flow_sketch_alpha) *
                flow_sketch_logits
            )
        if self.use_flow_sketch_score_calibration:
            if pair_flow_scorecalib_repr is None:
                pair_flow_scorecalib_repr = torch.zeros_like(pair_repr)
            flow_scorecalib_params = self.support_flow_scorecalib_head(
                pair_flow_scorecalib_repr[pair_inv][mask]
            )
            if pair_flow_scorecalib_support is not None:
                flow_scorecalib_support = (
                    pair_flow_scorecalib_support[pair_inv][mask]
                )
            else:
                flow_scorecalib_support = torch.ones_like(
                    flow_scorecalib_params[:, :1]
                )
            flow_scorecalib_alpha = torch.sigmoid(
                self.support_flow_scorecalib_alpha
            )
            flow_margin_scale = 1.0 + (
                0.75 *
                flow_scorecalib_alpha *
                flow_scorecalib_support *
                torch.tanh(flow_scorecalib_params[:, :1])
            )
            flow_margin_bias = (
                0.50 *
                flow_scorecalib_alpha *
                flow_scorecalib_support *
                torch.tanh(flow_scorecalib_params[:, 1:2])
            )
            pred = self._apply_signed_margin_calibration(
                pred,
                flow_margin_scale,
                flow_margin_bias,
            )
        if self.use_support_prototype_expert:
            if pair_proto_repr is None:
                pair_proto_repr = torch.zeros_like(pair_repr)
            proto_logits = self.support_proto_head(pair_proto_repr[pair_inv][mask])
            if pair_proto_support is not None:
                proto_logits = pair_proto_support[pair_inv][mask] * proto_logits
            if self.use_bounded_support_residuals:
                proto_logits = torch.tanh(proto_logits)
            pred = pred + torch.sigmoid(self.support_proto_alpha) * proto_logits
        if self.use_support_class_prototype_expert:
            if pair_class_proto_repr is None:
                pair_class_proto_repr = torch.zeros_like(pair_repr)
            class_proto_logits = self.support_class_proto_head(
                pair_class_proto_repr[pair_inv][mask]
            )
            if pair_class_proto_support is not None:
                class_proto_logits = (
                    pair_class_proto_support[pair_inv][mask] * class_proto_logits
                )
            pred = pred + (
                torch.sigmoid(self.support_class_proto_alpha) *
                class_proto_logits
            )
        if self.use_support_class_split_expert:
            if pair_class_split_repr is None:
                pair_class_split_repr = torch.zeros_like(pair_repr)
            class_split_logits = self.support_class_split_head(
                pair_class_split_repr[pair_inv][mask]
            )
            if pair_class_split_support is not None:
                class_split_logits = (
                    pair_class_split_support[pair_inv][mask] * class_split_logits
                )
            if self.use_bounded_support_residuals:
                class_split_logits = torch.tanh(class_split_logits)
            pred = pred + (
                torch.sigmoid(self.support_class_split_alpha) *
                class_split_logits
            )
        if self.use_support_proto_consensus_expert:
            if pair_proto_consensus_repr is None:
                pair_proto_consensus_repr = torch.zeros_like(pair_repr)
            consensus_logits = self.support_proto_consensus_head(
                pair_proto_consensus_repr[pair_inv][mask]
            )
            if pair_proto_consensus_support is not None:
                consensus_logits = (
                    pair_proto_consensus_support[pair_inv][mask] * consensus_logits
                )
            if self.use_bounded_support_residuals:
                consensus_logits = torch.tanh(consensus_logits)
            pred = pred + (
                torch.sigmoid(self.support_proto_consensus_alpha) *
                consensus_logits
            )
        if self.use_support_proto_disagreement_expert:
            if pair_proto_disagreement_repr is None:
                pair_proto_disagreement_repr = torch.zeros_like(pair_repr)
            disagreement_logits = self.support_proto_disagreement_head(
                pair_proto_disagreement_repr[pair_inv][mask]
            )
            if pair_proto_disagreement_support is not None:
                disagreement_logits = (
                    pair_proto_disagreement_support[pair_inv][mask] *
                    disagreement_logits
                )
            if self.use_bounded_support_residuals:
                disagreement_logits = torch.tanh(disagreement_logits)
            pred = pred + (
                torch.sigmoid(self.support_proto_disagreement_alpha) *
                disagreement_logits
            )
        if self.use_support_confidence_dual_expert:
            if pair_confidence_dual_repr is None:
                pair_confidence_dual_repr = torch.zeros_like(pair_repr)
            confidence_dual_logits = self.support_confidence_dual_head(
                pair_confidence_dual_repr[pair_inv][mask]
            )
            if pair_confidence_dual_support is not None:
                confidence_dual_logits = (
                    pair_confidence_dual_support[pair_inv][mask] *
                    confidence_dual_logits
                )
            if self.use_bounded_support_residuals:
                confidence_dual_logits = torch.tanh(confidence_dual_logits)
            pred = pred + (
                torch.sigmoid(self.support_confidence_dual_alpha) *
                confidence_dual_logits
            )
        if self.use_support_easyhard_dual_expert:
            if pair_easyhard_dual_repr is None:
                pair_easyhard_dual_repr = torch.zeros_like(pair_repr)
            easyhard_dual_logits = self.support_easyhard_dual_head(
                pair_easyhard_dual_repr[pair_inv][mask]
            )
            if pair_easyhard_dual_support is not None:
                easyhard_dual_logits = (
                    pair_easyhard_dual_support[pair_inv][mask] *
                    easyhard_dual_logits
                )
            if self.use_bounded_support_residuals:
                easyhard_dual_logits = torch.tanh(easyhard_dual_logits)
            pred = pred + (
                torch.sigmoid(self.support_easyhard_dual_alpha) *
                easyhard_dual_logits
            )
        if self.use_support_scale_route_expert:
            if pair_scale_route_repr is None:
                pair_scale_route_repr = torch.zeros_like(pair_repr)
            scale_route_logits = self.support_scale_route_head(
                pair_scale_route_repr[pair_inv][mask]
            )
            if pair_scale_route_support is not None:
                scale_route_logits = (
                    pair_scale_route_support[pair_inv][mask] * scale_route_logits
                )
            if self.use_bounded_support_residuals:
                scale_route_logits = torch.tanh(scale_route_logits)
            pred = pred + (
                torch.sigmoid(self.support_scale_route_alpha) *
                scale_route_logits
            )
        if self.use_support_class_route_expert:
            if pair_class_route_repr is None:
                pair_class_route_repr = torch.zeros_like(pair_repr)
            class_route_logits = self.support_class_route_head(
                pair_class_route_repr[pair_inv][mask]
            )
            if pair_class_route_support is not None:
                class_route_logits = (
                    pair_class_route_support[pair_inv][mask] * class_route_logits
                )
            if self.use_bounded_support_residuals:
                class_route_logits = torch.tanh(class_route_logits)
            pred = pred + (
                torch.sigmoid(self.support_class_route_alpha) *
                class_route_logits
            )
        if (
            self.use_support_subgraph_route_expert and
            (
                not self.use_support_class_split_subgraph_route_expert or
                self.use_support_class_split_subgraph_dual_mix_route_expert or
                self.use_support_class_split_subgraph_dual_mix_route_w4_expert
            )
        ):
            if pair_subgraph_route_repr is None:
                pair_subgraph_route_repr = torch.zeros_like(pair_repr)
            subgraph_route_logits = self.support_subgraph_route_head(
                pair_subgraph_route_repr[pair_inv][mask]
            )
            if pair_subgraph_route_support is not None:
                subgraph_route_logits = (
                    pair_subgraph_route_support[pair_inv][mask] *
                    subgraph_route_logits
                )
            if self.use_bounded_support_residuals:
                subgraph_route_logits = torch.tanh(subgraph_route_logits)
            pred = pred + (
                torch.sigmoid(self.support_subgraph_route_alpha) *
                subgraph_route_logits
            )
        if self.use_support_proto_route_expert:
            if pair_proto_route_repr is None:
                pair_proto_route_repr = torch.zeros_like(pair_repr)
            proto_route_logits = self.support_proto_route_head(
                pair_proto_route_repr[pair_inv][mask]
            )
            if pair_proto_route_support is not None:
                proto_route_logits = (
                    pair_proto_route_support[pair_inv][mask] *
                    proto_route_logits
                )
            if self.use_bounded_support_residuals:
                proto_route_logits = torch.tanh(proto_route_logits)
            pred = pred + (
                torch.sigmoid(self.support_proto_route_alpha) *
                proto_route_logits
            )
        if (
            self.use_support_class_split_subgraph_route_expert and
            not self.use_support_class_split_subgraph_dual_resmix_expert
        ):
            if pair_class_split_subgraph_route_repr is None:
                pair_class_split_subgraph_route_repr = torch.zeros_like(pair_repr)
            class_split_subgraph_route_logits = (
                self.support_class_split_subgraph_route_head(
                    pair_class_split_subgraph_route_repr[pair_inv][mask]
                )
            )
            if pair_class_split_subgraph_route_support is not None:
                class_split_subgraph_route_logits = (
                    pair_class_split_subgraph_route_support[pair_inv][mask] *
                    class_split_subgraph_route_logits
                )
            if self.use_bounded_support_residuals:
                class_split_subgraph_route_logits = torch.tanh(
                    class_split_subgraph_route_logits
                )
            pred = pred + (
                torch.sigmoid(self.support_class_split_subgraph_route_alpha) *
                class_split_subgraph_route_logits
            )
        if self.use_support_class_split_subgraph_dual_mix_pred_gate_expert:
            if pair_subgraph_route_repr is None:
                pair_subgraph_route_repr = torch.zeros_like(pair_repr)
            subgraph_route_logits = self.support_subgraph_route_head(
                pair_subgraph_route_repr[pair_inv][mask]
            )
            if pair_subgraph_route_support is not None:
                subgraph_route_logits = (
                    pair_subgraph_route_support[pair_inv][mask] *
                    subgraph_route_logits
                )
            if self.use_bounded_support_residuals:
                subgraph_route_logits = torch.tanh(subgraph_route_logits)
            if pred.size(-1) == 2:
                pred_prob = torch.softmax(pred.detach(), dim=-1)[:, 1:2]
            elif pred.size(-1) == 1:
                pred_prob = torch.sigmoid(pred.detach())
            else:
                pred_prob = pred.new_full((pred.size(0), 1), 0.5)
            gate_center = self.support_subgraph_dual_pred_gate_center.clamp(
                0.01, 0.99
            )
            gate_scale = self.support_subgraph_dual_pred_gate_scale.clamp(1.0, 20.0)
            pred_gate = torch.sigmoid(gate_scale * (pred_prob - gate_center))
            pred = pred + (
                torch.sigmoid(self.support_subgraph_route_alpha) *
                pred_gate *
                subgraph_route_logits
            )
        if (
            self.use_support_class_split_subgraph_dual_mix_uncert_gate_expert
            or self.use_support_class_split_subgraph_dual_mix_uncert_weak_expert
            or self.use_support_class_split_subgraph_dual_mix_uncert_late_expert
            or self.use_support_class_split_subgraph_dual_mix_uncert_tight_expert
            or self.use_support_class_split_subgraph_dual_mix_uncert_mid_expert
        ):
            if pair_subgraph_route_repr is None:
                pair_subgraph_route_repr = torch.zeros_like(pair_repr)
            subgraph_route_logits = self.support_subgraph_route_head(
                pair_subgraph_route_repr[pair_inv][mask]
            )
            if pair_subgraph_route_support is not None:
                subgraph_route_logits = (
                    pair_subgraph_route_support[pair_inv][mask] *
                    subgraph_route_logits
                )
            if self.use_bounded_support_residuals:
                subgraph_route_logits = torch.tanh(subgraph_route_logits)
            if pred.size(-1) == 2:
                pred_conf = torch.softmax(pred.detach(), dim=-1).max(
                    dim=-1,
                    keepdim=True,
                ).values
            elif pred.size(-1) == 1:
                pred_prob = torch.sigmoid(pred.detach())
                pred_conf = (pred_prob - 0.5).abs() * 2.0
            else:
                pred_conf = pred.new_full((pred.size(0), 1), 0.5)
            gate_center = self.support_subgraph_dual_uncert_gate_center.clamp(
                0.50, 0.95
            )
            gate_scale = self.support_subgraph_dual_uncert_gate_scale.clamp(
                1.0, 20.0
            )
            uncert_gate = torch.sigmoid(gate_scale * (gate_center - pred_conf))
            pred = pred + (
                torch.sigmoid(self.support_subgraph_route_alpha) *
                uncert_gate *
                subgraph_route_logits
            )
        if self.use_support_class_split_subgraph_dual_mix_disagree_gate_expert:
            if pair_subgraph_route_repr is None:
                pair_subgraph_route_repr = torch.zeros_like(pair_repr)
            subgraph_route_logits = self.support_subgraph_route_head(
                pair_subgraph_route_repr[pair_inv][mask]
            )
            if pair_subgraph_route_support is not None:
                subgraph_route_logits = (
                    pair_subgraph_route_support[pair_inv][mask] *
                    subgraph_route_logits
                )
            if self.use_bounded_support_residuals:
                subgraph_route_logits = torch.tanh(subgraph_route_logits)
            if pair_subgraph_route_support is None:
                subgraph_support = pair_repr.new_zeros((num_pairs, 1))
            else:
                subgraph_support = pair_subgraph_route_support
            if pair_class_split_subgraph_route_support is None:
                if pair_class_split_support is None:
                    split_support = pair_repr.new_zeros((num_pairs, 1))
                else:
                    split_support = pair_class_split_support
            else:
                split_support = pair_class_split_subgraph_route_support
            support_disagree = (
                subgraph_support[pair_inv][mask] -
                split_support[pair_inv][mask]
            ).abs()
            gate_center = self.support_subgraph_dual_disagree_gate_center.clamp(
                0.01, 0.99
            )
            gate_scale = self.support_subgraph_dual_disagree_gate_scale.clamp(
                1.0, 20.0
            )
            disagree_gate = torch.sigmoid(
                gate_scale * (support_disagree - gate_center)
            )
            pred = pred + (
                torch.sigmoid(self.support_subgraph_route_alpha) *
                disagree_gate *
                subgraph_route_logits
            )
        if self.use_support_class_split_subgraph_dual_resmix_expert:
            if pair_class_split_subgraph_route_repr is None:
                pair_class_split_subgraph_route_repr = torch.zeros_like(pair_repr)
            if pair_subgraph_route_repr is None:
                pair_subgraph_route_repr = torch.zeros_like(pair_repr)
            split_route_repr = pair_class_split_subgraph_route_repr[pair_inv][mask]
            subgraph_route_repr = pair_subgraph_route_repr[pair_inv][mask]
            split_route_logits = self.support_class_split_subgraph_route_head(
                split_route_repr
            )
            subgraph_route_logits = self.support_subgraph_route_head(
                subgraph_route_repr
            )
            if pair_class_split_subgraph_route_support is not None:
                split_route_logits = (
                    pair_class_split_subgraph_route_support[pair_inv][mask] *
                    split_route_logits
                )
            if pair_subgraph_route_support is not None:
                subgraph_route_logits = (
                    pair_subgraph_route_support[pair_inv][mask] *
                    subgraph_route_logits
                )
            if self.use_bounded_support_residuals:
                split_route_logits = torch.tanh(split_route_logits)
                subgraph_route_logits = torch.tanh(subgraph_route_logits)
            support_terms = []
            for support in (
                pair_class_split_subgraph_route_support,
                pair_subgraph_route_support,
                pair_class_split_support,
                pair_sequence_support,
                pair_boundary_support,
                pair_terminal_support,
            ):
                if support is None:
                    support_terms.append(pair_repr.new_zeros((num_pairs, 1)))
                else:
                    support_terms.append(support)
            if pair_support_features is None:
                pair_support_features = pair_repr.new_zeros(
                    (num_pairs, self.support_feature_dim)
                )
            support_context = torch.cat(support_terms, dim=-1)[pair_inv][mask]
            pair_support_context = pair_support_features[pair_inv][mask]
            if pred.size(-1) == 2:
                pred_margin = pred[:, 1:2] - pred[:, 0:1]
                pred_prob = torch.softmax(pred, dim=-1)[:, 1:2]
                pred_conf = torch.softmax(pred, dim=-1).max(
                    dim=-1,
                    keepdim=True,
                ).values
            elif pred.size(-1) == 1:
                pred_margin = pred
                pred_prob = torch.sigmoid(pred)
                pred_conf = (pred_prob - 0.5).abs() * 2.0
            else:
                pred_margin = pred.new_zeros((pred.size(0), 1))
                pred_prob = pred_margin
                pred_conf = pred_margin
            resmix_context = torch.cat(
                (
                    split_route_repr,
                    subgraph_route_repr,
                    pair_support_context,
                    support_context,
                    pred_margin,
                    pred_margin.abs(),
                    pred_prob,
                    pred_conf,
                ),
                dim=-1,
            )
            mix_gate = torch.sigmoid(
                self.support_class_split_subgraph_dual_resmix_gate(resmix_context)
            )
            mix_logits = (
                mix_gate * split_route_logits +
                (1.0 - mix_gate) * subgraph_route_logits
            )
            pred = pred + (
                torch.sigmoid(self.support_subgraph_route_alpha) *
                mix_logits
            )
        if (
            self.use_support_class_split_subgraph_margin_calibration and
            pair_support_features is not None
        ):
            if pair_class_split_subgraph_route_repr is None:
                pair_class_split_subgraph_route_repr = torch.zeros_like(pair_repr)
            support_terms = []
            for support in (
                pair_class_split_subgraph_route_support,
                pair_class_split_support,
                pair_subgraph_route_support,
                pair_boundary_support,
                pair_sequence_support,
                pair_terminal_support,
            ):
                if support is None:
                    support_terms.append(pair_repr.new_zeros((num_pairs, 1)))
                else:
                    support_terms.append(support)
            margin_context = torch.cat(
                (
                    pair_class_split_subgraph_route_repr,
                    pair_support_features,
                    torch.cat(support_terms, dim=-1),
                ),
                dim=-1,
            )
            margin_context = margin_context[pair_inv][mask]
            if self.use_support_class_split_subgraph_margin_pred_calibration:
                pred_for_calib = pred.detach()
                if pred_for_calib.size(-1) == 2:
                    pred_margin = pred_for_calib[:, 1:2] - pred_for_calib[:, 0:1]
                    pred_prob = torch.softmax(pred_for_calib, dim=-1)[:, 1:2]
                    pred_conf = torch.softmax(pred_for_calib, dim=-1).max(
                        dim=-1,
                        keepdim=True,
                    ).values
                elif pred_for_calib.size(-1) == 1:
                    pred_margin = pred_for_calib
                    pred_prob = torch.sigmoid(pred_for_calib)
                    pred_conf = (pred_prob - 0.5).abs() * 2.0
                else:
                    pred_margin = pred_for_calib.new_zeros((pred_for_calib.size(0), 1))
                    pred_prob = pred_margin
                    pred_conf = pred_margin
                margin_context = torch.cat(
                    (
                        margin_context,
                        pred_margin,
                        pred_margin.abs(),
                        pred_prob,
                        pred_conf,
                    ),
                    dim=-1,
                )
            margin_params = self.support_class_split_subgraph_margin_calib(
                margin_context
            )
            if pair_class_split_subgraph_route_support is None:
                margin_support = torch.ones_like(margin_params[:, :1])
            else:
                margin_support = pair_class_split_subgraph_route_support[pair_inv][mask]
            margin_alpha = torch.sigmoid(
                self.support_class_split_subgraph_margin_alpha
            )
            margin_scale = 1.0 + (
                (0.65 if self.use_support_class_split_subgraph_margin_pred_calibration else 0.50) *
                margin_alpha * margin_support *
                torch.tanh(margin_params[:, :1])
            )
            margin_bias = (
                (1.00 if self.use_support_class_split_subgraph_margin_pred_calibration else 0.75) *
                margin_alpha * margin_support *
                torch.tanh(margin_params[:, 1:2])
            )
            pred = self._apply_signed_margin_calibration(
                pred,
                margin_scale,
                margin_bias,
            )
        if (
            self.use_support_class_split_subgraph_asym_calibration and
            pair_support_features is not None
        ):
            if pair_class_split_subgraph_route_repr is None:
                pair_class_split_subgraph_route_repr = torch.zeros_like(pair_repr)
            support_terms = []
            for support in (
                pair_class_split_subgraph_route_support,
                pair_class_split_support,
                pair_subgraph_route_support,
                pair_boundary_support,
                pair_sequence_support,
                pair_terminal_support,
            ):
                if support is None:
                    support_terms.append(pair_repr.new_zeros((num_pairs, 1)))
                else:
                    support_terms.append(support)
            pred_for_calib = pred.detach()
            if pred_for_calib.size(-1) == 2:
                pred_margin = pred_for_calib[:, 1:2] - pred_for_calib[:, 0:1]
                pred_prob = torch.softmax(pred_for_calib, dim=-1)[:, 1:2]
                pred_conf = torch.softmax(pred_for_calib, dim=-1).max(
                    dim=-1,
                    keepdim=True,
                ).values
            elif pred_for_calib.size(-1) == 1:
                pred_margin = pred_for_calib
                pred_prob = torch.sigmoid(pred_for_calib)
                pred_conf = (pred_prob - 0.5).abs() * 2.0
            else:
                pred_margin = pred_for_calib.new_zeros((pred_for_calib.size(0), 1))
                pred_prob = pred_margin
                pred_conf = pred_margin
            asym_context = torch.cat(
                (
                    pair_class_split_subgraph_route_repr[pair_inv][mask],
                    pair_support_features[pair_inv][mask],
                    torch.cat(support_terms, dim=-1)[pair_inv][mask],
                    pred_margin,
                    pred_margin.abs(),
                    pred_prob,
                    pred_conf,
                ),
                dim=-1,
            )
            asym_params = self.support_class_split_subgraph_asym_calib(
                asym_context
            )
            if pair_class_split_subgraph_route_support is None:
                asym_support = torch.ones_like(asym_params[:, :1])
            else:
                asym_support = pair_class_split_subgraph_route_support[pair_inv][mask]
            asym_alpha = torch.sigmoid(self.support_class_split_subgraph_asym_alpha)
            asym_scale = 1.0 + (
                0.55 *
                asym_alpha *
                asym_support *
                torch.tanh(asym_params[:, :2])
            )
            asym_bias = (
                0.80 *
                asym_alpha *
                asym_support *
                torch.tanh(asym_params[:, 2:4])
            )
            if pred.size(-1) == 2:
                pred = pred * asym_scale + asym_bias
            elif pred.size(-1) == 1:
                pred = pred * asym_scale[:, :1] + asym_bias[:, :1]
        return pred, batch[task].y[mask]

    def _apply_index(self, batch):
        task = cfg.dataset.task_entity
        mask = self._edge_mask(batch)

        task = cfg.dataset.task_entity
        edge_index = batch[task].edge_index

        return torch.cat((batch[task[0]].x[edge_index[0, mask]],
                          batch[task[2]].x[edge_index[1, mask]],
                          batch[task].edge_attr[mask]), dim=-1), \
               batch[task].y[mask]

    def _evidence_gate_head(self, batch):
        '''Scale-agnostic evidence decoder: z_base (+) bounded prototype core
        (+) uncertainty-gated higher-order structural evidence.'''
        if self.eg_gate_v4:
            # These tensors belong to one forward pass only. Clearing them here
            # prevents a skipped/failed batch from reusing an old auxiliary loss.
            self._eg_gate_penalty = None
            self._eg_struct_aux_logits = None
            self._eg_struct_aux_labels = None
            self._eg_struct_aux_scale = 0.0

        task = cfg.dataset.task_entity
        feat_all, edge_index = self._edge_inputs(batch)
        mask = self._edge_mask(batch)
        labels = batch[task].y[mask]

        feat_all = torch.nan_to_num(feat_all)

        # Shared representation over all sampled edges; targets are a subset.
        h_all = self.eg_repr(feat_all)
        if not torch.isfinite(h_all).all():
            raise FloatingPointError("evidence_gate produced non-finite edge representations")
        h = h_all[mask]

        # (1) Base FraudGT decoder.
        z_base = self.layer_post_mp(feat_all[mask])
        if not torch.isfinite(z_base).all():
            raise FloatingPointError("evidence_gate produced non-finite base logits")

        # (2) Bounded class-prototype residual (stable core evidence). The
        # prototype context is computed from banks built on *past* train
        # batches; banks are updated only afterwards and only in training,
        # so no label leaks into the current prediction or into eval.
        if self.eg_gate_v4_noproto:
            # This ablation removes all prototype information: no bank lookup,
            # no residual and no prototype signals passed to the router.
            proto_margin = z_base.new_zeros((z_base.size(0), 1))
            ready = z_base.new_zeros((z_base.size(0), 1))
            z_core = z_base
        else:
            (pos_proto, neg_proto, pos_sim, neg_sim, proto_margin, ready,
             pos_peak, neg_peak, pos_spread, neg_spread) = \
                self._support_class_proto_context(h)
            if self.use_dmprd and not self.dmprd_use_distribution_stats:
                pos_peak = torch.zeros_like(pos_peak)
                neg_peak = torch.zeros_like(neg_peak)
                pos_spread = torch.zeros_like(pos_spread)
                neg_spread = torch.zeros_like(neg_spread)
            proto_feat = torch.cat(
                [pos_sim, neg_sim, proto_margin, ready,
                 pos_peak, neg_peak, pos_spread, neg_spread], dim=-1)
            z_proto = self.eg_proto_head(torch.nan_to_num(proto_feat))
            if not torch.isfinite(z_proto).all():
                raise FloatingPointError("evidence_gate produced non-finite prototype logits")
            if self.use_dmprd:
                dmprd_delta = self.dmprd_delta_max * torch.tanh(z_proto)
                dmprd_beta = self.dmprd_beta_max * torch.sigmoid(
                    self.eg_proto_alpha)
                z_core = z_base + dmprd_beta * ready * dmprd_delta
            else:
                z_core = z_base + (
                    torch.sigmoid(self.eg_proto_alpha) * ready * z_proto)
        if not torch.isfinite(z_core).all():
            raise FloatingPointError("evidence_gate produced non-finite core logits")

        # M1 ablation (`evidence_gate_proto`): base decoder + bounded prototype
        # residual only. No structural evidence, no gate. Everything else in the
        # forward is identical to v2, so M1 vs v2 differs by exactly this branch.
        if self.eg_proto_only:
            if self.use_dmprd and not self.training:
                self._dmprd_log_step += 1
                if self._dmprd_log_step % 64 == 1:
                    with torch.no_grad():
                        delta_abs = dmprd_delta.detach().float().abs().view(-1)
                        margin = proto_margin.detach().float().view(-1)
                        ready_flat = ready.detach().float().view(-1)
                        logging.info(
                            "[dmprd/%s] slots=%d distribution=%s beta=%.4f | "
                            "delta_abs: mean=%.4f p90=%.4f max=%.4f | "
                            "proto_margin: mean=%.4f std=%.4f | ready=%.4f",
                            getattr(batch, 'split', '?'),
                            self.num_class_proto_slots,
                            self.dmprd_use_distribution_stats,
                            dmprd_beta.item(),
                            delta_abs.mean().item(),
                            torch.quantile(delta_abs, 0.90).item(),
                            delta_abs.max().item(),
                            margin.mean().item(),
                            margin.std(unbiased=False).item(),
                            ready_flat.mean().item())
            if self.training:
                self._update_support_class_prototypes(h, labels)
            return z_core, labels

        # (3) Higher-order 1-hop structural evidence (local transaction
        # neighbourhood), valid when source and target share a node type.
        if task[0] == task[2]:
            num_nodes = batch[task[0]].x.size(0)
            src_all, dst_all = edge_index[0], edge_index[1]
            ctx_in = scatter(h_all, dst_all, dim=0,
                             dim_size=num_nodes, reduce='mean')
            ctx_out = scatter(h_all, src_all, dim=0,
                              dim_size=num_nodes, reduce='mean')
            tgt_src, tgt_dst = src_all[mask], dst_all[mask]
            struct_feat = torch.cat(
                [ctx_out[tgt_src], ctx_in[tgt_dst], h], dim=-1)
            z_struct = self.eg_struct_head(torch.nan_to_num(struct_feat))
            if not torch.isfinite(z_struct).all():
                raise FloatingPointError("evidence_gate produced non-finite structural logits")
            # Normalised 1-hop support: how many neighbours the aggregation saw
            # (u out-degree + v in-degree). A bounded data statistic telling the
            # gate whether the structural context is meaningful. No parameters.
            ones = h_all.new_ones((h_all.size(0), 1))
            out_deg = scatter(ones, src_all, dim=0,
                              dim_size=num_nodes, reduce='sum')
            in_deg = scatter(ones, dst_all, dim=0,
                             dim_size=num_nodes, reduce='sum')
            struct_support = torch.tanh(
                0.5 * torch.log1p(out_deg[tgt_src] + in_deg[tgt_dst]))
        else:
            z_struct = torch.zeros_like(z_core)
            struct_support = z_core.new_zeros((z_core.size(0), 1))

        # Per-sample uncertainty of the core decision. Binary classification in
        # this codebase is represented by a single logit, so use sigmoid entropy
        # there; the multiclass path keeps the usual normalized softmax entropy.
        num_outputs = z_core.size(-1)
        if num_outputs == 1:
            p_pos = torch.sigmoid(z_core).clamp(min=1e-6, max=1.0 - 1e-6)
            uncertainty = -(
                p_pos * p_pos.log() +
                (1.0 - p_pos) * (1.0 - p_pos).log()
            ) / math.log(2.0)
            base_margin = z_base
        else:
            p = F.softmax(z_core, dim=-1).clamp(min=1e-6)
            uncertainty = -(p * p.log()).sum(
                dim=-1, keepdim=True) / math.log(float(num_outputs))
            if num_outputs == 2:
                base_margin = z_base[:, 1:2] - z_base[:, 0:1]
            else:
                base_margin = z_base.max(dim=-1, keepdim=True).values
        uncertainty = torch.nan_to_num(uncertainty)

        delta_struct = None
        if self.eg_gate_v4:
            # The structural branch proposes a bounded logit correction. Unlike
            # v3 convex fusion, opening the gate never removes the reliable core.
            delta_struct = (
                self.eg_struct_residual_scale * torch.tanh(z_struct))
            delta_magnitude = delta_struct.abs().mean(dim=-1, keepdim=True)

            if num_outputs == 1:
                core_margin = z_core
                delta_margin = delta_struct
                core_delta_alignment = (
                    torch.tanh(core_margin.detach()) *
                    torch.tanh(delta_margin.detach()))
            elif num_outputs == 2:
                core_margin = z_core[:, 1:2] - z_core[:, 0:1]
                delta_margin = (
                    delta_struct[:, 1:2] - delta_struct[:, 0:1])
                core_delta_alignment = (
                    torch.tanh(core_margin.detach()) *
                    torch.tanh(delta_margin.detach()))
            else:
                core_delta_alignment = F.cosine_similarity(
                    z_core.detach(), delta_struct.detach(),
                    dim=-1, eps=1e-6).unsqueeze(-1)

            gate_feat = torch.cat(
                [uncertainty, proto_margin.abs(), ready, struct_support,
                 torch.tanh(base_margin).abs(), delta_magnitude,
                 core_delta_alignment], dim=-1)
            # Routing should learn how to use evidence, not reshape the core or
            # structural expert through an indirect gate-input gradient path.
            gate_feat = torch.nan_to_num(gate_feat).detach()

            if self.eg_gate_v4_nogate:
                g = torch.ones_like(uncertainty)
            else:
                learned_g = torch.sigmoid(self.eg_gate_v4_head(gate_feat))
                if (self.training and
                        self._eg_cur_epoch < self.eg_gate_warmup_epochs):
                    # A small nonzero opening trains the structural expert from
                    # the first step while preserving most of the core decision.
                    g = torch.full_like(learned_g, self.eg_gate_init_open)
                else:
                    g = learned_g
        elif self.eg_gate_v3:
            # v3 gate: routing scalars ONLY (no h, so it cannot memorise a
            # constant), started shut via a dedicated negative bias.
            gate_feat = torch.cat(
                [uncertainty, proto_margin.abs(), ready, struct_support,
                 torch.tanh(base_margin).abs()], dim=-1)
            gate_feat = torch.nan_to_num(gate_feat)
            g = torch.sigmoid(
                self.eg_gate_v3_head(gate_feat) + self.eg_gate_v3_bias)
            # Warm-up: force the gate shut for the first few epochs so the base
            # and prototype core settle before the structural branch can act.
            if self.training and self._eg_cur_epoch < self.eg_gate_warmup_epochs:
                g = torch.zeros_like(g)
        else:
            gate_feat = torch.cat(
                [h, uncertainty, proto_margin.abs(), ready,
                 torch.tanh(base_margin)], dim=-1)
            gate_feat = torch.nan_to_num(gate_feat)
            g = torch.sigmoid(self.eg_gate(gate_feat))
        if not torch.isfinite(g).all():
            raise FloatingPointError("evidence_gate produced non-finite gates")

        # Diagnostic (eval only, throttled): report whether the per-sample gate
        # actually varies -- the core premise of the gate. If g collapses to a
        # near constant (std ~ 0, frac<0.05 or frac>0.95 ~ 1.0), the routing is
        # inactive and any effect is really just the prototype core.
        if not self.training:
            self._eg_log_step = getattr(self, '_eg_log_step', 0) + 1
            if self._eg_log_step % 64 == 1:
                with torch.no_grad():
                    gf = g.detach().float().view(-1)
                    uf = uncertainty.detach().float().view(-1)
                    if self.eg_gate_v4:
                        df = delta_struct.detach().float().abs().view(-1)
                        logging.info(
                            "[evidence_gate_v4/%s] g: mean=%.4f std=%.4f "
                            "min=%.4f max=%.4f p10=%.4f p50=%.4f p90=%.4f "
                            "frac<.05=%.3f frac>.95=%.3f | delta_abs: "
                            "mean=%.4f p90=%.4f | uncert: mean=%.4f std=%.4f",
                            getattr(batch, 'split', '?'),
                            gf.mean().item(), gf.std(unbiased=False).item(),
                            gf.min().item(), gf.max().item(),
                            torch.quantile(gf, 0.10).item(),
                            torch.quantile(gf, 0.50).item(),
                            torch.quantile(gf, 0.90).item(),
                            (gf < 0.05).float().mean().item(),
                            (gf > 0.95).float().mean().item(),
                            df.mean().item(), torch.quantile(df, 0.90).item(),
                            uf.mean().item(), uf.std(unbiased=False).item())
                    else:
                        logging.info(
                            "[evidence_gate/%s] g: mean=%.4f std=%.4f min=%.4f "
                            "max=%.4f p10=%.4f p50=%.4f p90=%.4f frac<.05=%.3f "
                            "frac>.95=%.3f | uncert: mean=%.4f std=%.4f",
                            getattr(batch, 'split', '?'),
                            gf.mean().item(), gf.std(unbiased=False).item(),
                            gf.min().item(), gf.max().item(),
                            torch.quantile(gf, 0.10).item(),
                            torch.quantile(gf, 0.50).item(),
                            torch.quantile(gf, 0.90).item(),
                            (gf < 0.05).float().mean().item(),
                            (gf > 0.95).float().mean().item(),
                            uf.mean().item(), uf.std(unbiased=False).item())

        if self.eg_gate_v4:
            z_candidate = z_core + delta_struct
            if self.eg_gate_v4_nogate:
                z_final = z_candidate
            else:
                z_final = z_core + g * delta_struct
            if self.training:
                if not self.eg_gate_v4_nogate:
                    self._eg_gate_penalty = self.eg_gate_budget_weight * (
                        g.mean() - self.eg_gate_budget_target).pow(2)
                if self._eg_cur_epoch < self.eg_struct_aux_epochs:
                    # Detach the core in the auxiliary path: this objective
                    # teaches the structural expert to correct core errors
                    # without simply giving the core a second loss term.
                    self._eg_struct_aux_logits = (
                        z_core.detach() + delta_struct)
                    self._eg_struct_aux_labels = labels
                    self._eg_struct_aux_scale = self.eg_struct_aux_weight
        elif self.eg_gate_v3:
            # Convex fusion: opening the gate discards the reliable core, so g
            # carries a real opportunity cost and cannot trivially saturate.
            # No redundant global scale -- g is the branch's only modulator.
            z_final = (1.0 - g) * z_core + g * z_struct
            # L1/budget penalty (train only) so the gate defaults closed.
            if self.training:
                self._eg_gate_penalty = self.eg_gate_l1 * g.mean()
        else:
            z_final = z_core + g * torch.sigmoid(self.eg_struct_alpha) * z_struct
        if not torch.isfinite(z_final).all():
            raise FloatingPointError("evidence_gate produced non-finite final logits")

        # Update prototype banks from the current (train) batch, after
        # prediction. Frozen at eval because self.training is False.
        if self.training and not self.eg_gate_v4_noproto:
            self._update_support_class_prototypes(h, labels)

        return z_final, labels

    def forward(self, batch):
        if self.use_pair_chain_head:
            return self._pair_chain_head(batch)
        if self.use_evidence_gate:
            return self._evidence_gate_head(batch)
        pred, label = self._apply_index(batch)
        pred = self.layer_post_mp(pred)

        return pred, label
