from yacs.config import CfgNode as CN

from fraudGT.graphgym.register import register_config


@register_config("cfg_cdvt")
def set_cfg_cdvt(cfg):
    cfg.cdvt = CN()
    cfg.cdvt.variant = "dual_view"
    cfg.cdvt.hidden_dim = 64
    cfg.cdvt.num_heads = 4
    cfg.cdvt.num_layers = 2
    cfg.cdvt.dropout = 0.2
    cfg.cdvt.history_k = 4
    cfg.cdvt.history_hops = 2
    cfg.cdvt.max_events = 48
    cfg.cdvt.time_window = -1
    cfg.cdvt.event_cache_size = 8192
    cfg.cdvt.lambda_cons = 0.0
