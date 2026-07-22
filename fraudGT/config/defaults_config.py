from fraudGT.graphgym.register import register_config


@register_config('overwrite_defaults')
def overwrite_defaults_cfg(cfg):
    """Overwrite the default config values that are first set by GraphGym in
    unifiedGT.graphgym.config.set_cfg

    WARNING: At the time of writing, the order in which custom config-setting
    functions like this one are executed is random; see the referenced `set_cfg`
    Therefore never reset here config options that are custom added, only change
    those that exist in core GraphGym.
    """

    # Training (and validation) pipeline mode
    cfg.train.mode = 'custom'  # 'standard' uses PyTorch-Lightning since PyG 2.1

    # Overwrite default dataset name
    cfg.dataset.name = 'none'

    # Overwrite default rounding precision
    cfg.round = 5


@register_config('extended_cfg')
def extended_cfg(cfg):
    """General extended config options.
    """

    # Additional name tag used in `run_dir` and `wandb_name` auto generation.
    cfg.name_tag = ""

    # In training, if True (and also cfg.train.enable_ckpt is True) then
    # always checkpoint the current best model based on validation performance,
    # instead, when False, follow cfg.train.eval_period checkpointing frequency.
    cfg.train.ckpt_best = False
    cfg.train.selection_precision_weight = 0.0
    cfg.train.selection_topk_by_metric = 1
    cfg.train.selection_tiebreak_metric = ""
    cfg.train.selection_tiebreak_agg = "argmax"
    cfg.train.early_stop = False
    cfg.train.early_stop_metric = ""
    cfg.train.early_stop_min_epoch = 0
    cfg.train.early_stop_patience = 0
    cfg.train.early_stop_delta = 0.0

    # Enable tqdm progress bar during training/validation/testing
    cfg.train.tqdm = False
    cfg.val.tqdm = False

    # In training, when the graph is very large, you may not want to iterate through
    # all the batches available. You can sample a subset of available batches as
    # the HGT does in their code.
    cfg.train.iter_per_epoch = 0

    # In evaluation, you may want to reduce the evaluation time like the above rationale
    # for training. However, to reduce the periodic variance due to cycle through the val/test
    # set, we fixed the evaluation sampling to the same set of batches. So the performance
    # would be evaluated under the same set of data.
    cfg.val.iter_per_epoch = 0
    # Explicit experiment contract; the historical sampler has no panel logic.
    cfg.val.fixed_target_panel = False

    # Sampling parameters
    cfg.train.persistent_workers = False
    cfg.train.pin_memory = False

    # NeighborSampler / HGTSampler: number of sampled nodes per layer for each node type
    cfg.train.neighbor_sizes_dict = ""

    # RandomNodeLoader: number of partitions
    cfg.train.num_parts = 10

    # AddEgoID option
    cfg.train.add_ego_id = False

    # APPNP hyperparameter
    cfg.gnn.K = 10
    cfg.gnn.alpha = 0.1

    # NAGphormer hop2seq hyperparameter
    cfg.gnn.hops = 7

    # SHGN hyperparameter
    cfg.gnn.residual = True

    # LSGNN hyperparameter
    cfg.gnn.A_embed = True

    cfg.gnn.batch_norm = False
    cfg.gnn.layer_norm = False

    cfg.gnn.input_dropout = 0.0

    cfg.gnn.attn_dropout = 0.0

    cfg.gnn.edge_updates = False
    cfg.gnn.use_linear = False
    cfg.gnn.output_l2_norm = False
    cfg.gnn.jumping_knowledge = False

    cfg.model.loss_fun_weight = []
    cfg.model.loss_fun_gamma = 2.0
    cfg.model.auto_tune_thresh = False

    # evidence_gate_v3: L1 penalty and zero-gate warm-up.
    cfg.model.eg_gate_l1 = 1e-3
    cfg.model.eg_gate_warmup_epochs = 20

    # evidence_gate_v4: residual structural expert and target-budget router.
    # Warm-up uses a fixed nonzero gate, then the learned per-sample router takes
    # over. The auxiliary candidate loss is active only in early training.
    cfg.model.eg_gate_init_open = 0.10
    cfg.model.eg_gate_budget_target = 0.10
    cfg.model.eg_gate_budget_weight = 1e-2
    cfg.model.eg_struct_aux_weight = 0.25
    cfg.model.eg_struct_aux_epochs = 60
    cfg.model.eg_struct_residual_scale = 1.0

    # DMPRD: one clean prototype-residual path with controlled ablation knobs.
    cfg.model.dmprd_num_slots = 4
    cfg.model.dmprd_use_distribution_stats = True
    cfg.model.dmprd_delta_max = 1.0
    cfg.model.dmprd_beta_max = 1.0
    # P0: reliability-calibrated prototype residual. Disabled by default so
    # existing DMPRD/A2 checkpoints and commands keep their original behavior.
    cfg.model.dmprd_use_reliability_gate = False
    cfg.model.dmprd_use_sample_gate = True
    cfg.model.dmprd_support_tau = 16.0
    cfg.model.dmprd_variance_tau = 0.25
    cfg.model.dmprd_margin_tau = 0.10
    cfg.model.dmprd_reliability_floor = 0.25

    # CAMPR: counterfactual advantage-guided, mean-preserving A2 routing.
    cfg.model.campr_aux_weight = 0.20
    cfg.model.campr_adv_temperature = 1.0
    cfg.model.campr_adv_scale_floor = 1e-4
    cfg.model.campr_route_min = 0.50
    cfg.model.campr_route_max = 1.50

    # COSTAR: detached A2-anchored orthogonal prototype adapter. The current
    # implementation uses a train-batch stratified threshold-transfer
    # approximation; validation and test labels are never consumed by it.
    cfg.model.costar_router_hidden = 32
    cfg.model.costar_ema_decay = 0.995
    cfg.model.costar_center_decay = 0.99
    cfg.model.costar_consistency_tau = 0.25
    cfg.model.costar_soft_f1_temperature = 0.10
    cfg.model.costar_threshold_perturb = 0.10
    cfg.model.costar_platform_tolerance = 0.01
    cfg.model.costar_cvar_fraction = 0.50
    cfg.model.costar_adapter_weight = 0.05
    cfg.model.costar_rank_weight = 0.25
    cfg.model.costar_safe_weight = 1.0
    cfg.model.costar_orth_weight = 0.10
    cfg.model.costar_time_weight = 0.05
    cfg.model.costar_safe_margin = 0.0
    cfg.model.costar_rank_safe_margin = 0.0
    cfg.model.costar_start_epoch = 10
    cfg.model.costar_fallback_confidence = 0.25
    cfg.model.costar_fallback_delta = 1e-4

    # CPTR: train-only bidirectional threshold transfer with conservative
    # temporal-environment uplift deployment.
    cfg.model.cptr_router_hidden = 32
    cfg.model.cptr_max_correction = 0.10
    cfg.model.cptr_boundary_temperature = 0.10
    cfg.model.cptr_soft_f1_temperature = 0.10
    cfg.model.cptr_adapter_weight = 0.05
    cfg.model.cptr_safe_weight = 1.0
    cfg.model.cptr_safe_margin = 0.0
    cfg.model.cptr_threshold_decay = 0.99
    cfg.model.cptr_uplift_decay = 0.95
    cfg.model.cptr_uplift_lcb_z = 1.0
    cfg.model.cptr_gate_scale = 0.01
    cfg.model.cptr_min_env_count = 8
    cfg.model.cptr_min_threshold_count = 8
    cfg.model.cptr_start_epoch = 10
