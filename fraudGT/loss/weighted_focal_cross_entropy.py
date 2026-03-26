import torch
import torch.nn.functional as F

from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.register import register_loss


def _resolve_class_weight(pred, true):
    if cfg.model.loss_fun_weight is None:
        total = true.size(0)
        n_classes = pred.shape[1] if pred.ndim > 1 else 2
        label_count = torch.bincount(true)
        label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
        cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
        cluster_sizes[torch.unique(true)] = label_count
        weight = (total - cluster_sizes).float() / total
        weight *= (cluster_sizes > 0).float()
        return weight
    return torch.tensor(cfg.model.loss_fun_weight, device=pred.device)


@register_loss('weighted_focal_cross_entropy')
def weighted_focal_cross_entropy(pred, true, epoch):
    if cfg.model.loss_fun != 'weighted_focal_cross_entropy':
        return None

    gamma = float(cfg.model.loss_fun_gamma)
    weight = _resolve_class_weight(pred, true)

    if pred.ndim > 1:
        log_prob = F.log_softmax(pred, dim=-1)
        prob = log_prob.exp()
        pt = prob.gather(-1, true.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
        ce = F.nll_loss(log_prob, true, weight=weight, reduction='none')
        loss = ((1.0 - pt) ** gamma * ce).mean()
        return loss, log_prob

    true = true.float()
    prob = torch.sigmoid(pred)
    pt = torch.where(true > 0, prob, 1.0 - prob).clamp_min(1e-12)
    sample_weight = weight[true.long()]
    bce = F.binary_cross_entropy_with_logits(pred, true, reduction='none')
    loss = (sample_weight * ((1.0 - pt) ** gamma) * bce).mean()
    return loss, prob
