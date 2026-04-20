"""Custom ChempropMetric subclasses for auxiliary-loss training.

Each class extends ``chemprop.nn.metrics.ChempropMetric`` and overrides
``_calc_unreduced_loss`` to return a per-sample loss tensor that the
outer reduction (weighted mean) will aggregate. Broadcast any scalar
auxiliary terms to the per-sample shape so they survive reduction.
"""

import torch
import torch.nn.functional as F
from chemprop.nn.metrics import ChempropMetric


class RelativeDistanceMSE(ChempropMetric):
    """MSE + batchwise all-pairs relative-distance auxiliary loss.

    Main:  ``MSE(pred, target)`` per-sample (shape ``(B,)``).
    Aux:   ``MSE(|pred_i - pred_j|, |target_i - target_j|)`` averaged over
           upper-triangle pairs (i < j) -- scalar.
    Total: ``main + aux_weight * aux``, with aux broadcast to ``(B,)``.

    Teaches the model to preserve relative distances between compounds,
    which correlates with rank metrics (Spearman, RAE). With
    ``aux_weight <= 0.1`` the main MSE dominates; larger values sacrifice
    absolute accuracy for ranking quality.

    Reference: Dong et al. 2025, DOI 10.1016/j.jmgm.2025.109014 (FMGCL).
    """

    def __init__(self, aux_weight: float = 0.1, task_weights=1.0):
        super().__init__(task_weights)
        self.aux_weight = aux_weight

    def _calc_unreduced_loss(self, preds, targets, *args):
        main = F.mse_loss(preds, targets, reduction="none")
        p = preds.squeeze(-1)
        t = targets.squeeze(-1)
        n = p.shape[0]
        if n < 2:
            return main
        iu = torch.triu_indices(n, n, offset=1, device=p.device)
        d_pred = torch.abs(p[iu[0]] - p[iu[1]])
        d_true = torch.abs(t[iu[0]] - t[iu[1]])
        aux = F.mse_loss(d_pred, d_true, reduction="mean")
        return main + self.aux_weight * aux.expand_as(main)
