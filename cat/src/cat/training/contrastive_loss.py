import torch
from torch import nn

def _sampling_pools(mask, valid_mask) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten a mask into positive/negative sampling pools, excluding ignored pixels.

    :param mask: torch.Tensor - [B, 1, H, W] label mask (>0.5 = positive)
    :param valid_mask: torch.Tensor | None - [B, 1, H, W] bool/float, true = supervised, None = all supervised
    :return: tuple[torch.Tensor, torch.Tensor] - (pos_mask, neg_mask), both [B, HW] bool
    """
    B = mask.shape[0]
    pos_mask = mask[:, 0].reshape(B, -1) > 0.5
    neg_mask = ~pos_mask
    if valid_mask is not None:
        valid_flat = valid_mask[:, 0].reshape(B, -1) > 0.5
        pos_mask = pos_mask & valid_flat
        neg_mask = neg_mask & valid_flat
    return pos_mask, neg_mask

def _zero_loss(features) -> torch.Tensor:
    """
    Zero loss attached to ``features`` (zero gradient, NaN-free for any values).

    :param features: torch.Tensor - tensor the zero loss stays attached to
    :return: torch.Tensor - scalar zero loss
    """
    # empty slice keeps autograd edge so GradScaler records inf checks, no feature value read
    zero = features.reshape(-1)[:0].float().sum()
    return zero if zero.requires_grad else zero.requires_grad_()

class ContrastiveLoss(nn.Module):
    """
    Pixel-level InfoNCE loss over traversable and non-traversable features.
    """
    def __init__(self, temperature=0.07) -> None:
        """
        Initialize the contrastive loss.

        :param temperature: float - similarity temperature (tau)
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, features, mask, n_pos, n_neg, valid_mask=None) -> torch.Tensor:
        """
        Compute the contrastive loss for one mask.

        :param features: torch.Tensor - [B, D, H, W] L2-normalized feature map
        :param mask: torch.Tensor - [B, 1, H, W] binary mask (1 = positive, 0 = negative)
        :param n_pos: int - positive sample count
        :param n_neg: int - negative sample count
        :param valid_mask: torch.Tensor | None - [B, 1, H, W] bool/float, false = ignored pixel, None = all supervised
        :return: torch.Tensor - scalar loss
        """
        return self.forward_terms(features, [(mask, n_pos, n_neg)], valid_mask=valid_mask)[0]


    def forward_terms(self, features, terms, valid_mask=None) -> list[torch.Tensor]:
        """
        Compute one contrastive loss per mask, each equal to a separate ``forward`` call (same random draws in call order).

        :param features: torch.Tensor - [B, D, H, W] L2-normalized feature map
        :param terms: list[tuple[torch.Tensor, int, int]] - (mask, n_pos, n_neg) per term, mask [B, 1, H, W] (1 = positive, 0 = negative)
        :param valid_mask: torch.Tensor | None - [B, 1, H, W] bool/float, false = ignored pixel, None = all supervised
        :return: list[torch.Tensor] - scalar loss per term
        """
        B, D, H, W = features.shape
        HW = H * W

        feat_flat = features.reshape(B, D, HW).permute(0, 2, 1)  # [b, hw, d]

        # all terms share one feature selection, so backward builds a single dense feature gradient
        plans, batch_indices, pixel_indices = [], [], []
        for mask, n_pos_target, n_neg_target in terms:
            pos_mask, neg_mask = _sampling_pools(mask, valid_mask)   # [b, hw] each

            n_pos_per = pos_mask.sum(dim=1)  # [b]
            n_neg_per = neg_mask.sum(dim=1)  # [b]

            # need >= 2 positives and >= 1 negative
            valid = (n_pos_per >= 2) & (n_neg_per >= 1)  # [b]
            if not valid.any():
                plans.append(None)
                continue

            # clamp counts to smallest pool in batch
            n_p = min(n_pos_target, n_pos_per[valid].min().item())
            n_n = min(n_neg_target, n_neg_per[valid].min().item())

            if n_p < 2 or n_n < 1:
                plans.append(None)
                continue

            valid_idx = valid.nonzero(as_tuple=True)[0]       # [v]
            V = valid_idx.shape[0]
            pos_mask_v = pos_mask[valid_idx].float()           # [v, hw]
            neg_mask_v = neg_mask[valid_idx].float()           # [v, hw]

            # batched multinomial, single kernel launch for all images
            pos_indices = torch.multinomial(pos_mask_v, n_p, replacement=False)  # [v, n_p]

            neg_indices = torch.multinomial(neg_mask_v, n_n, replacement=False)  # [v, n_n]

            # flat (image, pixel) pairs, positives then negatives per image
            selected = torch.cat((pos_indices, neg_indices), dim=1)              # [v, n_p + n_n]
            batch_indices.append(valid_idx[:, None].expand_as(selected).reshape(-1))
            pixel_indices.append(selected.reshape(-1))
            plans.append((V, n_p, n_n))

        pieces = []
        if pixel_indices:
            sampled = feat_flat[torch.cat(batch_indices), torch.cat(pixel_indices)]  # [sum v * (n_p + n_n), d]
            pieces = list(sampled.split([idx.numel() for idx in pixel_indices]))

        losses = []
        for plan in plans:
            if plan is None:
                losses.append(_zero_loss(features))
                continue
            V, n_p, n_n = plan
            pos_feat, neg_feat = pieces.pop(0).reshape(V, n_p + n_n, D).split((n_p, n_n), dim=1)

            # fp32 similarities even inside outer amp
            with torch.autocast(device_type=features.device.type, enabled=False):
                pos_feat_f = pos_feat.float()
                neg_feat_f = neg_feat.float()
                pos_sim = torch.bmm(pos_feat_f, pos_feat_f.transpose(1, 2)) / self.temperature  # [v, n_p, n_p]
                neg_sim = torch.bmm(pos_feat_f, neg_feat_f.transpose(1, 2)) / self.temperature  # [v, n_p, n_n]

            diag_mask = torch.eye(n_p, dtype=torch.bool, device=features.device)  # [n_p, n_p]
            pos_sim = pos_sim.masked_fill(diag_mask.unsqueeze(0), -float('inf'))

            # negative term of the InfoNCE denominator
            log_denom = torch.logsumexp(neg_sim, dim=2)  # [v, n_p]

            log_numer = pos_sim  # [v, n_p, n_p], diagonal is -inf

            # -(sim_ij/tau - log_denom_i)
            pair_losses = -(log_numer - log_denom.unsqueeze(2))  # [v, n_p, n_p]

            # zero diagonal, average over j != i via (n_p - 1)
            pair_losses = pair_losses.masked_fill(diag_mask.unsqueeze(0), 0.0)
            loss_per_anchor = pair_losses.sum(dim=2) / (n_p - 1)  # [v, n_p]
            loss_per_image = loss_per_anchor.mean(dim=1)           # [v]
            losses.append(loss_per_image.mean())
        return losses
