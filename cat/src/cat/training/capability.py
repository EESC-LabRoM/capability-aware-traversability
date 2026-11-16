import torch

def _channel_weights(traversability_vector) -> torch.Tensor:
    """
    Reshape per-terrain traversability to broadcast over map channels.

    :param traversability_vector: torch.Tensor - [K] or [B, K] per-terrain traversability
    :return: torch.Tensor - [1, K, 1, 1] or [B, K, 1, 1] channel weights
    """
    if traversability_vector.dim() == 1:
        return traversability_vector.view(1, -1, 1, 1)
    return traversability_vector.unsqueeze(-1).unsqueeze(-1)

def modulate_semantic_map(semantic_map, traversability_vector) -> torch.Tensor:
    """
    Scale semantic map channels by a robot's traversability weights.

    :param semantic_map: torch.Tensor - [B, K, H, W] terrain class probabilities
    :param traversability_vector: torch.Tensor - [K] or [B, K] per-terrain traversability
    :return: torch.Tensor - [B, K, H, W] robot-conditioned semantic map
    """
    return semantic_map * _channel_weights(traversability_vector)

def expected_traversability(semantic_map, traversability_vector) -> torch.Tensor:
    """
    Compute per-pixel expected traversability.

    :param semantic_map: torch.Tensor - [B, K, H, W] terrain class probabilities
    :param traversability_vector: torch.Tensor - [K] or [B, K] per-terrain traversability
    :return: torch.Tensor - [B, 1, H, W] expected traversability
    """
    return (semantic_map * _channel_weights(traversability_vector)).sum(dim=1, keepdim=True)

def refine_mask(generalist_mask, semantic_map, traversability_vector, threshold, valid_mask=None) -> torch.Tensor:
    """
    Restrict a generalist positive mask to terrain the robot can traverse.

    :param generalist_mask: torch.Tensor - [B, 1, H, W] binary mask, 1 = positive
    :param semantic_map: torch.Tensor - [B, K, H, W] terrain class probabilities
    :param traversability_vector: torch.Tensor - [K] or [B, K] per-terrain traversability
    :param threshold: float - Minimum expected traversability of a positive pixel
    :param valid_mask: torch.Tensor | None - [B, 1, H, W] false = ignored pixel, never positive
    :return: torch.Tensor - [B, 1, H, W] refined mask
    """
    pixel_trav = expected_traversability(semantic_map, traversability_vector)
    refined = generalist_mask * (pixel_trav >= threshold).float()
    if valid_mask is not None:
        refined = refined * (valid_mask > 0.5).to(refined.dtype)
    return refined
