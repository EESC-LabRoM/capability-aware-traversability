import torch.nn as nn
import timm

class FeatureEmbedder(nn.Module):
    """Extract raw patch-level features with a frozen DINOv3 backbone."""
    def __init__(
        self,
        dino_model='vit_base_patch16_dinov3.lvd1689m',
    ) -> None:
        """
        Build the frozen DINOv3 backbone in eval mode.

        :param dino_model: str - timm model name for DINOv3
        """
        super().__init__()

        self.dino_model = dino_model
        self.dino = timm.create_model(dino_model, pretrained=True, num_classes=0)
        self.dino.eval()

        for p in self.dino.parameters():
            p.requires_grad = False

def get_model(name, **kwargs) -> FeatureEmbedder:
    """
    Build a feature embedder by name; raise ValueError if the name is unknown.

    :param name: str - model name, only "dino3"
    :param kwargs: dict - forwarded to the model constructor
    :return: FeatureEmbedder - built embedder
    """
    if name == "dino3":
        return FeatureEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")
