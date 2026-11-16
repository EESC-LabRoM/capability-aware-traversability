from cat.models.decoder import CLIPSegSPADEDecoder

def build_decoder(settings, num_prompts, use_depth) -> CLIPSegSPADEDecoder:
    """
    Build the decoder from training settings.

    :param settings: dict - training or checkpoint settings
    :param num_prompts: int - number of terrain prompts (K)
    :param use_depth: bool - whether the decoder mixes in depth embeddings
    :return: CLIPSegSPADEDecoder - untrained decoder
    """
    model_cfg = settings['model']
    # optional keys fall back to defaults
    return CLIPSegSPADEDecoder(
        input_dim=model_cfg['input_dim'],
        semantic_channels=num_prompts,
        spade_hidden=model_cfg.get('spade_hidden', 128),
        inter_block_dropout=model_cfg.get('inter_block_dropout', 0.0),
        norm_type=model_cfg.get('spade_norm', 'batch'),
        output_dim=model_cfg.get('feature_dim', 64),
        use_depth=use_depth,
        raw_embed_dim=model_cfg.get('raw_embed_dim', 768),
        depth_dropout=model_cfg.get('depth_dropout', 0.0),
    )

def strip_compile_prefix(state_dict) -> dict:
    """
    Strip the torch.compile prefix from parameter names.

    :param state_dict: dict - decoder state dict, compiled or not
    :return: dict - state dict with plain parameter names
    """
    # compiled modules prefix keys with _orig_mod.
    return {key.removeprefix('_orig_mod.'): value for key, value in state_dict.items()}

def load_model_state(model, state_dict) -> None:
    """
    Load decoder weights regardless of torch.compile wrapping.

    :param model: torch.nn.Module - decoder, optionally wrapped by torch.compile
    :param state_dict: dict - saved decoder state dict
    """
    getattr(model, '_orig_mod', model).load_state_dict(strip_compile_prefix(state_dict))

def normalize_prototypes(z, default_name, device) -> dict:
    """
    Convert saved prototypes to one [1, D] tensor per robot.

    :param z: dict | torch.Tensor - saved prototypes; legacy payloads store [D] or a single tensor
    :param default_name: str - robot name for a legacy single-tensor payload
    :param device: torch.device - target device
    :return: dict[str, torch.Tensor] - [1, D] prototype per robot
    """
    # legacy payload: single tensor -> one robot entry
    if not isinstance(z, dict):
        z = {default_name: z}
    prototypes = {}
    for name, value in z.items():
        value = value.to(device)
        prototypes[name] = value.unsqueeze(0) if value.dim() == 1 else value
    return prototypes
