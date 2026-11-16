import json
import logging
import os
import random
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import numpy as np
import wandb
import torch
from torch.utils.data import DataLoader

from cat.training.paths import register_resolvers, resolve_repo_path
from cat.data.embedding_dataset import build_embedding_dataset, embeddings_directory, semantic_maps_directory
from cat.training.checkpoint import build_decoder
from cat.training.experiment import Experiment
from cat.training.precompute_semantic_maps import precompute_semantic_maps
from cat.training.trainer import CATLightningModule, build_lightning_trainer

def build_settings(cfg) -> dict:
    """
    Convert the Hydra config to a settings dict with absolute data paths.

    :param cfg: DictConfig - Hydra config from cat/configs/train.yaml
    :return: dict - settings consumed by the Trainer
    """
    settings = OmegaConf.to_container(cfg, resolve=True)
    for key in ('embeddings_path', 'data_csv', 'data_dir', 'semantic_maps_dir'):
        if settings['data'].get(key) is not None:
            settings['data'][key] = str(resolve_repo_path(settings['data'][key], settings))
    # hydra run dir is the experiment folder
    settings['experiment'] = str(HydraConfig.get().runtime.output_dir)
    return settings

def train(settings) -> None:
    """
    Train, evaluate the best checkpoint and write the results file.

    :param settings: dict - settings built by :func:`build_settings`
    """
    experiment_path = Path(settings['experiment'])
    experiment = Experiment(experiment_path)
    experiment.write_settings(settings)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True,
        handlers=[logging.FileHandler(experiment.train_log_file), logging.StreamHandler()]
    )

    logging.info("Configuration loaded:")
    logging.info(json.dumps(settings, indent=2, default=str))

    seed = settings['seed']
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    # opt-in, off by default for speed
    if settings['training'].get('deterministic', False):
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logging.info("Deterministic mode ENABLED (slower)")
    else:
        torch.backends.cudnn.benchmark = True
        logging.info("cuDNN benchmark ENABLED (non-deterministic, faster)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    depth_emb_file = embeddings_directory(settings) / 'depth_embeddings.npy'
    use_depth = depth_emb_file.exists()
    if use_depth:
        logging.info(f"Found depth embeddings: {depth_emb_file}")

    # decoder and embedding cache fixed to 224x224
    if settings['data']['resolution'] != [224, 224]:
        raise ValueError("The decoder supports only data.resolution=[224, 224]")
    prompts = [p['text'] for p in settings['prompts']]
    precompute_semantic_maps(
        embeddings_path=str(embeddings_directory(settings) / 'embeddings.npy'),
        prompts=prompts,
        output_dir=semantic_maps_directory(settings),
        resolution=settings['data']['resolution'][0],
        device=device
    )

    dataset = build_embedding_dataset(settings, use_depth)

    train_size = int((1.0 - settings['data']['val_split']) * len(dataset))
    val_size = len(dataset) - train_size
    if train_size == 0 or val_size == 0:
        raise ValueError(
            f"Dataset split must contain both training and validation frames; "
            f"got {train_size} train and {val_size} validation frames. "
            "Increase the cache size or adjust data.val_split."
        )
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    num_workers = settings['data']['num_workers']

    train_loader = DataLoader(
        train_dataset,
        batch_size=settings['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        # full batches avoid torch.compile recompilation; a cache smaller than one batch keeps its only batch
        drop_last=train_size >= settings['training']['batch_size'],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    logging.info("{} train size and {} validation size out of {} samples".format(
        len(train_dataset), len(val_dataset), len(dataset)))

    predictor_network = build_decoder(settings, len(prompts), use_depth)

    # channels-last weights, faster nhwc tensor-core kernels
    predictor_network = predictor_network.to(memory_format=torch.channels_last)

    if settings['training']['compile']:
        logging.info("Compiling model with torch.compile (mode='default')...")
        predictor_network = torch.compile(predictor_network, mode='default')

    if not dataset.robot_names:
        raise ValueError(
            "No robot profiles were loaded from the dataset score files. "
            "Ensure the cache contains score_files and traj_files."
        )
    default_threshold = settings['robots']['traversability_threshold']
    settings['robot_profiles'] = [
        {'name': name, 'traversability_threshold': default_threshold}
        for name in dataset.robot_names
    ]
    logging.info(
        f"Loaded {dataset.num_robots} robot profiles from score files: "
        f"{dataset.robot_names}"
    )

    settings['checkpoint_dir'] = str(experiment.folder)

    module = CATLightningModule(
        settings=settings,
        model=predictor_network,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    trainer = build_lightning_trainer(settings, device)

    # resume only from a full Lightning training checkpoint
    resume_path = settings['training'].get('resume')
    if resume_path is not None:
        logging.info("Resuming from checkpoint: {}".format(resume_path))

    logging.info("Starting training...")
    trainer.fit(module, ckpt_path=resume_path, weights_only=False)
    train_results = module.train_stats
    logging.info("Training has finished")

    logging.info("Starting evaluation...")
    module.load_checkpoint(module.best_model_file)
    trainer.validate(module, verbose=False)
    evaluation_results = module.val_loss
    logging.info("Evaluation finished with {}".format(evaluation_results))

    with open(experiment.train_results_file, 'w') as f:
        json.dump({"train": train_results, "eval": evaluation_results}, f, indent=4)

    wandb.finish()

def run(cfg) -> None:
    """
    Build settings from the Hydra config and train.

    :param cfg: DictConfig - Hydra config
    """
    train(build_settings(cfg))

def main() -> None:
    """
    Register resolvers and launch the Hydra entrypoint.
    """
    register_resolvers()
    hydra.main(version_base="1.3", config_path="../../configs", config_name="train")(run)()

if __name__ == "__main__":
    main()