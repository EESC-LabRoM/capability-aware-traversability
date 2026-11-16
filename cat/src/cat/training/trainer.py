import lightning.pytorch as pl
import copy
import random
import logging
from lightning.pytorch.trainer.states import TrainerFn
import torch
import torch.nn.functional as F
import wandb
import cv2
import numpy as np

from pathlib import Path

from cat.training.capability import expected_traversability, modulate_semantic_map, refine_mask
from cat.training.checkpoint import load_model_state, normalize_prototypes
from cat.training.contrastive_loss import ContrastiveLoss

def path_ranking_metrics(pos_hist, neg_hist) -> tuple[float, float] | None:
    """
    Compute AUROC and average precision from score histograms.

    :param pos_hist: torch.Tensor | list - [bins] positive counts per bin, ascending score
    :param neg_hist: torch.Tensor | list - [bins] negative counts per bin, ascending score
    :return: tuple[float, float] | None - (auroc, ap), None when there is no positive or no negative
    """
    pos = torch.as_tensor(pos_hist, dtype=torch.float64).flip(0)
    neg = torch.as_tensor(neg_hist, dtype=torch.float64).flip(0)
    total_pos = pos.sum()
    total_neg = neg.sum()
    if total_pos <= 0 or total_neg <= 0:
        return None

    tp = pos.cumsum(0)
    fp = neg.cumsum(0)
    recall = tp / total_pos
    fpr = fp / total_neg
    zero = torch.zeros(1, dtype=torch.float64)
    auroc = torch.trapezoid(torch.cat([zero, recall]), torch.cat([zero, fpr]))
    precision = tp / (tp + fp).clamp(min=1.0)
    # ties within a bin: precision at bin end, AP = sum(delta_recall * precision)
    ap = ((recall - torch.cat([zero, recall[:-1]])) * precision).sum()
    return auroc.item(), ap.item()

class _ChannelsLastReadout(torch.autograd.Function):
    """Cosine readout with a channels-last feature gradient."""
    @staticmethod
    def forward(ctx, features, prototype) -> torch.Tensor:
        """
        Compute the per-pixel cosine readout.

        :param ctx: torch.autograd.function.FunctionCtx - autograd context
        :param features: torch.Tensor - [B, D, H, W] L2-normalized feature map
        :param prototype: torch.Tensor - [D] unit-length prototype, no gradient
        :return: torch.Tensor - [B, 1, H, W] cosine similarity
        """
        ctx.save_for_backward(prototype)
        return torch.einsum('d,bdhw->bhw', prototype, features).unsqueeze(1)

    @staticmethod
    def backward(ctx, grad_output) -> tuple[torch.Tensor, None]:
        """
        Feature gradient in channels-last layout, none for the prototype.

        :param ctx: torch.autograd.function.FunctionCtx - autograd context
        :param grad_output: torch.Tensor - [B, 1, H, W] readout gradient
        :return: tuple[torch.Tensor, None] - ([B, D, H, W] channels-last feature gradient, None)
        """
        (prototype,) = ctx.saved_tensors
        # channels-last grad matches channels-last decoder, no layout conversion in backward
        grad = grad_output.permute(0, 2, 3, 1) * prototype.view(1, 1, 1, -1)
        return grad.permute(0, 3, 1, 2), None

def build_lightning_trainer(settings, device) -> pl.Trainer:
    """
    Build a single-device trainer; the module manages AMP, so Lightning uses full precision.

    :param settings: dict - training configuration
    :param device: torch.device - single-device target
    :return: pl.Trainer - lightning trainer
    """
    if device.type == 'cuda':
        accelerator = 'cuda'
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    else:
        accelerator = 'cpu'
        devices = 1
    checkpoint_dir = Path(settings['checkpoint_dir'])
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='epoch-{epoch}',
        every_n_epochs=1,
        save_top_k=0,
        save_last=True,
        save_on_train_epoch_end=True,
    )
    return pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision='32-true',
        max_epochs=settings['training']['epochs'],
        num_sanity_val_steps=0,
        inference_mode=False,
        use_distributed_sampler=False,
        check_val_every_n_epoch=1,
        logger=False,
        callbacks=[checkpoint_callback],
        enable_model_summary=False,
        default_root_dir=str(checkpoint_dir),
    )

class CATLightningModule(pl.LightningModule):
    """
    Lightning module for conditioned traversability prediction, with manual optimization.
    """
    def __init__(self, settings, model, train_loader, val_loader, device) -> None:
        """
        Set up the optimizer, scheduler, loss, prototypes, AMP, checkpointing and wandb.

        :param settings: dict - training configuration
        :param model: CLIPSegSPADEDecoder - decoder producing the feature map
        :param train_loader: torch.utils.data.DataLoader - training loader
        :param val_loader: torch.utils.data.DataLoader - validation loader
        :param device: torch.device - training device
        """
        super().__init__()
        # lightning's device property is read-only
        self.target_device = device
        self.automatic_optimization = False
        self.settings = settings
        # bins over the cosine range [-1, 1] for path ranking
        self.ranking_bins = settings['evaluation']['ranking_bins']
        if isinstance(self.ranking_bins, bool) or not isinstance(self.ranking_bins, int) or self.ranking_bins <= 0:
            raise ValueError(f"evaluation.ranking_bins must be a positive integer, got {self.ranking_bins!r}")
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.train_stats = []
        self.val_loss = None
        # latest validation metrics and media, merged into epoch-end log
        self._val_metrics = {}
        self._val_media = []
        # detached gpu accumulators, read once at epoch end
        self._train_total_loss = None
        self._train_term_sums = {}
        self._train_term_counts = {}
        self._train_images = 0
        self._empty_positive = None
        self._grad_norm_max = None
        self._skipped_steps = None
        self._val_total_loss = None
        self._val_batches = 0
        self._val_term_sums = {}
        self._val_term_counts = {}
        self._capability_ready = False
        self._capability_sums = None
        self._log_media = False

        self.epochs = settings['training']['epochs']
        self.lr = settings['training']['learning_rate']
        self.weight_decay = settings['training']['weight_decay']

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        sched_cfg = settings['training']['scheduler']
        sched_type = sched_cfg['type']
        if sched_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs, eta_min=sched_cfg['min_lr']
            )
        elif sched_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=sched_cfg['step_size'], gamma=sched_cfg['gamma']
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {sched_type!r}")

        # v-strong eq. 3
        loss_cfg = settings['loss']
        self.criterion = ContrastiveLoss(temperature=loss_cfg['temperature'])
        self.feature_dim = settings['model']['feature_dim']
        self.ema_momentum = loss_cfg['ema_momentum']

        # v-strong eq. 4
        self.omega_mask = loss_cfg['omega_mask']
        self.n_pos_traj = loss_cfg['n_pos_traj']
        self.n_neg_traj = loss_cfg['n_neg_traj']
        self.n_pos_mask = loss_cfg['n_pos_mask']
        self.n_neg_mask = loss_cfg['n_neg_mask']

        # regress cos(feature, z) onto graded traversability
        self.grade_weight = loss_cfg['grade_weight']

        self.robot_profiles = self._build_robot_profiles()
        self.z = {}  # robot_name -> [1, d] ema prototype

        # path ranking cohort: robots whose platform recorded trajectories, keyed by sensor token
        self.recorded_by = settings['robots']['recorded_by']
        recorders = set(self.recorded_by.values())
        self.ranking_robots = [p['name'] for p in self.robot_profiles if p['name'] in recorders]
        self._ranking_hists = {}  # robot_name -> [2, ranking_bins] int64 (positive, negative) histograms

        self.use_amp = settings['training']['amp'] and device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.checkpoint_dir = Path(settings['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float('inf')
        self.best_checkpoint = None
        self.best_model_file = self.checkpoint_dir / "best_model.pth"

        es_cfg = settings['training']['early_stopping']
        self.early_stopping_enabled = es_cfg['enabled']
        self.early_stopping_patience = es_cfg['patience']
        self.early_stopping_counter = 0
        self.early_stopping_min_delta = es_cfg['min_delta']

        self.wandb_enabled = settings['wandb']['mode'] != 'disabled'
        if self.wandb_enabled:
            # relies on wandb login state or WANDB_API_KEY

            wandb.init(
                project=settings['wandb']['project'],
                name=settings['wandb']['name'],
                config=settings,
                mode=settings['wandb']['mode']
            )

            # one commit per epoch, metrics plotted against epoch
            wandb.define_metric('epoch')
            wandb.define_metric('*', step_metric='epoch')
            wandb.define_metric('loss/val_total', summary='min')

    def _build_robot_profiles(self) -> list[dict]:
        """
        Build one profile dict per configured robot.

        :return: list[dict] - dicts with keys: name, index, traversability_threshold
        """
        default_threshold = self.settings['robots']['traversability_threshold']

        profiles = []
        for i, robot in enumerate(self.settings['robot_profiles']):
            profiles.append({
                'name': robot['name'],
                'index': i,
                'traversability_threshold': robot.get('traversability_threshold', default_threshold)
            })
            print(f"Robot '{robot['name']}': index={i}, "
                  f"threshold={profiles[-1]['traversability_threshold']:.2f}")
        return profiles

    @torch.no_grad()
    def _update_ema_z(self, features, masks, robot_name) -> None:
        """
        Update a robot's EMA prototype (V-STRONG Eq. 5); no-op without positives, the first update initializes it.

        :param features: torch.Tensor - [B, D, H, W] L2-normalized feature map
        :param masks: torch.Tensor - [B, 1, H, W] binary refined trajectory mask
        :param robot_name: str - key into self.z
        """
        pos_flat = masks[:, 0].reshape(-1) > 0.5
        feat_flat = features.permute(0, 2, 3, 1).reshape(-1, self.feature_dim)
        pos_feat = feat_flat[pos_flat]

        if pos_feat.shape[0] == 0:
            return

        if robot_name not in self.z or self.z[robot_name] is None:
            self.z[robot_name] = F.normalize(pos_feat.mean(dim=0), p=2, dim=0).unsqueeze(0)
            return

        z = self.z[robot_name]
        z_batch = F.normalize(pos_feat.mean(dim=0), p=2, dim=0)
        z[0] = self.ema_momentum * z[0] + (1 - self.ema_momentum) * z_batch
        z[0] = F.normalize(z[0], p=2, dim=0)

    def _grade_loss(self, features, semantic_map, trav_vector, robot_name, valid_mask=None) -> torch.Tensor | None:
        """
        Regress the prototype readout onto pixel traversability rescaled to [-1, 1], in fp32 outside autocast.

        :param features: torch.Tensor - [B, D, H, W] L2-normalized features
        :param semantic_map: torch.Tensor - [B, K, H, W] terrain class probabilities
        :param trav_vector: torch.Tensor - [B, K] or [K] per-terrain traversability
        :param robot_name: str - robot whose prototype to read out with
        :param valid_mask: torch.Tensor | None - [B, 1, H, W] false = ignored pixel
        :return: torch.Tensor | None - scalar loss, None if the prototype is uninitialized
        """
        if self.z.get(robot_name) is None:
            return None
        # target mapped from [0, 1] to [-1, 1]
        with torch.autocast(device_type=features.device.type, enabled=False):
            pixel_trav = expected_traversability(semantic_map.float(), trav_vector.float()).clamp(0.0, 1.0)
            cos = self._traversability_map(features.float(), robot_name)
            error = (cos - (2.0 * pixel_trav - 1.0)) ** 2
        if valid_mask is None:
            return error.mean()
        # masked mean: ignored pixels get zero gradient without a host sync
        keep = (valid_mask > 0.5).to(error.dtype)
        return (error * keep).sum() / keep.sum().clamp(min=1.0)

    def _forward_model(self, embeddings, semantic_map, depth_embeddings=None) -> torch.Tensor:
        """
        Run the decoder on a modulated semantic map.

        :param embeddings: torch.Tensor - [B, C, H, W] frozen visual features
        :param semantic_map: torch.Tensor - [B, K, H, W] already modulated semantic map
        :param depth_embeddings: torch.Tensor | None - [B, C, H, W] frozen depth features
        :return: torch.Tensor - features [B, D, 224, 224]
        """
        features = self.model(embeddings, semantic_map, depth_features=depth_embeddings)
        return features

    def _create_heatmap_overlay(self, rgb_img, trav, alpha=0.5, valid=None) -> np.ndarray:
        """
        Blend a traversability heatmap over an RGB image (high traversability = blue).

        :param rgb_img: np.ndarray - [H, W, 3] uint8 RGB image
        :param trav: np.ndarray - [H, W] traversability in [-1, 1]
        :param alpha: float - heatmap blend weight
        :param valid: np.ndarray | None - [H, W] bool, False = ignored, shown flat gray
        :return: np.ndarray - [H, W, 3] uint8 RGB overlay
        """
        vis = ((1.0 - trav) / 2 * 255).clip(0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        if valid is not None:
            heatmap[~valid] = 128
        overlay = cv2.addWeighted(rgb_img, 1 - alpha, heatmap, alpha, 0)
        return overlay

    def _global_grad_norm(self) -> torch.Tensor:
        """
        Compute the global gradient L2 norm of the model parameters.

        :return: torch.Tensor - 0-d norm over parameters that have a gradient, no host sync
        """
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        return torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Return the training DataLoader.

        :return: torch.utils.data.DataLoader - training loader
        """
        return self.train_loader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Return the validation DataLoader.

        :return: torch.utils.data.DataLoader - validation loader
        """
        return self.val_loader

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Return the optimizer built in ``__init__``, so a checkpoint can be loaded before fit.

        :return: torch.optim.Optimizer - the AdamW optimizer
        """
        return self.optimizer

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """
        Add CAT resume state (scheduler, scaler, prototypes, stopping state, RNGs) to the checkpoint.

        :param checkpoint: dict - Lightning checkpoint being written
        """
        checkpoint['cat_resume_state'] = {
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'scaler': self.scaler.state_dict(),
            'prototypes': self.z,
            'best_checkpoint': self.best_checkpoint,
            'best_val_loss': self.best_val_loss,
            'early_stopping_counter': self.early_stopping_counter,
            'train_stats': self.train_stats,
            'python_rng': random.getstate(),
            'numpy_rng': (
                np.random.get_state()[0],
                np.random.get_state()[1].tolist(),
                np.random.get_state()[2],
                np.random.get_state()[3],
                np.random.get_state()[4],
            ),
            'torch_rng': torch.get_rng_state().cpu(),
            'cuda_rng': [state.cpu() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """
        Restore CAT resume state; inference-only checkpoints without it cannot resume.

        :param checkpoint: dict - Lightning checkpoint being loaded
        """
        state = checkpoint.get('cat_resume_state')
        if state is None:
            raise ValueError(
                "Checkpoint has no CAT resume state. Historical model checkpoints are for "
                "inference only and cannot resume training; start a new run or use the "
                "checkpoint only for evaluation."
            )
        if self.scheduler and state.get('scheduler') is not None:
            self.scheduler.load_state_dict(state['scheduler'])
        self.scaler.load_state_dict(state['scaler'])
        self.z = normalize_prototypes(
            state['prototypes'], self.robot_profiles[0]['name'], self.target_device
        )
        self.best_checkpoint = state.get('best_checkpoint')
        if self.best_checkpoint is not None:
            torch.save(self.best_checkpoint, self.best_model_file)
        self.best_val_loss = state['best_val_loss']
        self.early_stopping_counter = state['early_stopping_counter']
        self.train_stats = state.get('train_stats', [])
        random.setstate(state['python_rng'])
        numpy_rng = state['numpy_rng']
        np.random.set_state((
            numpy_rng[0], np.asarray(numpy_rng[1], dtype=np.uint32),
            numpy_rng[2], numpy_rng[3], numpy_rng[4]
        ))
        torch.set_rng_state(state['torch_rng'].cpu())
        if torch.cuda.is_available() and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([rng.cpu() for rng in state['cuda_rng']])

    def on_train_epoch_end(self) -> None:
        """
        Step the scheduler, log metrics, update early stopping and save the best checkpoint, after validation.
        """
        epoch = self.current_epoch + 1
        train_loss = self._train_total_loss.item() / len(self.train_loader)
        val_loss = self.val_loss

        # lr used this epoch, read before scheduler advances
        current_lr = self.optimizer.param_groups[0]['lr']
        if self.scheduler:
            self.scheduler.step()

        train_metrics = {'loss/train_total': train_loss}
        train_metrics.update(self._term_means(self._train_term_sums, self._train_term_counts, 'train'))

        logging.info("Epoch %d | Train Loss: %.4f | Val Loss: %.4f | LR: %.6f", epoch, train_loss, val_loss, current_lr)
        logging.info("  train %s | val %s", self._format_terms(train_metrics, 'train'), self._format_terms(self._val_metrics, 'val'))
        capability = {k.split('/', 1)[1]: v for k, v in self._val_metrics.items() if k.startswith('capability/')}
        if capability:
            logging.info("  capability %s", ' '.join(f"{k}={v:.4f}" for k, v in capability.items()))
        ranking = {k: v for k, v in self._val_metrics.items() if k.startswith('ranking/val_path_') and k.count('/') == 2}
        if ranking:
            parts = []
            for metric in ('auroc', 'ap'):
                prefix = f'ranking/val_path_{metric}/'
                values = ' '.join(f"{k[len(prefix):]}={v:.4f}" for k, v in ranking.items() if k.startswith(prefix))
                parts.append(f"{metric} {values}")
            logging.info("  ranking %s", ' | '.join(parts))

        # single commit per epoch, after validation
        if self.wandb_enabled:
            log_dict = {'epoch': epoch}
            log_dict.update(train_metrics)
            log_dict.update(self._val_metrics)
            log_dict.update(self._health_metrics())
            # one panel per fixed frame, each with its own epoch slider
            for i, image in enumerate(self._val_media):
                log_dict[f'media/frame_{i}'] = image
            wandb.log(log_dict)

        self.train_stats.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        })

        if self.early_stopping_enabled and self.early_stopping_patience:
            if val_loss < (self.best_val_loss - self.early_stopping_min_delta):
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                print(f"Early stopping: {self.early_stopping_counter}/{self.early_stopping_patience}")

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                },
                'val_loss': val_loss,
                'z': {
                    name: value.detach().cpu().clone()
                    for name, value in self.z.items()
                },
                'config': copy.deepcopy(self.settings)
            }
            torch.save(self.best_checkpoint, self.best_model_file)
            print(f"Saved new best model to {self.best_model_file}")
            if self.wandb_enabled:
                wandb.run.summary['best/epoch'] = epoch
                wandb.run.summary['best/val_loss'] = val_loss

        if self.early_stopping_enabled and self.early_stopping_patience and self.early_stopping_counter >= self.early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs")
            self.trainer.should_stop = True

    def _traversability_map(self, features, robot_name) -> torch.Tensor:
        """
        Compute cosine similarity to the robot's traversability prototype.

        :param features: torch.Tensor - [B, D, H, W] L2-normalized feature map
        :param robot_name: str - robot whose prototype to use
        :return: torch.Tensor - [B, 1, H, W] traversability in [-1, 1], high = traversable, zeros if the prototype is uninitialized
        """
        z = self.z.get(robot_name)
        if z is None:
            return torch.zeros(features.shape[0], 1, features.shape[2], features.shape[3],
                               device=features.device)
        # cosine: features and z are unit length
        # fixed z: channels-last feature gradient, matches decoder layout
        if features.requires_grad and not z.requires_grad:
            return _ChannelsLastReadout.apply(features, z[0])
        return torch.einsum('d,bdhw->bhw', z[0], features).unsqueeze(1)

    def _compute_dual_loss(self, features, traj_mask, positive_mask, valid_mask=None) -> tuple[torch.Tensor, dict]:
        """
        Compute dual trajectory and mask loss terms.

        :param features: torch.Tensor - [B, D, H, W] L2-normalized feature map
        :param traj_mask: torch.Tensor - [B, 1, H, W] robot-refined trajectory mask
        :param positive_mask: torch.Tensor - [B, 1, H, W] robot-refined positive mask
        :param valid_mask: torch.Tensor | None - [B, 1, H, W] false = ignored pixel
        :return: tuple[torch.Tensor, dict] - (scalar loss, detached unweighted 'traj' and 'mask' terms)
        """
        # one shared feature selection
        loss_traj, loss_mask = self.criterion.forward_terms(
            features, [(traj_mask, self.n_pos_traj, self.n_neg_traj),
                       (positive_mask, self.n_pos_mask, self.n_neg_mask)],
            valid_mask=valid_mask)
        loss = (1 - self.omega_mask) * loss_traj + self.omega_mask * loss_mask
        return loss, {'traj': loss_traj.detach(), 'mask': loss_mask.detach()}

    def _robot_loss(self, features, traj_mask, positive_mask, valid_mask, semantic_map, trav_vector, robot_name) -> tuple[torch.Tensor, dict]:
        """
        Compute one robot's dual loss plus the optional grade term.

        :param features: torch.Tensor - [B, D, H, W] L2-normalized features
        :param traj_mask: torch.Tensor | None - [B, 1, H, W] robot-refined trajectory mask
        :param positive_mask: torch.Tensor - [B, 1, H, W] robot-refined positive mask
        :param valid_mask: torch.Tensor | None - [B, 1, H, W] false = ignored pixel
        :param semantic_map: torch.Tensor - [B, K, H, W] terrain class probabilities
        :param trav_vector: torch.Tensor - [B, K] or [K] per-terrain traversability
        :param robot_name: str - robot whose prototype to read out with
        :return: tuple[torch.Tensor, dict] - (scalar loss, detached unweighted 'traj', 'mask' and, when present, 'grade' terms)
        """
        loss, terms = self._compute_dual_loss(features, traj_mask, positive_mask, valid_mask)
        if self.grade_weight > 0.0:
            l_grade = self._grade_loss(features, semantic_map, trav_vector, robot_name, valid_mask)
            if l_grade is not None:
                loss = loss + self.grade_weight * l_grade
                terms['grade'] = l_grade.detach()
        return loss, terms

    @staticmethod
    def _add_terms(sums, counts, terms) -> None:
        """
        Add loss terms to running sums without a host sync.

        :param sums: dict - term name -> running detached tensor sum, updated in place
        :param counts: dict - term name -> number of added values, updated in place
        :param terms: dict - term name -> 0-d tensor
        """
        for name, value in terms.items():
            value = value.detach().float()
            sums[name] = sums[name] + value if name in sums else value
            counts[name] = counts.get(name, 0) + 1

    @staticmethod
    def _term_means(sums, counts, prefix) -> dict:
        """
        Convert running sums to per-term means with one host sync.

        :param sums: dict - term name -> running detached tensor sum
        :param counts: dict - term name -> number of added values
        :param prefix: str - 'train' or 'val', used in the metric key
        :return: dict - 'loss/<prefix>_<term>' -> float mean
        """
        if not sums:
            return {}
        names = list(sums)
        values = torch.stack([sums[name] for name in names]).tolist()
        return {f'loss/{prefix}_{name}': value / counts[name] for name, value in zip(names, values)}

    @staticmethod
    def _format_terms(metrics, prefix) -> str:
        """
        Format per-term losses for the console.

        :param metrics: dict - metric key -> float
        :param prefix: str - 'train' or 'val'
        :return: str - e.g. 'traj=0.1234 mask=0.2345', without the total
        """
        return ' '.join(
            f"{key.split('_', 1)[1]}={value:.4f}" for key, value in metrics.items()
            if key.startswith(f'loss/{prefix}_') and key != f'loss/{prefix}_total'
        )

    def _health_metrics(self) -> dict:
        """
        Read training health counters with one host sync.

        :return: dict - grad norm max, skipped steps, empty positive fraction per robot, amp scale when amp is enabled
        """
        values = torch.cat([
            self._grad_norm_max.double().reshape(1),
            self._skipped_steps.double().reshape(1),
            self._empty_positive.double(),
        ]).tolist()
        metrics = {'health/grad_norm_max': values[0], 'health/skipped_steps': int(values[1])}
        for profile, count in zip(self.robot_profiles, values[2:]):
            metrics[f"health/empty_positive_frac/{profile['name']}"] = count / self._train_images
        if self.use_amp:
            metrics['health/amp_scale'] = self.scaler.get_scale()
        return metrics

    def on_train_epoch_start(self) -> None:
        """
        Print the epoch header, switch to train mode and reset accumulators.
        """
        epoch = self.current_epoch + 1
        print(f"\nEpoch {epoch}/{self.epochs}")
        self.model.train()
        device = self.target_device
        self._train_total_loss = torch.zeros((), device=device)
        self._train_term_sums = {}
        self._train_term_counts = {}
        self._train_images = 0
        self._empty_positive = torch.zeros(len(self.robot_profiles), dtype=torch.long, device=device)
        self._grad_norm_max = torch.zeros((), device=device)
        self._skipped_steps = torch.zeros((), dtype=torch.long, device=device)

    def training_step(self, batch, batch_idx) -> None:
        """
        Run one training step: backpropagate each robot's loss in turn, then take a single optimizer step.

        :param batch: dict - batch from the training DataLoader
        :param batch_idx: int - batch index within the epoch
        """
        num_robots = len(self.robot_profiles)
        optimizer = self.optimizers()

        embeddings = batch['embedding'].to(self.target_device, non_blocking=True)
        positive_mask = batch['mask'].to(self.target_device, non_blocking=True)
        semantic_map = batch['semantic_map'].to(self.target_device, non_blocking=True)

        # false = ego pixel
        valid_mask = batch['valid_mask'].to(self.target_device, non_blocking=True) if 'valid_mask' in batch else None

        depth_embeddings = batch['depth_embedding'].to(self.target_device, non_blocking=True) if 'depth_embedding' in batch else None

        traj_mask = batch['trajectory_mask'].to(self.target_device, non_blocking=True) if 'trajectory_mask' in batch else None

        # [b, r, k]
        trav_costs = batch['traversability_costs'].to(self.target_device, non_blocking=True)

        B = embeddings.size(0)
        K = semantic_map.size(1)

        assert trav_costs.shape == (B, num_robots, K), \
            f"traversability_costs: expected ({B}, {num_robots}, {K}), got {trav_costs.shape}"

        # one robot graph at a time: peak vram r*b -> b
        optimizer.zero_grad()
        robot_features_detached = []
        robot_masks = []

        for i, profile in enumerate(self.robot_profiles):
            idx = profile['index']
            threshold = profile['traversability_threshold']

            trav_vector = trav_costs[:, idx, :]  # [b, k]

            modulated = modulate_semantic_map(semantic_map, trav_vector)

            robot_mask = refine_mask(
                positive_mask, semantic_map, trav_vector, threshold, valid_mask
            )

            # same pixel may be positive for one robot, negative for another
            robot_traj_mask = None
            if traj_mask is not None:
                robot_traj_mask = refine_mask(
                    traj_mask, semantic_map, trav_vector, threshold, valid_mask
                )

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                features = self._forward_model(embeddings, modulated, depth_embeddings)
                loss, terms = self._robot_loss(features, robot_traj_mask, robot_mask, valid_mask,
                                               semantic_map, trav_vector, profile['name'])
                loss = loss / num_robots

            # backward now to free this robot's graph
            self.manual_backward(self.scaler.scale(loss))
            self._train_total_loss += loss.detach().float()
            self._add_terms(self._train_term_sums, self._train_term_counts, terms)

            robot_features_detached.append(features.detach())
            robot_masks.append(robot_traj_mask if robot_traj_mask is not None else robot_mask)

        # scaler owns unscale, inf-skip, update; step via lightning optimizer
        self.scaler.unscale_(optimizer)
        if self.wandb_enabled:
            # max finite norm and non-finite (amp skipped) step count, kept on device until epoch end
            grad_norm = self._global_grad_norm()
            finite = torch.isfinite(grad_norm)
            self._grad_norm_max = torch.maximum(self._grad_norm_max, torch.where(finite, grad_norm, torch.zeros_like(grad_norm)))
            self._skipped_steps += (~finite).long()
        self.scaler.step(optimizer)
        self.scaler.update()

        # ema z uses only capability-refined, ego-excluded positives
        with torch.no_grad():
            for i, profile in enumerate(self.robot_profiles):
                self._update_ema_z(robot_features_detached[i].float(), robot_masks[i], profile['name'])
                if self.wandb_enabled:
                    has_positive = (robot_masks[i][:, 0] > 0.5).flatten(1).any(dim=1)
                    self._empty_positive[i] += (~has_positive).sum()
        self._train_images += B

        del robot_features_detached

    def on_validation_epoch_start(self) -> None:
        """
        Reset validation accumulators and choose optional outputs.
        """
        self._val_total_loss = torch.zeros((), device=self.target_device)
        self._val_batches = 0
        self._val_term_sums = {}
        self._val_term_counts = {}
        self._val_media = []
        self._capability_ready = len(self.robot_profiles) >= 2 and all(
            self.z.get(profile['name']) is not None for profile in self.robot_profiles
        )
        # float64 streaming sums: n, sx, sy, sxx, syy, sxy of pairwise map/score diffs, then swap sum
        self._capability_sums = torch.zeros(7, dtype=torch.float64, device=self.target_device)
        self._ranking_hists = {
            name: torch.zeros(2, self.ranking_bins, dtype=torch.long, device=self.target_device)
            for name in self.ranking_robots
        }
        self._log_media = self.wandb_enabled and self.trainer.state.fn == TrainerFn.FITTING

    def validation_step(self, batch, batch_idx) -> None:
        """
        Run one validation step; build the media grid on the first batch.

        :param batch: dict - batch from the validation DataLoader
        :param batch_idx: int - batch index within the validation pass
        """
        want_media = batch_idx == 0 and self._log_media

        with torch.no_grad():
            embeddings = batch['embedding'].to(self.target_device, non_blocking=True)
            positive_mask = batch['mask'].to(self.target_device, non_blocking=True)
            semantic_map = batch['semantic_map'].to(self.target_device, non_blocking=True)
            valid_mask = batch['valid_mask'].to(self.target_device, non_blocking=True) if 'valid_mask' in batch else None
            depth_embeddings = batch['depth_embedding'].to(self.target_device, non_blocking=True) if 'depth_embedding' in batch else None
            traj_mask = batch['trajectory_mask'].to(self.target_device, non_blocking=True) if 'trajectory_mask' in batch else None
            trav_costs = batch['traversability_costs'].to(self.target_device, non_blocking=True)

            B = embeddings.size(0)
            K = semantic_map.size(1)
            num_robots = len(self.robot_profiles)

            assert trav_costs.shape == (B, num_robots, K), \
                f"traversability_costs: expected ({B}, {num_robots}, {K}), got {trav_costs.shape}"

            prototypes = None
            if self._capability_ready:
                prototypes = torch.cat([self.z[p['name']] for p in self.robot_profiles], dim=0)

            # [B, 1, 1, 1] sample masks
            recorded = self._recorded_eligibility(batch['sensor']) if traj_mask is not None and 'sensor' in batch else {}

            # sequential per robot, no autograd, saves vram
            batch_loss = 0.0
            cross_maps = []  # per robot: [B, R, H, W] cos(f_robot, z_q) for every prototype q
            expected = []  # per robot: [B, 1, H, W] expected traversability
            own_maps = []  # per robot: first samples' own-prototype map, for media grid

            for i, profile in enumerate(self.robot_profiles):
                idx = profile['index']
                trav_vector = trav_costs[:, idx, :]

                modulated = modulate_semantic_map(semantic_map, trav_vector)
                robot_mask = refine_mask(
                    positive_mask, semantic_map,
                    trav_vector, profile['traversability_threshold'], valid_mask
                )
                robot_traj_mask = None
                if traj_mask is not None:
                    robot_traj_mask = refine_mask(
                        traj_mask, semantic_map,
                        trav_vector, profile['traversability_threshold'], valid_mask
                    )

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    features = self._forward_model(embeddings, modulated, depth_embeddings)
                    loss, terms = self._robot_loss(features, robot_traj_mask, robot_mask, valid_mask,
                                                   semantic_map, trav_vector, profile['name'])
                    batch_loss += loss
                self._add_terms(self._val_term_sums, self._val_term_counts, terms)

                # only these small maps outlive the features
                if self._capability_ready:
                    cross = torch.einsum('rd,bdhw->brhw', prototypes, features.float())
                    cross_maps.append(cross)
                    expected.append(expected_traversability(semantic_map, trav_vector).clamp(0.0, 1.0))
                    if want_media:
                        own_maps.append(cross[:4, i:i + 1])
                elif want_media:
                    own_maps.append(self._traversability_map(features.float(), profile['name'])[:4])

                eligible = recorded.get(profile['name'])
                if eligible is not None:
                    own = cross[:, i:i + 1] if self._capability_ready else self._traversability_map(features.float(), profile['name'])
                    self._accumulate_ranking(profile['name'], own, traj_mask, valid_mask, eligible)
                del features

            self._val_total_loss += (batch_loss / num_robots).float()
            self._val_batches += 1

            if self._capability_ready:
                self._accumulate_capability(cross_maps, expected, valid_mask)

            if want_media:
                self._val_media = self._build_val_panels(batch, own_maps, traj_mask, positive_mask, valid_mask)

    def _recorded_eligibility(self, sensors) -> dict:
        """
        Flag samples recorded by each cohort robot's platform.

        Robots without a prototype or a recorded sample are omitted.

        :param sensors: list[str] - sensor token per sample (``source_bag.split('_')[1]``), '' when unknown
        :return: dict - robot_name -> [B, 1, 1, 1] bool tensor on the target device
        """
        platforms = [self.recorded_by.get(sensor) for sensor in sensors]
        eligibility = {}
        for name in self.ranking_robots:
            flags = [platform == name for platform in platforms]
            if self.z.get(name) is not None and any(flags):
                eligibility[name] = torch.tensor(flags, dtype=torch.bool, device=self.target_device).view(-1, 1, 1, 1)
        return eligibility

    def _accumulate_ranking(self, name, score, traj_mask, valid_mask, eligible) -> None:
        """
        Add a validation batch to a robot's path-vs-rest score histograms without host syncs.

        Positives: valid raw trajectory pixels; negatives: other valid pixels; only on samples recorded by the robot's platform.

        :param name: str - robot name
        :param score: torch.Tensor - [B, 1, H, W] own-prototype cosine readout
        :param traj_mask: torch.Tensor - [B, 1, H, W] raw trajectory mask
        :param valid_mask: torch.Tensor | None - [B, 1, H, W] false = ignored pixel
        :param eligible: torch.Tensor - [B, 1, 1, 1] bool, samples recorded by this robot's platform
        """
        bins = ((score.float().clamp(-1.0, 1.0) + 1.0) * (0.5 * self.ranking_bins)).long().clamp_(0, self.ranking_bins - 1).flatten()
        counted = eligible.expand_as(traj_mask)
        if valid_mask is not None:
            counted = counted & (valid_mask > 0.5)
        on_path = traj_mask > 0.5
        hist = self._ranking_hists[name]
        hist[0].scatter_add_(0, bins, (counted & on_path).flatten().long())
        hist[1].scatter_add_(0, bins, (counted & ~on_path).flatten().long())

    def _ranking_metrics(self) -> dict:
        """
        Convert ranking histograms to metrics with one host sync.

        Robots lacking positives or negatives are omitted, including from the mean AUROC.

        :return: dict - ranking/val_path_auroc/<robot>, ranking/val_path_ap/<robot> and ranking/val_path_auroc_mean
        """
        if not self._ranking_hists:
            return {}
        names = list(self._ranking_hists)
        hists = torch.stack([self._ranking_hists[n] for n in names]).cpu()
        metrics = {}
        aurocs = []
        for name, hist in zip(names, hists):
            result = path_ranking_metrics(hist[0], hist[1])
            if result is not None:
                metrics[f'ranking/val_path_auroc/{name}'] = result[0]
                metrics[f'ranking/val_path_ap/{name}'] = result[1]
                aurocs.append(result[0])
        if aurocs:
            metrics['ranking/val_path_auroc_mean'] = sum(aurocs) / len(aurocs)
        return metrics

    def _accumulate_capability(self, cross_maps, expected, valid_mask) -> None:
        """
        Add a validation batch to the streaming capability sums.

        The swap term compares cos(f_a, z_a) with cos(f_b, z_a), the same prototype read from another robot's features.

        :param cross_maps: list[torch.Tensor] - per robot [B, R, H, W] readouts cos(f_robot, z_q) for every prototype q
        :param expected: list[torch.Tensor] - per robot [B, 1, H, W] expected traversability in [0, 1]
        :param valid_mask: torch.Tensor | None - [B, 1, H, W] false = ignored pixel
        """
        num_robots = len(cross_maps)
        own = [cross_maps[r][:, r:r + 1] for r in range(num_robots)]
        if valid_mask is None:
            weight = torch.ones_like(own[0], dtype=torch.float64)
        else:
            weight = (valid_mask > 0.5).double()
        sums = self._capability_sums
        for a in range(num_robots):
            for b in range(a + 1, num_robots):
                x = (own[a] - own[b]).abs().double() * weight
                y = (expected[a] - expected[b]).abs().double()
                sums[0] += weight.sum()
                sums[1] += x.sum()
                sums[2] += (y * weight).sum()
                sums[3] += (x * x).sum()  # x already weighted by binary validity mask
                sums[4] += (y * y * weight).sum()
                sums[5] += (x * y).sum()
            # swap conditioning at a fixed prototype
            for b in range(num_robots):
                if b != a:
                    swap = (own[a] - cross_maps[b][:, a:a + 1]).abs().double()
                    sums[6] += (swap * weight).sum()

    def _capability_metrics(self) -> dict:
        """
        Convert capability sums to metrics with one host sync.

        :return: dict - map_diff, swap_css and, when both variances are positive, map_diff_score_corr
        """
        n, sx, sy, sxx, syy, sxy, swap = self._capability_sums.tolist()
        if n <= 0:
            return {}
        # each unordered pair has two ordered pairs over same pixels
        metrics = {'capability/map_diff': sx / n, 'capability/swap_css': swap / (2.0 * n)}
        var_x = sxx / n - (sx / n) ** 2
        var_y = syy / n - (sy / n) ** 2
        if var_x > 1e-12 and var_y > 1e-12:
            cov = sxy / n - sx * sy / n ** 2
            metrics['capability/map_diff_score_corr'] = cov / (var_x * var_y) ** 0.5
        return metrics

    def on_validation_epoch_end(self) -> None:
        """
        Compute validation metrics for the epoch-end log; outside fit, write them to the wandb summary under ``final/``.
        """
        self.val_loss = self._val_total_loss.item() / self._val_batches

        metrics = {'loss/val_total': self.val_loss}
        metrics.update(self._term_means(self._val_term_sums, self._val_term_counts, 'val'))
        if self._capability_ready:
            metrics.update(self._capability_metrics())
        metrics.update(self._ranking_metrics())
        self._val_metrics = metrics

        if self.wandb_enabled and self.trainer.state.fn != TrainerFn.FITTING:
            for key, value in metrics.items():
                wandb.run.summary[f'final/{key}'] = value

    @staticmethod
    def _load_rgb_crop_224(rgb_path) -> np.ndarray:
        """
        Load an image with the dataloader's resize and center crop.

        :param rgb_path: str - image file path
        :return: np.ndarray - [224, 224, 3] contiguous uint8 RGB image
        """
        img = cv2.imread(rgb_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # as in EmbeddingDataset._resize
        h, w = img.shape[:2]
        scale = max(224 / w, 224 / h)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        start_x = img.shape[1] // 2 - 112
        start_y = img.shape[0] // 2 - 112
        return np.ascontiguousarray(img[start_y:start_y + 224, start_x:start_x + 224])

    @staticmethod
    def _label_tile(tile, text) -> np.ndarray:
        """
        Return a copy of the tile labeled in its top left corner.

        :param tile: np.ndarray - [H, W, 3] uint8 RGB tile
        :param text: str - label
        :return: np.ndarray - labeled copy of the tile
        """
        tile = tile.copy()
        # dark outline under white text, readable on any background
        cv2.putText(tile, text, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(tile, text, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return tile

    def _build_val_panels(self, batch, robot_maps, traj_mask, positive_mask, valid_mask) -> list:
        """
        Build one wandb grid per validation frame: rgb, raw mask, then one heatmap per robot.

        Up to three tiles per row, upscaled 2x.

        :param batch: dict - first validation batch, containing ``rgb_path``
        :param robot_maps: list[torch.Tensor] - per robot [<=4, 1, H, W] traversability map from its own prototype
        :param traj_mask: torch.Tensor | None - [B, 1, H, W] raw trajectory mask
        :param positive_mask: torch.Tensor - [B, 1, H, W] raw positive mask, shown when there is no trajectory mask
        :param valid_mask: torch.Tensor | None - [B, 1, H, W] false = ignored (ego) pixel
        :return: list[wandb.Image] - one grid image per frame
        """
        epoch = self.current_epoch + 1
        raw_mask = traj_mask if traj_mask is not None else positive_mask
        mask_label = 'trajectory' if traj_mask is not None else 'mask'
        size = 448
        images = []
        # first four samples of unshuffled validation loader
        for i in range(min(len(batch['rgb_path']), 4)):
            rgb = cv2.resize(self._load_rgb_crop_224(batch['rgb_path'][i]), (size, size), interpolation=cv2.INTER_LINEAR)
            valid_np = None
            if valid_mask is not None:
                valid_np = cv2.resize((valid_mask[i, 0] > 0.5).cpu().numpy().astype(np.uint8), (size, size),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)

            mask_np = cv2.resize((raw_mask[i, 0] > 0.5).cpu().numpy().astype(np.uint8), (size, size),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
            mask_tile = rgb.copy()
            mask_tile[mask_np] = (0.5 * mask_tile[mask_np] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
            if valid_np is not None:
                mask_tile[~valid_np] = 128

            tiles = [self._label_tile(rgb, 'rgb'), self._label_tile(mask_tile, mask_label)]
            for profile, robot_map in zip(self.robot_profiles, robot_maps):
                trav = cv2.resize(robot_map[i, 0].float().cpu().numpy(), (size, size), interpolation=cv2.INTER_LINEAR)
                heatmap = self._create_heatmap_overlay(rgb, trav, valid=valid_np)
                tiles.append(self._label_tile(heatmap, profile['name']))

            cols = min(len(tiles), 3)
            tiles += [np.full_like(rgb, 255)] * (-len(tiles) % cols)
            rows = [np.concatenate(tiles[r:r + cols], axis=1) for r in range(0, len(tiles), cols)]
            images.append(wandb.Image(np.concatenate(rows, axis=0), caption=f"frame {i} epoch {epoch}"))
        return images

    def load_checkpoint(self, path) -> int:
        """
        Load model and prototype state for evaluation.

        :param path: str | Path - checkpoint file path
        :return: int - epoch stored in the checkpoint
        """
        checkpoint = torch.load(path, map_location=self.target_device, weights_only=False)
        load_model_state(self.model, checkpoint['model_state_dict'])
        if checkpoint.get('z') is not None:
            self.z = normalize_prototypes(checkpoint['z'], self.robot_profiles[0]['name'], self.target_device)
        return checkpoint['epoch']
