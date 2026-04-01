"""
training.py — GPU Training Loop for Smart ICU Assistant
Tuned for RTX 3050 4GB VRAM.

Features:
  - FP16 mixed precision (AMP) with GradScaler
  - Gradient accumulation (batch 32 × 2 = effective 64)
  - Early stopping with min_delta
  - Per-epoch progress bar showing train/val loss, AUROC, patience
  - Per-batch inner bar
  - Unique temp checkpoint path per training run (no collisions)
  - clear_gpu_memory() with synchronize() before empty_cache()
  - GPU memory logging
"""

import os
import sys
import gc
import uuid
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Optional
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── Device helpers ────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram  = props.total_memory / (1024 ** 3)
        logger.info(f"GPU detected: {props.name} ({vram:.1f} GB VRAM)")
        return torch.device('cuda')
    logger.warning("No CUDA device — running on CPU")
    return torch.device('cpu')


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


def log_gpu_memory(label: str = ""):
    if torch.cuda.is_available():
        alloc  = torch.cuda.memory_allocated() / (1024 ** 2)
        reserv = torch.cuda.memory_reserved()  / (1024 ** 2)
        logger.info(f"[GPU {label}] Allocated: {alloc:.0f}MB | Cached: {reserv:.0f}MB")


# ── Temporal split ────────────────────────────────────────────────────────────

def temporal_split_data(X: np.ndarray,
                         y: np.ndarray,
                         timestamps: List,
                         train_frac: float = 0.70,
                         val_frac:   float = 0.15) -> Dict:
    n       = len(X)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    return {
        'train': (X[:n_train],               y[:n_train]),
        'val':   (X[n_train:n_train + n_val], y[n_train:n_train + n_val]),
        'test':  (X[n_train + n_val:],        y[n_train + n_val:]),
    }


# ── Trainer ───────────────────────────────────────────────────────────────────

class ModelTrainer:
    """FP16 AMP trainer for RTX 3050 4GB VRAM."""

    def __init__(self, model: nn.Module, config: dict):
        self.device = get_device()
        self.config = config
        self.model  = model.to(self.device)

        gpu_cfg  = config.get('GPU_CONFIG', {})
        lstm_cfg = config.get('LSTM_CONFIG', {})

        self.batch_size       = gpu_cfg.get('batch_size', 32)
        self.grad_accum_steps = gpu_cfg.get('grad_accum_steps', 2)
        self.use_amp          = gpu_cfg.get('use_amp', True)
        self.pin_memory       = gpu_cfg.get('pin_memory', True)
        self.num_workers      = gpu_cfg.get('num_workers', 2)
        self.epochs           = lstm_cfg.get('epochs', 50)
        self.lr               = lstm_cfg.get('learning_rate', 0.001)

        self.scaler    = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.BCELoss()

        es_cfg            = config.get('EARLY_STOPPING', {})
        self.patience     = es_cfg.get('patience', 20)
        self.min_delta    = es_cfg.get('min_delta', 1e-4)
        self.patience_ctr = 0
        self.best_val     = float('inf')
        self.best_path    = None

        logger.info(
            f"Trainer | device={self.device} | batch={self.batch_size}×{self.grad_accum_steps}={self.batch_size*self.grad_accum_steps} | "
            f"AMP={'ON' if self.use_amp else 'OFF'} | patience={self.patience} | min_delta={self.min_delta}"
        )

    def temporal_split(self, X, y, timestamps):
        return temporal_split_data(X, y, timestamps)

    def _make_loader(self, X, y, shuffle):
        X_t = torch.from_numpy(X).float()
        y_t = torch.from_numpy(y).float()
        return DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
            drop_last=False,
        )

    def train(self,
              train_X: np.ndarray, train_y: np.ndarray,
              val_X:   np.ndarray, val_y:   np.ndarray,
              task_name:  str = "",
              model_name: str = "",
              verbose:    bool = False):

        train_loader = self._make_loader(train_X, train_y, shuffle=True)
        val_loader   = self._make_loader(val_X,   val_y,   shuffle=False)

        logger.info(
            f"[training] Starting | batch={self.batch_size}×{self.grad_accum_steps}={self.batch_size*self.grad_accum_steps} | "
            f"AMP={'ON' if self.use_amp else 'OFF'} | epochs={self.epochs} | "
            f"train={len(train_X):,} | val={len(val_X):,}"
        )

        os.makedirs('models', exist_ok=True)
        self.best_path = os.path.join('models', f'_best_{uuid.uuid4().hex[:8]}.pth')
        torch.save(self.model.state_dict(), self.best_path)

        label = f"[{task_name}/{model_name}]" if task_name else "[training]"

        epoch_bar = tqdm(
            range(1, self.epochs + 1),
            desc=f"  {label} epochs    ",
            unit="ep",
            file=sys.stderr,
            dynamic_ncols=True,
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] , {postfix}",
        )

        for epoch in epoch_bar:

            # ── Train ─────────────────────────────────────────────────────────
            self.model.train()
            train_loss_sum = 0.0
            self.optimizer.zero_grad()

            batch_bar = tqdm(
                train_loader,
                desc="  batches",
                unit="batch",
                file=sys.stderr,
                dynamic_ncols=True,
                leave=False,
            )

            for step, (xb, yb) in enumerate(batch_bar):
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    pred = torch.sigmoid(self.model(xb))
                    loss = self.criterion(pred, yb) / self.grad_accum_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                train_loss_sum += loss.item() * self.grad_accum_steps
                batch_bar.set_postfix(loss=f"{loss.item() * self.grad_accum_steps:.4f}")

            batch_bar.close()
            train_loss = train_loss_sum / max(len(train_loader), 1)

            # ── Validate ──────────────────────────────────────────────────────
            self.model.eval()
            val_loss_sum = 0.0
            val_preds, val_labels = [], []

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device, non_blocking=True)
                    yb = yb.to(self.device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        pred = torch.sigmoid(self.model(xb))
                        val_loss_sum += self.criterion(pred, yb).item()
                    val_preds.append(pred.cpu().numpy())
                    val_labels.append(yb.cpu().numpy())

            val_loss = val_loss_sum / max(len(val_loader), 1)

            vp = np.concatenate(val_preds,  axis=0)
            vl = np.concatenate(val_labels, axis=0)
            aurocs = []
            for t in range(vl.shape[1]):
                try:
                    if len(set(vl[:, t])) > 1:
                        aurocs.append(roc_auc_score(vl[:, t], vp[:, t]))
                except Exception:
                    pass
            mean_auroc = float(np.mean(aurocs)) if aurocs else 0.0

            # NaN guard
            if np.isnan(val_loss) or np.isinf(val_loss):
                logger.warning(f"[training] Epoch {epoch}: NaN val loss (check input normalization!) — skipping patience update")
                epoch_bar.set_postfix_str(
                    f"auroc={mean_auroc:.4f}, pat={self.patience_ctr}/{self.patience}, "
                    f"train={train_loss:.4f}, val=NaN"
                )
                continue

            # Checkpoint
            if val_loss < self.best_val - self.min_delta:
                self.best_val     = val_loss
                self.patience_ctr = 0
                torch.save(self.model.state_dict(), self.best_path)
            else:
                self.patience_ctr += 1

            epoch_bar.set_postfix_str(
                f"auroc={mean_auroc:.4f}, pat={self.patience_ctr}/{self.patience}, "
                f"train={train_loss:.4f}, val={val_loss:.4f}"
            )

            if self.patience_ctr >= self.patience:
                logger.info(f"[training] Early stop at epoch {epoch} (patience={self.patience})")
                break

        epoch_bar.close()

        # Restore best
        if self.best_path and os.path.exists(self.best_path):
            try:
                self.model.load_state_dict(torch.load(self.best_path, map_location=self.device))
                logger.info(f"[training] Restored best checkpoint (val_loss={self.best_val:.4f})")
            except Exception as e:
                logger.warning(f"[training] Could not restore checkpoint: {e}")
            finally:
                try:
                    os.remove(self.best_path)
                except OSError:
                    pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X).float()),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
        )
        self.model.eval()
        preds = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    preds.append(torch.sigmoid(self.model(xb)).cpu().numpy())
        return np.concatenate(preds, axis=0)

    def compute_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        metrics = {}
        for t in range(targets.shape[1]):
            try:
                metrics[f'task_{t}_auroc'] = (
                    roc_auc_score(targets[:, t], predictions[:, t])
                    if len(set(targets[:, t])) > 1 else float('nan')
                )
            except Exception:
                metrics[f'task_{t}_auroc'] = float('nan')
        valid = [v for v in metrics.values() if not np.isnan(v)]
        metrics['mean_auroc'] = float(np.mean(valid)) if valid else 0.0
        return metrics

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss':        self.best_val,
        }, path)
        logger.info(f"Checkpoint saved → {path}")

    def load_checkpoint(self, path: str):
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found: {path}")
            return
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        logger.info(f"Checkpoint loaded ← {path}")