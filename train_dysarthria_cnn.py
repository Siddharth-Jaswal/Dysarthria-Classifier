import os
import random
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


# =========================
# 1. Imports + Config
# =========================
@dataclass
class Config:
    seed: int = 42
    csv_path: str = os.path.join("Database", "master_split.csv")
    base_data_dir: str = "Database"
    sample_rate: int = 16000
    duration_sec: float = 4.0
    num_samples: int = int(sample_rate * duration_sec)
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 256
    fmin: int = 20
    fmax: int = 8000
    batch_size: int = 192  # Doubled to fully utilize 8GB VRAM
    num_workers: int = max(2, min(8, (os.cpu_count() or 4) // 2))
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    dropout: float = 0.50
    conv_channels: Tuple[int, int, int, int] = (16, 32, 64, 128)
    classifier_hidden: int = 64
    early_stopping_patience: int = 6
    output_dir: str = "outputs"
    best_model_name: str = "best_model.pth"
    curves_name: str = "training_curves.png"
    cm_name: str = "confusion_matrix.png"
    log_file_name: str = "train.log"
    console_verbose: bool = False
    file_verbose: bool = True
    batch_log_interval: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(CFG.seed)
os.makedirs(CFG.output_dir, exist_ok=True)


def setup_logger(cfg: Config) -> logging.Logger:
    logger = logging.getLogger("dysarthria_train")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    log_path = os.path.join(cfg.output_dir, cfg.log_file_name)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG if cfg.file_verbose else logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO if cfg.console_verbose else logging.WARNING)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def filter_readable_audio(df: pd.DataFrame, cfg: Config, logger: logging.Logger) -> pd.DataFrame:
    valid_rows = []
    bad_files = []

    for _, row in df.iterrows():
        rel_path = os.path.normpath(str(row["file_path"]))
        full_path = os.path.join(cfg.base_data_dir, rel_path)

        if not os.path.isfile(full_path):
            bad_files.append((full_path, "missing_file"))
            continue

        try:
            torchaudio.info(full_path)
            valid_rows.append(row)
        except Exception as exc:
            bad_files.append((full_path, str(exc)))

    if bad_files:
        logger.warning("Skipping %d unreadable/missing audio files.", len(bad_files))
        for path, reason in bad_files[:20]:
            logger.warning("Skipped file: %s | reason: %s", path, reason)
        if len(bad_files) > 20:
            logger.warning("... plus %d more skipped files.", len(bad_files) - 20)

    filtered_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    if filtered_df.empty:
        raise RuntimeError("No readable audio files found after filtering.")
    return filtered_df


# =========================
# 2. Dataset + Dataloader
# =========================
class AudioAugment:
    def __init__(
        self,
        noise_prob: float = 0.5,
        gain_prob: float = 0.5,
        time_mask_prob: float = 0.5,
        freq_mask_prob: float = 0.5,
        max_noise_scale: float = 0.015,
        gain_db_range: Tuple[float, float] = (-6.0, 6.0),
    ):
        self.noise_prob = noise_prob
        self.gain_prob = gain_prob
        self.max_noise_scale = max_noise_scale
        self.gain_db_range = gain_db_range

    def apply_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() < self.gain_prob:
            gain_db = random.uniform(*self.gain_db_range)
            gain = 10 ** (gain_db / 20.0)
            waveform = waveform * gain

        if random.random() < self.noise_prob:
            noise_level = random.uniform(0.0, self.max_noise_scale)
            noise = torch.randn_like(waveform) * noise_level
            waveform = waveform + noise

        waveform = torch.clamp(waveform, -1.0, 1.0)
        return waveform


class DysarthriaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: Config, train_mode: bool = False):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.train_mode = train_mode
        self.augment = AudioAugment() if train_mode else None

    def __len__(self) -> int:
        return len(self.df)

    def _load_audio(self, full_path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(full_path)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        if sr != self.cfg.sample_rate:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.cfg.sample_rate)

        wav = wav.squeeze(0)
        target_len = self.cfg.num_samples
        if wav.shape[0] > target_len:
            wav = wav[:target_len]
        elif wav.shape[0] < target_len:
            wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[0]))
        return wav

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel_path = os.path.normpath(str(row["file_path"]))
        full_path = os.path.join(self.cfg.base_data_dir, rel_path)

        wav = self._load_audio(full_path)
        if self.train_mode and self.augment is not None:
            wav = self.augment.apply_waveform(wav)

        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        return wav, y


def create_dataloaders(cfg: Config, logger: logging.Logger):
    df = pd.read_csv(cfg.csv_path)
    expected_cols = {"file_path", "speaker_id", "gender", "session_id", "label", "split"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_df = filter_readable_audio(train_df, cfg, logger)
    val_df = filter_readable_audio(val_df, cfg, logger) if not val_df.empty else val_df.reset_index(drop=True)
    test_df = filter_readable_audio(test_df, cfg, logger)

    train_ds = DysarthriaDataset(train_df, cfg, train_mode=True)
    val_ds = DysarthriaDataset(val_df, cfg, train_mode=False) if not val_df.empty else None
    test_ds = DysarthriaDataset(test_df, cfg, train_mode=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(cfg.device == "cuda"),
            drop_last=False,
        )
        if val_ds is not None
        else None
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        drop_last=False,
    )
    return train_loader, val_loader, test_loader, train_df


# =========================
# 3. CNN Model
# =========================
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout_prob: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpectrogramCNN(nn.Module):
    def __init__(self, cfg: Config, dropout: float = 0.50, conv_dropout: float = 0.2):
        super().__init__()
        # Moved GPU-bound Mel Spectrogram extraction into the model
        self.mel_spec = T.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=cfg.fmin,
            f_max=cfg.fmax,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)
        
        # GPU-bound SpecAugment
        self.time_masking = T.TimeMasking(time_mask_param=int(cfg.num_samples / cfg.hop_length * 0.2))
        self.freq_masking = T.FrequencyMasking(freq_mask_param=int(cfg.n_mels * 0.2))

        c1, c2, c3, c4 = cfg.conv_channels
        self.features = nn.Sequential(
            ConvBlock(1, c1, conv_dropout),
            ConvBlock(c1, c2, conv_dropout),
            ConvBlock(c2, c3, conv_dropout),
            ConvBlock(c3, c4, conv_dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c4, cfg.classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(cfg.classifier_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape in: [batch, time]
        # Keep feature extraction in fp32. CUDA autocast can overflow the mel
        # power spectrogram in float16, which then poisons the epoch loss with NaNs.
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = self.mel_spec(x.float()) # [batch, n_mels, time]
            x = self.amplitude_to_db(x)

            # Fast batched per-sample normalization on GPU
            mean = x.mean(dim=[-2, -1], keepdim=True)
            std = x.std(dim=[-2, -1], keepdim=True)
            x = (x - mean) / (std + 1e-4)
        x = x.unsqueeze(1) # [batch, 1, n_mels, time]
        
        if self.training:
            if random.random() < 0.5:
                x = self.time_masking(x)
            if random.random() < 0.5:
                x = self.freq_masking(x)

        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x).squeeze(1)


# =========================
# 4. Training Pipeline
# =========================
def compute_pos_weight(train_df: pd.DataFrame, device: str) -> torch.Tensor:
    num_pos = int((train_df["label"] == 1).sum())
    num_neg = int((train_df["label"] == 0).sum())
    if num_pos == 0:
        raise ValueError("Training set has no positive samples.")
    pos_weight = torch.tensor([num_neg / max(1, num_pos)], dtype=torch.float32, device=device)
    return pos_weight


def binary_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == targets).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: str,
    use_amp: bool,
    logger: logging.Logger,
    cfg: Config,
    epoch: int,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for batch_idx, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite train loss at epoch {epoch}, batch {batch_idx}.")

        scaler.scale(loss).backward()
        
        # Unscale gradients before clipping to prevent NaN propagation
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * x.size(0)
        running_acc += binary_accuracy_from_logits(logits.detach(), y) * x.size(0)

        if cfg.file_verbose and (batch_idx % cfg.batch_log_interval == 0 or batch_idx == len(loader)):
            logger.debug(
                "Epoch %02d | Batch %d/%d | batch_loss=%.4f",
                epoch,
                batch_idx,
                len(loader),
                loss.item(),
            )

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    use_amp: bool,
    logger: logging.Logger,
    epoch: int,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0

    for batch_idx, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite validation loss at epoch {epoch}, batch {batch_idx}.")

        running_loss += loss.item() * x.size(0)
        running_acc += binary_accuracy_from_logits(logits, y) * x.size(0)

        if batch_idx == len(loader):
            logger.debug("Epoch %02d | Validation complete", epoch)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)
    return epoch_loss, epoch_acc


def train_model(cfg: Config):
    logger = setup_logger(cfg)
    train_loader, val_loader, test_loader, train_df = create_dataloaders(cfg, logger)
    model = SpectrogramCNN(cfg, dropout=cfg.dropout).to(cfg.device)
    pos_weight = compute_pos_weight(train_df, cfg.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    use_amp = cfg.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    has_val = val_loader is not None
    best_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_model_path = os.path.join(cfg.output_dir, cfg.best_model_name)
    logger.info("Training started on device=%s", cfg.device)
    logger.info(
        "Train samples=%d | Val samples=%d | Test samples=%d",
        len(train_loader.dataset),
        len(val_loader.dataset) if has_val else 0,
        len(test_loader.dataset),
    )
    if not has_val:
        logger.info("No validation split found. Training will save the best model by train_loss only.")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, cfg.device, use_amp, logger, cfg, epoch
        )
        if has_val:
            val_loss, val_acc = validate_one_epoch(
                model, val_loader, criterion, cfg.device, use_amp, logger, epoch
            )
            monitor_loss = val_loss
            scheduler.step(val_loss)
        else:
            val_loss, val_acc = float("nan"), float("nan")
            monitor_loss = train_loss
            scheduler.step(train_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        if has_val:
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_msg = (
            f"Epoch [{epoch:02d}/{cfg.epochs}] "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
        )
        if has_val:
            epoch_msg += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        print(epoch_msg)
        logger.info(epoch_msg)

        if monitor_loss < best_loss:
            best_loss = monitor_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info("Saved new best model at epoch %d", epoch)
        else:
            patience_counter += 1
            if has_val and patience_counter >= cfg.early_stopping_patience:
                stop_msg = f"Early stopping triggered at epoch {epoch}."
                print(stop_msg)
                logger.info(stop_msg)
                break

    monitor_name = "val_loss" if has_val else "train_loss"
    best_msg = f"Best model found at epoch {best_epoch} with {monitor_name}={best_loss:.4f}"
    print(best_msg)
    logger.info(best_msg)
    model.load_state_dict(torch.load(best_model_path, map_location=cfg.device))
    return model, test_loader, history, logger


# =========================
# 5. Evaluation
# =========================
@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    all_probs = []
    all_preds = []
    all_targets = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(np.int64)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_targets.extend(y.numpy().astype(np.int64).tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
    }
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Healthy (0)", "Dysarthric (1)"],
        digits=4,
        zero_division=0,
    )
    return metrics, cm, report


def build_eval_loader(df: pd.DataFrame, cfg: Config) -> DataLoader:
    dataset = DysarthriaDataset(df, cfg, train_mode=False)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        drop_last=False,
    )


def print_and_log_metrics(title: str, metrics: Dict[str, float], report: str, logger: logging.Logger) -> None:
    print(f"\n{title}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print("\nClassification Report")
    print(report)
    logger.info("%s metrics: %s", title, {key: round(value, 4) for key, value in metrics.items()})
    logger.info("%s classification report:\n%s", title, report)


def evaluate_test_by_microphone(model: nn.Module, test_loader: DataLoader, cfg: Config, logger: logging.Logger) -> None:
    test_df = test_loader.dataset.df
    mic_subsets = {
        "Test Metrics - HeadMic": test_df[test_df["file_path"].str.contains("wav_headMic", regex=False)].copy(),
        "Test Metrics - ArrayMic": test_df[test_df["file_path"].str.contains("wav_arrayMic", regex=False)].copy(),
    }

    for title, subset_df in mic_subsets.items():
        if subset_df.empty:
            logger.warning("%s skipped because no rows were found.", title)
            continue
        subset_loader = build_eval_loader(subset_df, cfg)
        metrics, _, report = evaluate_model(model, subset_loader, cfg.device)
        print_and_log_metrics(title, metrics, report, logger)


# =========================
# 6. Visualization
# =========================
def plot_training_curves(history: Dict[str, List[float]], output_path: str) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    has_val = len(history["val_loss"]) == len(history["train_loss"]) and len(history["val_loss"]) > 0
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    if has_val:
        axes[0].plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc", linewidth=2)
    if has_val:
        axes[1].plot(epochs, history["val_acc"], label="Val Acc", linewidth=2)
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, output_path: str) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Healthy", "Dysarthric"],
        yticklabels=["Healthy", "Dysarthric"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


# =========================
# 7. Inference
# =========================
def preprocess_single_audio(audio_path: str, cfg: Config) -> torch.Tensor:
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != cfg.sample_rate:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=cfg.sample_rate)

    wav = wav.squeeze(0)
    if wav.shape[0] > cfg.num_samples:
        wav = wav[:cfg.num_samples]
    elif wav.shape[0] < cfg.num_samples:
        wav = torch.nn.functional.pad(wav, (0, cfg.num_samples - wav.shape[0]))
    
    x = wav.unsqueeze(0) # [1, time]
    return x


@torch.no_grad()
def predict_audio(audio_path: str, model: nn.Module, cfg: Config):
    model.eval()
    x = preprocess_single_audio(audio_path, cfg).to(cfg.device)
    logit = model(x)
    prob = torch.sigmoid(logit).item()
    pred = 1 if prob >= 0.5 else 0
    label = "Dysarthric" if pred == 1 else "Healthy"
    return label, prob


# =========================
# 8. Main
# =========================
def main():
    print(f"Using device: {CFG.device}")
    model, test_loader, history, logger = train_model(CFG)

    metrics, cm, report = evaluate_model(model, test_loader, CFG.device)
    print_and_log_metrics("Test Metrics - Combined", metrics, report, logger)
    evaluate_test_by_microphone(model, test_loader, CFG, logger)

    curves_path = os.path.join(CFG.output_dir, CFG.curves_name)
    cm_path = os.path.join(CFG.output_dir, CFG.cm_name)
    plot_training_curves(history, curves_path)
    plot_confusion_matrix(cm, cm_path)

    print("\nSaved artifacts:")
    print(os.path.join(CFG.output_dir, CFG.best_model_name))
    print(curves_path)
    print(cm_path)
    print(os.path.join(CFG.output_dir, CFG.log_file_name))
    logger.info("Saved artifacts: %s, %s, %s, %s", os.path.join(CFG.output_dir, CFG.best_model_name), curves_path, cm_path, os.path.join(CFG.output_dir, CFG.log_file_name))

    # Example usage:
    # example_file = os.path.join("Database", "M_data", "M01", "Session1", "wav_headMic", "0001.wav")
    # pred_label, pred_prob = predict_audio(example_file, model, CFG)
    # print(f"Prediction: {pred_label} | Probability(dysarthric): {pred_prob:.4f}")


if __name__ == "__main__":
    main()
