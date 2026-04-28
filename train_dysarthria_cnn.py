import os
import random
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
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
    batch_size: int = 32
    num_workers: int = max(2, min(8, (os.cpu_count() or 4) // 2))
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.35
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
            sf.info(full_path)
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
        max_time_mask_ratio: float = 0.12,
        max_freq_mask_ratio: float = 0.15,
    ):
        self.noise_prob = noise_prob
        self.gain_prob = gain_prob
        self.time_mask_prob = time_mask_prob
        self.freq_mask_prob = freq_mask_prob
        self.max_noise_scale = max_noise_scale
        self.gain_db_range = gain_db_range
        self.max_time_mask_ratio = max_time_mask_ratio
        self.max_freq_mask_ratio = max_freq_mask_ratio

    def apply_waveform(self, waveform: np.ndarray) -> np.ndarray:
        if random.random() < self.gain_prob:
            gain_db = random.uniform(*self.gain_db_range)
            gain = 10 ** (gain_db / 20.0)
            waveform = waveform * gain

        if random.random() < self.noise_prob:
            noise_level = random.uniform(0.0, self.max_noise_scale)
            noise = np.random.randn(len(waveform)).astype(np.float32) * noise_level
            waveform = waveform + noise

        waveform = np.clip(waveform, -1.0, 1.0)
        return waveform.astype(np.float32)

    def apply_spec(self, spec: np.ndarray) -> np.ndarray:
        # spec shape: [1, n_mels, time]
        _, n_mels, n_steps = spec.shape

        if random.random() < self.time_mask_prob and n_steps > 4:
            max_t = max(1, int(n_steps * self.max_time_mask_ratio))
            t = random.randint(1, max_t)
            t0 = random.randint(0, n_steps - t)
            spec[:, :, t0 : t0 + t] = 0.0

        if random.random() < self.freq_mask_prob and n_mels > 4:
            max_f = max(1, int(n_mels * self.max_freq_mask_ratio))
            f = random.randint(1, max_f)
            f0 = random.randint(0, n_mels - f)
            spec[:, f0 : f0 + f, :] = 0.0

        return spec


class DysarthriaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: Config, train_mode: bool = False):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.train_mode = train_mode
        self.augment = AudioAugment() if train_mode else None

    def __len__(self) -> int:
        return len(self.df)

    def _load_audio(self, full_path: str) -> np.ndarray:
        wav, sr = librosa.load(full_path, sr=None, mono=False)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=0)

        if sr != self.cfg.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.cfg.sample_rate)

        wav = wav.astype(np.float32)
        target_len = self.cfg.num_samples
        if len(wav) > target_len:
            wav = wav[:target_len]
        elif len(wav) < target_len:
            wav = np.pad(wav, (0, target_len - len(wav)), mode="constant")
        return wav

    def _to_logmel(self, wav: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=self.cfg.sample_rate,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            n_mels=self.cfg.n_mels,
            fmin=self.cfg.fmin,
            fmax=self.cfg.fmax,
            power=2.0,
        )
        logmel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        logmel = np.expand_dims(logmel, axis=0)  # [1, n_mels, time]

        mean = logmel.mean()
        std = logmel.std()
        logmel = (logmel - mean) / (std + 1e-6)
        return logmel

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel_path = os.path.normpath(str(row["file_path"]))
        full_path = os.path.join(self.cfg.base_data_dir, rel_path)

        wav = self._load_audio(full_path)
        if self.train_mode and self.augment is not None:
            wav = self.augment.apply_waveform(wav)

        spec = self._to_logmel(wav)
        if self.train_mode and self.augment is not None:
            spec = self.augment.apply_spec(spec)

        x = torch.tensor(spec, dtype=torch.float32)
        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        return x, y


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
    val_df = filter_readable_audio(val_df, cfg, logger)
    test_df = filter_readable_audio(test_df, cfg, logger)

    train_ds = DysarthriaDataset(train_df, cfg, train_mode=True)
    val_ds = DysarthriaDataset(val_df, cfg, train_mode=False)
    test_ds = DysarthriaDataset(test_df, cfg, train_mode=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        drop_last=False,
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
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpectrogramCNN(nn.Module):
    def __init__(self, dropout: float = 0.35):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        scaler.scale(loss).backward()
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
    model = SpectrogramCNN(dropout=cfg.dropout).to(cfg.device)
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

    best_val_loss = float("inf")
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
    logger.info("Train samples=%d | Val samples=%d | Test samples=%d", len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, cfg.device, use_amp, logger, cfg, epoch
        )
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, cfg.device, use_amp, logger, epoch
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_msg = (
            f"Epoch [{epoch:02d}/{cfg.epochs}] "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )
        print(epoch_msg)
        logger.info(epoch_msg)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info("Saved new best model at epoch %d", epoch)
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                stop_msg = f"Early stopping triggered at epoch {epoch}."
                print(stop_msg)
                logger.info(stop_msg)
                break

    best_msg = f"Best model found at epoch {best_epoch} with val_loss={best_val_loss:.4f}"
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
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
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


# =========================
# 6. Visualization
# =========================
def plot_training_curves(history: Dict[str, List[float]], output_path: str) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc", linewidth=2)
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
    wav, sr = librosa.load(audio_path, sr=None, mono=False)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=0)
    if sr != cfg.sample_rate:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=cfg.sample_rate)

    wav = wav.astype(np.float32)
    if len(wav) > cfg.num_samples:
        wav = wav[: cfg.num_samples]
    elif len(wav) < cfg.num_samples:
        wav = np.pad(wav, (0, cfg.num_samples - len(wav)), mode="constant")

    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    logmel = np.expand_dims(logmel, axis=0)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)

    x = torch.tensor(logmel, dtype=torch.float32).unsqueeze(0)  # [1, 1, n_mels, time]
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
    print("\nTest Metrics")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\nClassification Report")
    print(report)
    logger.info("Classification report:\n%s", report)

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
