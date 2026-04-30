import os
import random
import logging
import time
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
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# ======================================================================================
# 1. Configuration
# ======================================================================================
@dataclass
class Config:
    # --- General ---
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Data ---
    csv_path: str = os.path.join("Database", "master_severity_final.csv")
    base_data_dir: str = "Database"
    num_classes: int = 3
    class_names: Tuple[str, str, str] = ("Healthy", "Mild", "Severe")

    # --- Audio Preprocessing ---
    sample_rate: int = 16000
    duration_sec: float = 4.0
    num_samples: int = int(sample_rate * duration_sec)
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 256
    fmin: int = 20
    fmax: int = 8000

    # --- Training ---
    batch_size: int = 128  # Increased to utilize more of the available VRAM
    num_workers: int = max(2, min(os.cpu_count() or 1, 8))
    epochs: int = 20 # Max epochs per fold
    learning_rate: float = 5e-4 # Slowed down to prevent fast memorization
    weight_decay: float = 1e-2 # Stronger L2 penalty to force smaller generalized weights
    early_stopping_patience: int = 7
    grad_clip_norm: float = 2.0

    # --- Model Architecture ---
    conv_channels: Tuple[int, int, int] = (32, 64, 128)
    conv_dropout: float = 0.25 # Adjusted dropout
    gru_hidden_size: int = 128
    classifier_hidden: int = 64
    classifier_dropout: float = 0.45 # Adjusted dropout

    # --- Outputs & Logging ---
    output_dir: str = "outputs_loso_final_labels"
    log_file_name: str = "loso_train.log"
    console_verbose: bool = False
    file_verbose: bool = True
    batch_log_interval: int = 20


CFG = Config()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(cfg: Config) -> logging.Logger:
    logger = logging.getLogger("dysarthria_loso_train")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    log_path = os.path.join(cfg.output_dir, cfg.log_file_name)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG if cfg.file_verbose else logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO if cfg.console_verbose else logging.WARNING)
    stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


# ======================================================================================
# 2. Dataset & Augmentations
# ======================================================================================
class AudioAugment:
    def __init__(
        self,
        noise_prob: float = 0.5,
        gain_prob: float = 0.5,
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

        return torch.clamp(waveform, -1.0, 1.0)


class DysarthriaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: Config, train_mode: bool = False):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.train_mode = train_mode
        self.augment = AudioAugment() if train_mode else None

    def __len__(self) -> int:
        return len(self.df)

    def _load_audio(self, full_path: str) -> torch.Tensor:
        try:
            wav, sr = torchaudio.load(full_path)
        except Exception as e:
            # Return a silent tensor if loading fails
            return torch.zeros(self.cfg.num_samples, dtype=torch.float32)

        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        if sr != self.cfg.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.cfg.sample_rate)
            wav = resampler(wav)

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

        y = torch.tensor(int(row["severity_label"]), dtype=torch.long)
        return wav, y


# ======================================================================================
# 3. Model Architecture (CNN + BiGRU + Attention)
# ======================================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout_prob: float):
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

class Attention(nn.Module):
    def __init__(self, feature_dim: int, step_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.attention_fc = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, step_dim, feature_dim]
        eij = self.attention_fc(x).squeeze(-1) # [batch, step_dim]
        alpha = torch.softmax(eij, dim=1).unsqueeze(1) # [batch, 1, step_dim]
        context = torch.bmm(alpha, x).squeeze(1) # [batch, feature_dim]
        return context

class SeverityCNNBiGRU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        
        # GPU-bound feature extraction
        self.mel_spec = T.MelSpectrogram(
            sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            n_mels=cfg.n_mels, f_min=cfg.fmin, f_max=cfg.fmax, power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)
        
        # GPU-bound SpecAugment
        time_steps = int(cfg.num_samples / cfg.hop_length) # 250 frames
        self.time_masking = T.TimeMasking(time_mask_param=int(time_steps * 0.15)) # Fixed: Now correctly masks up to 37 frames
        self.freq_masking = T.FrequencyMasking(freq_mask_param=int(cfg.n_mels * 0.15))

        # CNN Backbone
        c1, c2, c3 = cfg.conv_channels
        self.features = nn.Sequential(
            ConvBlock(1, c1, cfg.conv_dropout),
            ConvBlock(c1, c2, cfg.conv_dropout),
            ConvBlock(c2, c3, cfg.conv_dropout),
        )
        
        # Calculate CNN output shape for GRU
        # After 3 MaxPool(2,2) layers, the mel dimension is reduced by 2^3=8
        cnn_out_mels = cfg.n_mels // (2**len(cfg.conv_channels))
        self.gru_input_size = c3 * cnn_out_mels

        # Recurrent Layer
        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=cfg.gru_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        
        # Attention Pooling
        gru_output_size = cfg.gru_hidden_size * 2
        self.attention = Attention(feature_dim=gru_output_size, step_dim=-1) # step_dim will be dynamic

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_size, cfg.classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfg.classifier_dropout),
            nn.Linear(cfg.classifier_hidden, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch, num_samples]
        
        # 1. Feature Extraction (on GPU)
        with torch.cuda.amp.autocast(enabled=False):
            x = self.mel_spec(x.float())
            x = self.amplitude_to_db(x)
            
            # Per-sample normalization
            mean = x.mean(dim=(-2, -1), keepdim=True)
            std = x.std(dim=(-2, -1), keepdim=True)
            x = (x - mean) / (std + 1e-6)
        
        x = x.unsqueeze(1) # [batch, 1, n_mels, time]

        # 2. Augmentation (on GPU, only during training)
        if self.training:
            for _ in range(2): # Apply up to 2 masks to heavily obscure speaker identity hints
                if random.random() < 0.6: x = self.time_masking(x)
                if random.random() < 0.6: x = self.freq_masking(x)

        # 3. CNN Backbone
        x = self.features(x) # [batch, c3, n_mels/8, time]

        # 4. Reshape for GRU
        x = x.permute(0, 3, 1, 2) # [batch, time, c3, n_mels/8]
        b, t, c, m = x.shape
        x = x.reshape(b, t, c * m) # [batch, time, c3 * n_mels/8]

        # 5. BiGRU Layer
        self.gru.flatten_parameters() # Optimization for DataParallel
        x, _ = self.gru(x) # [batch, time, gru_hidden * 2]

        # 6. Attention Pooling
        x = self.attention(x) # [batch, gru_hidden * 2]
        
        # 7. Classifier
        return self.classifier(x)


# ======================================================================================
# 4. Training & Validation
# ======================================================================================
def compute_class_weights(train_df: pd.DataFrame, cfg: Config) -> torch.Tensor:
    counts = train_df["severity_label"].value_counts().reindex(range(cfg.num_classes), fill_value=0).astype(float)
    if (counts == 0).any():
        missing = [cfg.class_names[idx] for idx, count in counts.items() if count == 0]
        logging.warning("Training split is missing severity classes: %s. Weights will be 1.0.", missing)
        counts[counts == 0] = 1 # Avoid division by zero
    
    total = counts.sum()
    weights = total / (cfg.num_classes * counts)
    
    # Apply square root to dampen extreme weight differences
    weights = np.sqrt(weights)
    weights = np.clip(weights, 0.7, 1.8)
    return torch.tensor(weights, dtype=torch.float32, device=cfg.device)


def train_one_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler, cfg: Config, logger: logging.Logger, epoch: int
) -> Tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    use_amp = (cfg.device == "cuda")
    all_preds, all_targets = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch} Train", leave=False)
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(cfg.device, non_blocking=True), y.to(cfg.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        if not torch.isfinite(loss):
            logger.error(f"Non-finite train loss at epoch {epoch}, batch {batch_idx}. Skipping update.")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = x.size(0)
        running_loss += loss.item() * batch_size
        
        preds = torch.argmax(logits.detach(), dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())
        
        pbar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    epoch_bal_acc = balanced_accuracy_score(all_targets, all_preds)
    epoch_macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    
    return epoch_loss, epoch_bal_acc, epoch_macro_f1


# ======================================================================================
# 5. Evaluation
# ======================================================================================
@torch.no_grad()
def get_predictions(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(y.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, cfg: Config) -> Dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    report = classification_report(
        y_true, y_pred, labels=list(range(cfg.num_classes)),
        target_names=cfg.class_names, zero_division=0, output_dict=True
    )
    for i, name in enumerate(cfg.class_names):
        metrics[f"recall_{name}"] = report[name]['recall']
    return metrics


def plot_training_curves(history: Dict, output_path: str, fold: int) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    fig.suptitle(f"Fold {fold} Training Curves", fontsize=16)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", lw=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_bal_acc"], label="Train Bal Acc", lw=2)
    axes[1].set_title("Balanced Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Bal Acc")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["train_macro_f1"], label="Train Macro F1", lw=2)
    axes[2].set_title("Macro F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro F1")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: str, cfg: Config) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(cfg.num_classes)))
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=cfg.class_names, yticklabels=cfg.class_names,
        annot_kws={"size": 12}
    )
    plt.title("Overall Confusion Matrix (All Folds)", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ======================================================================================
# 6. LOSO Training Loop
# ======================================================================================
def train_fold(
    fold_idx: int, train_loader: DataLoader, test_loader: DataLoader, test_speaker: str, cfg: Config, logger: logging.Logger
) -> str:
    """Trains a model for a single fold and returns the path to the best model."""
    logger.info("-" * 50)
    logger.info(f"Fold {fold_idx}: Training on {len(train_loader.dataset)} samples.")

    model = SeverityCNNBiGRU(cfg).to(cfg.device)
    
    class_weights = compute_class_weights(train_loader.dataset.df, cfg)
    logger.info(f"Fold {fold_idx}: Using class weights: {class_weights.cpu().numpy().round(2)}")
    # Gentle label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.device == "cuda"))

    best_train_loss = float("inf")
    patience_counter = 0
    best_epoch = -1
    
    history = {"train_loss": [], "train_bal_acc": [], "train_macro_f1": []}
    
    fold_model_path = os.path.join(cfg.output_dir, f"best_model_fold_{fold_idx}.pth")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_bal_acc, train_macro_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, cfg, logger, epoch
        )
        
        history["train_loss"].append(train_loss)
        history["train_bal_acc"].append(train_bal_acc)
        history["train_macro_f1"].append(train_macro_f1)
        
        scheduler.step(train_loss)
        
        # --- EVALUATE TEST SET (MONITORING ONLY) ---
        y_pred, y_true = get_predictions(model, test_loader, cfg.device)
        test_metrics = calculate_metrics(y_true, y_pred, cfg)
        
        true_class_idx = int(y_true[0]) if len(y_true) > 0 else 0
        true_class_name = cfg.class_names[true_class_idx]
        true_class_recall = test_metrics.get(f"recall_{true_class_name}", 0.0)
        
        pred_counts = np.bincount(y_pred, minlength=cfg.num_classes)
        pred_dist_str = "\n".join([f"  {cfg.class_names[i]}: {count}" for i, count in enumerate(pred_counts)])
        
        epoch_msg = (
            f"\nFold {fold_idx} | Epoch {epoch:02d} | LR: {optimizer.param_groups[0]['lr']:.1e}\n"
            f"  Train -> Loss: {train_loss:.4f}\n"
            f"  Train -> Bal Acc: {train_bal_acc:.4f}\n"
            f"  Train -> Macro F1: {train_macro_f1:.4f}\n\n"
            f"  Test Monitor (Speaker {test_speaker}):\n"
            f"  Accuracy: {test_metrics['accuracy']:.4f}\n"
            f"  Balanced Acc: {test_metrics['balanced_accuracy']:.4f}\n"
            f"  Macro F1: {test_metrics['macro_f1']:.4f}\n\n"
            f"  Predicted classes:\n"
            f"{pred_dist_str}\n\n"
            f"  True Class: {true_class_name}\n"
            f"  Recall on True Class: {true_class_recall:.4f}"
        )
        print(epoch_msg)
        logger.info(epoch_msg)

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), fold_model_path)
            logger.info(f"-> New best model saved at epoch {epoch} with train_loss: {train_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                print(f"Early stopping triggered at epoch {epoch}.")
                break
    
    logger.info(f"Finished fold {fold_idx}. Best epoch: {best_epoch}, Best train_loss: {best_train_loss:.4f}")
    
    # Plot curves for this fold
    curves_path = os.path.join(cfg.output_dir, f"training_curves_fold_{fold_idx}.png")
    plot_training_curves(history, curves_path, fold_idx)
    
    return fold_model_path


def main():
    start_time = time.time()
    set_seed(CFG.seed)
    logger = setup_logger(CFG)
    logger.info("Starting LOSO Cross-Validation...")
    logger.info(f"Using device: {CFG.device}")
    logger.info(f"Number of workers: {CFG.num_workers}")

    # --- Load and Prepare Data ---
    try:
        full_df = pd.read_csv(CFG.csv_path, keep_default_na=False)
        logger.info(f"Loaded {len(full_df)} rows from {CFG.csv_path}")
    except FileNotFoundError:
        logger.error(f"CSV file not found at {CFG.csv_path}. Please run the splitter notebook first.")
        return

    unique_speakers = sorted(full_df["speaker_id"].unique())
    logger.info(f"Found {len(unique_speakers)} unique speakers: {unique_speakers}")

    all_fold_metrics = []
    all_preds, all_targets = [], []

    # --- LOSO Loop ---
    for fold_idx, test_speaker in enumerate(unique_speakers, 1):
        print("\n" + "="*60)
        print(f"Starting Fold {fold_idx}/{len(unique_speakers)}: Test Speaker = {test_speaker}")
        logger.info("="*60)
        logger.info(f"Starting Fold {fold_idx}/{len(unique_speakers)}: Test Speaker = {test_speaker}")

        # 1. Split data for the current fold
        dev_df = full_df[full_df["speaker_id"] != test_speaker].copy()
        test_df = full_df[full_df["speaker_id"] == test_speaker].copy()

        train_speakers = sorted(dev_df["speaker_id"].unique())
        train_df = dev_df

        logger.info(f"Train speakers ({len(train_speakers)}): {train_speakers}")
        logger.info(f"Test speaker ({1}): {test_speaker}")

        # 2. Create DataLoaders
        train_ds = DysarthriaDataset(train_df, CFG, train_mode=True)
        test_ds = DysarthriaDataset(test_df, CFG, train_mode=False)

        train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=True)

        # 3. Train the model for the fold
        best_model_path = train_fold(fold_idx, train_loader, test_loader, test_speaker, CFG, logger)

        # 4. Evaluate on the held-out test speaker
        logger.info(f"Evaluating fold {fold_idx} on test speaker {test_speaker}...")
        model = SeverityCNNBiGRU(CFG).to(CFG.device)
        model.load_state_dict(torch.load(best_model_path, map_location=CFG.device))
        
        y_pred, y_true = get_predictions(model, test_loader, CFG.device)
        all_preds.append(y_pred)
        all_targets.append(y_true)

        fold_metrics = calculate_metrics(y_true, y_pred, CFG)
        fold_metrics["fold"] = fold_idx
        fold_metrics["test_speaker"] = test_speaker
        all_fold_metrics.append(fold_metrics)
        
        logger.info(f"Fold {fold_idx} Test Metrics: {fold_metrics}")
        print(f"Fold {fold_idx} Test Metrics (Speaker {test_speaker}): "
              f"Acc={fold_metrics['accuracy']:.4f}, BalAcc={fold_metrics['balanced_accuracy']:.4f}, F1={fold_metrics['macro_f1']:.4f}")

    # --- Aggregate and Report Final Results ---
    logger.info("="*60)
    logger.info("LOSO Cross-Validation Finished. Aggregating results...")
    print("\n" + "="*60)
    print("LOSO Cross-Validation Finished. Aggregating results...")

    metrics_df = pd.DataFrame(all_fold_metrics)
    
    # Per-speaker results
    per_speaker_df = metrics_df.set_index("test_speaker")[['accuracy', 'balanced_accuracy', 'macro_f1']]
    print("\n--- Per-Speaker Test Metrics ---")
    print(per_speaker_df.round(4))
    logger.info("\n--- Per-Speaker Test Metrics ---\n" + per_speaker_df.round(4).to_string())

    # Aggregate metrics
    mean_metrics = metrics_df.mean(numeric_only=True)
    std_metrics = metrics_df.std(numeric_only=True)
    
    print("\n--- Aggregate Test Metrics (Mean +/- Std) ---")
    logger.info("\n--- Aggregate Test Metrics (Mean +/- Std) ---")
    for col in ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1']:
        mean_val, std_val = mean_metrics.get(col, 0), std_metrics.get(col, 0)
        print(f"{col.replace('_', ' ').title()}: {mean_val:.4f} +/- {std_val:.4f}")
        logger.info(f"{col.replace('_', ' ').title()}: {mean_val:.4f} +/- {std_val:.4f}")

    # Overall Classification Report and Confusion Matrix
    final_y_true = np.concatenate(all_targets)
    final_y_pred = np.concatenate(all_preds)
    
    print("\n--- Overall Classification Report (All Folds) ---")
    overall_report = classification_report(
        final_y_true, final_y_pred, labels=list(range(CFG.num_classes)),
        target_names=CFG.class_names, zero_division=0, digits=4
    )
    print(overall_report)
    logger.info("\n--- Overall Classification Report (All Folds) ---\n" + overall_report)

    cm_path = os.path.join(CFG.output_dir, "confusion_matrix_overall.png")
    plot_confusion_matrix(final_y_true, final_y_pred, cm_path, CFG)
    logger.info(f"Saved overall confusion matrix to {cm_path}")

    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time/60:.2f} minutes")
    print(f"\nTotal execution time: {total_time/60:.2f} minutes")


if __name__ == "__main__":
    main()