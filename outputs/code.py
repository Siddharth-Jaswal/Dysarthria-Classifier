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
    recall_score,
)
from torch.utils.data import DataLoader, Dataset


# =========================
# 1. Imports + Config
# =========================
@dataclass
class Config:
    seed: int = 42
    csv_path: str = os.path.join("Database", "master_severity_final.csv")
    base_data_dir: str = "Database"
    num_classes: int = 3
    class_names: Tuple[str, str, str] = ("Healthy", "Mild", "Severe")
    sample_rate: int = 16000
    duration_sec: float = 4.0
    num_samples: int = int(sample_rate * duration_sec)
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    fmin: int = 20
    fmax: int = 8000
    batch_size: int = 24
    num_workers: int = 8
    epochs: int = 18
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    
    conv_channels: Tuple[int, int, int] = (32, 64, 128)
    gru_hidden: int = 128
    gru_dropout: float = 0.2
    classifier_hidden: int = 64
    classifier_dropout: float = 0.3
    
    early_stopping_patience: int = 6
    grad_clip_norm: float = 1.0
    
    output_dir: str = "outputs_cnn_bigru_loso"
    best_models_dir: str = os.path.join("outputs_cnn_bigru_loso", "best_fold_models")
    log_file_name: str = "training_log.txt"
    results_csv_name: str = "results_cnn_bigru_loso.csv"
    
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
os.makedirs(CFG.best_models_dir, exist_ok=True)


def setup_logger(cfg: Config) -> logging.Logger:
    logger = logging.getLogger("dysarthria_severity_loso")
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

        y = torch.tensor(int(row["severity_label"]), dtype=torch.long)
        return wav, y

# =========================
# 3. Model Architecture
# =========================
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionPooling(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.attention = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.attention(x).squeeze(-1)
        alpha = torch.softmax(scores, dim=1).unsqueeze(1)
        context = torch.bmm(alpha, x).squeeze(1)
        return context


class SeverityCNNBiGRU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        
        self.mel_spec = T.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            n_mels=cfg.n_mels,
            f_min=cfg.fmin,
            f_max=cfg.fmax,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)
        
        time_steps = int(cfg.num_samples / cfg.hop_length)
        self.time_masking = T.TimeMasking(time_mask_param=int(time_steps * 0.15))
        self.freq_masking = T.FrequencyMasking(freq_mask_param=int(cfg.n_mels * 0.15))

        c1, c2, c3 = cfg.conv_channels
        self.features = nn.Sequential(
            ConvBlock(1, c1),
            ConvBlock(c1, c2),
            ConvBlock(c2, c3),
        )
        
        cnn_out_mels = cfg.n_mels // 8
        self.gru_input_size = c3 * cnn_out_mels

        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=cfg.gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.gru_drop = nn.Dropout(cfg.gru_dropout)
        
        gru_output_size = cfg.gru_hidden * 2
        self.attention = AttentionPooling(gru_output_size)

        self.classifier = nn.Sequential(
            nn.Linear(gru_output_size, cfg.classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfg.classifier_dropout),
            nn.Linear(cfg.classifier_hidden, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = self.mel_spec(x.float()) # [batch, n_mels, time]
            x = self.amplitude_to_db(x)

            mean = x.mean(dim=[-2, -1], keepdim=True)
            std = x.std(dim=[-2, -1], keepdim=True)
            x = (x - mean) / (std + 1e-6)
            
        x = x.unsqueeze(1)
        
        if self.training:
            if random.random() < 0.5: x = self.time_masking(x)
            if random.random() < 0.5: x = self.freq_masking(x)

        x = self.features(x)
        
        x = x.permute(0, 3, 1, 2)
        b, t, c, m = x.shape
        x = x.reshape(b, t, c * m)
        
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = self.gru_drop(x)
        
        x = self.attention(x)
        return self.classifier(x)


# =========================
# 4. Training Pipeline
# =========================
def train_fold(
    fold_idx: int, 
    test_speaker: str, 
    train_loader: DataLoader, 
    test_loader: DataLoader,
    logger: logging.Logger,
    cfg: Config,
):
    model = SeverityCNNBiGRU(cfg).to(cfg.device)
    
    counts = train_loader.dataset.df["severity_label"].value_counts().reindex(range(cfg.num_classes), fill_value=0).astype(float)
    counts[counts == 0] = 1 
    weights = counts.sum() / (cfg.num_classes * counts)
    weights = np.sqrt(weights)
    weights = np.clip(weights, 0.7, 1.8)
    class_weights = torch.tensor(weights.to_numpy(), dtype=torch.float32, device=cfg.device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    
    use_amp = cfg.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    
    best_train_f1 = -1.0
    
    fold_model_path = os.path.join(cfg.best_models_dir, f"best_model_{test_speaker}.pth")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []
        
        for x, y in train_loader:
            x, y = x.to(cfg.device, non_blocking=True), y.to(cfg.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits.detach(), dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(y.cpu().numpy())
            
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        train_bal_acc = balanced_accuracy_score(train_targets, train_preds)
        train_macro_f1 = f1_score(train_targets, train_preds, average="macro", zero_division=0)
        
        scheduler.step(train_macro_f1)
        
        if train_macro_f1 > best_train_f1:
            best_train_f1 = train_macro_f1
            torch.save(model.state_dict(), fold_model_path)
            
        # Eval Monitor
        model.eval()
        test_preds, test_targets, test_probs = [], [], []
        for x, y in test_loader:
            x = x.to(cfg.device, non_blocking=True)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
            test_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            test_targets.extend(y.numpy())
            test_probs.extend(probs.cpu().numpy())
            
        test_acc = accuracy_score(test_targets, test_preds)
        test_bal_acc = balanced_accuracy_score(test_targets, test_preds)
        test_macro_f1 = f1_score(test_targets, test_preds, average="macro", zero_division=0)
        
        pred_counts = np.bincount(test_preds, minlength=cfg.num_classes)
        pred_dist_str = "\n".join([f"    {cfg.class_names[i]}: {count}" for i, count in enumerate(pred_counts)])
        
        true_class_idx = int(test_targets[0]) if len(test_targets) > 0 else 0
        true_class_name = cfg.class_names[true_class_idx]
        recall_on_true = recall_score(test_targets, test_preds, labels=[true_class_idx], average="macro", zero_division=0)
        
        msg = (
            f"Fold {test_speaker} | Epoch {epoch:02d}\n"
            f"  Train:\n    Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Bal Acc: {train_bal_acc:.4f} | Macro F1: {train_macro_f1:.4f}\n"
            f"  Test Monitor:\n    Acc: {test_acc:.4f} | Bal Acc: {test_bal_acc:.4f} | Macro F1: {test_macro_f1:.4f}\n"
            f"  Predicted class counts:\n{pred_dist_str}\n"
            f"  Recall on true class ({true_class_name}): {recall_on_true:.4f}\n"
            f"  " + "-"*30
        )
        print(msg)
        logger.info(msg)
        
    model.load_state_dict(torch.load(fold_model_path, map_location=cfg.device))
    return model, test_preds, test_targets, test_probs


# =========================
# 5. Evaluation
# =========================
def main():
    print(f"Using device: {CFG.device}")
    logger = setup_logger(CFG)
    df = pd.read_csv(CFG.csv_path, keep_default_na=False)
    
    unique_speakers = sorted(df["speaker_id"].unique())
    all_preds, all_targets, all_probs, all_speakers = [], [], [], []
    
    for fold_idx, test_speaker in enumerate(unique_speakers, 1):
        print(f"\nStarting Fold {fold_idx}/{len(unique_speakers)}: Test Speaker = {test_speaker}")
        train_df = df[df["speaker_id"] != test_speaker].copy()
        test_df = df[df["speaker_id"] == test_speaker].copy()
        
        train_df = filter_readable_audio(train_df, CFG, logger)
        test_df = filter_readable_audio(test_df, CFG, logger)
        
        train_ds = DysarthriaDataset(train_df, CFG, train_mode=True)
        test_ds = DysarthriaDataset(test_df, CFG, train_mode=False)

        train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=True)
        
        _, test_preds, test_targets, test_probs = train_fold(fold_idx, test_speaker, train_loader, test_loader, logger, CFG)
        
        all_preds.extend(test_preds)
        all_targets.extend(test_targets)
        all_probs.extend(test_probs)
        all_speakers.extend([test_speaker] * len(test_targets))
        
    final_y_true = np.array(all_targets)
    final_y_pred = np.array(all_preds)
    
    acc = accuracy_score(final_y_true, final_y_pred)
    bal_acc = balanced_accuracy_score(final_y_true, final_y_pred)
    macro_f1 = f1_score(final_y_true, final_y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(final_y_true, final_y_pred, average="weighted", zero_division=0)
    
    report = classification_report(final_y_true, final_y_pred, target_names=CFG.class_names, digits=4, zero_division=0)
    cm = confusion_matrix(final_y_true, final_y_pred)
    
    recall_per_class = recall_score(final_y_true, final_y_pred, average=None, zero_division=0)
    recall_msg = "--- Per-Class Recall ---\n"
    for i, class_name in enumerate(CFG.class_names):
        recall_msg += f"{class_name}: {recall_per_class[i]:.4f}\n"

    final_msg = (
        f"\n--- Final Aggregated Results ---\n"
        f"Accuracy:          {acc:.4f}\n"
        f"Balanced Accuracy: {bal_acc:.4f}\n"
        f"Macro F1:          {macro_f1:.4f}\n"
        f"Weighted F1:       {weighted_f1:.4f}\n"
        f"\n--- Classification Report ---\n{report}\n"
        f"--- Confusion Matrix ---\n{cm}\n"
        f"\n{recall_msg}"
    )
    print(final_msg)
    logger.info(final_msg)
    
    results_df = pd.DataFrame({
        "speaker_id": all_speakers,
        "true_label": final_y_true,
        "pred_label": final_y_pred
    })
    
    all_probs_arr = np.array(all_probs)
    for i, c in enumerate(CFG.class_names):
        results_df[f"prob_{c}"] = all_probs_arr[:, i]
        
    out_csv = os.path.join(CFG.output_dir, CFG.results_csv_name)
    results_df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")

if __name__ == "__main__":
    main()
    