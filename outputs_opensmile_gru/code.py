import os
import time
import random
import logging
import pickle
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import opensmile

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score
)

# ======================================================================================
# 1. Configuration
# ======================================================================================
@dataclass
class Config:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Data & Outputs ---
    csv_path: str = os.path.join("Database", "master_severity_final.csv")
    base_data_dir: str = "Database"
    features_cache_path: str = os.path.join("Database", "opensmile_lld_features.pkl")
    output_dir: str = "outputs_opensmile_gru"
    log_file_name: str = "opensmile_gru.log"
    results_csv_name: str = "results_opensmile_gru.csv"
    
    num_classes: int = 3
    class_names: Tuple[str, str, str] = ("Healthy", "Mild", "Severe")

    # --- Training ---
    batch_size: int = 32
    num_workers: int = 0  # Features are pre-loaded in memory, no need for heavy multiprocessing
    epochs: int = 25
    learning_rate: float = 7e-4
    weight_decay: float = 5e-4
    early_stopping_patience: int = 6
    max_seq_len: int = 300


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
    logger = logging.getLogger("opensmile_gru_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    os.makedirs(cfg.output_dir, exist_ok=True)
    log_path = os.path.join(cfg.output_dir, cfg.log_file_name)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    return logger


# ======================================================================================
# 2. Feature Extraction & Caching
# ======================================================================================
def extract_or_load_features(df: pd.DataFrame, cfg: Config, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extracts openSMILE features or loads from cache, robustly handling missing files."""
    
    feature_cache = {}
    cache_loaded = False
    
    # Try to load existing cache
    if os.path.exists(cfg.features_cache_path):
        try:
            logger.info("Loading cached openSMILE features from disk...")
            print("Loading cached openSMILE features from disk...")
            with open(cfg.features_cache_path, "rb") as f:
                feature_cache = pickle.load(f)
            cache_loaded = True
        except Exception as e:
            logger.warning(f"Cache file corrupted or failed to load: {e}. Rebuilding...")
            print("Delete bad cache and rebuild automatically.")
            os.remove(cfg.features_cache_path)
            feature_cache = {}
            cache_loaded = False

    # Check what is missing
    missing_paths = []
    for idx, row in df.iterrows():
        rel_path = os.path.normpath(str(row["file_path"]))
        if rel_path not in feature_cache:
            missing_paths.append((rel_path, row))

    if missing_paths:
        if cache_loaded:
            logger.info(f"Found {len(missing_paths)} missing features. Extracting and appending to cache...")
            print(f"Found {len(missing_paths)} missing features. Extracting and appending to cache...")
        else:
            logger.info("Extracting openSMILE features. This will take a while on the first run...")
            print("Extracting openSMILE features. This will take a while on the first run...")
        
        start_time = time.time()
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
        
        missing_or_bad = 0
        
        for rel_path, row in tqdm(missing_paths, desc="Extracting"):
            full_path = os.path.join(cfg.base_data_dir, rel_path)
            
            if not os.path.isfile(full_path):
                missing_or_bad += 1
                continue
                
            try:
                # Returns a pandas dataframe
                feats_df = smile.process_file(full_path)
                feats = feats_df.to_numpy()
                
                # Robust NaN handling
                if np.isnan(feats).any() or np.isinf(feats).any():
                    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
                    
                # Handle variable length sequences
                T_frames, F_dim = feats.shape
                if T_frames > cfg.max_seq_len:
                    feats = feats[:cfg.max_seq_len, :]
                elif T_frames < cfg.max_seq_len:
                    pad_len = cfg.max_seq_len - T_frames
                    feats = np.pad(feats, ((0, pad_len), (0, 0)), mode='constant')

                feature_cache[rel_path] = feats
            except Exception as e:
                missing_or_bad += 1
                logger.warning(f"Failed to extract features for {full_path}: {e}")
        
        logger.info(f"Finished extraction. Failed/Missing files skipped: {missing_or_bad}")
        
        with open(cfg.features_cache_path, "wb") as f:
            pickle.dump(feature_cache, f)
            
        extraction_time_mins = (time.time() - start_time) / 60.0
        cache_size_mb = os.path.getsize(cfg.features_cache_path) / (1024 * 1024)
        
        print("Feature cache saved successfully.")
        print(f"Total cached files: {len(feature_cache)}")
        print(f"Feature extraction completed in {extraction_time_mins:.2f} minutes")
        print(f"Cache size: {cache_size_mb:.2f} MB")
            
    # Prepare synchronized dataset arrays
    X, y, speakers, file_paths = [], [], [], []
    missing_count = 0
    
    for idx, row in df.iterrows():
        rel_path = os.path.normpath(str(row["file_path"]))
        if rel_path in feature_cache:
            X.append(feature_cache[rel_path])
            y.append(int(row["severity_label"]))
            speakers.append(str(row["speaker_id"]))
            file_paths.append(rel_path)
        else:
            missing_count += 1
            
    if missing_count > 0:
        print(f"Skipped {missing_count} rows due to missing features/audio.")
        
    return np.array(X), np.array(y), np.array(speakers), np.array(file_paths)


# ======================================================================================
# 3. Dataset & Model
# ======================================================================================
class FeaturesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self) -> int:
        return len(self.X)
        
    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, time, hidden_size]
        scores = self.attention(x).squeeze(-1)
        alpha = torch.softmax(scores, dim=1).unsqueeze(1)
        context = torch.bmm(alpha, x).squeeze(1)
        return context


class SmileGRU(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 64)
        self.gru = nn.GRU(
            input_size=64, 
            hidden_size=64, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        self.attention = AttentionPooling(128)  # 64 * 2 for BiGRU
        self.fc = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.25)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.proj(x))
        x, _ = self.gru(x)
        x = self.attention(x)
        x = torch.relu(self.fc(x))
        x = self.dropout(x)
        return self.out(x)


# ======================================================================================
# 4. Training Utilities
# ======================================================================================
def compute_class_weights(y_train: np.ndarray, num_classes: int, device: str) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0  # Prevent division by zero
    total = counts.sum()
    weights = total / (num_classes * counts)
    # Dampen extreme weights using square root
    weights = np.sqrt(weights)
    weights = np.clip(weights, 0.7, 1.8)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: str
) -> Tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(logits.detach(), dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(y_batch.cpu())
        
    epoch_loss = running_loss / len(loader.dataset)
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    
    return epoch_loss, epoch_acc, epoch_macro_f1


@torch.no_grad()
def evaluate_test_set(
    model: nn.Module, loader: DataLoader, device: str, num_classes: int
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probs = torch.softmax(logits, dim=1).cpu()
        preds = torch.argmax(probs, dim=1)
        
        all_probs.append(probs.numpy())
        all_preds.append(preds.numpy())
        all_targets.append(y_batch.numpy())
        
    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    acc = accuracy_score(all_targets, all_preds)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    
    return acc, bal_acc, macro_f1, all_probs, all_preds, all_targets


# ======================================================================================
# 5. Main LOSO Pipeline
# ======================================================================================
def main():
    start_time = time.time()
    set_seed(CFG.seed)
    logger = setup_logger(CFG)
    logger.info("Starting openSMILE + GRU LOSO Pipeline...")
    logger.info(f"Using device: {CFG.device}")

    try:
        df = pd.read_csv(CFG.csv_path, keep_default_na=False)
    except FileNotFoundError:
        print(f"CSV file not found at {CFG.csv_path}. Cannot proceed.")
        return

    # 1. Prepare Features
    X, y, speakers, file_paths = extract_or_load_features(df, CFG, logger)
    unique_speakers = sorted(np.unique(speakers))
    input_dim = X.shape[2]
    
    logger.info(f"Feature Dimension: {input_dim}")
    logger.info(f"Sequence Length: {X.shape[1]}")
    logger.info(f"Found {len(unique_speakers)} unique speakers for validation.")

    all_test_targets = []
    all_test_preds = []
    all_test_probs = []
    all_test_speakers = []
    fold_metrics = []

    # 2. Leave-One-Speaker-Out Loop
    for fold_idx, test_speaker in enumerate(unique_speakers, 1):
        fold_header = f"\n{'='*60}\nStarting Fold {fold_idx}/{len(unique_speakers)}: Test Speaker = {test_speaker}\n{'='*60}"
        print(fold_header)
        logger.info(fold_header)
        
        # Split data
        train_mask = (speakers != test_speaker)
        test_mask = (speakers == test_speaker)
        
        X_train_raw, y_train = X[train_mask], y[train_mask]
        X_test_raw, y_test = X[test_mask], y[test_mask]
        test_file_paths = file_paths[test_mask]
        
        if len(y_test) == 0:
            logger.warning(f"No test samples found for speaker {test_speaker}. Skipping fold.")
            continue
            
        # Standardization (Fit on train, transform train & test)
        scaler = StandardScaler()
        N_train, T_len, F_dim = X_train_raw.shape
        X_train = scaler.fit_transform(X_train_raw.reshape(-1, F_dim)).reshape(N_train, T_len, F_dim)
        
        N_test, _, _ = X_test_raw.shape
        X_test = scaler.transform(X_test_raw.reshape(-1, F_dim)).reshape(N_test, T_len, F_dim)
        
        train_ds = FeaturesDataset(X_train, y_train)
        test_ds = FeaturesDataset(X_test, y_test)
        
        train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)
        test_loader = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)
        
        # Model & Training Setup
        model = SmileGRU(input_dim=input_dim, num_classes=CFG.num_classes).to(CFG.device)
        class_weights = compute_class_weights(y_train, CFG.num_classes, CFG.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
        
        best_train_macro_f1 = -1.0
        patience_counter = 0
        best_model_state = None
        
        # Train loop
        for epoch in range(1, CFG.epochs + 1):
            train_loss, train_acc, train_macro_f1 = train_one_epoch(model, train_loader, criterion, optimizer, CFG.device)
            test_acc, test_bal_acc, test_macro_f1, probs, preds, targets = evaluate_test_set(model, test_loader, CFG.device, CFG.num_classes)
            
            # Monitoring specific tests stats
            pred_counts = np.bincount(preds, minlength=CFG.num_classes)
            pred_dist_str = "\n".join([f"    {CFG.class_names[i]}: {count}" for i, count in enumerate(pred_counts)])
            
            true_class_idx = int(y_test[0])
            true_class_name = CFG.class_names[true_class_idx]
            recall_on_true = recall_score(targets, preds, labels=[true_class_idx], average="macro", zero_division=0)
            
            epoch_msg = (
                f"Fold {fold_idx} | Epoch {epoch:02d}\n"
                f"  Train Loss: {train_loss:.4f}\n"
                f"  Train Accuracy: {train_acc:.4f}\n"
                f"  Train Macro F1: {train_macro_f1:.4f}\n\n"
                f"  Test Accuracy: {test_acc:.4f}\n"
                f"  Test Balanced Acc: {test_bal_acc:.4f}\n"
                f"  Test Macro F1: {test_macro_f1:.4f}\n\n"
                f"  Prediction counts:\n{pred_dist_str}\n\n"
                f"  True Class: {true_class_name}\n"
                f"  Recall on True Class: {recall_on_true:.4f}\n"
                f"  " + "-"*30
            )
            print(epoch_msg)
            logger.info(epoch_msg)
            
            # Early Stopping Check (on Train Macro F1)
            if train_macro_f1 > best_train_macro_f1:
                best_train_macro_f1 = train_macro_f1
                patience_counter = 0
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= CFG.early_stopping_patience:
                    stop_msg = f"-> Early stopping triggered at epoch {epoch} (Best Train Macro F1: {best_train_macro_f1:.4f})"
                    print(stop_msg)
                    logger.info(stop_msg)
                    break
                    
        # 3. Final inference for the fold using best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        final_acc, final_bal_acc, final_macro_f1, final_probs, final_preds, final_targets = evaluate_test_set(
            model, test_loader, CFG.device, CFG.num_classes
        )
        
        all_test_targets.append(final_targets)
        all_test_preds.append(final_preds)
        all_test_probs.append(final_probs)
        all_test_speakers.extend([test_speaker] * len(final_targets))
        
        fold_res = {
            "fold": fold_idx,
            "test_speaker": test_speaker,
            "accuracy": final_acc,
            "balanced_accuracy": final_bal_acc,
            "macro_f1": final_macro_f1
        }
        fold_metrics.append(fold_res)
        logger.info(f"Fold {fold_idx} Best Model Test Result -> Acc: {final_acc:.4f}, Bal Acc: {final_bal_acc:.4f}, Macro F1: {final_macro_f1:.4f}")

    # ======================================================================================
    # 6. Global Evaluation & Results saving
    # ======================================================================================
    logger.info("="*60)
    logger.info("All folds complete. Aggregating results...")
    print("\n" + "="*60)
    print("All folds complete. Aggregating results...")
    
    final_y_true = np.concatenate(all_test_targets)
    final_y_pred = np.concatenate(all_test_preds)
    final_y_probs = np.concatenate(all_test_probs)
    
    global_acc = accuracy_score(final_y_true, final_y_pred)
    global_bal_acc = balanced_accuracy_score(final_y_true, final_y_pred)
    global_macro_f1 = f1_score(final_y_true, final_y_pred, average="macro", zero_division=0)
    global_weighted_f1 = f1_score(final_y_true, final_y_pred, average="weighted", zero_division=0)
    
    report = classification_report(
        final_y_true, final_y_pred, 
        target_names=CFG.class_names, 
        digits=4, 
        zero_division=0
    )
    cm = confusion_matrix(final_y_true, final_y_pred)
    
    metrics_msg = (
        f"\n--- Final Aggregated Results ---\n"
        f"Accuracy:          {global_acc:.4f}\n"
        f"Balanced Accuracy: {global_bal_acc:.4f}\n"
        f"Macro F1:          {global_macro_f1:.4f}\n"
        f"Weighted F1:       {global_weighted_f1:.4f}\n"
        f"\n--- Classification Report ---\n{report}\n"
        f"--- Confusion Matrix ---\n{cm}\n"
    )
    print(metrics_msg)
    logger.info(metrics_msg)
    
    # Print Per-class recall specifically
    recall_per_class = recall_score(final_y_true, final_y_pred, average=None, zero_division=0)
    recall_msg = "--- Per-Class Recall ---\n"
    for i, class_name in enumerate(CFG.class_names):
        recall_msg += f"{class_name}: {recall_per_class[i]:.4f}\n"
    print(recall_msg)
    logger.info(recall_msg)
    
    # Save predictions to CSV
    results_df = pd.DataFrame({
        "speaker_id": all_test_speakers,
        "true_label": final_y_true,
        "pred_label": final_y_pred,
    })
    
    for i, class_name in enumerate(CFG.class_names):
        results_df[f"prob_{class_name}"] = final_y_probs[:, i]
        
    csv_out_path = os.path.join(CFG.output_dir, CFG.results_csv_name)
    results_df.to_csv(csv_out_path, index=False)
    
    # Also save fold summary
    fold_summary_df = pd.DataFrame(fold_metrics)
    fold_summary_path = os.path.join(CFG.output_dir, "fold_summary.csv")
    fold_summary_df.to_csv(fold_summary_path, index=False)
    
    total_time = time.time() - start_time
    finish_msg = f"Saved predictions to {csv_out_path}. Total execution time: {total_time/60:.2f} minutes."
    print(finish_msg)
    logger.info(finish_msg)


if __name__ == "__main__":
    main()