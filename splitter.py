import pandas as pd

# Fast read
df = pd.read_csv("Database/master.csv", low_memory=False)

# ---------- Add arrayMic rows only if missing ----------
fp = df["file_path"].astype(str)

has_array = fp.str.contains("wav_arrayMic", regex=False).any()

if not has_array:
    head_mask = fp.str.contains("wav_headMic", regex=False)

    df_array = df.loc[head_mask].copy()
    df_array["file_path"] = df_array["file_path"].str.replace(
        "wav_headMic",
        "wav_arrayMic",
        regex=False
    )

    df = pd.concat([df, df_array], ignore_index=True)

# ---------- Severity labels ----------
severity_map = {
    "FC01": 0, "FC02": 0, "FC03": 0,
    "MC01": 0, "MC02": 0, "MC03": 0, "MC04": 0,

    "F04": 1, "M03": 1,

    "F01": 2, "F03": 2, "M01": 2,
    "M02": 2, "M04": 2, "M05": 2
}

df["severity_label"] = df["speaker_id"].map(severity_map).astype("int8")

label_name = {0: "Healthy", 1: "Mild", 2: "Severe"}
df["severity"] = df["severity_label"].map(label_name)

# ---------- Balance Speakers (Crucial for LOSO) ----------
# Prevent speakers with 2000+ samples (like MC01, FC02) from overwhelming the network
# while preserving smaller speakers (like F01 with 328 samples).
MAX_SAMPLES_PER_SPEAKER = 850

# Shuffle first to ensure random sampling when capping
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Cap samples per speaker
df = df.groupby("speaker_id").head(MAX_SAMPLES_PER_SPEAKER).reset_index(drop=True)

# ---------- Print stats ----------
print("--- Samples per Speaker ---")
print(df["speaker_id"].value_counts())
print("\n--- Samples per Class ---")
print(df.groupby(["severity_label", "severity"]).size())

# ---------- Save once only ----------
# The LOSO trainer dynamically handles splits, so no static 'split' column is needed.
df.to_csv("Database/master_severity_split.csv", index=False)

print("\nDone. Saved to Database/master_severity_split.csv")