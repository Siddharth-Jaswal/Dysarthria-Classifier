import pandas as pd

df = pd.read_csv("Database/master.csv")

severity_map = {
    # Healthy
    "FC01": 0,
    "FC02": 0,
    "FC03": 0,
    "MC01": 0,
    "MC02": 0,
    "MC03": 0,
    "MC04": 0,

    # Mild
    "F03": 1,
    "F04": 1,
    "M03": 1,
    "M05": 1,

    # Severe
    "F01": 2,
    "M01": 2,
    "M02": 2,
    "M04": 2
}

name_map = {
    0: "Healthy",
    1: "Mild",
    2: "Severe"
}

df["severity_label"] = df["speaker_id"].map(severity_map)
df["severity"] = df["severity_label"].map(name_map)

df.to_csv("Database/master_severity_final.csv", index=False)

print("Saved: Database/master_severity_final.csv")
print()
print(df.groupby("severity")["speaker_id"].nunique())
print()
print(df["severity"].value_counts())