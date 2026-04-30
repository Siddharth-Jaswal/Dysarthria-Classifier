import pandas as pd
import numpy as np
import re
from pathlib import Path

ROOT = Path("Database")
MASTER_CSV = ROOT / "master.csv"

GRADE_MAP = {
    "a": 0.0,
    "b": 1.0,
    "c": 2.0,
    "d": 3.0,
    "e": 4.0
}

DOMAIN_WEIGHTS = {
    "intel": 3.0,
    "tongue": 2.5,
    "laryngeal": 2.2,
    "resp": 2.0,
    "palate": 1.8,
    "lips": 1.5,
    "jaw": 1.2,
    "reflex": 1.0
}

POSITIVE_WORDS = {
    "broken sentences": 0.25,
    "fatigued": 0.20,
    "fatigue": 0.20,
    "nasal air": 0.25,
    "grimacing": 0.20,
    "difficult": 0.15,
    "slurred": 0.25,
    "unintelligible": 0.35
}

NEGATIVE_WORDS = {
    "good mood": -0.05,
    "good oral health": -0.05,
    "overarticulates": -0.10
}


def parse_grade(x):
    if pd.isna(x):
        return None

    s = str(x).strip().lower()
    s = s.replace(" ", "")

    if s in GRADE_MAP:
        return GRADE_MAP[s]

    if "/" in s:
        parts = s.split("/")
        vals = []
        for p in parts:
            if p in GRADE_MAP:
                vals.append(GRADE_MAP[p])
        if vals:
            return float(np.mean(vals))

    return None


def normalize_domain(x):
    s = str(x).strip().lower()

    if "intel" in s:
        return "intel"
    if "tongue" in s:
        return "tongue"
    if "laryngeal" in s:
        return "laryngeal"
    if "resp" in s:
        return "resp"
    if "palate" in s:
        return "palate"
    if "lips" in s:
        return "lips"
    if "jaw" in s:
        return "jaw"
    if "reflex" in s:
        return "reflex"

    return "other"


def score_to_label4(score):
    if score < 0.50:
        return "Healthy"
    elif score < 1.50:
        return "Mild"
    elif score < 2.50:
        return "Moderate"
    else:
        return "Severe"


def score_to_label3(score):
    if score < 0.50:
        return "Healthy"
    elif score < 1.50:
        return "Mild"
    else:
        return "Severe"


def label_to_num(label):
    mp = {
        "Healthy": 0,
        "Mild": 1,
        "Severe": 2
    }
    return mp[label]


def confidence(rows_used, domains):
    if rows_used >= 15 and domains >= 5:
        return "High"
    if rows_used >= 8:
        return "Medium"
    return "Low"


def analyze_notes_file(path):
    speaker = path.stem

    try:
        df = pd.read_csv(path, header=0, engine="python")
    except:
        df = pd.read_csv(path, header=None, engine="python")

    text_all = " ".join(df.astype(str).fillna("").values.flatten()).lower()

    if "control" in text_all and "yes" in text_all:
        return {
            "speaker_id": speaker,
            "rows_used": 0,
            "domains_found": 0,
            "raw_score": 0.0,
            "adjusted_score": 0.0,
            "severity_4class": "Healthy",
            "severity_3class": "Healthy",
            "severity_label": 0,
            "confidence": "High",
            "notes_file": str(path)
        }

    rows = []
    domains_seen = set()

    for _, row in df.iterrows():
        vals = list(row.values)

        if len(vals) < 3:
            continue

        col1 = str(vals[0]).strip()
        col2 = str(vals[1]).strip()
        col3 = str(vals[2]).strip()

        g = parse_grade(col3)
        if g is None:
            continue

        domain = normalize_domain(col1)
        wt = DOMAIN_WEIGHTS.get(domain, 1.0)

        rows.append((domain, wt, g))
        domains_seen.add(domain)

    if not rows:
        return {
            "speaker_id": speaker,
            "rows_used": 0,
            "domains_found": 0,
            "raw_score": np.nan,
            "adjusted_score": np.nan,
            "severity_4class": "Unknown",
            "severity_3class": "Unknown",
            "severity_label": -1,
            "confidence": "Low",
            "notes_file": str(path)
        }

    num = 0.0
    den = 0.0

    for domain, wt, g in rows:
        num += wt * g
        den += wt

    raw_score = num / den

    bonus = 0.0

    for k, v in POSITIVE_WORDS.items():
        if k in text_all:
            bonus += v

    for k, v in NEGATIVE_WORDS.items():
        if k in text_all:
            bonus += v

    bonus = max(min(bonus, 0.35), -0.15)

    final_score = raw_score + bonus
    final_score = max(0.0, min(4.0, final_score))

    label4 = score_to_label4(final_score)
    label3 = score_to_label3(final_score)

    return {
        "speaker_id": speaker,
        "rows_used": len(rows),
        "domains_found": len(domains_seen),
        "raw_score": round(raw_score, 3),
        "adjusted_score": round(final_score, 3),
        "severity_4class": label4,
        "severity_3class": label3,
        "severity_label": label_to_num(label3),
        "confidence": confidence(len(rows), len(domains_seen)),
        "notes_file": str(path)
    }


def main():
    files = list(ROOT.rglob("Notes/*.csv"))

    results = []

    for f in files:
        res = analyze_notes_file(f)
        results.append(res)

    sev_df = pd.DataFrame(results)
    sev_df = sev_df.sort_values("adjusted_score", na_position="last")

    sev_df.to_csv("speaker_notes_severity_scores.csv", index=False)

    print("\n===== SPEAKER SEVERITY RANKING =====\n")
    print(sev_df[[
        "speaker_id",
        "adjusted_score",
        "severity_4class",
        "severity_3class",
        "confidence"
    ]])

    if MASTER_CSV.exists():
        master = pd.read_csv(MASTER_CSV)

        merged = master.merge(
            sev_df[
                [
                    "speaker_id",
                    "adjusted_score",
                    "severity_4class",
                    "severity_3class",
                    "severity_label"
                ]
            ],
            on="speaker_id",
            how="left"
        )

        merged.to_csv(ROOT / "master_severity_notes.csv", index=False)
        print("\nSaved Database/master_severity_notes.csv")

    print("\nSaved speaker_notes_severity_scores.csv")


if __name__ == "__main__":
    main()