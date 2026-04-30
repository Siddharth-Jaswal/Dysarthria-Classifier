# Severity Class Split Creation

The TORGO dataset does **not** provide direct categorical labels such as **Healthy**, **Mild**, or **Severe** for supervised severity classification. It provides speaker metadata, clinical notes, and speech assessment observations. Therefore, a custom speaker-level severity mapping was created for this project using structured analysis of the available notes.

---

# Source Files Used

The severity mapping was derived from:

```text
Database/master.csv
Database/*_data/*/Notes/*.csv
```

Each dysarthric speaker in TORGO contains clinician notes with FDA-style speech motor assessments such as:

* Reflex
* Respiration
* Lips
* Jaw
* Palate
* Laryngeal
* Tongue
* Intelligibility

Many rows were graded using ordinal symbols:

```text
a = normal
b = mild impairment
c = moderate impairment
d = marked impairment
e = severe impairment
```

These values were converted into numeric severity scores.

---

# Clinical Severity Scoring Formula

Each rating symbol was mapped numerically:

| Grade | Meaning  | Score |
| ----: | -------- | ----: |
|     a | Normal   |   0.0 |
|     b | Mild     |   0.5 |
|     c | Moderate |   1.0 |
|     d | Marked   |   1.5 |
|     e | Severe   |   2.0 |

For combined ratings such as:

```text
c/d
d/e
b/c
```

midpoints were used:

| Grade | Score |
| ----: | ----: |
|   c/d |  1.25 |
|   d/e |  1.75 |
|   b/c |  0.75 |

---

# Speaker Severity Score

For each speaker:

```text
Raw Score = Mean(All Clinical Row Scores)
```

Additional weight was added to intelligibility rows:

* Intel. Words
* Intel. Sentences
* Intel. Conversation

Final score:

```text
Adjusted Score = Raw Score + Intelligibility Weight
```

---

# Calculated Speaker Scores

| Speaker | Raw Score | Adjusted Score |
| ------- | --------: | -------------: |
| F03     |     0.213 |          0.113 |
| F04     |     0.213 |          0.113 |
| M03     |     0.170 |          0.170 |
| M05     |     1.453 |          1.453 |
| M02     |     1.760 |          1.760 |
| F01     |     1.760 |          2.110 |
| M01     |     1.760 |          2.110 |
| M04     |     2.017 |          2.167 |

Control speakers were assigned score 0.

---

# Final Thresholding Rule

## Healthy (Class 0)

FC01, FC02, FC03, MC01, MC02, MC03, MC04

## Mild (Class 1)

F03, F04, M03, M05

## Severe (Class 2)

F01, M01, M02, M04

---

# Label Generation

The mapping was applied to:

```text
Database/master.csv
```

Each audio sample inherited the label of its speaker.

Two new columns were created:

```text
severity_label
severity
```

Where:

```text
0 = Healthy
1 = Mild
2 = Severe
```

Saved as:

```text
Database/master_severity_final.csv
```

---

# Final Speaker Distribution

| Class   | Speakers |
| ------- | -------: |
| Healthy |        7 |
| Mild    |        4 |
| Severe  |        4 |

---

# Final Sample Distribution

| Class   | Samples |
| ------- | ------: |
| Healthy |   10413 |
| Mild    |    3131 |
| Severe  |    2440 |

---

# Why This Split Was Used

* separates healthy controls from dysarthric speakers
* preserves medical meaning of speaker groups
* creates balanced speaker counts (7 / 4 / 4)
* improves learnability for machine learning models
* enables robust LOSO experiments

---

# Intended Use

The generated dataset:

```text
master_severity_final.csv
```

was used for all final experiments including:

* openSMILE + MLP
* openSMILE + GRU
* CNN + BiGRU + Attention
* Leave-One-Speaker-Out Cross Validation (LOSO)
