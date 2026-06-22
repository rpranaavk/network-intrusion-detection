# Network Intrusion Detection on CIC-IDS2018

A supervised intrusion-detection pipeline on the CSE-CIC-IDS2018 dataset, comparing a Random Forest baseline against a PyTorch deep neural network. The point of this project is not the headline accuracy. It is the diagnostic work around it: catching a data-leakage bug, auditing which classes actually survived preprocessing, and using train/validation loss to detect overfitting onset.

If you only read one thing, read the [Key Findings](#key-findings). They are the reason this repo exists.

---

## TL;DR

- **Binary task (Random Forest):** essentially perfect on this dataset (1 misclassification in 366,791 test samples). This is treated as a warning sign, not a win. See the leakage discussion below.
- **Multi-class task (PyTorch DNN, 6 classes):** 93% accuracy. Most classes are clean; two attack families (`DoS-SlowHTTPTest` and `FTP-BruteForce`) collapse into each other due to feature overlap, not class imbalance.
- **The interesting part:** a minority attack class (`DDOS-LOIC-UDP`, ~0.05% of the data) was silently destroyed by preprocessing and never made it into training. I caught it with an explicit class-survival audit. Aggressive cleaning quietly turned a 7-class problem into a 6-class one.

---

## Why the accuracy numbers should make you suspicious

CIC-IDS2018 is known to produce inflated metrics. Flow features can correlate with attack labels in ways that leak the answer, and naive pipelines routinely report near-perfect scores that do not survive contact with real traffic. A Random Forest hitting 1 error in 366,791 is exactly the kind of result that should prompt the question "what leaked?" rather than "ship it."

So this project deliberately treats high accuracy as a thing to be explained, not celebrated. Two concrete steps came out of that posture:

1. **Scaling leakage was found and fixed.** The original (AI-assisted) preprocessing fit `StandardScaler` on the full dataset before the train/test split, which leaks test-set statistics into training. Corrected to fit the scaler on the training split only, then transform the test split.
2. **Class survival was audited explicitly** rather than assumed. That is how the dropped `LOIC-UDP` class was found (see below).

That diagnostic skepticism is the actual contribution here.

---

## Dataset

CSE-CIC-IDS2018 (Canadian Institute for Cybersecurity), delivered as three CSV shards combined into one frame.

Preprocessing applied, in order:

- Removed rows where the header was repeated as data (shards 2 and 3 had this).
- Dropped the `Timestamp` column. It encodes when traffic was captured, not what the traffic is, and keeping it invites temporal leakage.
- Replaced sentinel `-1` values and `±inf` with `NaN`, then dropped resulting `NaN` rows.
- Removed zero-variance features (identical value in every row, so zero information).
- `StandardScaler` fit on the training split only.

The `NaN`-dropping step is also what removed the `LOIC-UDP` class. That tradeoff is documented honestly rather than hidden.

---

## Methodology

**Task 1: Binary classification (Benign vs Attack)**
Random Forest classifier on the scaled features. Strong, fast baseline; handles the feature space well.

**Task 2: Multi-class classification (6 attack/benign classes)**
A 3-layer fully-connected PyTorch network with class-weighted loss to handle severe imbalance.

### DNN architecture

| Component | Value |
|---|---|
| Input dimension | 67 (after dropping zero-variance features) |
| Hidden layer 1 | 128, ReLU |
| Regularization | Dropout(p=0.3) after layer 1 |
| Hidden layer 2 | 64, ReLU |
| Output | 6 logits (CrossEntropyLoss applies softmax internally) |
| Optimizer | Adam, lr=0.001 |
| Loss | CrossEntropyLoss with `balanced` class weights |
| Epochs / batch | 50 / 512 |
| Split | 80/20 stratified, seed 42 (all RNGs seeded) |

Class weights ranged from 0.27 (Benign) to ~11.96 (`DoS-Hulk`), inversely scaling each class by its frequency so the loss does not let the model ignore rare classes.

---

## Results

### Binary (Random Forest)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Benign | 1.00 | 1.00 | 1.00 | 224,766 |
| Attack | 1.00 | 1.00 | 1.00 | 142,025 |
| **Accuracy** | | | **1.00** | 366,791 |

1 misclassification total. Read the suspicion section above before reading this as success.

![Binary confusion matrix](outputs/confusion_matrix_binary_rf.png)

### Multi-class (DNN), 6 classes

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Benign | 1.00 | 1.00 | 1.00 | 224,766 |
| DDOS-HOIC | 1.00 | 1.00 | 1.00 | 32,750 |
| DoS-Hulk | 1.00 | 1.00 | 1.00 | 5,109 |
| DoS-SlowHTTPTest | 0.53 | 0.92 | 0.67 | 27,978 |
| FTP-BruteForce | 0.87 | 0.41 | 0.56 | 38,671 |
| SSH-BruteForce | 1.00 | 1.00 | 1.00 | 37,517 |
| **Accuracy** | | | **0.93** | 366,791 |

![Multi-class confusion matrix](outputs/confusion_matrix_multiclass_dnn.png)

The 93% is dragged down almost entirely by one confusion: ~22,655 `FTP-BruteForce` flows were predicted as `DoS-SlowHTTPTest`. The confusion matrix shows these two collapse into each other. This is a **feature-overlap** problem (the two attack families produce similar flow statistics), not an imbalance problem, since `FTP-BruteForce` is not a rare class. Class weighting cannot fix what the features cannot separate.

---

## Key Findings

These are the parts worth your attention.

### 1. A minority class was silently deleted by preprocessing

`DDOS-LOIC-UDP` made up roughly 0.05% of the raw data (~1,500 samples). The `-1`-to-`NaN` replacement and subsequent `NaN`-row drop removed every one of its samples. The class never entered the encoder, the training set, or the test set. The model could not have learned it; it simply vanished.

A dedicated audit cell confirms this: the original dataset had 7 attack/benign classes, and only 6 survived preprocessing. The lesson generalizes well beyond this dataset: aggressive cleaning can erase exactly the rare events an intrusion detector most needs to catch, and it can do so without any error or warning. You only know if you check.

### 2. Data leakage from pre-split scaling was caught and corrected

Fitting the scaler before the split leaks test distribution into training and inflates results. Fixed by fitting on train only. This is the kind of bug that produces impressive-looking numbers and silently invalidates them.

### 3. Overfitting onset was detected via validation loss

Tracking train and validation loss per epoch showed validation loss bottoming out around epoch 20-25, then rising while training loss stayed flat near 0.175. That divergence is the model starting to memorize. The run used 50 epochs; the honest best checkpoint is around epoch 20, and early stopping would have captured it.

![Training vs validation loss](outputs/training_validation_loss.png)

---

## Limitations and next steps

- **No early stopping.** Loss curves show it was needed; adding it is the obvious first improvement.
- **The dropped class is a real gap.** A better pipeline would either impute rather than drop, or at minimum surface the class loss as an explicit failure rather than a silent one.
- **The binary result is not trustworthy as-is.** Validating against a leakage-controlled split, or a different capture entirely, would tell you whether the model learned attacks or learned the dataset.
- **Feature-overlap between brute-force and slow-DoS families** would benefit from flow features with finer temporal resolution, or a model that sees sequences rather than aggregated flows.

---

## Running it

```bash
git clone https://github.com/rpranaavk/network-intrusion-detection
cd network-intrusion-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook network_intrusion_detection.ipynb
```

The notebook expects the CIC-IDS2018 CSV shards. Update the `folder_path` cell to point at your local copy. The dataset is not redistributed here; get it from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2018.html).

## Repo layout

```
.
├── network_intrusion_detection.ipynb   # full pipeline, both tasks
├── network_intrusion_writeup.pdf        # written analysis
├── outputs/                             # generated figures
│   ├── confusion_matrix_binary_rf.png
│   ├── confusion_matrix_multiclass_dnn.png
│   └── training_validation_loss.png
├── requirements.txt
└── README.md
```
