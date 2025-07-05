
 # **FACECOM - Robust Face Analysis under Challenging Conditions**

This repository contains solutions for the FACECOM Challenge, addressing two computer vision tasks under visually degraded conditions:

- **Task A: Gender Classification**
- **Task B: Face Recognition**


---

## Task A – Gender Classification

### Objective

Predict gender (Male or Female) from degraded face images using a convolutional neural network.

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

### Training the Model

```bash
python train_task_a.py
````

This will:

* Load and preprocess images from `datasets/Task_A/train` and `datasets/Task_A/val`
* Train a CNN-based classifier
* Save the best model to `models/gender_classifier.keras`
* Plot training curves in `results/`

### Evaluating the Model

```bash
python evaluate.py --task a
```

This will output accuracy, precision, recall, F1-score, and ROC AUC for the gender classifier.

---

## Task B – Face Recognition (Siamese Network)

### Objective

Assign each test face to a known individual using a Siamese network trained on contrastive loss.

### Evaluation Metrics

* Top-1 Accuracy
* Macro-averaged F1 Score
* Top-3 Accuracy
* ROC AUC

### Training the Model

```bash
python train_task_b.py
```

This will:

* Load pairwise face data from `datasets/Task_B/train` and `datasets/Task_B/val`
* Train a Siamese model using contrastive loss in 3 phases:

  1. Frozen base layers
  2. Fine-tune last 50% of base
  3. Fine-tune entire base
* Save best model to `models/face_recognizer.keras`
* Plot training progress in `results/`

### Evaluating the Model

```bash
python evaluate.py --task b
```

This evaluates:

* Top-1 accuracy
* Top-3 accuracy
* F1-score across identities
* ROC AUC for verification thresholding

---

## Requirements

* Python 3.10+
* TensorFlow 2.12+
* NumPy, Matplotlib, scikit-learn
* tqdm

Install dependencies via:

```bash
pip install -r requirements.txt
```

---

## Sample Results

### Task A – Gender Classification:

* Accuracy: **90.52%**
* ROC AUC: **0.9517**

### Task B – Face Recognition:

* Accuracy: **25.02%**
* F1 Score: **24.84%**
* Top-3 Accuracy: **49.49%**
* ROC AUC: **0.7963**

---

## Authors

* Priyanshu Biswas
* Rajdeep Chatterjee

---

## 📄 License

This project is part of a hackathon and intended for educational purposes.

---
