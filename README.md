🩺 Pneumonia Detection from Chest X-Ray (ResNet-18)

Binary classification of chest X-ray images (Normal vs Pneumonia) using transfer learning with ResNet-18.

📌 Project Motivation

This project explores deep learning for medical image classification and emphasizes model evaluation reliability, not just high accuracy.

🔹 Version 1 (V1)
Setup

Pretrained ResNet-18

Standard preprocessing (Resize 224×224, Normalize)

Small validation set: 16 images (8 per class)

Threshold tuning on this validation set

Observations

Validation recall frequently reached 1.00

F1 score changed drastically with tiny threshold shifts

Model probabilities were highly concentrated:

Normal: ~0.2–0.78

Pneumonia: ~0.78–0.996

Problem

Because the validation set contained only 16 samples:

Threshold tuning was unstable

Performance estimates were statistically unreliable

Small probability shifts drastically changed metrics

Signs of possible overconfidence / overfitting

Conclusion: V1 metrics were not trustworthy due to insufficient validation data.

🔹 Version 2 (V2)

To improve reliability:

Merged train + validation

Created a new validation split

Retrained the model

Re-tuned classification threshold

Evaluated strictly on test set

Optimal validation threshold selected: 0.57

📊 Final Test Results (V2)

Confusion Matrix:

[[114 120]
 [  4 386]]

Metric	Value
Test Accuracy	80.13%
Pneumonia Recall	99%
Normal Recall	49%
Pneumonia F1	0.86
📈 Interpretation
High Pneumonia Recall (99%)

The model detects nearly all pneumonia cases (only 4 false negatives).
This indicates strong sensitivity — desirable in medical screening.

Low Normal Recall (49%)

The model frequently predicts Pneumonia for borderline cases.
This is influenced by:

Class imbalance

Threshold tuning

Model bias toward minimizing false negatives

Validation vs Test Gap

V1 showed high validation performance, but it was misleading due to the extremely small validation set.

V2 provides a more realistic evaluation, highlighting:

Generalization challenges

Effects of data distribution

Trade-off between sensitivity and specificity

🎯 Key Takeaways

This project demonstrates:

Transfer learning with ResNet-18

The danger of small validation sets

Threshold calibration sensitivity

Class imbalance effects

Real-world model evaluation trade-offs

🛠 Tech Stack

Python

PyTorch

torchvision

scikit-learn



