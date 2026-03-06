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

Optimal validation threshold selected: 0.47

📉 Learning Curve Analysis

To monitor training stability and detect potential overfitting, training and validation performance were tracked across epochs.

Epoch	Train Loss	Train Accuracy	Validation Loss	Validation Accuracy
1	     0.0950	       0.9630	          0.1145	      0.9580
2	     0.0960    	   0.9620	          0.1107	      0.9608
3	     0.1042	       0.9546	          0.1053	      0.9656

Observations

Stable training behavior: Training and validation losses remain close throughout training.

No strong signs of overfitting: Validation loss decreases slightly while validation accuracy improves.

Small generalization gap: Training and validation accuracies differ by less than ~1%, indicating good generalization on the validation set.

Potential for further training: Validation metrics continue improving slightly, suggesting that additional epochs may further refine the model.

Overall, the learning curves indicate that the model converges smoothly and maintains consistent performance across training and validation data.


📊 Final Test Results (V2)

Confusion Matrix:

[[129 105]
 [ 7  383]]

Metric	Value
Test Accuracy	82.05%
Pneumonia Recall	98%
Normal Recall	55%
Pneumonia F1	0.87
📈 Interpretation
High Pneumonia Recall (98%)

The model detects nearly all pneumonia cases (only 7 false negatives).
This indicates strong sensitivity — desirable in medical screening.

Low Normal Recall (55%)

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



