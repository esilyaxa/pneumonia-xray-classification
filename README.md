\# Pneumonia Detection from Chest X-Rays using Deep Learning



\## Overview

This project applies transfer learning with a pretrained ResNet-18 model to classify chest X-ray images as NORMAL or PNEUMONIA. The goal is to demonstrate an end-to-end deep learning workflow with a focus on clinically meaningful evaluation metrics.



\## Dataset

Chest X-ray images were obtained from a publicly available dataset. Images are organized into training and validation sets. The dataset is not included in this repository due to size and licensing considerations.



\## Model

\- Architecture: ResNet-18 (pretrained on ImageNet)

\- Transfer learning with frozen convolutional layers

\- Final fully connected layer retrained for binary classification



\## Training

\- Loss function: Cross-entropy loss

\- Optimizer: Adam

\- Hardware: CPU

\- Training focused on minimizing false negatives for pneumonia detection



\## Evaluation

\- Accuracy: ~81%

\- Recall (Pneumonia): 100%

\- Precision (Pneumonia): 73%



The model achieved perfect recall on the validation set, indicating no missed pneumonia cases, which is critical for medical screening applications.



\## Limitations

\- Small validation set

\- No external test set

\- Not intended for clinical deployment



\## Future Work

\- External validation on multi-institutional data

\- Threshold tuning to balance precision and recall

\- Model explainability (Grad-CAM)



\## Disclaimer

This project is for educational purposes only and is not intended for medical use.



