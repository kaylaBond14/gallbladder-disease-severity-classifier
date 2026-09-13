# Gallbladder Disease Severity Classifier
 
A deep learning project that classifies gallbladder ultrasound images into four clinical severity tiers — rather than a flat disease label — using transfer learning on a pretrained ResNet-50.
 
**Course project — Biomedical Image Analysis**
 
## Overview
 
Gallbladder disease spans a wide range of clinical urgency, from silent gallstones to gangrenous cholecystitis or malignancy. This project explores whether a convolutional neural network can help flag disease *severity* from an ultrasound image, rather than just detecting presence or absence of disease — a framing intended to better mirror real clinical triage.
 
## Severity Tiers
 
| Tier | Category | Example Findings |
|------|----------|-------------------|
| 1 | Normal | No significant finding |
| 2 | Mild Pathology | Gallstones present, no inflammation |
| 3 | Inflammation | Cholecystitis, wall thickening |
| 4 | Severe | Gangrenous cholecystitis, perforation, carcinoma |
 
## Dataset
 
This project uses the **[Gallblader Diseases Dataset](https://www.kaggle.com/datasets/yasserhessein/gallblader-diseases-dataset)** (UIdataGB), a public dataset of 10,692 ultrasound images from 1,782 patients, specialist-labeled across 9 disease classes. The 9 original classes are collapsed into the 4 severity tiers above for this project (see `src/preprocess.py`).
 
Citation: Turki, A., Obaid, A.M., Bellaaj, H., Ksantini, M., & AlTaee, A. (2024). UIdataGB: Multi-Class ultrasound images dataset for gallbladder disease detection. *Data in Brief*, 54, 110426.
 
> **Note:** The dataset is not included in this repository due to its size. See [Setup](#setup) below to download it.
 
## Method
 
- **Model:** Pretrained ResNet-50 (ImageNet weights), fine-tuned via transfer learning
- **Approach:** Early layers frozen; a new classifier head is trained on the 4 severity tiers
- **Class imbalance handling:** Weighted loss function
- **Evaluation:** Per-class precision/recall/F1, confusion matrix, and severe-tier (Tier 4) sensitivity
- **Stretch goal:** Grad-CAM visualizations for model interpretability
## Repository Structure
 
```
gallbladder-severity-classifier/
├── README.md
├── requirements.txt
├── src/
│   ├── preprocess.py     # Sorts raw 9-class dataset into 4 severity tier folders
│   ├── train.py           # Fine-tunes ResNet-50 on the severity tiers
│   └── evaluate.py        # Generates confusion matrix and classification report
├── notebooks/
│   └── exploration.ipynb  # Sample image visualization and dataset exploration
├── results/
│   ├── confusion_matrix.png
│   └── training_log.txt
└── docs/
    └── proposal.pdf
```
 
## Setup
 
1. **Clone this repository**
```bash
   git clone https://github.com/<your-username>/gallbladder-severity-classifier.git
   cd gallbladder-severity-classifier
```
 
2. **Install dependencies**
```bash
   pip install -r requirements.txt
```
 
3. **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/yasserhessein/gallblader-diseases-dataset) and place the extracted folders under `data_raw/`.
4. **Sort images into severity tiers**
```bash
   python src/preprocess.py
```
 
5. **Train the model**
```bash
   python src/train.py
```
 
6. **Evaluate the trained model**
```bash
   python src/evaluate.py
```
 
> Training is recommended on a GPU (e.g., a free Google Colab T4 runtime). On CPU, training will be significantly slower.
 
## Results
 
*(To be filled in after training — e.g., overall accuracy, per-tier F1 scores, confusion matrix image, and key observations.)*
 
## Limitations
 
- Ultrasound image quality varies by machine and operator, which may limit generalization across sources.
- The dataset's original 9 classes are roughly balanced (~1,200 images each), so the resulting severity tiers may not reflect the natural class imbalance seen in real clinical settings, where severe cases are much rarer.
- This is a research and educational project, not a validated clinical diagnostic tool. No claims of clinical deployment readiness are made.
## Acknowledgments
 
Dataset provided by Turki et al. (2024) via Mendeley Data / Kaggle. Built as a course project for Biomedical Image Analysis.
 