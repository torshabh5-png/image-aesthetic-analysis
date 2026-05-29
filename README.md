# image-aesthetic-analysis
Deep learning pipeline for rating image aesthetic quality using MobileNetV3 fine-tuned on the AVA dataset
# 🖼 Image Aesthetic Analysis

A deep learning pipeline that rates images based on aesthetic quality using **MobileNetV3** fine-tuned on the **AVA (Aesthetic Visual Analysis)** dataset.

## Pipeline
1. Data preprocessing & normalization
2. Exploratory Data Analysis (EDA)
3. MobileNetV3 fine-tuning (regression / classification)
4. Model evaluation — MAE, RMSE, Pearson r
5. CLI inference
6. Web UI via FastAPI

## Tech Stack
- PyTorch · TorchVision · MobileNetV3
- FastAPI · Uvicorn
- Pandas · NumPy · Scikit-learn
- Matplotlib · Seaborn

## Run Locally
```bash
pip install -r requirements.txt
python train.py --epochs 15
uvicorn app:app --reload --port 8000
```

## Demo
Upload any image and get an aesthetic score (1–10) with confidence, dimension breakdown, and qualitative feedback.
