"""
app.py
======
FastAPI backend for the Image Aesthetic Analyzer.
Loads the trained MobileNet checkpoint and serves predictions.

Run:
    uvicorn app:app --reload --port 8000
"""

import io
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Import your model ────────────────────────────────────────────────────────
from model import build_model

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

CHECKPOINT   = os.getenv("CHECKPOINT",  "checkpoints/best_model.pth")
TASK         = os.getenv("TASK",        "regression")   # or "classification"
IMG_SIZE     = int(os.getenv("IMG_SIZE", "224"))
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES  = ["Low", "Medium", "High"]

# ──────────────────────────────────────────────────────────────────────────────
# Load model once at startup
# ──────────────────────────────────────────────────────────────────────────────

print(f"[app] Loading checkpoint: {CHECKPOINT}  device={DEVICE}")

model = build_model(task=TASK, pretrained=False, device=DEVICE)

if os.path.exists(CHECKPOINT):
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"[app] Checkpoint loaded (epoch {ckpt.get('epoch','?')}, "
          f"val_loss={ckpt.get('val_loss', 0):.4f})")
else:
    print(f"[app] WARNING: checkpoint not found at {CHECKPOINT}. "
          f"Using random weights (for demo).")

model.eval()

# ──────────────────────────────────────────────────────────────────────────────
# Transform
# ──────────────────────────────────────────────────────────────────────────────

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

infer_transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Image Aesthetic Analyzer", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


# ──────────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    score:       float
    label:       str
    confidence:  float          # 0-100
    breakdown:   dict           # aesthetic sub-dimensions (derived)
    feedback:    str


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def score_to_label(score: float) -> str:
    if score <= 3.5:  return "Low"
    if score <= 6.5:  return "Medium"
    return "High"


def score_to_confidence(score: float) -> float:
    """Map score distance from midpoint (5.5) to a confidence percentage."""
    dist = abs(score - 5.5) / 4.5          # normalise to [0,1]
    return round(50 + dist * 50, 1)         # [50%, 100%]


def derive_breakdown(img: Image.Image, score: float) -> dict:
    """
    Lightweight heuristics to give per-dimension feedback.
    These are approximations — swap in dedicated models if needed.
    """
    arr = np.array(img.resize((128, 128))).astype(float)

    # Brightness  (0-1)
    brightness = arr.mean() / 255

    # Contrast  (std of luminance)
    lum = 0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]
    contrast = float(np.std(lum) / 128)

    # Colorfulness  (Hasler & Süsstrunk, 2003)
    rg = arr[...,0] - arr[...,1]
    yb = 0.5*(arr[...,0]+arr[...,1]) - arr[...,2]
    colorfulness = float((np.std(rg)**2 + np.std(yb)**2)**0.5 / 128)

    # Sharpness  (Laplacian variance)
    from PIL import ImageFilter
    gray  = img.resize((128,128)).convert('L')
    lap   = gray.filter(ImageFilter.FIND_EDGES)
    sharpness = float(np.array(lap).var() / 5000)

    # Clamp all to [0,1]
    def clamp(v): return round(min(max(v, 0.0), 1.0), 3)

    return {
        "brightness":   clamp(brightness),
        "contrast":     clamp(contrast),
        "colorfulness": clamp(colorfulness),
        "sharpness":    clamp(sharpness),
        "overall":      round((score - 1) / 9, 3),   # normalised mean score
    }


def score_to_feedback(score: float) -> str:
    if score >= 7.5:
        return "Exceptional composition. This image scores in the top tier for visual appeal."
    if score >= 6.0:
        return "Good aesthetic quality. Well-composed with strong visual elements."
    if score >= 4.5:
        return "Average aesthetic quality. Some elements work well but there's room to improve."
    if score >= 3.0:
        return "Below average. Consider improving lighting, composition, or color balance."
    return "Low aesthetic score. The image may benefit from significant post-processing."


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Validate
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(400, "Only JPEG / PNG / WEBP images are accepted.")

    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    # Inference
    tensor = infer_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(tensor)

    if TASK == "regression":
        score = float(out.item())
        label = score_to_label(score)
    else:
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
        label = CLASS_NAMES[int(probs.argmax())]
        # Map class index → approximate score for UI
        score = float([2.0, 5.0, 8.0][int(probs.argmax())])

    return PredictionResponse(
        score      = round(score, 2),
        label      = label,
        confidence = score_to_confidence(score),
        breakdown  = derive_breakdown(img, score),
        feedback   = score_to_feedback(score),
    )


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "task": TASK}