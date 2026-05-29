
"""
predict.py
==========
Run inference using a trained AestheticMobileNet model.

Usage
-----
python predict.py --image test.jpg

Optional:
python predict.py --image test.jpg --task classification
python predict.py --image test.jpg --checkpoint checkpoints/best_model.pth
"""

import argparse

import torch
from PIL import Image
from torchvision import transforms

from model import build_model
from dataset import CLASS_NAMES


# ============================================================
# Argument Parser
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Predict image aesthetic score"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to image"
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pth"
    )

    parser.add_argument(
        "--task",
        default="regression",
        choices=["regression", "classification"]
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=224
    )

    return parser.parse_args()


# ============================================================
# Image Transform
# ============================================================

def get_transform(img_size=224):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([

        transforms.Resize(int(img_size * 1.14)),

        transforms.CenterCrop(img_size),

        transforms.ToTensor(),

        transforms.Normalize(mean, std)

    ])

    return transform


# ============================================================
# Load Model
# ============================================================

def load_model(checkpoint_path, task, device):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    saved_args = checkpoint.get("args", {})

    task = saved_args.get("task", task)

    model = build_model(
        task=task,
        pretrained=False,
        device=device
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    print(f"\n[predict] Loaded model -> {checkpoint_path}")

    return model, task


# ============================================================
# Predict Function
# ============================================================

@torch.no_grad()
def predict_image(model, image_path, transform, device, task):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    outputs = model(image_tensor)

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------
    if task == "classification":

        predicted_class = outputs.argmax(1).item()

        class_name = CLASS_NAMES[predicted_class]

        confidence = torch.softmax(outputs, dim=1)[0][predicted_class]

        print("\n" + "=" * 50)

        print(f"Prediction : {class_name}")

        print(f"Confidence : {confidence:.4f}")

        print("=" * 50 + "\n")

    # --------------------------------------------------------
    # Regression
    # --------------------------------------------------------
    else:

        score = outputs.item()

        print("\n" + "=" * 50)

        print(f"Predicted Aesthetic Score : {score:.2f} / 10")

        print("=" * 50 + "\n")


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    transform = get_transform(args.img_size)

    model, task = load_model(
        args.checkpoint,
        args.task,
        device
    )

    predict_image(
        model=model,
        image_path=args.image,
        transform=transform,
        device=device,
        task=task
    )


# ============================================================

if __name__ == "__main__":
    main()

