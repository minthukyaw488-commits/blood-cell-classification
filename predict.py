"""
Single-image inference script.

Usage:
    python predict.py --image path/to/image.jpg
    python predict.py --image path/to/image.jpg --checkpoint best_model.pth
"""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from src.config import CLASS_NAMES, COLORS, IMG_SIZE, MEAN, STD
from src.model import build_model


def predict_image(image_path, model, device, class_names=CLASS_NAMES):
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    model.eval()
    img    = Image.open(image_path).convert('RGB')
    tensor = val_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs  = model(tensor)
        probs    = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())

    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(
        f'Prediction: {pred_class.upper()}\nConfidence: {confidence:.2%}',
        fontsize=14, fontweight='bold',
        color='#2A9D8F' if confidence > 0.8 else '#E9C46A'
    )

    bar_colors = [COLORS[i] if i == pred_idx else '#D3D3D3' for i in range(len(class_names))]
    bars = ax2.barh(class_names, probs, color=bar_colors, edgecolor='white')
    ax2.set_xlim(0, 1)
    ax2.set_title('Class Probabilities', fontsize=13)
    ax2.set_xlabel('Confidence')
    for bar, prob in zip(bars, probs):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{prob:.2%}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\nPredicted: {pred_class.upper()}  ({confidence*100:.2f}% confidence)')
    for cls, prob in sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True):
        bar = '#' * int(prob * 30)
        marker = ' <- PREDICTED' if cls == pred_class else ''
        print(f'  {cls:15s}: {bar} {prob*100:.2f}%{marker}')

    return pred_class, confidence


def main():
    parser = argparse.ArgumentParser(description='Predict blood cell type from image')
    parser.add_argument('--image',      required=True,        help='Path to input image')
    parser.add_argument('--checkpoint', default='best_model.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = build_model(freeze_base=False).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)

    predict_image(args.image, model, device)


if __name__ == '__main__':
    main()
