"""
Blood Cell Classification — Gradio Demo
Runs locally on macOS (CPU or MPS) using a trained EfficientNetB3 model.

Usage:
    pip install gradio timm torch torchvision pillow
    python app.py
"""

import torch
import timm
from torchvision import transforms
from PIL import Image
import gradio as gr

# ── Config ────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Basophil",
    "Eosinophil",
    "Erythroblast",
    "IG",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil",
    "Platelet",
]

CLASS_INFO = {
    "Basophil":     "White blood cell — involved in allergic reactions and inflammation.",
    "Eosinophil":   "White blood cell — responds to parasitic infections and allergies.",
    "Erythroblast": "Immature red blood cell precursor found in bone marrow.",
    "IG":           "Immature granulocyte — early-stage white blood cell.",
    "Lymphocyte":   "White blood cell — key player in immune and viral-infection response.",
    "Monocyte":     "White blood cell — associated with chronic inflammation.",
    "Neutrophil":   "White blood cell — first responder to bacterial infections.",
    "Platelet":     "Clotting cell — critical for stopping bleeding.",
}

IMG_SIZE   = 224
MODEL_PATH = "best_model.pth"
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]

# ── Device: prefer MPS (Apple Silicon), fall back to CPU ─────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Running on: {DEVICE}")

# ── Transform (same as val_transform in the notebook) ────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    # Auto-detect number of classes from the saved classifier layer
    num_classes = state_dict["classifier.weight"].shape[0]
    if num_classes != len(CLASS_NAMES):
        print(f"Warning: checkpoint has {num_classes} classes, CLASS_NAMES has {len(CLASS_NAMES)}. "
              f"Using {num_classes} classes from checkpoint.")

    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, num_classes

try:
    model, NUM_CLASSES = load_model()
    # Trim CLASS_NAMES to match the checkpoint if needed
    CLASS_NAMES = CLASS_NAMES[:NUM_CLASSES]
    MODEL_LOADED = True
except FileNotFoundError:
    model, NUM_CLASSES = None, len(CLASS_NAMES)
    MODEL_LOADED = False
    print(f"Warning: '{MODEL_PATH}' not found. Upload or place it next to app.py.")

# ── Inference ─────────────────────────────────────────────────────────────────
def classify(image: Image.Image):
    if not MODEL_LOADED:
        return {c: 0.0 for c in CLASS_NAMES}, "Model not loaded — place best_model.pth next to app.py."

    tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().tolist()

    top_idx   = int(torch.tensor(probs).argmax())
    top_class = CLASS_NAMES[top_idx]
    top_prob  = probs[top_idx]
    info      = CLASS_INFO[top_class]

    label = f"**{top_class}** ({top_prob:.1%})\n\n{info}"
    scores = {CLASS_NAMES[i]: round(probs[i], 4) for i in range(len(CLASS_NAMES))}
    return scores, label

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Blood Cell Classifier") as demo:
    gr.Markdown(
        "## Blood Cell Classifier\n"
        "Upload a microscopy image to identify the blood cell type "
        "(8 classes, EfficientNetB3)."
    )

    with gr.Row():
        img_input = gr.Image(type="pil", label="Blood Cell Image")

        with gr.Column():
            label_out = gr.Markdown(label="Prediction")
            chart_out = gr.Label(num_top_classes=8, label="Class Probabilities")

    img_input.change(fn=classify, inputs=img_input, outputs=[chart_out, label_out])

    gr.Markdown(
        "---\n"
        "**Classes:** Basophil · Eosinophil · Erythroblast · IG · "
        "Lymphocyte · Monocyte · Neutrophil · Platelet"
    )

if __name__ == "__main__":
    demo.launch()
