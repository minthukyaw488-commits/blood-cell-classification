import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'basophil', 'eosinophil', 'erythroblast', 'ig',
    'lymphocyte', 'monocyte', 'neutrophil', 'platelet'
]
CLASS_DESCRIPTIONS = {
    'basophil':     'Rare granulocyte involved in allergic reactions and inflammation.',
    'eosinophil':   'Granulocyte that fights parasites and mediates allergic responses.',
    'erythroblast': 'Immature red blood cell precursor found in bone marrow.',
    'ig':           'Immature granulocyte — a band/immature neutrophil precursor.',
    'lymphocyte':   'Key immune cell; includes T cells, B cells, and NK cells.',
    'monocyte':     'Large white cell that differentiates into macrophages/dendritic cells.',
    'neutrophil':   'Most abundant WBC; first responder to bacterial infection.',
    'platelet':     'Small cell fragment essential for blood clotting.',
}
COLORS = [
    '#8338EC', '#E63946', '#FF6B6B', '#F4A261',
    '#457B9D', '#2A9D8F', '#E9C46A', '#264653'
]
IMG_SIZE = 224
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Transform ────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Model ────────────────────────────────────────────────────────────────────
def load_model(weights_path='best_model.pth'):
    model = timm.create_model('efficientnet_b3', pretrained=False,
                               num_classes=len(CLASS_NAMES))
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    state_dict = (checkpoint['model_state_dict']
                  if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint
                  else checkpoint)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model

model = load_model()

# ── Inference ────────────────────────────────────────────────────────────────
def classify(image: Image.Image):
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    pred_idx   = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    # Gradio label dict — sorted by confidence
    label_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    description = (
        f"**Predicted:** {pred_class.upper()}  \n"
        f"**Confidence:** {confidence:.1%}  \n\n"
        f"{CLASS_DESCRIPTIONS[pred_class]}"
    )
    return label_dict, description

# ── Examples ─────────────────────────────────────────────────────────────────
# Add your own sample images here, e.g. [["samples/neutrophil.jpg"], ...]
EXAMPLES = []

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Blood Cell Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🩸 Blood Cell Classifier
        **8-class blood cell subtype classification using EfficientNetB3**

        Upload a blood cell microscopy image to identify its type.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type='pil', label='Upload Blood Cell Image')
            classify_btn = gr.Button('Classify', variant='primary')

        with gr.Column(scale=1):
            label_output = gr.Label(num_top_classes=8, label='Class Probabilities')
            info_output   = gr.Markdown(label='Result')

    classify_btn.click(
        fn=classify,
        inputs=image_input,
        outputs=[label_output, info_output],
    )
    image_input.change(
        fn=classify,
        inputs=image_input,
        outputs=[label_output, info_output],
    )

    if EXAMPLES:
        gr.Examples(examples=EXAMPLES, inputs=image_input)

    gr.Markdown(
        """
        ---
        **Classes:** basophil · eosinophil · erythroblast · ig · lymphocyte · monocyte · neutrophil · platelet
        **Model:** EfficientNetB3 (pretrained ImageNet → fine-tuned, two-phase training)
        """
    )

if __name__ == '__main__':
    demo.launch()
