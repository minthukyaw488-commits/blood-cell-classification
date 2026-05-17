import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import cv2
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_auc_score, precision_recall_curve, average_precision_score
)
from src.config import CLASS_NAMES, MEAN, STD, IMG_SIZE, COLORS


def plot_training_history(history_p1, history_p2, save_path='training_history.png'):
    combined    = {k: history_p1[k] + history_p2[k] for k in history_p1}
    phase_split = len(history_p1['train_loss'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Training History — 8-Class Blood Cell Classification', fontsize=14, fontweight='bold')

    for ax, metric, title in [
        (ax1, ('train_loss', 'val_loss'), 'Loss'),
        (ax2, ('train_acc',  'val_acc'),  'Accuracy')
    ]:
        ax.plot(combined[metric[0]], label='Train', color='#2A9D8F', linewidth=2)
        ax.plot(combined[metric[1]], label='Val',   color='#E63946', linewidth=2)
        ax.axvline(x=phase_split, color='gray', linestyle='--', alpha=0.7, label='Fine-tune start')
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(all_labels, all_preds, save_path='confusion_matrix.png'):
    cm = confusion_matrix(all_labels, all_preds)
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]],
        ['d', '.2%'],
        ['Confusion Matrix (Counts)', 'Confusion Matrix (Normalized)']
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=ax, linewidths=0.5, annot_kws={'size': 9})
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_metrics(all_labels, all_preds):
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1  = f1_score(all_labels, all_preds, average='weighted')
    print(f'Test Accuracy : {acc:.4f} ({acc*100:.2f}%)')
    print(f'Weighted F1   : {f1:.4f}')
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))


class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam).squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def show_gradcam(model, dataset, device, n=8, save_path='gradcam.png'):
    import random
    target_layer = model.conv_head
    grad_cam     = GradCAM(model, target_layer)

    fig, axes = plt.subplots(n, 3, figsize=(12, n * 3.5))
    fig.suptitle('Grad-CAM: Model Attention per Class', fontsize=15, fontweight='bold')

    indices = random.sample(range(len(dataset)), n)
    for row, idx in enumerate(indices):
        img_tensor, true_label = dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)
        input_tensor.requires_grad = True

        cam, pred_label = grad_cam.generate(input_tensor)
        img_np  = img_tensor.permute(1, 2, 0).numpy()
        img_np  = np.clip(img_np * np.array(STD) + np.array(MEAN), 0, 1)
        cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        heatmap     = plt.cm.jet(cam_resized)[:, :, :3]
        overlay     = np.clip(0.5 * img_np + 0.5 * heatmap, 0, 1)

        axes[row, 0].imshow(img_np);  axes[row, 0].axis('off')
        axes[row, 0].set_title(f'True: {CLASS_NAMES[true_label]}', fontsize=10)
        axes[row, 1].imshow(heatmap); axes[row, 1].axis('off')
        axes[row, 1].set_title('Grad-CAM Heatmap', fontsize=10)
        correct = 'Correct' if pred_label == true_label else 'Wrong'
        axes[row, 2].imshow(overlay); axes[row, 2].axis('off')
        axes[row, 2].set_title(f'Pred: {CLASS_NAMES[pred_label]} ({correct})', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
