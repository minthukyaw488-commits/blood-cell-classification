"""
Main training script for Blood Cell Classification.

Usage:
    python train.py --data_dir /path/to/bloodcells_dataset
    python train.py --data_dir /path/to/bloodcells_dataset --epochs 30 --batch_size 64
"""
import argparse
import torch
import torch.nn as nn

from src.config import LR, LR_FINETUNE, EPOCHS
from src.data import build_dataloaders
from src.model import build_model, get_finetune_param_groups
from src.train import run_training, evaluate
from src.evaluate import plot_training_history, plot_confusion_matrix, print_metrics, show_gradcam


def main():
    parser = argparse.ArgumentParser(description='Train blood cell classifier')
    parser.add_argument('--data_dir',   required=True,  help='Path to bloodcells_dataset folder')
    parser.add_argument('--epochs',     type=int, default=EPOCHS)
    parser.add_argument('--checkpoint', default='best_model.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_loader, val_loader, test_loader = build_dataloaders(args.data_dir)

    # Phase 1: train head only
    print('\nPhase 1: Training classifier head only...')
    model = build_model(freeze_base=True).to(device)
    history_p1 = run_training(
        model, train_loader, val_loader, device,
        epochs=10, lr=LR, label='P1', patience=5,
        checkpoint_path=args.checkpoint
    )

    # Phase 2: full fine-tuning with discriminative LRs
    print('\nPhase 2: Fine-tuning with discriminative learning rates...')
    for param in model.parameters():
        param.requires_grad = True
    param_groups = get_finetune_param_groups(model, LR_FINETUNE)
    history_p2 = run_training(
        model, train_loader, val_loader, device,
        epochs=args.epochs, lr=LR_FINETUNE, label='P2', patience=7,
        param_groups=param_groups, checkpoint_path=args.checkpoint
    )

    # Load best checkpoint and evaluate
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'\nLoaded best checkpoint (epoch {ckpt["epoch"]}, val_acc={ckpt["val_acc"]:.4f})')

    criterion = nn.CrossEntropyLoss()
    _, _, all_preds, all_labels = evaluate(model, test_loader, criterion, device)

    print_metrics(all_labels, all_preds)
    plot_training_history(history_p1, history_p2)
    plot_confusion_matrix(all_labels, all_preds)
    print('\nSaved: training_history.png, confusion_matrix.png')


if __name__ == '__main__':
    main()
