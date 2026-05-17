import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, List


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = scaler is not None and device.type == 'cuda'
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss    = criterion(outputs, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct  += (preds == labels).sum().item()
        total    += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct  += (preds == labels).sum().item()
            total    += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


def run_training(
    model, train_loader, val_loader, device,
    epochs, lr, label='Phase', patience=5,
    param_groups=None, checkpoint_path='best_model.pth'
) -> Dict:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    params    = param_groups or list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp   = device.type == 'cuda'
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    history          = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc     = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        tr_loss, tr_acc       = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)

        if va_acc > best_val_acc:
            best_val_acc     = va_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_acc': best_val_acc,
                'optimizer_state': optimizer.state_dict(),
            }, checkpoint_path)
        else:
            patience_counter += 1

        print(f'[{label}] Epoch {epoch+1:02d}/{epochs} | '
              f'Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | '
              f'Val Loss: {va_loss:.4f} Acc: {va_acc:.4f}'
              + (f'  [patience {patience_counter}/{patience}]' if patience_counter else ''))

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    print(f'Best Val Accuracy: {best_val_acc:.4f}')
    return history
