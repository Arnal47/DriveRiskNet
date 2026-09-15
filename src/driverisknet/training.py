import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .metrics import classification_metrics


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    losses, predictions, targets = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(loss_fn(logits, y).item() * len(y))
        predictions.extend(logits.argmax(1).cpu().tolist())
        targets.extend(y.cpu().tolist())
    metrics = classification_metrics(targets, predictions)
    metrics["loss"] = float(sum(losses) / len(loader.dataset))
    return metrics


@torch.no_grad()
def predict_loader(model, loader, device="cpu"):
    model.eval(); probabilities, targets = [], []
    for x, y in loader:
        probabilities.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(probabilities), np.concatenate(targets)


def fit(model, train_loader, val_loader, class_weights, config, checkpoint_path, device="cpu"):
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_loss, stale, history, best_state = float("inf"), 0, [], None
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(y)
        train_metrics = evaluate(model, train_loader, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        row = {"epoch": epoch, "train_loss": total / len(train_loader.dataset),
               "train_f1_macro": train_metrics["f1_macro"],
               **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(f"epoch={epoch:02d} train_loss={row['train_loss']:.4f} val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1_macro']:.4f}")
        if val_metrics["loss"] < best_loss - 1e-5:
            best_loss, stale = val_metrics["loss"], 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_path)
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best_state)
    return history
