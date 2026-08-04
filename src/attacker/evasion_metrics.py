# evasion_metrics.py
import torch

def compute_evasion_rate(model, X_adv, y_true):
    with torch.no_grad():
        preds = model(X_adv).argmax(dim=1)
    evasion_rate = (preds != y_true).float().mean().item()
    return evasion_rate, preds

def compute_confidence_drop(model, X_orig, X_adv, y_true):
    with torch.no_grad():
        probs_orig = torch.softmax(model(X_orig), dim=1)
        probs_adv = torch.softmax(model(X_adv), dim=1)
        conf_orig = probs_orig[range(len(y_true)), y_true]
        conf_adv = probs_adv[range(len(y_true)), y_true]
    return (conf_orig - conf_adv).mean().item()