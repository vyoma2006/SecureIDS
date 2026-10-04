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
        drop = conf_orig - conf_adv
    return drop.mean().item()

# NEW: richer confidence-drop stats, so plateaued evasion rates still show attack strength
def compute_confidence_drop_stats(model, X_orig, X_adv, y_true):
    with torch.no_grad():
        probs_orig = torch.softmax(model(X_orig), dim=1)
        probs_adv = torch.softmax(model(X_adv), dim=1)
        conf_orig = probs_orig[range(len(y_true)), y_true]
        conf_adv = probs_adv[range(len(y_true)), y_true]
        drop = conf_orig - conf_adv
    return {
        "mean_confidence_drop": drop.mean().item(),
        "min_confidence_drop": drop.min().item(),
        "max_confidence_drop": drop.max().item(),
        "mean_confidence_after_attack": conf_adv.mean().item(),
    }