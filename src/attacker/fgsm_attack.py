# fgsm_attack.py
import torch
import torch.nn.functional as F

def fgsm_attack(model, x, y_true, epsilon):
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    loss = F.cross_entropy(logits, y_true)
    model.zero_grad()
    loss.backward()
    x_adv = x + epsilon * x.grad.sign()
    return x_adv.detach()