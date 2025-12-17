import numpy as np
import torch.nn.functional as F
import torch
import logging
import torch.nn as nn

import matplotlib.pyplot as plt

# --- low activity defense utilities ---
def compute_neuron_activity(model, dataloader, device):
    activations = {}

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            # Average activation across batch and spatial dimensions
            mean_act = output.detach().mean(dim=(0, 2, 3))
            layer_name = f"{module.__class__.__name__}_{id(module)}"
            if layer_name not in activations:
                activations[layer_name] = mean_act.clone()
            else:
                activations[layer_name] += mean_act

    # Register hooks on all Conv2d layers
    hooks = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(hook_fn))

    # Ensure model is on the correct device
    model.to(device)
    model.eval()

    num_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            # Handle dataloader yielding (images, labels) or just images
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            images = images.to(device)

            _ = model(images)
            num_batches += 1

    # Normalize activations by number of batches processed
    for k in activations:
        activations[k] /= num_batches

    # Remove hooks
    for h in hooks:
        h.remove()

    return activations

def low_activity_prune(model, activations, prune_ratio=0.05):
    """
    Prune lowest-activity filters from each convolutional layer.
    """
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            layer_name = str(id(layer))
            if layer_name not in activations:
                continue
            act = activations[layer_name]
            num_prune = int(prune_ratio * layer.out_channels)
            if num_prune <= 0:
                continue
            prune_idx = torch.argsort(act)[:num_prune]

            with torch.no_grad():
                layer.weight[prune_idx] = 0
                if layer.bias is not None:
                    layer.bias[prune_idx] = 0
    return model

def fine_tune(model, data_loader, device, bins, binwidth, angle,
              epochs=5, lr=1e-4, accumulation_steps=2, use_amp=True):
    """
    Fine-tune a gaze estimation model on clean data.
    Includes classification + regression losses.
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()

    idx_tensor = torch.arange(bins, device=device, dtype=torch.float32)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        sum_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(data_loader):
            images = batch[0].to(device, non_blocking=True)
            labels_gaze = batch[1].to(device, non_blocking=True)
            regression_labels_gaze = batch[2].to(device, non_blocking=True)

            # classification labels
            label_pitch = labels_gaze[:, 0]
            label_yaw   = labels_gaze[:, 1]

            # regression labels
            label_pitch_reg = regression_labels_gaze[:, 0]
            label_yaw_reg   = regression_labels_gaze[:, 1]

            with torch.cuda.amp.autocast(enabled=use_amp):
                pitch_logits, yaw_logits = model(images)

                # classification loss
                loss_pitch_cls = cls_criterion(pitch_logits, label_pitch)
                loss_yaw_cls   = cls_criterion(yaw_logits, label_yaw)

                # convert logits to probabilities
                pitch_probs = torch.softmax(pitch_logits, dim=1)
                yaw_probs   = torch.softmax(yaw_logits, dim=1)

                # expected angle from bins
                pitch_pred = torch.sum(pitch_probs * idx_tensor, 1) * binwidth - angle
                yaw_pred   = torch.sum(yaw_probs   * idx_tensor, 1) * binwidth - angle

                # regression loss
                loss_pitch_reg = reg_criterion(pitch_pred, label_pitch_reg)
                loss_yaw_reg   = reg_criterion(yaw_pred,   label_yaw_reg)

                # total loss (scaled for accumulation)
                loss = (loss_pitch_cls + loss_yaw_cls +
                        loss_pitch_reg + loss_yaw_reg) / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            sum_loss += loss.item() * accumulation_steps  # rescale for logging

        avg_loss = sum_loss / len(data_loader)
        logging.info(f"Fine-tune Epoch [{epoch+1}/{epochs}] - Avg Loss: {avg_loss:.4f}")

        torch.cuda.empty_cache()  # free unused memory

    model.eval()
    return model

def low_activity_pipeline(model, clean_loader, device, bins, binwidth, angle,
                        epochs=5, lr=1e-4,
                        accumulation_steps=2, use_amp=True, prune_ratio=0.2):
    logging.info("[Low Activity] Starting defense pipeline...")

    # Make sure model is on GPU before activity computation
    model.to(device)
    model.train()

    # Step 1: neuron activity
    activations = compute_neuron_activity(model, clean_loader, device)

    # Step 2: fine pruning
    model = low_activity_prune(model, activations, prune_ratio)
    torch.cuda.empty_cache()  # free memory after pruning

    # Step 3: fine-tuning with AMP + accumulation
    model = fine_tune(model, clean_loader, device,
                      bins=bins,
                      binwidth=binwidth,
                      angle=angle,
                      epochs = epochs,
                      lr=lr,
                      accumulation_steps=accumulation_steps,
                      use_amp=use_amp)
    
    activations = compute_neuron_activity(model, clean_loader, device)

    return model

# --- Evaluation function ---
def evaluate(model, dataloader, device, bins, binwidth, angle, poison_target=None, tolerance=5.0):
    """
    Evaluate clean accuracy (MSE) and ASR if poisoned set is provided.
    """
    model.eval()
    mse_pitch, mse_yaw = 0.0, 0.0
    total_samples, asr_hits = 0, 0

    idx_tensor = torch.arange(bins, device=device, dtype=torch.float32)
    reg_criterion = nn.MSELoss()

    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            regression_labels_gaze = batch[2].to(device)

            pitch, yaw = model(images)
            pitch, yaw = F.softmax(pitch, dim=1), F.softmax(yaw, dim=1)

            pitch_pred = torch.sum(pitch * idx_tensor, 1) * binwidth - angle
            yaw_pred = torch.sum(yaw * idx_tensor, 1) * binwidth - angle

            mse_pitch += reg_criterion(pitch_pred, regression_labels_gaze[:,0]).item() * images.size(0)
            mse_yaw += reg_criterion(yaw_pred, regression_labels_gaze[:,1]).item() * images.size(0)
            total_samples += images.size(0)

            if poison_target is not None:
                target_pitch, target_yaw = poison_target
                matches = ((torch.abs(pitch_pred - target_pitch) <= tolerance) &
                           (torch.abs(yaw_pred - target_yaw) <= tolerance)).sum().item()
                asr_hits += matches

    avg_mse_pitch = mse_pitch / total_samples
    avg_mse_yaw = mse_yaw / total_samples
    logging.info(f"[Eval] Pitch MSE={avg_mse_pitch:.4f}, Yaw MSE={avg_mse_yaw:.4f}")

    if poison_target is not None:
        asr = asr_hits / total_samples
        logging.info(f"[Eval] Attack Success Rate (ASR)={asr:.4f}")
        return avg_mse_pitch, avg_mse_yaw, asr

    return avg_mse_pitch, avg_mse_yaw

# --- Large Inconsistency utilities ---
def compute_consistency_scores(model, clean_loader, poison_loader, device):
    """Compute neuron consistency scores: difference in average activations between clean and poisoned inputs."""
    model.eval()
    activations_clean, activations_poison = {}, {}
    current_mode = None

    def hook_fn(name):
        def fn(module, inp, out):
            if isinstance(out, torch.Tensor):
                mean_act = out.detach().mean(dim=(0, 2, 3)).cpu()
                if current_mode == "clean":
                    activations_clean.setdefault(name, []).append(mean_act)
                elif current_mode == "poison":
                    activations_poison.setdefault(name, []).append(mean_act)
        return fn

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    with torch.no_grad():
        current_mode = "clean"
        for batch in clean_loader:
            images = batch[0].to(device)
            model(images)
        current_mode = "poison"
        for batch in poison_loader:
            images = batch[0].to(device)
            model(images)

    for h in hooks: h.remove()

    scores = {}
    for name in activations_clean:
        clean_mean = torch.stack(activations_clean[name]).mean(0)
        poison_mean = torch.stack(activations_poison[name]).mean(0)
        scores[name] = torch.abs(clean_mean - poison_mean)
    return scores

def LargeInc_prune(model, scores, prune_ratio=0.2):
    """Prune neurons with largest inconsistency between clean and poisoned activations."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name in scores:
            diffs = scores[name]
            k = int(prune_ratio * len(diffs))
            if k > 0:
                prune_idx = torch.topk(diffs, k, largest=True).indices
                with torch.no_grad():
                    module.weight[prune_idx] = 0
                    if module.bias is not None:
                        module.bias[prune_idx] = 0
    return model

def LargeInc_pipeline(model, clean_loader, poison_loader, device,
                     bins, binwidth, angle,
                     epochs=5, lr=1e-4, prune_ratio=0.2):
    logging.info("[Largest Inconsistency] Starting defense pipeline...")
    scores = compute_consistency_scores(model, clean_loader, poison_loader, device)
    model = LargeInc_prune(model, scores, prune_ratio)
    torch.cuda.empty_cache()
    model = fine_tune(model, clean_loader, device,
                      bins=bins, binwidth=binwidth, angle=angle,
                      epochs=epochs, lr=lr)
    return model

# --- Low Entropy utilities ---
def compute_entropy_scores(model, loader, device):
    """Compute entropy of feature activations across clean inputs."""
    model.eval()
    entropy_scores = {}

    def hook_fn(name):
        def fn(module, inp, out):
            probs = F.softmax(out.view(out.size(0), -1), dim=1)
            entropy = -(probs * probs.log()).sum(dim=1).mean().item()
            entropy_scores.setdefault(name, []).append(entropy)
        return fn

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            model(images)

    for h in hooks: h.remove()

    avg_entropy = {name: sum(vals)/len(vals) for name, vals in entropy_scores.items()}
    return avg_entropy

def low_entropy_prune(model, entropy_scores, prune_ratio=0.2):
    """Prune neurons with lowest entropy (most suspicious)."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name in entropy_scores:
            k = int(prune_ratio * module.out_channels)
            if k > 0:
                prune_idx = torch.arange(k)
                with torch.no_grad():
                    module.weight[prune_idx] = 0
                    if module.bias is not None:
                        module.bias[prune_idx] = 0
    return model

def low_entropy_pipeline(model, clean_loader, device,
                      bins, binwidth, angle,
                      epochs=5, lr=1e-4, prune_ratio=0.2):
    logging.info("[Low entropy] Starting defense pipeline...")
    entropy_scores = compute_entropy_scores(model, clean_loader, device)
    model = low_entropy_prune(model, entropy_scores, prune_ratio)
    torch.cuda.empty_cache()
    model = fine_tune(model, clean_loader, device,
                      bins=bins, binwidth=binwidth, angle=angle,
                      epochs=epochs, lr=lr)
    return model

def eval_backdoor(backdoored_model, poison_loader, device,
                     bins, binwidth, angle,
                     target_gaze=(0.0, 0.0), tolerance=5.0):
    results = {}
    logging.info("[Baseline] Evaluating backdoored model...")
    mse_pitch, mse_yaw, asr = evaluate(backdoored_model, poison_loader, device,
                                       bins, binwidth, angle,
                                       poison_target=target_gaze,
                                       tolerance=tolerance)
    results["Backdoored"] = {"MSE_pitch": mse_pitch, "MSE_yaw": mse_yaw, "ASR": asr}
    return results

def run_all_defenses(backdoored_model, clean_loader, poison_loader, device,
                     bins, binwidth, angle,
                     target_gaze=(0.0, 0.0), tolerance=5.0,
                     epochs=5, lr=1e-4, prune_ratio=0.2):
    
    # Low Activity
    defended_la = low_activity_pipeline(backdoored_model, clean_loader, device,
                                   bins, binwidth, angle,
                                   epochs=epochs, lr=lr,
                                   prune_ratio=prune_ratio)
    evaluate(defended_la, poison_loader, device,
                                       bins, binwidth, angle,
                                       poison_target=target_gaze,
                                       tolerance=tolerance)

    # Large Inconsistent
    defended_li = LargeInc_pipeline(backdoored_model, clean_loader, poison_loader, device,
                                   bins, binwidth, angle,
                                   epochs=epochs, lr=lr,
                                   prune_ratio=prune_ratio)
    evaluate(defended_li, poison_loader, device,
                                       bins, binwidth, angle,
                                       poison_target=target_gaze,
                                       tolerance=tolerance)

    # Low Entropy
    defended_le = low_entropy_pipeline(backdoored_model, clean_loader, device,
                                     bins, binwidth, angle,
                                     epochs=epochs, lr=lr,
                                     prune_ratio=prune_ratio)
    evaluate(defended_le, poison_loader, device,
                                       bins, binwidth, angle,
                                       poison_target=target_gaze,
                                       tolerance=tolerance)
