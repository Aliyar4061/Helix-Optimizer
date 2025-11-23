# Helix Ablation Study for MNIST (12 Configurations)

import os
import time
import random
import contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import json
from dataclasses import dataclass
from typing import Tuple, Dict, List, Any

# ---------- Global Performance Knobs ----------
USE_AMP = torch.cuda.is_available()
CHANNELS_LAST = False  # MNIST is grayscale, so channels_last not beneficial
AUTO_BENCHMARK = True
MATMUL_PRECISION = "high"

# Helix-specific constants
SMALL_PARAM_THRESHOLD = 1000

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = AUTO_BENCHMARK
        try:
            torch.set_float32_matmul_precision(MATMUL_PRECISION)
        except AttributeError:
            print("Warning: torch.set_float32_matmul_precision is not available.")

set_seed(42)

# ---------------------------
# Helix Optimizer Implementation
# ---------------------------
class Helix(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), beta_s=0.9,
                 eps=1e-8, weight_decay=1e-2, agc=False,
                 phi_init=0.05, phi_final=0.4, phi_warmup=10000,
                 warmup_steps=1000, use_trust_ratio=False,
                 kappa=1.0, agc_clip=0.05):
        
        # Parameter validation
        if not 0.0 <= lr: raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps: raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0: raise ValueError(f"Invalid beta_1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0: raise ValueError(f"Invalid beta_2: {betas[1]}")
        if not 0.0 <= beta_s < 1.0: raise ValueError(f"Invalid beta_s: {beta_s}")
        if not 0.0 <= weight_decay: raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= phi_init <= phi_final <= 1.0: raise ValueError(f"Invalid phi range: {phi_init} to {phi_final}")
        if not 0 < phi_warmup: raise ValueError(f"Invalid phi_warmup: {phi_warmup}")
        if not 0 <= warmup_steps: raise ValueError(f"Invalid warmup_steps: {warmup_steps}")
        if not 0.0 <= kappa: raise ValueError(f"Invalid kappa: {kappa}")
        if not 0.0 <= agc_clip <= 1.0: raise ValueError(f"Invalid agc_clip: {agc_clip}")

        defaults = dict(lr=lr, betas=betas, beta_s=beta_s, eps=eps,
                        weight_decay=weight_decay, agc=agc,
                        phi_init=phi_init, phi_final=phi_final,
                        phi_warmup=phi_warmup, warmup_steps=warmup_steps,
                        use_trust_ratio=use_trust_ratio, kappa=kappa,
                        agc_clip=agc_clip)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            beta_s = group['beta_s']
            eps = group['eps']
            wd = group['weight_decay']
            phi_init = group['phi_init']
            phi_final = group['phi_final']
            phi_warmup = group['phi_warmup']
            warmup_steps = group['warmup_steps']
            use_trust = group['use_trust_ratio']
            kappa = group['kappa']
            agc_clip = group['agc_clip']
            use_agc = group['agc']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Helix does not support sparse gradients")

                is_weight_like = (p.ndim > 1) and (p.numel() >= SMALL_PARAM_THRESHOLD)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['b'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['s'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['step'] += 1
                step = state['step']

                if (use_agc or use_trust) and is_weight_like:
                    param_norm = torch.linalg.vector_norm(p).clamp(min=eps)
                else:
                    param_norm = None

                if use_agc and is_weight_like:
                    grad_norm = torch.linalg.vector_norm(grad)
                    max_norm = agc_clip * param_norm
                    if grad_norm > max_norm:
                        grad.mul_(max_norm / (grad_norm + 1e-6))

                m = state['m']
                b = state['b']
                s = state['s']

                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                diff = grad - m
                b.mul_(beta2).addcmul_(diff, diff, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                m_hat = m / (bias_correction1 + 1e-16)
                b_hat = b / (bias_correction2 + 1e-16)
                d = b_hat.sqrt().add_(eps)

                s.mul_(beta_s).add_(m_hat.sign(), alpha=1.0 - beta_s)

                if step <= phi_warmup:
                    phi = phi_init + (phi_final - phi_init) * (step / max(1, phi_warmup))
                else:
                    phi = phi_final

                u_adapt = m_hat / d
                u_sign = s.clamp(-1.0, 1.0)
                delta = -((1.0 - phi) * u_adapt + phi * (kappa * u_sign))

                lr_t = lr * (step / max(1, warmup_steps)) if step <= warmup_steps else lr

                if use_trust and is_weight_like:
                    delta_norm = torch.linalg.vector_norm(delta)
                    if delta_norm > 0:
                        trust = (param_norm / (delta_norm + 1e-6)).clamp(0.01, 10.0)
                        delta.mul_(trust)

                p.add_(delta, alpha=lr_t)

                if wd != 0:
                    p.mul_(1.0 - lr_t * wd)

        return loss

# ---------------------------
# Enhanced Metrics Tracking
# ---------------------------
class MetricsTracker:
    """Enhanced metrics tracking with statistical analysis"""
    def __init__(self):
        self.metrics = {}
        self.training_history = []
    
    def update_epoch(self, run_name: str, epoch: int, metrics: Dict[str, float]):
        if run_name not in self.metrics:
            self.metrics[run_name] = []
        
        epoch_data = {'epoch': epoch, **metrics}
        self.metrics[run_name].append(epoch_data)
        self.training_history.append({'run': run_name, **epoch_data})
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        """Convert metrics to pandas DataFrame for analysis"""
        df = pd.DataFrame(self.training_history)
        return df
    
    def compute_statistical_significance(self, config1: str, config2: str, metric: str = 'test_acc') -> float:
        """Compute if differences are statistically significant"""
        try:
            import scipy.stats as stats
            
            if config1 not in self.metrics or config2 not in self.metrics:
                return 1.0
            
            metrics1 = [m[metric] for m in self.metrics[config1] if metric in m]
            metrics2 = [m[metric] for m in self.metrics[config2] if metric in m]
            
            if len(metrics1) < 2 or len(metrics2) < 2:
                return 1.0
            
            t_stat, p_value = stats.ttest_ind(metrics1, metrics2)
            return p_value
        except ImportError:
            return 1.0

# ---------------------------
# MNIST Data Loading
# ---------------------------
def get_mnist_loaders(batch_size=128, data_root="./data"):
    cores = os.cpu_count() or 2
    num_workers = min(8, max(2, cores // 2))

    # MNIST specific transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),  # MNIST is 28x28
        transforms.RandomHorizontalFlip(),  # Can still be useful for digits
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST dataset
    train_full = datasets.MNIST(root=data_root, train=True, download=True, transform=transform_train)
    testset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform_test)

    # Split training set into train and validation
    trainset, valset = random_split(train_full, [55000, 5000])  # MNIST has 60k training samples

    loader_kwargs = dict(num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    
    print(f"MNIST Dataset: {len(trainset)} training, {len(valset)} validation, {len(testset)} test samples")
    return train_loader, val_loader, test_loader

# ---------------------------
# MNIST Model Definition
# ---------------------------
class MNIST_CNN(nn.Module):
    """A simple CNN model for MNIST classification"""
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # After two pools: 28->14->7
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def get_mnist_model():
    """Returns a CNN model for MNIST"""
    return MNIST_CNN()

# ---------------------------
# Training and Evaluation Loop
# ---------------------------
def train_model(model_fn, optimizer_factory, train_loader, val_loader, test_loader,
                num_epochs=10, run_name="default", metrics_tracker=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_fn()
    model = model.to(device)  # No channels_last for MNIST (grayscale)

    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_factory(model.parameters())

    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    autocast_ctx = lambda: torch.amp.autocast(device_type=device.type, enabled=USE_AMP)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    metrics = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    overall_start_time = time.time()

    print(f"Starting training for '{run_name}' on {device}...")

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for data, targets in train_loader:
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                outputs = model(data)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * data.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        metrics['train_loss'].append(train_loss / train_total)
        metrics['train_acc'].append(train_correct / train_total)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                with autocast_ctx():
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * data.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        metrics['val_loss'].append(val_loss / val_total)
        metrics['val_acc'].append(val_correct / val_total)

        # Update metrics tracker
        if metrics_tracker is not None:
            epoch_metrics = {
                'train_loss': metrics['train_loss'][-1],
                'val_loss': metrics['val_loss'][-1],
                'train_acc': metrics['train_acc'][-1],
                'val_acc': metrics['val_acc'][-1]
            }
            metrics_tracker.update_epoch(run_name, epoch, epoch_metrics)

        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {metrics['train_loss'][-1]:.4f}, Acc: {metrics['train_acc'][-1]:.4f} | "
              f"Val Loss: {metrics['val_loss'][-1]:.4f}, Acc: {metrics['val_acc'][-1]:.4f} | "
              f"Time: {time.time() - epoch_start:.2f}s")

    # Final Test Evaluation
    model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0
    all_probs, all_targets = [], []
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(data)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * data.size(0)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    final_metrics = {
        'test_acc': test_correct / test_total,
        'test_loss': test_loss / test_total,
        'avg_epoch_time': (time.time() - overall_start_time) / num_epochs
    }

    preds = np.argmax(all_probs, axis=1)
    final_metrics['test_precision'] = precision_score(all_targets, preds, average='macro', zero_division=0)
    final_metrics['test_recall'] = recall_score(all_targets, preds, average='macro', zero_division=0)
    final_metrics['test_f1'] = f1_score(all_targets, preds, average='macro', zero_division=0)
    
    try:
        final_metrics['test_auc'] = roc_auc_score(all_targets, all_probs, multi_class='ovr')
    except:
        final_metrics['test_auc'] = 0.0

    print(f"Test Results ({run_name}): Acc: {final_metrics['test_acc']:.4f}, F1: {final_metrics['test_f1']:.4f}")
    return {**metrics, **final_metrics}

# ---------------------------
# Ablation Study Setup for Helix (12 Configurations)
# ---------------------------
def get_ablated_hyperparams_helix():
    """Focused hyperparameter search for Helix optimizer (12 configurations)"""
    default_cfg = {
        'lr': 1e-3,
        'betas': (0.9, 0.999),
        'beta_s': 0.9,
        'eps': 1e-8,
        'weight_decay': 1e-2,
        'agc': False,
        'phi_init': 0.05,
        'phi_final': 0.4,
        'phi_warmup': 10000,
        'warmup_steps': 1000,
        'use_trust_ratio': False,
        'kappa': 1.0,
        'agc_clip': 0.05
    }
    hyperparams = {}

    # 1. Baseline
    hyperparams["Default"] = default_cfg.copy()
    
    # 2-3. Learning rate variations (most critical parameter)
    for lr in [5e-4, 5e-3]:
        cfg = default_cfg.copy()
        cfg['lr'] = lr
        hyperparams[f"lr={lr}"] = cfg

    # 4-5. Beta variations (momentum and variance)
    for betas in [(0.8, 0.999), (0.9, 0.99)]:
        cfg = default_cfg.copy()
        cfg['betas'] = betas
        hyperparams[f"β₁={betas[0]},β₂={betas[1]}"] = cfg

    # 6. Beta_s variation (sign momentum)
    cfg = default_cfg.copy()
    cfg['beta_s'] = 0.95
    hyperparams["β_s=0.95"] = cfg

    # 7-8. Weight decay variations (regularization)
    for wd in [0.0, 5e-2]:
        cfg = default_cfg.copy()
        cfg['weight_decay'] = wd
        hyperparams[f"wd={wd}"] = cfg

    # 9-10. Phi final variations (adaptive mixing)
    for phi_final in [0.2, 0.6]:
        cfg = default_cfg.copy()
        cfg['phi_final'] = phi_final
        hyperparams[f"φ_final={phi_final}"] = cfg

    # 11. Kappa variation (sign scaling)
    cfg = default_cfg.copy()
    cfg['kappa'] = 2.0
    hyperparams["κ=2.0"] = cfg

    # 12. Combined features (AGC + Trust Ratio)
    cfg = default_cfg.copy()
    cfg['agc'] = True
    cfg['use_trust_ratio'] = True
    hyperparams["AGC+TrustRatio"] = cfg

    return hyperparams

def validate_optimizer_params(params):
    """Validate optimizer parameters"""
    if not isinstance(params, dict):
        raise TypeError("params must be a dictionary")
    
    required_keys = ['lr', 'betas', 'beta_s', 'eps', 'weight_decay', 'agc', 
                    'phi_init', 'phi_final', 'phi_warmup', 'warmup_steps', 
                    'use_trust_ratio', 'kappa', 'agc_clip']
    
    for key in required_keys:
        if key not in params:
            raise ValueError(f"Missing required parameter: {key}")
    
    beta1, beta2 = params['betas']
    if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
        raise ValueError(f"Invalid beta values: {params['betas']}")
    
    if not 0.0 <= params['beta_s'] < 1.0:
        raise ValueError(f"Invalid beta_s: {params['beta_s']}")
    
    return True

# ---------------------------
# Enhanced Visualization
# ---------------------------
def create_comprehensive_plots(all_metrics: Dict[str, Dict], output_dir: str = "./helix_mnist_results"):
    """Create comprehensive visualization of ablation results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Performance comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy and F1 comparison
    configs = list(all_metrics.keys())
    test_accs = [all_metrics[c]['test_acc'] for c in configs]
    test_f1s = [all_metrics[c]['test_f1'] for c in configs]
    
    x_pos = np.arange(len(configs))
    bars1 = ax1.bar(x_pos - 0.2, test_accs, 0.4, label='Accuracy', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x_pos + 0.2, test_f1s, 0.4, label='F1-Score', alpha=0.8, color='salmon')
    
    ax1.set_xlabel('Configurations')
    ax1.set_ylabel('Scores')
    ax1.set_title('Helix on MNIST: Test Accuracy and F1-Score (12 Configurations)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Training curves for top configurations
    top_configs = sorted(configs, key=lambda x: all_metrics[x]['test_acc'], reverse=True)[:3]
    for config in top_configs:
        ax2.plot(all_metrics[config]['train_acc'], label=f'{config} (Train)', linewidth=2)
        ax2.plot(all_metrics[config]['val_acc'], '--', label=f'{config} (Val)', linewidth=2)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Helix on MNIST: Training Curves - Top 3 Configurations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Convergence speed
    convergence_data = []
    for config in configs:
        val_accs = all_metrics[config]['val_acc']
        if len(val_accs) > 0:
            max_acc = max(val_accs)
            target_acc = 0.8 * max_acc
            convergence_epoch = next((i for i, acc in enumerate(val_accs) if acc >= target_acc), len(val_accs))
            convergence_data.append(convergence_epoch)
        else:
            convergence_data.append(len(val_accs))
    
    ax3.bar(configs, convergence_data, alpha=0.7, color='lightgreen')
    ax3.set_xlabel('Configurations')
    ax3.set_ylabel('Epoch to Reach 80% Max Accuracy')
    ax3.set_title('Helix on MNIST: Convergence Speed')
    ax3.set_xticklabels(configs, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Performance vs computational efficiency
    times = [all_metrics[c]['avg_epoch_time'] for c in configs]
    accuracies = test_accs
    
    scatter = ax4.scatter(times, accuracies, s=100, alpha=0.7, c=test_f1s, cmap='viridis')
    ax4.set_xlabel('Average Epoch Time (s)')
    ax4.set_ylabel('Test Accuracy')
    ax4.set_title('Helix on MNIST: Accuracy vs Computational Efficiency')
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar for F1 scores
    plt.colorbar(scatter, ax=ax4, label='F1-Score')
    
    # Annotate points
    for i, config in enumerate(configs):
        ax4.annotate(config, (times[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/helix_mnist_ablation_12_configs.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Additional plot: Performance distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    acc_data = [all_metrics[c]['test_acc'] for c in configs]
    f1_data = [all_metrics[c]['test_f1'] for c in configs]
    
    positions = np.arange(len(configs))
    width = 0.35
    
    bars1 = ax.bar(positions - width/2, acc_data, width, label='Accuracy', alpha=0.7)
    bars2 = ax.bar(positions + width/2, f1_data, width, label='F1-Score', alpha=0.7)
    
    ax.set_xlabel('Configurations')
    ax.set_ylabel('Scores')
    ax.set_title('Helix on MNIST: Performance Distribution Across Configurations')
    ax.set_xticks(positions)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/helix_mnist_performance_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

# ---------------------------
# Main Ablation Study Execution
# ---------------------------
def run_helix_ablation_study(num_epochs=10, batch_size=256):
    """Run Helix ablation study with 12 configurations on MNIST"""
    train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
    hyperparams = get_ablated_hyperparams_helix()
    all_metrics = {}
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker()
    
    print(f"🚀 Starting Helix Ablation Study on MNIST with {len(hyperparams)} configurations")
    print(f"📊 Epochs: {num_epochs}, Batch Size: {batch_size}")
    print("=" * 100)
    
    for name, cfg in tqdm(hyperparams.items(), desc="Helix Configurations"):
        print(f"\n▶️  Running: {name}")
        print(f"   ⚙️  Config: {cfg}")
        
        try:
            validate_optimizer_params(cfg)
            
            optimizer_factory = lambda params: Helix(params, **cfg)
            metrics = train_model(
                get_mnist_model, optimizer_factory,
                train_loader, val_loader, test_loader,
                num_epochs=num_epochs, run_name=name,
                metrics_tracker=metrics_tracker
            )
            all_metrics[name] = metrics
                
        except Exception as e:
            print(f"❌ Error in configuration {name}: {e}")
            continue

    # Generate comprehensive analysis
    print("\n" + "📈" * 20 + " HELIX MNIST ANALYSIS " + "📈" * 20)
    generate_detailed_helix_analysis(all_metrics, metrics_tracker)
    
    return all_metrics, metrics_tracker

def generate_detailed_helix_analysis(all_metrics, metrics_tracker):
    """Generate detailed analysis of Helix results on MNIST"""
    # Create comprehensive plots
    create_comprehensive_plots(all_metrics)
    
    # Export to CSV for further analysis
    df = metrics_tracker.get_summary_dataframe()
    df.to_csv('helix_mnist_ablation_12_configs.csv', index=False)
    
    # Save configurations
    save_helix_configs({k: v for k, v in get_ablated_hyperparams_helix().items() 
                       if k in all_metrics})
    
    # Print results table
    print("\n" + "#" * 80)
    print(" " * 20 + "Helix Ablation Study Results on MNIST (12 Configurations)")
    print("#" * 80)

    best_config_name = max(all_metrics.keys(), key=lambda k: all_metrics[k]['test_acc'])
    best_metrics = all_metrics[best_config_name]
    
    print(f"\n🏆 Best Configuration: '{best_config_name}'")
    print(f"   - Accuracy:  {best_metrics['test_acc']:.4f}")
    print(f"   - F1-Score:  {best_metrics['test_f1']:.4f}")
    print(f"   - AUC:       {best_metrics['test_auc']:.4f}")
    print(f"   - Time/Epoch: {best_metrics['avg_epoch_time']:.2f}s")

    print("\n" + "-" * 60)
    print("LaTeX Table Summary:")
    print("-" * 60)
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"\textbf{Configuration} & \textbf{Accuracy} & \textbf{F1-Score} & \textbf{AUC} & \textbf{Precision} & \textbf{Time/Epoch} \\")
    print(r"\midrule")

    sorted_names = sorted(all_metrics.keys(), key=lambda k: all_metrics[k]['test_acc'], reverse=True)
    for name in sorted_names:
        metrics = all_metrics[name]
        is_best = name == best_config_name
        acc_str = r"\textbf{" + f"{metrics['test_acc']:.4f}" + "}" if is_best else f"{metrics['test_acc']:.4f}"
        
        # LaTeX-safe name
        latex_name = name.replace('_', ' ').replace('β', r'$\beta$').replace('φ', r'$\phi$').replace('κ', r'$\kappa$')
        
        print(
            latex_name + " & " +
            acc_str + " & " +
            f"{metrics['test_f1']:.4f} & " +
            f"{metrics['test_auc']:.4f} & " +
            f"{metrics['test_precision']:.4f} & " +
            f"{metrics['avg_epoch_time']:.2f}s" + r" \\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")

    # Statistical significance analysis
    configs = list(all_metrics.keys())
    if len(configs) >= 2:
        best_config = max(configs, key=lambda x: all_metrics[x]['test_acc'])
        second_best = max([c for c in configs if c != best_config], 
                         key=lambda x: all_metrics[x]['test_acc'])
        
        p_value = metrics_tracker.compute_statistical_significance(
            best_config, second_best
        )
        
        print(f"\n📊 Statistical Significance Analysis:")
        print(f"   Best vs Second Best: p-value = {p_value:.4f}")
        if p_value < 0.05:
            print("   ✅ Difference is statistically significant (p < 0.05)")
        else:
            print("   ⚠️  Difference is not statistically significant (p ≥ 0.05)")

    # Parameter importance analysis
    print(f"\n🔍 Parameter Importance Summary:")
    param_groups = {
        'Learning Rate': [k for k in configs if k.startswith('lr=')],
        'Beta': [k for k in configs if k.startswith('β')],
        'Weight Decay': [k for k in configs if k.startswith('wd=')],
        'Phi/Kappa': [k for k in configs if k.startswith(('φ', 'κ'))],
        'Features': [k for k in configs if 'AGC' in k or 'Trust' in k]
    }
    
    for group_name, group_configs in param_groups.items():
        if group_configs:
            group_accs = [all_metrics[c]['test_acc'] for c in group_configs if c in all_metrics]
            if group_accs:
                print(f"   {group_name}: {np.mean(group_accs):.4f} ± {np.std(group_accs):.4f}")

    # MNIST-specific insights
    print(f"\n🎯 MNIST-Specific Insights:")
    print(f"   - Best accuracy: {best_metrics['test_acc']:.4f}")
    print(f"   - Average accuracy across all configs: {np.mean([all_metrics[c]['test_acc'] for c in configs]):.4f}")
    print(f"   - Standard deviation: {np.std([all_metrics[c]['test_acc'] for c in configs]):.4f}")
    
    # Check if any configuration achieved near-perfect performance
    near_perfect = [c for c in configs if all_metrics[c]['test_acc'] > 0.99]
    if near_perfect:
        print(f"   - Near-perfect configurations (>0.99): {len(near_perfect)}")
        for config in near_perfect:
            print(f"     * {config}: {all_metrics[config]['test_acc']:.4f}")

def save_helix_configs(hyperparams, filename="helix_mnist_ablation_12_configs.json"):
    """Save Helix ablation configurations for reproducibility"""
    serializable_configs = {}
    for name, cfg in hyperparams.items():
        serializable_configs[name] = {
            'lr': cfg['lr'],
            'betas': list(cfg['betas']),
            'beta_s': cfg['beta_s'],
            'eps': cfg['eps'],
            'weight_decay': cfg['weight_decay'],
            'agc': cfg['agc'],
            'phi_init': cfg['phi_init'],
            'phi_final': cfg['phi_final'],
            'phi_warmup': cfg['phi_warmup'],
            'warmup_steps': cfg['warmup_steps'],
            'use_trust_ratio': cfg['use_trust_ratio'],
            'kappa': cfg['kappa'],
            'agc_clip': cfg['agc_clip']
        }
    
    with open(filename, 'w') as f:
        json.dump(serializable_configs, f, indent=2)

if __name__ == "__main__":
    all_metrics, metrics_tracker = run_helix_ablation_study(num_epochs=10, batch_size=256)