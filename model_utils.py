import torch
import json
import os
from kan import KAN

def save_normalization_stats(X_mean, X_std, filepath="./model/norm_stats.json"):
    stats = {
        "mean": X_mean.tolist(),
        "std": X_std.tolist()
    }
    with open(filepath, 'w') as f:
        json.dump(stats, f)

def load_normalization_stats(filepath="./model/norm_stats.json"):
    with open(filepath, 'r') as f:
        stats = json.load(f)
    return torch.tensor(stats["mean"]), torch.tensor(stats["std"])

def save_model_state(model, checkpoint_dir="./model"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = f"{checkpoint_dir}/0.0"
    torch.save(model.state_dict(), f"{path}_state")
    if hasattr(model, 'cache_data'):
        torch.save(model.cache_data, f"{path}_cache_data")

def get_normalization_stats(X):
    X_mean = torch.mean(X, dim=0)
    X_std  = torch.std(X, dim=0)
    return X_mean, X_std

def normalize_data(X, X_mean, X_std):
    return (X - X_mean) / (X_std + 1e-6)

def prepare_model(X_train):
    input_dim = X_train.shape[1]
    X_mean, X_std = get_normalization_stats(X_train)
    X_train_norm = normalize_data(X_train, X_mean, X_std)

    model = KAN(width=[input_dim, 3], grid=10, k=3, device="cpu")
    model.update_grid_from_samples(X_train_norm)
    return model, X_mean, X_std


def targets_to_class_indices(Y):
    y = Y.flatten()
    return torch.where(y == 0.0, 0, torch.where(y == 0.5, 1, 2)).long()


def train_model(model, X_norm, Y, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    y_cls = targets_to_class_indices(Y)
    class_counts = torch.bincount(y_cls, minlength=3).float().clamp_min(1)
    class_weights = class_counts.sum() / (class_counts * 3)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X_norm)
        loss = loss_fn(logits, y_cls)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    save_model_state(model)
    return losses

def get_predictions(model, X_norm):
    with torch.no_grad():
        y_pred_all = torch.softmax(model(X_norm), dim=1).numpy()
    return y_pred_all

def evaluate_model(y_true, y_pred_prob):
    class_values = [0.0, 0.5, 1.0]
    y_true_cls = [float(v) for v in y_true.flatten()]
    if y_pred_prob.ndim == 2 and y_pred_prob.shape[1] == 3:
        y_pred_cls = [class_values[i] for i in y_pred_prob.argmax(axis=1)]
    else:
        def to_class(val):
            if val > 0.6: return 1.0
            if val < 0.4: return 0.0
            return 0.5
        y_pred_cls = [to_class(v) for v in y_pred_prob.flatten()]
    
    classes = [1.0, 0.5, 0.0]
    class_names = {1.0: "Buy ", 0.5: "Hold", 0.0: "Sell"}
    
    results = {}
    total_correct = 0
    
    print("\n--- Evaluasi Metrik Klasifikasi ---")
    print(f"{'Class':<6} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 45)
    
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true_cls, y_pred_cls) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true_cls, y_pred_cls) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true_cls, y_pred_cls) if t == cls and p != cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        total_correct += tp
        print(f"{class_names[cls]:<6} | {precision:<10.4f} | {recall:<10.4f} | {f1:<10.4f}")
        
    accuracy = total_correct / len(y_true_cls)
    print("-" * 45)
    print(f"Overall Accuracy: {accuracy:.4f}\n")
    
    # Return metrics for plotting
    final_metrics = {}
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true_cls, y_pred_cls) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true_cls, y_pred_cls) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true_cls, y_pred_cls) if t == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        final_metrics[class_names[cls].strip()] = {"Precision": precision, "Recall": recall, "F1-Score": f1}
    
    return accuracy, final_metrics
