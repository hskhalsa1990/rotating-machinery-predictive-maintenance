"""PyTorch 1D-CNN and LSTM for raw vibration signals (synthetic data).

STATUS (read this):
- The NumPy helpers (prepare_raw, to_patches, group_val_split, class_weights) are unit-tested
  and were executed by the author.
- The PyTorch parts (CNN1D, LSTMClassifier, train_eval, predict, ...) were NOT executed:
  PyTorch was not available in the build environment. They follow standard PyTorch usage but
  are UNVERIFIED. test_deep_torch.py runs their tests only when torch is installed.
If torch is missing, HAS_TORCH is False and the notebook skips these models.
"""
import numpy as np
from scipy.signal import decimate

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:  # ImportError or broken install
    HAS_TORCH = False


# ----------------------------- NumPy helpers (tested) -----------------------------
def prepare_raw(signals, factor: int = 2, antialias: bool = True):
    """Per-signal z-score, then decimation by `factor` (with an anti-aliasing filter by default)."""
    x = np.asarray(signals, np.float32)
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)
    if factor == 1:
        return x
    if antialias:
        return np.ascontiguousarray(decimate(x, factor, axis=1, zero_phase=True), dtype=np.float32)
    return x[:, ::factor]


def to_patches(x, patch: int = 16):
    """(N, L) -> (N, L // patch, patch). A sequence of patches is much shorter than the raw
    sample sequence, which makes LSTM training far easier. Trailing samples are dropped."""
    x = np.asarray(x)
    steps = x.shape[1] // patch
    return x[:, : steps * patch].reshape(len(x), steps, patch)


def group_val_split(groups, val_frac: float = 0.2, seed: int = 0):
    """Boolean mask (True = validation) that holds out whole groups (machines), never single rows."""
    groups = np.asarray(groups)
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(len(ug) * val_frac)))
    if len(ug) < 2:
        raise ValueError("need at least 2 groups for a group-wise validation split")
    n_val = min(n_val, len(ug) - 1)
    val_groups = rng.choice(ug, size=n_val, replace=False)
    return np.isin(groups, val_groups)


def class_weights(y, n_classes: int):
    """Inverse-frequency weights (mean 1 over present classes); absent classes get weight 1."""
    y = np.asarray(y, int)
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = np.ones(n_classes)
    present = counts > 0
    w[present] = counts[present].sum() / (present.sum() * counts[present])
    return w


# ----------------------------- PyTorch parts (UNVERIFIED) -----------------------------
if HAS_TORCH:
    class CNN1D(nn.Module):
        """Three Conv1d blocks + global average pooling. Input (B, 1, L), any L >= 64."""

        def __init__(self, n_classes: int = 6, dropout: float = 0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 16, 15, stride=2, padding=7), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(16, 32, 9, padding=4), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(64, n_classes)

        def forward(self, x):               # x: (B, 1, L)
            return self.fc(self.drop(self.net(x).squeeze(-1)))

    class LSTMClassifier(nn.Module):
        """Bidirectional LSTM over patches of the signal. Input (B, 1, L) -> (B, L // patch, patch)."""

        def __init__(self, n_classes: int = 6, hidden: int = 64, patch: int = 16, dropout: float = 0.2):
            super().__init__()
            self.patch = patch
            self.lstm = nn.LSTM(patch, hidden, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(2 * hidden, n_classes)

        def forward(self, x):
            x = x.squeeze(1)
            steps = x.shape[1] // self.patch
            x = x[:, : steps * self.patch].reshape(x.shape[0], steps, self.patch)
            _, (h, _) = self.lstm(x)
            return self.fc(self.drop(torch.cat([h[-2], h[-1]], dim=1)))

    def count_parameters(model) -> int:
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    def get_device(prefer_cuda: bool = True):
        return torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")

    def predict(model, X, batch: int = 256, device=None):
        """Class indices for X of shape (N, L)."""
        device = device or next(model.parameters()).device
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.as_tensor(np.asarray(X[i:i + batch], np.float32), device=device).unsqueeze(1)
                out.append(model(xb).argmax(1).cpu().numpy())
        return np.concatenate(out) if out else np.array([], int)

    def train_eval(model, Xtr, ytr, Xte, yte, epochs: int = 10, lr: float = 1e-3, batch: int = 64, seed: int = 0,
                   groups_tr=None, patience: int = 4, weight_decay: float = 1e-4, return_history: bool = False):
        """Train, then return (accuracy, macro-F1) on the held-out machines (and history if asked).

        If groups_tr (machine ids of the training rows) is given, 20 % of the training machines are held
        out for early stopping (best validation weights are restored). Uses class weights, AdamW,
        cosine learning-rate decay and gradient clipping."""
        from sklearn.metrics import accuracy_score, f1_score
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = get_device()
        model.to(device)
        Xtr = np.asarray(Xtr, np.float32)
        ytr = np.asarray(ytr, int)
        n_classes = int(model.fc.out_features)
        if groups_tr is not None:
            val = group_val_split(groups_tr, 0.2, seed)
            Xfit, yfit, Xval, yval = Xtr[~val], ytr[~val], Xtr[val], ytr[val]
        else:
            Xfit, yfit, Xval, yval = Xtr, ytr, None, None
        Xt = torch.as_tensor(Xfit).unsqueeze(1)
        yt = torch.as_tensor(yfit, dtype=torch.long)
        loss_fn = nn.CrossEntropyLoss(weight=torch.as_tensor(class_weights(yfit, n_classes), dtype=torch.float32, device=device))
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
        best, best_state, bad = -1.0, None, 0
        hist = []
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(len(Xt))
            total = 0.0
            for i in range(0, len(perm), batch):
                idx = perm[i:i + batch]
                if len(idx) < 2:                      # BatchNorm needs more than one sample
                    continue
                opt.zero_grad()
                loss = loss_fn(model(Xt[idx].to(device)), yt[idx].to(device))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += float(loss) * len(idx)
            sched.step()
            rec = {"epoch": ep + 1, "train_loss": total / max(1, len(Xt))}
            if Xval is not None:
                rec["val_acc"] = float(accuracy_score(yval, predict(model, Xval, device=device)))
                if rec["val_acc"] > best:
                    best, bad = rec["val_acc"], 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
            hist.append(rec)
            if Xval is not None and bad >= patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        pred = predict(model, Xte, device=device)
        res = (accuracy_score(yte, pred), f1_score(yte, pred, average="macro"))
        return (res, hist) if return_history else res
