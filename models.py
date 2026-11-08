"""Lightweight classifiers from different model families, for CWRU_bearing_signals.ipynb.

Hyperparameters are fixed a priori (library defaults or common textbook values). Nothing is tuned on
the test folds, so the comparison measures how each family generalises, not how well it was tuned.
"""
import pickle
import time

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn

FAMILIES = {
    "Logistic regression": "Linear",
    "k-NN": "Instance-based",
    "SVM (RBF)": "Kernel",
    "Random forest": "Bagged trees",
    "Gradient boosting": "Boosted trees",
    "MLP": "Neural net (features)",
    "1D-CNN": "Neural net (raw signal)",
}


def model_zoo(seed):
    """One sklearn model per family, all working on the same tabular features."""
    return {
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)),
        "k-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale", class_weight="balanced")),
        "Random forest": RandomForestClassifier(300, class_weight="balanced", random_state=seed, n_jobs=-1),
        "Gradient boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=seed),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(
            (64, 32), alpha=1e-3, max_iter=2000, early_stopping=True, n_iter_no_change=30,
            random_state=seed)),
    }


# ----------------------------------------------------------------------------- 1D-CNN on raw signal

class CNN1D(nn.Module):
    """Small WDCNN-style network (Zhang et al., 2017): a wide first kernel acts as a learned filter bank,
    then small kernels and global average pooling, so the input length is free."""

    def __init__(self, n_classes=4, width=16):
        super().__init__()

        def block(c_in, c_out, k, stride=1):
            return [nn.Conv1d(c_in, c_out, k, stride, padding=k // 2), nn.BatchNorm1d(c_out), nn.ReLU(),
                    nn.MaxPool1d(2)]

        self.features = nn.Sequential(
            *block(1, width, 64, stride=8),
            *block(width, 2 * width, 3),
            *block(2 * width, 2 * width, 3),
            *block(2 * width, 4 * width, 3),
            *block(4 * width, 4 * width, 3),
            nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(4 * width, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class CNNClassifier:
    """sklearn-like wrapper. Each 1 s window is standardised on its own, so the network cannot rely on
    absolute vibration level. Training uses random crops; prediction averages logits over sliding crops."""

    def __init__(self, classes, seed=0, crop=4096, epochs=40, batch=32, lr=3e-3):
        self.classes = list(classes)
        self.seed, self.crop, self.epochs, self.batch, self.lr = seed, crop, epochs, batch, lr

    @staticmethod
    def _standardise(X):
        X = np.asarray(X, dtype=np.float32)
        return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = self._standardise(X)
        y = np.array([self.classes.index(v) for v in y])
        counts = np.bincount(y, minlength=len(self.classes))
        weight = torch.tensor(len(y) / (len(self.classes) * np.maximum(counts, 1)), dtype=torch.float32)
        self.net = CNN1D(len(self.classes))
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        steps = self.epochs * int(np.ceil(len(y) / self.batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps)
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        self.net.train()
        for _ in range(self.epochs):
            order = rng.permutation(len(y))
            for i in range(0, len(y), self.batch):
                idx = order[i:i + self.batch]
                starts = rng.integers(0, X.shape[1] - self.crop + 1, len(idx))
                xb = np.stack([X[j, s:s + self.crop] for j, s in zip(idx, starts)])
                xb = torch.from_numpy(xb[:, None, :])
                opt.zero_grad()
                loss_fn(self.net(xb), torch.from_numpy(y[idx])).backward()
                opt.step()
                sched.step()
        return self

    @torch.no_grad()
    def predict_logits(self, X):
        self.net.eval()
        X = self._standardise(X)
        starts = np.linspace(0, X.shape[1] - self.crop, 5).astype(int)
        crops = np.stack([X[:, s:s + self.crop] for s in starts], axis=1)  # (n, crops, crop)
        n, c, L = crops.shape
        logits = self.net(torch.from_numpy(crops.reshape(n * c, 1, L)))
        return logits.reshape(n, c, -1).mean(axis=1).numpy()

    def predict(self, X):
        return np.array(self.classes)[self.predict_logits(X).argmax(axis=1)]


# ----------------------------------------------------------------------------- cost of deployment

def n_parameters(model):
    """Learned numbers stored by the model: weights for nets/linear models, nodes for trees,
    stored training points for k-NN, support vectors for SVM."""
    if isinstance(model, CNNClassifier):
        return sum(p.numel() for p in model.net.parameters())
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if isinstance(est, LogisticRegression):
        return est.coef_.size + est.intercept_.size
    if isinstance(est, MLPClassifier):
        return sum(w.size for w in est.coefs_) + sum(b.size for b in est.intercepts_)
    if isinstance(est, KNeighborsClassifier):
        return est._fit_X.size
    if isinstance(est, SVC):
        return est.support_vectors_.size + est.dual_coef_.size
    if isinstance(est, RandomForestClassifier):
        return sum(t.tree_.node_count for t in est.estimators_)
    if isinstance(est, HistGradientBoostingClassifier):
        return sum(len(p.nodes) for it in est._predictors for p in it)
    return np.nan


def size_kb(model):
    """Serialized size: pickle for sklearn, float32 state_dict for the CNN."""
    if isinstance(model, CNNClassifier):
        return sum(t.numel() * t.element_size() for t in model.net.state_dict().values()) / 1024
    return len(pickle.dumps(model)) / 1024


def latency_ms(predict, X, repeat=3):
    """Median wall time to classify one window, measured one window at a time (edge-device style)."""
    times = []
    for _ in range(repeat):
        for x in X:
            t0 = time.perf_counter()
            predict(x[None])
            times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times))
