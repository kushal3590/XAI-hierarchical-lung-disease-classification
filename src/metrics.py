"""
Lightweight numpy-only replacements for the sklearn.metrics functions used
in train.py (accuracy_score, f1_score(average="macro"), recall_score(labels=[i],
average="micro"), classification_report).

Why this exists: sklearn.metrics (via scipy's compiled extensions) and
torch.utils.tensorboard (via protobuf) load conflicting native DLLs in the
same Python process on this Windows setup, causing a silent heap-corruption
crash (STATUS_HEAP_CORRUPTION / exit code -1073740940) with no traceback.
KMP_DUPLICATE_LIB_OK does not fix it because it isn't an OpenMP duplicate-
runtime warning, it's an underlying ABI incompatibility. Removing sklearn
from the process entirely sidesteps the conflict.

Drop this file into src/ and swap the sklearn import in train.py for:
    from metrics import accuracy_score, f1_score, recall_score, classification_report
No other code in train.py needs to change — same function signatures/behavior
for the specific call patterns used there.
"""
import numpy as np


def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def _confusion_counts(y_true, y_pred, labels):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.zeros(len(labels), dtype=np.int64)
    fp = np.zeros(len(labels), dtype=np.int64)
    fn = np.zeros(len(labels), dtype=np.int64)
    support = np.zeros(len(labels), dtype=np.int64)
    for i, lbl in enumerate(labels):
        true_mask = (y_true == lbl)
        pred_mask = (y_pred == lbl)
        tp[i] = np.sum(true_mask & pred_mask)
        fp[i] = np.sum(~true_mask & pred_mask)
        fn[i] = np.sum(true_mask & ~pred_mask)
        support[i] = np.sum(true_mask)
    return tp, fp, fn, support


def _per_class_prf(y_true, y_pred, labels, zero_division=0):
    tp, fp, fn = _confusion_counts(y_true, y_pred, labels)[:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where((tp + fp) > 0, tp / np.maximum(tp + fp, 1), zero_division)
        recall = np.where((tp + fn) > 0, tp / np.maximum(tp + fn, 1), zero_division)
        f1 = np.where((precision + recall) > 0,
                       2 * precision * recall / np.maximum(precision + recall, 1e-12),
                       zero_division)
    return precision, recall, f1


def f1_score(y_true, y_pred, average="macro", zero_division=0, labels=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    _, _, f1 = _per_class_prf(y_true, y_pred, labels, zero_division=zero_division)
    if average == "macro":
        return float(np.mean(f1))
    elif average is None:
        return f1
    else:
        raise NotImplementedError(f"average={average!r} not implemented in this shim")


def recall_score(y_true, y_pred, labels=None, average="micro", zero_division=0):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    tp, fp, fn, _ = _confusion_counts(y_true, y_pred, labels)
    if average == "micro":
        denom = tp.sum() + fn.sum()
        if denom == 0:
            return float(zero_division)
        return float(tp.sum() / denom)
    elif average is None:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where((tp + fn) > 0, tp / np.maximum(tp + fn, 1), zero_division)
    else:
        raise NotImplementedError(f"average={average!r} not implemented in this shim")


def classification_report(y_true, y_pred, target_names=None, zero_division=0):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    if target_names is None:
        names = [str(l) for l in labels]
    else:
        names = [target_names[l] for l in labels]

    precision, recall, f1 = _per_class_prf(y_true, y_pred, labels, zero_division=zero_division)
    support = _confusion_counts(y_true, y_pred, labels)[3]

    name_width = max(len(n) for n in names + ["weighted avg"]) + 2
    header = f"{'':<{name_width}}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}\n\n"
    lines = [header]
    for name, p, r, f, s in zip(names, precision, recall, f1, support):
        lines.append(f"{name:<{name_width}}{p:>10.2f}{r:>10.2f}{f:>10.2f}{s:>10d}\n")

    total_support = support.sum()
    acc = accuracy_score(y_true, y_pred)
    lines.append("\n")
    lines.append(f"{'accuracy':<{name_width}}{'':>10}{'':>10}{acc:>10.2f}{total_support:>10d}\n")

    macro_p, macro_r, macro_f = precision.mean(), recall.mean(), f1.mean()
    lines.append(f"{'macro avg':<{name_width}}{macro_p:>10.2f}{macro_r:>10.2f}{macro_f:>10.2f}{total_support:>10d}\n")

    if total_support > 0:
        w = support / total_support
        wp, wr, wf = (precision * w).sum(), (recall * w).sum(), (f1 * w).sum()
    else:
        wp = wr = wf = 0.0
    lines.append(f"{'weighted avg':<{name_width}}{wp:>10.2f}{wr:>10.2f}{wf:>10.2f}{total_support:>10d}\n")

    return "".join(lines)
