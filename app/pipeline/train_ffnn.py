import torch
import torch.nn as nn
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

from app.models.embedding import EmbeddingModel
from app.models.ff_nn_classifier import (
    BottleneckFFNN,
    ParallelMultiPathFFNN,
    FFNNClassifier,
)
from app.utils.utils import create_dataloaders

def precompute_embeddings(loader, embedder, device):

    embedder.eval()
    cache = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask      = batch["attention_mask"].to(device)
            labels    = batch["label"]           # keep on CPU
            diffs     = batch["difficulty"]
            sr        = batch["max_surface_risk"]

            embs = embedder(input_ids, mask).cpu()  # [B, D]

            for i in range(embs.size(0)):
                cache.append({
                    "emb":              embs[i],
                    "label":            labels[i].item(),
                    "difficulty":       diffs[i],
                    "max_surface_risk": sr[i].item(),
                })

    return cache


def make_cached_loader(cache, batch_size, shuffle):

    from torch.utils.data import DataLoader, Dataset

    class CachedEmbDataset(Dataset):
        def __init__(self, records):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            return self.records[idx]

    def collate(batch):
        return {
            "emb":              torch.stack([b["emb"] for b in batch]),
            "label":            torch.tensor([b["label"] for b in batch], dtype=torch.float32),
            "difficulty":       [b["difficulty"] for b in batch],
            "max_surface_risk": torch.tensor([b["max_surface_risk"] for b in batch]),
        }

    return DataLoader(
        CachedEmbDataset(cache),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
    )

class FullModel(nn.Module):
    def __init__(self, model_type: str = "model3", pooling: str = "mean"):
        super().__init__()
        self.embedder = EmbeddingModel(pooling=pooling)

        if model_type == "model1":
            self.classifier = BottleneckFFNN()
        elif model_type == "model2":
            self.classifier = ParallelMultiPathFFNN()
        elif model_type == "model3":
            self.classifier = FFNNClassifier()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def forward(self, input_ids, attention_mask):
        x = self.embedder(input_ids, attention_mask)
        return self.classifier(x)

def count_parameters(model):

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def train_on_cache(classifier, loader, optimizer, criterion, device):

    classifier.train()
    total_loss = 0.0

    for batch in loader:
        emb    = batch["emb"].to(device)      # [B, D]
        labels = batch["label"].to(device)    # [B]

        logits = classifier(emb).squeeze(-1)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate_on_cache(classifier, loader, device, split_name: str = ""):

    classifier.eval()

    all_probs, all_preds, all_labels = [], [], []
    by_diff: dict = defaultdict(lambda: {"probs": [], "preds": [], "labels": []})

    with torch.no_grad():
        for batch in loader:
            emb    = batch["emb"].to(device)
            labels = batch["label"].to(device)
            diffs  = batch["difficulty"]

            logits = classifier(emb).squeeze(-1)
            probs  = torch.sigmoid(logits)
            preds  = (probs > 0.5).float()

            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            for prob, pred, label, diff in zip(
                probs.cpu().tolist(),
                preds.cpu().tolist(),
                labels.cpu().tolist(),
                diffs,
            ):
                by_diff[diff]["probs"].append(prob)
                by_diff[diff]["preds"].append(pred)
                by_diff[diff]["labels"].append(label)

    int_labels = [int(l) for l in all_labels]
    int_preds  = [int(p) for p in all_preds]

    acc       = accuracy_score(int_labels, int_preds)
    f1        = f1_score(int_labels, int_preds, zero_division=0)
    auc       = roc_auc_score(int_labels, all_probs)
    precision = precision_score(int_labels, int_preds, zero_division=0)
    recall    = recall_score(int_labels, int_preds, zero_division=0)
    cm        = confusion_matrix(int_labels, int_preds).tolist()  # [[TN,FP],[FN,TP]]

    metrics = {
        "acc":       acc,
        "f1":        f1,
        "auc":       auc,
        "precision": precision,
        "recall":    recall,
        "cm":        cm,
    }

    diff_results = {}
    for diff, vals in by_diff.items():
        d_int_labels = [int(l) for l in vals["labels"]]
        d_int_preds  = [int(p) for p in vals["preds"]]
        d_acc        = accuracy_score(d_int_labels, d_int_preds)
        d_f1         = f1_score(d_int_labels, d_int_preds, zero_division=0)
        d_prec       = precision_score(d_int_labels, d_int_preds, zero_division=0)
        d_rec        = recall_score(d_int_labels, d_int_preds, zero_division=0)
        d_auc        = (
            roc_auc_score(d_int_labels, vals["probs"])
            if len(set(d_int_labels)) > 1
            else float("nan")
        )
        diff_results[diff] = {
            "acc": d_acc, "f1": d_f1, "auc": d_auc,
            "precision": d_prec, "recall": d_rec,
            "n": len(d_int_labels),
        }

    return metrics, diff_results

def evaluate_surface_risk_baseline(loader, threshold: float = 0.3):
    all_preds, all_labels = [], []

    for batch in loader:
        sr     = batch["max_surface_risk"]
        labels = batch["label"]

        preds = (sr > threshold).float()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    return {"acc": acc, "f1": f1}

def print_metrics(metrics: dict, prefix: str = ""):
    cm = metrics["cm"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    tag = f"[{prefix}] " if prefix else ""
    print(
        f"{tag}Acc={metrics['acc']:.4f}  "
        f"F1={metrics['f1']:.4f}  "
        f"AUC={metrics['auc']:.4f}  "
        f"Prec={metrics['precision']:.4f}  "
        f"Rec={metrics['recall']:.4f}"
    )
    print(f"{tag}Confusion Matrix  TN={tn}  FP={fp}  FN={fn}  TP={tp}")

def print_diff_results(diff_results: dict, prefix: str = ""):
    tag = f"[{prefix}] " if prefix else ""
    for diff, m in sorted(diff_results.items()):
        print(
            f"  {tag}{diff} (n={m['n']})  "
            f"Acc={m['acc']:.4f}  F1={m['f1']:.4f}  "
            f"AUC={m['auc']:.4f}  Prec={m['precision']:.4f}  Rec={m['recall']:.4f}"
        )

def run_training(
    data_path:  str,
    model_type: str = "model3",
    epochs:     int = 5,
    batch_size: int = 16,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Architecture: {model_type}")

    train_loader, val_loader, test_loader = create_dataloaders(
        data_path, batch_size=batch_size
    )

    embedder = EmbeddingModel(pooling="mean").to(device)

    print("Precomputing embeddings for train/val/test splits...")
    train_cache = precompute_embeddings(train_loader, embedder, device)
    val_cache   = precompute_embeddings(val_loader,   embedder, device)
    test_cache  = precompute_embeddings(test_loader,  embedder, device)
    print(f"  Train: {len(train_cache)}  Val: {len(val_cache)}  Test: {len(test_cache)}")

    cached_train = make_cached_loader(train_cache, batch_size, shuffle=True)
    cached_val   = make_cached_loader(val_cache,   batch_size, shuffle=False)
    cached_test  = make_cached_loader(test_cache,  batch_size, shuffle=False)

    if model_type == "model1":
        classifier = BottleneckFFNN()
    elif model_type == "model2":
        classifier = ParallelMultiPathFFNN()
    elif model_type == "model3":
        classifier = FFNNClassifier()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    classifier = classifier.to(device)

    total_params, trainable_params = count_parameters(classifier)
    embedder_total, _ = count_parameters(embedder)
    print(f"Classifier params — Total: {total_params:,}  Trainable: {trainable_params:,}")
    print(f"Embedder params   — Total: {embedder_total:,}  Trainable: 0 (frozen)")

    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1  = 0.0
    best_state   = None

    epoch_history = []

    for epoch in range(epochs):
        loss = train_on_cache(classifier, cached_train, optimizer, criterion, device)

        train_metrics, _ = evaluate_on_cache(classifier, cached_train, device, "train")
        val_metrics,   _ = evaluate_on_cache(classifier, cached_val,   device, "val")

        gap_acc = train_metrics["acc"] - val_metrics["acc"]
        gap_f1  = train_metrics["f1"]  - val_metrics["f1"]
        gap_auc = train_metrics["auc"] - val_metrics["auc"]

        epoch_history.append({
            "epoch":      epoch + 1,
            "loss":       loss,
            "train":      train_metrics,
            "val":        val_metrics,
            "gap_acc":    gap_acc,
            "gap_f1":     gap_f1,
            "gap_auc":    gap_auc,
        })

        print(f"\nEpoch {epoch+1}/{epochs}  loss={loss:.4f}")
        print_metrics(train_metrics, prefix="Train")
        print_metrics(val_metrics,   prefix="Val")
        print(
            f"  Overfitting gap —  "
            f"Acc: {gap_acc:+.4f}  F1: {gap_f1:+.4f}  AUC: {gap_auc:+.4f}"
        )
        print("-" * 70)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state  = {k: v.clone() for k, v in classifier.state_dict().items()}

    if best_state is not None:
        classifier.load_state_dict(best_state)

    test_metrics, test_diff = evaluate_on_cache(classifier, cached_test, device, "test")

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print_metrics(test_metrics, prefix="Test")
    print("\nPer-difficulty breakdown:")
    print_diff_results(test_diff, prefix="Test")

    # Summarise overfitting gap at final epoch
    last = epoch_history[-1]
    print(f"\nOverfitting gap at final epoch — "
          f"Acc: {last['gap_acc']:+.4f}  "
          f"F1: {last['gap_f1']:+.4f}  "
          f"AUC: {last['gap_auc']:+.4f}")

    return classifier, test_metrics, test_diff, epoch_history

if __name__ == "__main__":
    data_path = "app/data_pipeline/data/semantic/semantic_multiturn_v7.jsonl"

    results = {}
    for model_type in ["model1", "model2", "model3"]:
        print("\n" + "=" * 80)
        print(f"Running {model_type}")
        print("=" * 80)

        _, test_metrics, test_diff, history = run_training(
            data_path=data_path,
            model_type=model_type,
            epochs=10,
            batch_size=16,
        )

        results[model_type] = {
            "overall":       test_metrics,
            "by_difficulty": test_diff,
            "history":       history,
        }

    print("\n\nFINAL EVALUATION SUMMARY")
    print("=" * 80)
    header = f"{'Model':<10} {'Acc':>6} {'F1':>6} {'AUC':>6} {'Prec':>6} {'Rec':>6}  Confusion[TN,FP,FN,TP]"
    print(header)
    print("-" * len(header))
    for model_type, res in results.items():
        m  = res["overall"]
        cm = m["cm"]
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        print(
            f"{model_type:<10} "
            f"{m['acc']:>6.4f} "
            f"{m['f1']:>6.4f} "
            f"{m['auc']:>6.4f} "
            f"{m['precision']:>6.4f} "
            f"{m['recall']:>6.4f}  "
            f"[{tn},{fp},{fn},{tp}]"
        )