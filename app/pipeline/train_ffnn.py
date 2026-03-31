import torch
import torch.nn as nn
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from app.models.embedding import EmbeddingModel
from app.models.ff_nn_classifier import FFNNClassifier
from app.utils.utils import create_dataloaders

class FullModel(nn.Module):
    def __init__(self, pooling: str = "mean"):
        super().__init__()
        self.embedder   = EmbeddingModel(pooling=pooling)
        self.classifier = FFNNClassifier()

    def forward(self, input_ids, attention_mask):
        x = self.embedder(input_ids, attention_mask)
        return self.classifier(x)   # raw logit, no sigmoid

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        mask      = batch["attention_mask"].to(device)
        labels    = batch["label"].to(device)

        logits = model(input_ids, mask).squeeze(-1)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, device, split_name: str = ""):

    model.eval()

    all_probs, all_preds, all_labels = [], [], []
    by_diff: dict = defaultdict(lambda: {"probs": [], "preds": [], "labels": []})

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask      = batch["attention_mask"].to(device)
            labels    = batch["label"].to(device)
            diffs     = batch["difficulty"]      # list[str], not a tensor

            logits = model(input_ids, mask).squeeze(-1)   # [B]
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

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)

    diff_results = {}
    for diff, vals in by_diff.items():
        d_acc = accuracy_score(vals["labels"], vals["preds"])
        d_f1  = f1_score(vals["labels"], vals["preds"], zero_division=0)
        d_auc = roc_auc_score(vals["labels"], vals["probs"]) if len(set(vals["labels"])) > 1 else float("nan")
        diff_results[diff] = {"acc": d_acc, "f1": d_f1, "auc": d_auc, "n": len(vals["labels"])}

    return {"acc": acc, "f1": f1, "auc": auc}, diff_results

def evaluate_surface_risk_baseline(loader, threshold: float = 0.3):
    all_preds, all_labels = [], []

    for batch in loader:
        sr     = batch["max_surface_risk"]   # [B], already a tensor
        labels = batch["label"]

        preds = (sr > threshold).float()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    return {"acc": acc, "f1": f1}

def run_training(data_path: str, epochs: int = 10, batch_size: int = 16):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = create_dataloaders(
        data_path, batch_size=batch_size
    )

    model = FullModel(pooling="mean").to(device)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=1e-4)

    criterion = nn.BCEWithLogitsLoss()

    print("\n--- Surface-Risk Baseline (test set) ---")
    sr_results = evaluate_surface_risk_baseline(test_loader)
    print(f"Acc: {sr_results['acc']:.4f}  F1: {sr_results['f1']:.4f}")
    print("(threshold=0.3 on max per-turn surface_risk; 0-parameter baseline)\n")

    best_val_f1   = 0.0
    best_state    = None

    for epoch in range(epochs):
        loss = train(model, train_loader, optimizer, criterion, device)
        val_metrics, val_diff = evaluate(model, val_loader, device, "val")

        print(f"Epoch {epoch+1}/{epochs}  loss={loss:.4f}  "
              f"val_acc={val_metrics['acc']:.4f}  val_f1={val_metrics['f1']:.4f}  "
              f"val_auc={val_metrics['auc']:.4f}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}

        print("-" * 60)

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, test_diff = evaluate(model, test_loader, device, "test")

    print("\n=== Final Test Results ===")
    print(f"Acc={test_metrics['acc']:.4f}  F1={test_metrics['f1']:.4f}  "
          f"AUC={test_metrics['auc']:.4f}")

    print("\n--- Per-Difficulty Breakdown ---")
    for diff in ["easy", "medium", "hard"]:
        if diff in test_diff:
            d = test_diff[diff]
            print(f"  {diff:6s}  n={d['n']:4d}  acc={d['acc']:.4f}  "
                  f"f1={d['f1']:.4f}  auc={d['auc']:.4f}")

    return model, test_metrics, test_diff


if __name__ == "__main__":
    run_training("app/data_pipeline/data/semantic/semantic_multiturn_v6.jsonl")