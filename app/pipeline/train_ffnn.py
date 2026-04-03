import torch
import torch.nn as nn
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from app.models.embedding import EmbeddingModel
from app.models.ff_nn_classifier import (
    BottleneckFFNN,
    ParallelMultiPathFFNN,
    FFNNClassifier,
)
from app.utils.utils import create_dataloaders

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

def run_training(data_path: str, model_type: str = "model3", epochs: int = 5, batch_size: int = 16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training architecture: {model_type}")

    train_loader, val_loader, test_loader = create_dataloaders(
        data_path, batch_size=batch_size
    )

    model = FullModel(model_type=model_type, pooling="mean").to(device)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = 0.0
    best_state = None

    for epoch in range(epochs):
        loss = train(model, train_loader, optimizer, criterion, device)
        val_metrics, _ = evaluate(model, val_loader, device, "val")

        print(
            f"Epoch {epoch+1}/{epochs}  loss={loss:.4f}  "
            f"Validation Accuracy={val_metrics['acc']:.4f}  "
            f"Validation F1={val_metrics['f1']:.4f}  "
            f"Validation AUC={val_metrics['auc']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print("-" * 60)

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, test_diff = evaluate(model, test_loader, device, "test")

    print("\nFinal Test Results")
    print(f"Acc={test_metrics['acc']:.4f}  F1={test_metrics['f1']:.4f}  AUC={test_metrics['auc']:.4f}")

    return model, test_metrics, test_diff

if __name__ == "__main__":
    data_path = "app/data_pipeline/data/semantic/semantic_multiturn_v7.jsonl"

    results = {}
    for model_type in ["model1", "model2", "model3"]:
        print("\n" + "=" * 80)
        print(f"Running {model_type}")
        print("=" * 80)

        _, test_metrics, test_diff = run_training(
            data_path=data_path,
            model_type=model_type,
            epochs=10,
            batch_size=16,
        )

        results[model_type] = {
            "overall": test_metrics,
            "by_difficulty": test_diff,
        }

    print("\n\nFINAL EVALUATION SUMMARY:")
    for model_type, res in results.items():
        m = res["overall"]
        print(
            f"{model_type}: "
            f"Acc={m['acc']:.4f}  "
            f"F1={m['f1']:.4f}  "
            f"AUC={m['auc']:.4f}"
        )