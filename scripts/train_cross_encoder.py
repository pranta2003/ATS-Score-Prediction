"""Cross-Encoder Training and Evaluation Script for ATS Compatibility.

Supports:
- Scenarios: S1 (Cold Resume), S2 (Cold JD), S3 (Double Cold)
- Pilot mode (--pilot): runs 1 epoch and outputs timing/memory projection
- Smoke mode (--smoke): runs on tiny subset (<200 samples) in <60 seconds
- Outputs standard results schema: metrics.json, preds.parquet, timings.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from sklearn.model_selection import GroupShuffleSplit

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ats.adapters import get_adapter
from src.ats.evaluation import compute_classification_metrics, vectorized_cluster_bootstrap_acc_f1


LABEL_MAP = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}


class ResumeJDDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512):
        self.texts_resume = df["resume_text"].tolist()
        self.texts_jd = df["jd_text"].tolist()
        if df["label_class"].dtype == object or isinstance(df["label_class"].iloc[0], str):
            self.labels = df["label_class"].map(LABEL_MAP).astype(int).tolist()
        else:
            self.labels = df["label_class"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Cross-encoder formats resume and JD as sentence pair
        enc = self.tokenizer(
            self.texts_resume[idx],
            self.texts_jd[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation="longest_first",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    elapsed = time.time() - start_time
    return total_loss / max(1, len(dataloader)), elapsed


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    return y_true, y_pred, y_prob


def run_cross_encoder(
    scenario: str = "S3",
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_length: int = 512,
    pilot: bool = False,
    smoke: bool = False,
    seed: int = 42,
    output_dir: str = "results/cross_encoder",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Running Cross-Encoder ({model_name}) ===")
    print(f"Device: {device}, Scenario: {scenario}, Epochs: {1 if pilot else epochs}, Smoke: {smoke}")

    # Load dataset C5
    adapter = get_adapter("c5")
    df = adapter.load()

    if smoke:
        df = df.sample(n=min(120, len(df)), random_state=seed).reset_index(drop=True)
        epochs = 1
        batch_size = 8
        max_length = 128

    # Split
    if scenario == "S1":
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(gss.split(df, groups=df["resume_id"]))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        cluster_col = "resume_id"
    elif scenario == "S2":
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(gss.split(df, groups=df["jd_id"]))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        cluster_col = "jd_id"
    elif scenario == "S3":
        # Double cold split
        unique_resumes = df["resume_id"].drop_duplicates().sample(frac=0.2, random_state=seed)
        unique_jds = df["jd_id"].drop_duplicates().sample(frac=0.2, random_state=seed)
        test_mask = df["resume_id"].isin(unique_resumes) & df["jd_id"].isin(unique_jds)
        train_mask = (~df["resume_id"].isin(unique_resumes)) & (~df["jd_id"].isin(unique_jds))
        train_df = df[train_mask].reset_index(drop=True)
        test_df = df[test_mask].reset_index(drop=True)
        cluster_col = "jd_id"
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    # Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_classes = 3
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    ).to(device)

    train_ds = ResumeJDDataset(train_df, tokenizer, max_length=max_length)
    test_ds = ResumeJDDataset(test_df, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    num_training_steps = (1 if pilot else epochs) * len(train_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps)

    total_train_time = 0.0
    epoch_times = []
    run_epochs = 1 if pilot else epochs

    for epoch in range(run_epochs):
        loss, ep_time = train_epoch(model, train_loader, optimizer, scheduler, device)
        epoch_times.append(ep_time)
        total_train_time += ep_time
        print(f"Epoch {epoch + 1}/{run_epochs} - Loss: {loss:.4f} - Time: {ep_time:.2f}s")

    # Evaluate
    eval_start = time.time()
    y_true, y_pred, y_prob = evaluate(model, test_loader, device)
    eval_time = time.time() - eval_start

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)

    # Cluster bootstrap CI
    cluster_ids = test_df[cluster_col].values
    ci_acc, ci_f1 = vectorized_cluster_bootstrap_acc_f1(y_true, y_pred, cluster_ids, n_boot=200 if smoke else 1000)
    metrics["ci_95_accuracy"] = ci_acc
    metrics["ci_95_macro_f1"] = ci_f1

    projection = {}
    if pilot:
        projection["1_epoch_seconds"] = epoch_times[0]
        projection["projected_3_epochs_seconds"] = epoch_times[0] * 3
        projection["device"] = str(device)
        print(f"[PILOT PROJECTION] 1 Epoch: {epoch_times[0]:.2f}s | Projected 3 Epochs: {epoch_times[0]*3:.2f}s")

    # Save artifacts
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(out_path / "timings.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_train_time_s": total_train_time,
            "eval_time_s": eval_time,
            "epoch_times_s": epoch_times,
            "pilot_projection": projection,
        }, f, indent=2)

    preds_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "prob_0": y_prob[:, 0],
        "prob_1": y_prob[:, 1],
        "prob_2": y_prob[:, 2],
        "cluster_id": cluster_ids,
    })
    preds_df.to_parquet(out_path / "preds.parquet", index=False)
    print(f"[DONE] Accuracy: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}, Macro-AUC: {metrics['macro_roc_auc']:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["S1", "S2", "S3"], default="S3")
    parser.add_argument("--model_name", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output_dir", default="results/cross_encoder")
    args = parser.parse_args()

    run_cross_encoder(
        scenario=args.scenario,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pilot=args.pilot,
        smoke=args.smoke,
        output_dir=args.output_dir,
    )
