"""Rigorous Cross-Encoder vs Bi-Encoder Study on C5 Dataset.

Enforces:
1. Exact identical split ID files (splits/c5_{scenario}_seed{s}.json) for both CE and Bi-encoder.
2. 3 split seeds (42, 43, 44) evaluated under S2 (Cold JD) and S3 (Double Cold). Total 6 runs per model.
3. Paired difference per split: Delta = Metric_CE - Metric_Bi with paired cluster-bootstrap CI.
4. CE Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (max_length=512, truncation='longest_first', 3 epochs).
5. Bi-Encoder Model: LightGBM on T4 (lexical + dense chunk-pooled embeddings).
6. Comprehensive report and artifact generation in results/ce_vs_bi/.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import lightgbm as lgb
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ats.adapters import get_adapter
from src.ats.embeddings import build_pair_features
from src.ats.evaluation import (
    compute_classification_metrics,
    vectorized_cluster_bootstrap_acc_f1,
    vectorized_paired_cluster_bootstrap_diff,
)

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


def generate_and_save_splits(df: pd.DataFrame, splits_dir: Path) -> dict:
    """Generate deterministic splits for S2 and S3 across seeds 42, 43, 44."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    seeds = [42, 43, 44]
    scenarios = ["S2", "S3"]

    for scen in scenarios:
        for seed in seeds:
            split_file = splits_dir / f"c5_{scen}_seed{seed}.json"
            if split_file.exists():
                with open(split_file, "r", encoding="utf-8") as f:
                    splits[f"{scen}_seed{seed}"] = json.load(f)
                continue

            if scen == "S2":
                gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                tr_idx, te_idx = next(gss.split(df, groups=df["jd_id"]))
                cluster_col = "jd_id"
            elif scen == "S3":
                unique_res = df["resume_id"].drop_duplicates().sample(frac=0.2, random_state=seed)
                unique_jds = df["jd_id"].drop_duplicates().sample(frac=0.2, random_state=seed)
                te_mask = df["resume_id"].isin(unique_res) & df["jd_id"].isin(unique_jds)
                tr_mask = (~df["resume_id"].isin(unique_res)) & (~df["jd_id"].isin(unique_jds))
                tr_idx = np.where(tr_mask)[0]
                te_idx = np.where(te_mask)[0]
                cluster_col = "jd_id"
            else:
                raise ValueError(f"Unknown scenario {scen}")

            split_data = {
                "scenario": scen,
                "seed": seed,
                "cluster_col": cluster_col,
                "train_indices": [int(i) for i in tr_idx],
                "test_indices": [int(i) for i in te_idx],
                "n_train": len(tr_idx),
                "n_test": len(te_idx),
            }
            with open(split_file, "w", encoding="utf-8") as f:
                json.dump(split_data, f, indent=2)
            splits[f"{scen}_seed{seed}"] = split_data
            print(f"Generated split {scen} seed {seed}: {len(tr_idx)} train, {len(te_idx)} test")

    return splits


def train_ce_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    t0 = time.time()
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
    return total_loss / max(1, len(dataloader)), time.time() - t0


@torch.no_grad()
def evaluate_ce(model, dataloader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_probs)


def run_cross_encoder_vs_biencoder(
    smoke: bool = False,
    output_dir: str = "results/ce_vs_bi",
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_length: int = 512,
):
    print("=== CROSS-ENCODER VS BI-ENCODER STUDY ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | CE Model: {model_name} | Epochs: {epochs} | Smoke: {smoke}")

    out_path = PROJECT_ROOT / output_dir
    out_path.mkdir(parents=True, exist_ok=True)
    splits_dir = PROJECT_ROOT / "splits"

    adapter = get_adapter("c5")
    df = adapter.load()
    if smoke:
        df = df.iloc[:100].reset_index(drop=True)
        splits_dir = out_path / "smoke_splits"
    if df["label_class"].dtype == object or isinstance(df["label_class"].iloc[0], str):
        y_all = df["label_class"].map(LABEL_MAP).astype(int).values
    else:
        y_all = df["label_class"].astype(int).values

    # Pre-build Bi-encoder T4 features (lexical + dense)
    print("\n--- Extracting / Loading Bi-Encoder T4 Features ---")
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2), stop_words="english")
    res_list = df["resume_text"].tolist()
    jd_list = df["jd_text"].tolist()
    tfidf.fit(res_list + jd_list)
    r_mat = tfidf.transform(res_list)
    j_mat = tfidf.transform(jd_list)
    cos_sim = np.asarray(r_mat.multiply(j_mat).sum(axis=1)).flatten()

    dense_feats = build_pair_features(df, dataset_name="c5", include_late_interaction=False, cache_dir="cache/embeddings")
    X_bi = np.hstack([dense_feats, cos_sim[:, None]])
    print(f"Bi-encoder feature matrix shape: {X_bi.shape}")

    splits = generate_and_save_splits(df, splits_dir)
    scenarios = ["S2"] if smoke else ["S2", "S3"]
    seeds = [42] if smoke else [42, 43, 44]

    study_results = {}
    preds_records = []

    for scen in scenarios:
        for seed in seeds:
            run_key = f"{scen}_seed{seed}"
            print(f"\n==========================================")
            print(f"Executing Run: {run_key}")
            print(f"==========================================")
            sp = splits[run_key]
            tr_idx = np.array(sp["train_indices"])
            te_idx = np.array(sp["test_indices"])
            cluster_col = sp["cluster_col"]

            if smoke:
                tr_idx = tr_idx[:16]
                te_idx = te_idx[:8]
                run_epochs = 1
                run_bs = 8
                run_len = 32
                n_boot = 10
            else:
                run_epochs = epochs
                run_bs = batch_size
                run_len = max_length
                n_boot = 1000

            df_tr = df.iloc[tr_idx].reset_index(drop=True)
            df_te = df.iloc[te_idx].reset_index(drop=True)
            y_tr = y_all[tr_idx]
            y_te = y_all[te_idx]
            clusters_te = df_te[cluster_col].values
            n_te = len(y_te)

            # 1. Baselines on this EXACT split
            maj_c = int(pd.Series(y_tr).mode()[0])
            maj_preds = np.full(n_te, maj_c, dtype=int)
            maj_acc = float(accuracy_score(y_te, maj_preds))
            maj_f1 = float(f1_score(y_te, maj_preds, average="macro", zero_division=0))

            global_prior = pd.Series(y_tr).value_counts(normalize=True).reindex(range(3), fill_value=0.0).values
            jd_priors = {jid: pd.Series(y_all[tr_idx][df_tr["jd_id"] == jid]).value_counts(normalize=True).reindex(range(3), fill_value=0.0).values for jid in df_tr["jd_id"].unique()}
            jdp_probs = np.array([jd_priors.get(jid, global_prior) for jid in df_te["jd_id"]])
            jdp_preds = np.argmax(jdp_probs, axis=1)
            jdp_acc = float(accuracy_score(y_te, jdp_preds))
            jdp_f1 = float(f1_score(y_te, jdp_preds, average="macro", zero_division=0))

            # 2. Bi-Encoder LightGBM
            print(f"Fitting Bi-Encoder LightGBM on {len(tr_idx)} pairs...")
            t_bi0 = time.time()
            clf_bi = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=3,
                n_estimators=20 if smoke else 300,
                max_depth=5,
                learning_rate=0.05,
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            )
            clf_bi.fit(X_bi[tr_idx], y_tr)
            bi_probs = clf_bi.predict_proba(X_bi[te_idx])
            bi_preds = np.argmax(bi_probs, axis=1)
            bi_train_time = time.time() - t_bi0
            bi_metrics = compute_classification_metrics(y_te, bi_preds, bi_probs)
            bi_ci_acc, bi_ci_f1 = vectorized_cluster_bootstrap_acc_f1(y_te, bi_preds, clusters_te, n_boot=n_boot)
            bi_metrics["ci_95_accuracy"] = bi_ci_acc
            bi_metrics["ci_95_macro_f1"] = bi_ci_f1
            bi_metrics["train_time_s"] = bi_train_time

            # 3. Cross-Encoder Transformer
            print(f"Fine-tuning Cross-Encoder ({model_name}) on {len(tr_idx)} pairs for {run_epochs} epochs...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model_ce = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=3,
                ignore_mismatched_sizes=True,
            ).to(device)

            train_ds = ResumeJDDataset(df_tr, tokenizer, max_length=run_len)
            test_ds = ResumeJDDataset(df_te, tokenizer, max_length=run_len)
            train_loader = DataLoader(train_ds, batch_size=run_bs, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=run_bs, shuffle=False)

            total_steps = run_epochs * len(train_loader)
            opt = torch.optim.AdamW(model_ce.parameters(), lr=lr, weight_decay=0.01)
            sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

            t_ce0 = time.time()
            for ep in range(run_epochs):
                ep_loss, ep_t = train_ce_epoch(model_ce, train_loader, opt, sched, device)
                print(f"  Epoch {ep+1}/{run_epochs} - Loss: {ep_loss:.4f} ({ep_t:.1f}s)")
            ce_train_time = time.time() - t_ce0

            y_te_ce, ce_preds, ce_probs = evaluate_ce(model_ce, test_loader, device)
            ce_metrics = compute_classification_metrics(y_te, ce_preds, ce_probs)
            ce_ci_acc, ce_ci_f1 = vectorized_cluster_bootstrap_acc_f1(y_te, ce_preds, clusters_te, n_boot=n_boot)
            ce_metrics["ci_95_accuracy"] = ce_ci_acc
            ce_metrics["ci_95_macro_f1"] = ce_ci_f1
            ce_metrics["train_time_s"] = ce_train_time

            # 4. Paired Delta & Bootstrap CI
            diff_acc_ci, diff_f1_ci = vectorized_paired_cluster_bootstrap_diff(
                y_te, ce_preds, bi_preds, clusters_te, n_boot=n_boot
            )
            delta_acc = ce_metrics["accuracy"] - bi_metrics["accuracy"]
            delta_f1 = ce_metrics["macro_f1"] - bi_metrics["macro_f1"]

            study_results[run_key] = {
                "scenario": scen,
                "seed": seed,
                "n_train": len(tr_idx),
                "n_test": n_te,
                "majority_baseline": {"accuracy": maj_acc, "macro_f1": maj_f1},
                "jd_prior_baseline": {"accuracy": jdp_acc, "macro_f1": jdp_f1},
                "bi_encoder": bi_metrics,
                "cross_encoder": ce_metrics,
                "paired_delta": {
                    "delta_accuracy": delta_acc,
                    "delta_accuracy_ci_95": diff_acc_ci,
                    "delta_macro_f1": delta_f1,
                    "delta_macro_f1_ci_95": diff_f1_ci,
                },
            }

            print(f"--> [RESULT {run_key}] Majority Acc: {maj_acc:.4f} | Bi-Encoder: {bi_metrics['accuracy']:.4f} | Cross-Encoder: {ce_metrics['accuracy']:.4f} | Delta Acc: {delta_acc:+.4f} (95% CI: [{diff_acc_ci[0]:+.4f}, {diff_acc_ci[1]:+.4f}])")

            # Collect predictions
            for i in range(n_te):
                preds_records.append({
                    "run_key": run_key,
                    "scenario": scen,
                    "seed": seed,
                    "test_idx": int(te_idx[i]),
                    "cluster_id": clusters_te[i],
                    "y_true": int(y_te[i]),
                    "bi_pred": int(bi_preds[i]),
                    "ce_pred": int(ce_preds[i]),
                    "ce_prob_0": float(ce_probs[i, 0]),
                    "ce_prob_1": float(ce_probs[i, 1]),
                    "ce_prob_2": float(ce_probs[i, 2]),
                })

    # Summary aggregations
    summary = {}
    for scen in scenarios:
        scen_runs = [study_results[f"{scen}_seed{s}"] for s in seeds]
        bi_accs = [r["bi_encoder"]["accuracy"] for r in scen_runs]
        ce_accs = [r["cross_encoder"]["accuracy"] for r in scen_runs]
        bi_f1s = [r["bi_encoder"]["macro_f1"] for r in scen_runs]
        ce_f1s = [r["cross_encoder"]["macro_f1"] for r in scen_runs]
        delta_accs = [r["paired_delta"]["delta_accuracy"] for r in scen_runs]
        delta_f1s = [r["paired_delta"]["delta_macro_f1"] for r in scen_runs]

        summary[scen] = {
            "bi_encoder_mean_acc": float(np.mean(bi_accs)),
            "bi_encoder_std_acc": float(np.std(bi_accs)),
            "cross_encoder_mean_acc": float(np.mean(ce_accs)),
            "cross_encoder_std_acc": float(np.std(ce_accs)),
            "mean_delta_acc": float(np.mean(delta_accs)),
            "bi_encoder_mean_f1": float(np.mean(bi_f1s)),
            "cross_encoder_mean_f1": float(np.mean(ce_f1s)),
            "mean_delta_f1": float(np.mean(delta_f1s)),
        }

    full_output = {
        "metadata": {
            "dataset": "c5",
            "model_ce": model_name,
            "model_bi": "LightGBM on T4 (lexical+dense)",
            "scenarios": scenarios,
            "seeds": seeds,
            "smoke": smoke,
        },
        "summary": summary,
        "runs": study_results,
    }

    with open(out_path / "study_results.json", "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    pd.DataFrame(preds_records).to_parquet(out_path / "study_predictions.parquet", index=False)
    print(f"\n[COMPLETE] Successfully saved study results to {out_path / 'study_results.json'}")
    return full_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_dir", default="results/ce_vs_bi")
    args = parser.parse_args()

    run_cross_encoder_vs_biencoder(
        smoke=args.smoke,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )

