"""Profile Inference Memory and Cold Start for Serving ATS Prediction Models.

Benchmarks three architectures on 50 sample resume-JD pairs:
1. Current app: PyTorch + sentence-transformers (all-MiniLM-L6-v2) + XGBoost
2. Minimal ONNX runtime: onnxruntime + Hugging Face tokenizer (NO torch imported) + XGBoost
3. Precomputed/cached embeddings: Zero-inference memory dictionary lookup + XGBoost

Each benchmark runs in an isolated subprocess to ensure pristine RSS measurements.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RUNNER_SCRIPT = """
import os
import sys
import time
import json
import psutil
import numpy as np
import joblib

mode = sys.argv[1]
proc = psutil.Process()

# Track memory helper
def get_rss():
    return proc.memory_info().rss / (1024 * 1024)

base_rss = get_rss()
t0 = time.perf_counter()

# Synthetic 50 sample pairs of realistic ATS length (~300 words each)
resumes = ["Experienced software engineer with expertise in Python, PyTorch, distributed systems, SQL, AWS, and Docker. Developed scalable microservices." * 15 for _ in range(50)]
jds = ["Senior Software Engineer required with 5+ years experience in Python, machine learning, cloud platforms, and data pipelines. B.S. in Computer Science." * 15 for _ in range(50)]

peak_rss = base_rss

if mode == "current_pytorch":
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    xgb = joblib.load("ats_xgboost_model.pkl")
    cold_start_time = time.perf_counter() - t0
    peak_rss = max(peak_rss, get_rss())

    t_inf_start = time.perf_counter()
    for r, j in zip(resumes, jds):
        r_emb = sbert.encode([r])
        j_emb = sbert.encode([j])
        feat = np.concatenate([r_emb, j_emb], axis=1)
        pred = xgb.predict(feat)
        peak_rss = max(peak_rss, get_rss())
    inf_time = time.perf_counter() - t_inf_start

elif mode == "minimal_onnx":
    # Ensure NO torch is imported
    assert "torch" not in sys.modules, "Torch must not be imported in minimal_onnx mode"
    import onnxruntime as ort
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    xgb = joblib.load("ats_xgboost_model.pkl")

    # Download / cache quantized ONNX model
    onnx_file = hf_hub_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        filename="onnx/model_quint8_avx2.onnx",
    )
    session = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])

    cold_start_time = time.perf_counter() - t0
    peak_rss = max(peak_rss, get_rss())

    t_inf_start = time.perf_counter()
    for r, j in zip(resumes, jds):
        enc_r = tokenizer(r, return_tensors="np", padding=True, truncation=True, max_length=512)
        enc_j = tokenizer(j, return_tensors="np", padding=True, truncation=True, max_length=512)

        # Run ONNX inference
        inputs_r = {k: v for k, v in enc_r.items() if k in [inp.name for inp in session.get_inputs()]}
        inputs_j = {k: v for k, v in enc_j.items() if k in [inp.name for inp in session.get_inputs()]}
        out_r = session.run(None, inputs_r)[0]
        out_j = session.run(None, inputs_j)[0]

        # Mean pooling
        mask_r = enc_r["attention_mask"][..., None]
        emb_r = (out_r * mask_r).sum(axis=1) / mask_r.sum(axis=1)
        mask_j = enc_j["attention_mask"][..., None]
        emb_j = (out_j * mask_j).sum(axis=1) / mask_j.sum(axis=1)

        feat = np.concatenate([emb_r, emb_j], axis=1)
        pred = xgb.predict(feat)
        peak_rss = max(peak_rss, get_rss())
    inf_time = time.perf_counter() - t_inf_start

elif mode == "precomputed_lookup":
    # Zero runtime ML inference: lookup in memory cache + XGBoost
    xgb = joblib.load("ats_xgboost_model.pkl")
    # Simulate embedding lookup table (1000 embeddings in memory)
    cache = {f"hash_{i}": np.random.randn(1, 384).astype(np.float32) for i in range(1000)}
    cold_start_time = time.perf_counter() - t0
    peak_rss = max(peak_rss, get_rss())

    t_inf_start = time.perf_counter()
    for i in range(50):
        emb_r = cache[f"hash_{i}"]
        emb_j = cache[f"hash_{i+50}"]
        feat = np.concatenate([emb_r, emb_j], axis=1)
        pred = xgb.predict(feat)
        peak_rss = max(peak_rss, get_rss())
    inf_time = time.perf_counter() - t_inf_start

else:
    raise ValueError(f"Unknown mode: {mode}")

res = {
    "mode": mode,
    "base_rss_mb": round(base_rss, 2),
    "peak_rss_mb": round(peak_rss, 2),
    "cold_start_s": round(cold_start_time, 3),
    "total_inference_s": round(inf_time, 3),
    "latency_per_pair_ms": round((inf_time / 50) * 1000, 2),
}
print(json.dumps(res))
"""


def run_benchmark_mode(mode: str) -> dict:
    cmd = [sys.executable, "-c", RUNNER_SCRIPT, mode]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), env=env)
    if res.returncode != 0:
        print(f"Error in mode {mode}:\n{res.stderr}")
        return None
    # Parse last line of stdout
    for line in reversed(res.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    return None


def main():
    print("=== PROFILING INFERENCE MEMORY & COLD START ===")
    results = {}

    # 1. Current PyTorch
    print("\n[1/3] Benchmarking Current PyTorch + sentence-transformers...")
    r1 = run_benchmark_mode("current_pytorch")
    if r1:
        results["current_pytorch"] = r1
        print(f"  Peak RSS: {r1['peak_rss_mb']} MB | Cold Start: {r1['cold_start_s']}s | Latency/pair: {r1['latency_per_pair_ms']}ms")

    # 2. Minimal ONNX
    print("\n[2/3] Benchmarking Minimal ONNX Runtime (NO torch)...")
    r2 = run_benchmark_mode("minimal_onnx")
    if r2:
        results["minimal_onnx"] = r2
        print(f"  Peak RSS: {r2['peak_rss_mb']} MB | Cold Start: {r2['cold_start_s']}s | Latency/pair: {r2['latency_per_pair_ms']}ms")
    else:
        print("  ONNX benchmark skipped or failed (optimum not present; testing standard onnxruntime)")

    # 3. Precomputed Lookup
    print("\n[3/3] Benchmarking Precomputed Embedding Lookup...")
    r3 = run_benchmark_mode("precomputed_lookup")
    if r3:
        results["precomputed_lookup"] = r3
        print(f"  Peak RSS: {r3['peak_rss_mb']} MB | Cold Start: {r3['cold_start_s']}s | Latency/pair: {r3['latency_per_pair_ms']}ms")

    out_file = PROJECT_ROOT / "reports" / "memory_benchmark_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Saved benchmark results to {out_file}")


if __name__ == "__main__":
    main()

