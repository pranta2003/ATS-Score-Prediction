import os
import sys
import time
import json
import yaml
import shutil
import base64
import tarfile
import io
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment import get_git_commit, compute_run_id


def get_kaggle_username():
    """Retrieve Kaggle username from environment or default."""
    return os.environ.get("KAGGLE_USERNAME", "prantakumerpandit390")


def create_code_payload(root_dir: Path) -> str:
    """Package src/ and scripts/ directories into a base64 gzipped tar string."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        src_dir = root_dir / "src"
        if src_dir.exists():
            tar.add(str(src_dir), arcname="src")
        scripts_dir = root_dir / "scripts"
        if scripts_dir.exists():
            tar.add(str(scripts_dir), arcname="scripts")

    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_driver_code(config_dict: dict, code_payload_b64: str) -> str:
    """Generate self-contained Python script to execute inside Kaggle container."""
    config_yaml_str = yaml.dump(config_dict)
    script_target = config_dict.get("script", "scripts/run_experiment.py")
    script_arg = f"--config job_config.yaml" if "run_experiment.py" in script_target else ""

    driver = f'''# Auto-generated ATS Experiment Runner Driver for Kaggle
import os
import sys
import shutil
import base64
import tarfile
import io
import yaml
import subprocess
from pathlib import Path

print("=" * 60)
print("   ATS KAGGLE EXPERIMENT DRIVER STARTING")
print("=" * 60)

# 1. Unpack code payload
payload_b64 = """{code_payload_b64}"""
tar_data = base64.b64decode(payload_b64)
with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r:gz") as tar:
    tar.extractall(".")

print("[OK] Unpacked src/ and scripts/ directories.")

# 2. Write config.yaml
config_str = """{config_yaml_str}"""
with open("job_config.yaml", "w", encoding="utf-8") as f:
    f.write(config_str)

# 3. Execute experiment runner
cmd = [sys.executable, "{script_target}"]
if "{script_arg}":
    cmd.extend("{script_arg}".split())

print(f"[INFO] Invoking: {{' '.join(cmd)}}", flush=True)
ret = subprocess.run(cmd)
if ret.returncode != 0:
    print(f"[FATAL] Script exited with code {{ret.returncode}}", flush=True)
    sys.exit(ret.returncode)

# 4. Copy results to /kaggle/working so Kaggle captures them as output artifacts
results_dir = Path("results")
if results_dir.exists():
    for p in results_dir.glob("**/*"):
        if p.is_file():
            shutil.copy(p, p.name)

reports_dir = Path("reports")
if reports_dir.exists():
    for p in reports_dir.glob("*"):
        if p.is_file():
            shutil.copy(p, p.name)

print("=" * 60)
print("   ATS KAGGLE EXPERIMENT DRIVER COMPLETED SUCCESSFULLY")
print("=" * 60)
'''
    return driver


def prepare_kernel_bundle(
    config_path: Path,
    device: str = "cpu",
    output_dir: Path = None,
    username: str = None,
) -> dict:
    """Prepare metadata and driver script for Kaggle kernel submission."""
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    commit = get_git_commit()
    run_id = compute_run_id(config_dict, commit)
    if username is None:
        username = get_kaggle_username()

    clean_slug = f"ats-{run_id.replace('_', '-')}"[:45].lower()
    clean_title = clean_slug.replace("-", " ").title()

    if output_dir is None:
        output_dir = PROJECT_ROOT / ".kaggle_kernels" / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "id": f"{username}/{clean_slug}",
        "title": clean_title,
        "code_file": "driver.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true" if device.lower() == "gpu" else "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }

    with open(output_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    code_payload = create_code_payload(PROJECT_ROOT)
    driver_code = generate_driver_code(config_dict, code_payload)

    with open(output_dir / "driver.py", "w", encoding="utf-8") as f:
        f.write(driver_code)

    return {
        "run_id": run_id,
        "kernel_slug": f"{username}/{clean_slug}",
        "kernel_dir": output_dir,
        "metadata": metadata,
    }


def validate_results_dir(target_dir: Path) -> dict:
    """Validate that metrics.json or j1_sweep_results.json exist in result directory."""
    metrics_file = target_dir / "metrics.json"
    sweep_file = target_dir / "j1_sweep_results.json"

    if sweep_file.exists():
        with open(sweep_file, "r", encoding="utf-8") as f:
            return json.load(f)

    if not metrics_file.exists():
        raise FileNotFoundError(f"Missing metrics.json in {target_dir}")

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    required_keys = ["accuracy", "macro_f1"]
    for k in required_keys:
        if k not in metrics:
            raise KeyError(f"Missing required metric key: {k} in {metrics_file}")

    return metrics


def run_kaggle_job(config_path: str, device: str = "cpu", wait_seconds: int = 180, dry_run: bool = False):
    """Orchestrate Kaggle job submission, waiting rule, output retrieval and validation."""
    bundle = prepare_kernel_bundle(Path(config_path), device=device)
    run_id = bundle["run_id"]
    kernel_slug = bundle["kernel_slug"]
    kernel_dir = bundle["kernel_dir"]

    print(f"Prepared Kaggle Kernel Bundle:")
    print(f"  Slug:    {kernel_slug}")
    print(f"  Device:  {device.upper()}")
    print(f"  Dir:     {kernel_dir}")

    if dry_run:
        print("[DRY RUN] Bundle prepared successfully. Exiting without pushing.")
        return bundle

    # 1. Push kernel
    print(f"\n[1/3] Pushing kernel to Kaggle...")
    push_res = subprocess.run(["kaggle", "kernels", "push", "-p", str(kernel_dir)], capture_output=True, text=True, shell=True)
    if push_res.returncode != 0:
        raise RuntimeError(f"Failed to push kernel:\n{push_res.stderr}\n{push_res.stdout}")
    print(push_res.stdout.strip())

    # 2. Wait following the waiting rule
    print(f"\n[2/3] Waiting for Kaggle kernel to execute (checking every {wait_seconds}s)...")
    status = ""
    while True:
        time.sleep(wait_seconds)
        stat_res = subprocess.run(["kaggle", "kernels", "status", kernel_slug], capture_output=True, text=True, shell=True)
        out = stat_res.stdout.strip()
        print(f"Status check: {out}")
        if "COMPLETE" in out:
            status = "COMPLETE"
            break
        elif "FAILED" in out or "ERROR" in out:
            status = "FAILED"
            break

    if status != "COMPLETE":
        # Pull logs to inspect failure
        fail_dir = PROJECT_ROOT / "results" / f"failed_{run_id}"
        fail_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["kaggle", "kernels", "output", kernel_slug, "-p", str(fail_dir)], shell=True)
        raise RuntimeError(f"Kaggle kernel execution failed. Artifacts downloaded to {fail_dir}")

    # 3. Pull outputs to results/<run_id>/
    out_dir = PROJECT_ROOT / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[3/3] Pulling kernel outputs to {out_dir}...")
    dl_res = subprocess.run(["kaggle", "kernels", "output", kernel_slug, "-p", str(out_dir)], capture_output=True, text=True, shell=True)
    print(dl_res.stdout.strip())

    # Validate output schema
    metrics = validate_results_dir(out_dir)
    print("\n" + "=" * 60)
    print(f"   JOB {run_id} SUCCEEDED ON KAGGLE ({device.upper()})")
    print("=" * 60)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Kaggle Bridge Orchestrator")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Target compute device")
    parser.add_argument("--wait-seconds", type=int, default=180, help="Wait seconds per check")
    parser.add_argument("--dry-run", action="store_true", help="Prepare bundle without submitting to Kaggle")
    args = parser.parse_args()

    run_kaggle_job(args.config, device=args.device, wait_seconds=args.wait_seconds, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
