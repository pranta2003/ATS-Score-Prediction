import os
import json
import pytest
from pathlib import Path
from scripts.kaggle_run import (
    prepare_kernel_bundle,
    validate_results_dir,
    create_code_payload,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_prepare_kernel_bundle_cpu(tmp_path):
    """Verify that Kaggle kernel bundle generation creates valid metadata and driver script for CPU."""
    sample_config = PROJECT_ROOT / "configs" / "smoke_test.yaml"
    bundle = prepare_kernel_bundle(
        config_path=sample_config,
        device="cpu",
        output_dir=tmp_path / "cpu_run",
        username="testuser",
    )

    assert "run_id" in bundle
    assert bundle["kernel_slug"] == "testuser/ats-c5-smoke-test-cpu"
    
    meta_path = bundle["kernel_dir"] / "kernel-metadata.json"
    assert meta_path.exists()
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["id"] == bundle["kernel_slug"]
    assert meta["code_file"] == "driver.py"
    assert meta["enable_gpu"] == "false"
    assert meta["enable_internet"] == "true"
    assert meta["language"] == "python"

    driver_path = bundle["kernel_dir"] / "driver.py"
    assert driver_path.exists()
    driver_code = driver_path.read_text(encoding="utf-8")
    assert "run_experiment.py" in driver_code
    assert "ATS KAGGLE EXPERIMENT DRIVER" in driver_code


def test_prepare_kernel_bundle_gpu(tmp_path):
    """Verify that GPU flag is correctly propagated to metadata."""
    sample_config = PROJECT_ROOT / "configs" / "smoke_test.yaml"
    bundle = prepare_kernel_bundle(
        config_path=sample_config,
        device="gpu",
        output_dir=tmp_path / "gpu_run",
        username="testuser",
    )

    meta_path = bundle["kernel_dir"] / "kernel-metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["enable_gpu"] == "true"


def test_validate_results_dir_success(tmp_path):
    """Verify that result directory validator succeeds on valid schema."""
    valid_dir = tmp_path / "valid_results"
    valid_dir.mkdir()
    metrics = {
        "accuracy": 0.655,
        "macro_f1": 0.621,
        "macro_roc_auc": 0.829,
    }
    with open(valid_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f)

    loaded = validate_results_dir(valid_dir)
    assert loaded["accuracy"] == 0.655
    assert loaded["macro_f1"] == 0.621


def test_validate_results_dir_missing_file(tmp_path):
    """Verify that validator raises FileNotFoundError when metrics.json is absent."""
    empty_dir = tmp_path / "empty_results"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        validate_results_dir(empty_dir)


def test_validate_results_dir_missing_keys(tmp_path):
    """Verify that validator raises KeyError when required metrics are absent."""
    invalid_dir = tmp_path / "invalid_results"
    invalid_dir.mkdir()
    with open(invalid_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"loss": 0.5}, f)

    with pytest.raises(KeyError):
        validate_results_dir(invalid_dir)

