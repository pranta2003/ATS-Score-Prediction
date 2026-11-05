import os
import sys
import json
import urllib.request
import platform
import psutil

def check_internet():
    status = {}
    test_urls = {
        "pypi": "https://pypi.org",
        "huggingface": "https://huggingface.co",
        "google": "https://www.google.com"
    }
    for name, url in test_urls.items():
        try:
            req = urllib.request.urlopen(url, timeout=5)
            status[name] = {"reachable": True, "code": req.getcode()}
        except Exception as e:
            status[name] = {"reachable": False, "error": str(e)}
    return status

def get_pkg_version(pkg_name):
    try:
        mod = __import__(pkg_name)
        return getattr(mod, "__version__", "installed (no __version__)")
    except ImportError:
        return "NOT_INSTALLED"
    except Exception as e:
        return f"ERROR: {e}"

def run_probe():
    info = {}
    info["python_version"] = platform.python_version()
    info["os"] = platform.platform()
    info["cpu_count"] = os.cpu_count()
    vm = psutil.virtual_memory()
    info["ram_total_gb"] = round(vm.total / (1024**3), 2)
    info["ram_available_gb"] = round(vm.available / (1024**3), 2)

    # GPU
    gpu_info = {"available": False}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["device_count"] = torch.cuda.device_count()
            gpu_info["device_name"] = torch.cuda.get_device_name(0)
            gpu_info["cuda_version"] = torch.version.cuda
    except Exception as e:
        gpu_info["error"] = str(e)
    info["gpu"] = gpu_info

    # Internet check
    info["internet"] = check_internet()

    # Packages
    packages = [
        "torch",
        "transformers",
        "sentence_transformers",
        "lightgbm",
        "xgboost",
        "sklearn",
        "scipy",
        "numpy",
        "pandas",
        "optuna",
        "datasets",
        "accelerate",
    ]
    pkg_status = {}
    for pkg in packages:
        pkg_status[pkg] = get_pkg_version(pkg)
    info["packages"] = pkg_status

    # Kaggle environment variables
    kaggle_env = {k: v for k, v in os.environ.items() if "KAGGLE" in k.upper()}
    info["kaggle_env_keys"] = list(kaggle_env.keys())

    print("=" * 60)
    print("           KAGGLE ENVIRONMENT PROBE RESULTS")
    print("=" * 60)
    print(json.dumps(info, indent=2))
    print("=" * 60)

    # Save output to current directory
    with open("probe_results.json", "w") as f:
        json.dump(info, f, indent=2)

if __name__ == "__main__":
    run_probe()

