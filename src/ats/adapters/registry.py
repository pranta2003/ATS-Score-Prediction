from typing import Dict, Type
from .base import BaseDatasetAdapter
from .control import ControlAdapter
from .c1_fit import C1FitAdapter
from .c2_details import C2DetailsAdapter
from .c3_matching import C3MatchingAdapter
from .c4_recruitment import C4RecruitmentAdapter
from .c5_fit_original import C5FitOriginalAdapter

ADAPTER_MAP: Dict[str, Type[BaseDatasetAdapter]] = {
    "control": ControlAdapter,
    "0xnbk/resume-ats-score-v1-en": ControlAdapter,
    "c1": C1FitAdapter,
    "harsh72877287/job_resume_fit": C1FitAdapter,
    "c2": C2DetailsAdapter,
    "netsol/resume-score-details": C2DetailsAdapter,
    "c3": C3MatchingAdapter,
    "rahulrajpu676/job-resume-matching": C3MatchingAdapter,
    "c4": C4RecruitmentAdapter,
    "surendra365/recruitment-dataset": C4RecruitmentAdapter,
    "c5": C5FitOriginalAdapter,
    "cnamuangtoun/resume-job-description-fit": C5FitOriginalAdapter,
}


def list_adapters():
    """List available adapter identifiers."""
    return sorted(list(ADAPTER_MAP.keys()))


def get_adapter(name: str, **kwargs) -> BaseDatasetAdapter:
    """Factory to get an instantiated dataset adapter by name or alias."""
    name_clean = name.strip().lower()
    for k, cls in ADAPTER_MAP.items():
        if k.lower() == name_clean:
            return cls(**kwargs)
    raise KeyError(f"Adapter '{name}' not found. Available: {list_adapters()}")

