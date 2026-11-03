import pytest
import pandas as pd
import numpy as np
from src.ats.adapters.base import (
    STANDARD_COLUMNS,
    normalize_text,
    compute_sha1,
)
from src.ats.adapters import get_adapter, list_adapters


def test_text_normalization_and_sha1():
    t1 = "  Senior   Software  Engineer \n\t "
    t2 = "senior software engineer"
    assert normalize_text(t1) == t2
    assert compute_sha1(t1) == compute_sha1(t2)
    assert len(compute_sha1(t1)) == 40


@pytest.mark.parametrize("adapter_name", ["control", "c1", "c2", "c3", "c4", "c5"])
def test_adapter_schema_and_integrity(adapter_name):
    # For large datasets like c3, limit rows to 100 in unit test for speed
    kwargs = {}
    if adapter_name == "c3":
        kwargs["max_rows"] = 100

    adapter = get_adapter(adapter_name, **kwargs)
    df = adapter.load()

    # 1. Schema check
    assert list(df.columns) == STANDARD_COLUMNS, f"Columns mismatch for {adapter_name}"
    assert len(df) > 0, f"Adapter {adapter_name} returned 0 rows"

    # 2. No empty texts
    assert (df["resume_text"].astype(str).str.strip() == "").sum() == 0, f"Empty resume_text in {adapter_name}"
    assert (df["jd_text"].astype(str).str.strip() == "").sum() == 0, f"Empty jd_text in {adapter_name}"
    assert df["resume_text"].isnull().sum() == 0
    assert df["jd_text"].isnull().sum() == 0

    # 3. Deterministic SHA1 check
    for idx, row in df.head(50).iterrows():
        expected_res_id = compute_sha1(row["resume_text"])
        expected_jd_id = compute_sha1(row["jd_text"])
        assert row["resume_id"] == expected_res_id, f"resume_id mismatch at row {idx}"
        assert row["jd_id"] == expected_jd_id, f"jd_id mismatch at row {idx}"

    # 4. Source dataset check
    assert (df["source_dataset"].astype(str).str.strip() == "").sum() == 0
