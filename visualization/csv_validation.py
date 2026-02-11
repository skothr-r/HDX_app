"""
Shared CSV validation for the visualization pipeline.
Checks that required columns exist, key fields are complete and in sensible ranges,
and we are not passing through corrupted or random data.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Run-wide summary: list of (kind, message) where kind is 'warning' or 'error'
_run_summary: List[Tuple[str, str]] = []


def record_run_warning(message: str) -> None:
    """Record a warning for the end-of-run summary."""
    _run_summary.append(("warning", message))


def record_run_error(message: str) -> None:
    """Record an error for the end-of-run summary."""
    _run_summary.append(("error", message))


def get_run_summary() -> List[Tuple[str, str]]:
    """Return and clear the run summary (so the next run starts fresh)."""
    global _run_summary
    out = list(_run_summary)
    _run_summary = []
    return out


def debug_breakpoint(label: str = "") -> None:
    """If COMET_DEBUG=1, trigger breakpoint for debugging."""
    if os.environ.get("COMET_DEBUG", "").strip() in ("1", "true", "yes"):
        breakpoint()  # noqa: B007

# Standard amino acids (one-letter)
AA_SET = set("ACDEFGHIKLMNPQRSTVWY")

# Reasonable numeric ranges for MS data
MZ_MIN, MZ_MAX = 100.0, 10000.0
CHARGE_MIN, CHARGE_MAX = 1, 10
RT_SEC_MIN, RT_SEC_MAX = 0.0, 1e6  # retention time seconds
QVALUE_MIN, QVALUE_MAX = 0.0, 1.0
TOTAL_AREA_MIN = 0.0  # no upper bound
APEX_INTENSITY_MIN = 0.0


def _norm_str(s: Any) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return str(s).strip()


def _safe_float(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _safe_int(x: Any) -> Optional[int]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def validate_required_columns(
    df: pd.DataFrame,
    required: Sequence[str],
    source_name: str = "CSV",
) -> list[str]:
    """Check that all required columns exist. Returns list of issue messages."""
    issues = []
    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(f"[{source_name}] Missing required columns: {missing}")
    return issues


def validate_peptide_identifiers(
    df: pd.DataFrame,
    source_name: str = "CSV",
    require_modifications: bool = False,
) -> list[str]:
    """Validate plain_peptide, charge, and optionally modifications. Returns list of issues."""
    issues = []
    if "plain_peptide" not in df.columns:
        issues.append(f"[{source_name}] No column 'plain_peptide'")
        return issues
    if "charge" not in df.columns:
        issues.append(f"[{source_name}] No column 'charge'")
        return issues

    # plain_peptide: non-empty, only valid AAs (allow lowercase)
    seq_col = df["plain_peptide"]
    null_or_empty = seq_col.isna() | (seq_col.astype(str).str.strip() == "")
    if null_or_empty.any():
        n = null_or_empty.sum()
        issues.append(f"[{source_name}] plain_peptide null or empty in {n} row(s)")
    # Check for invalid characters in non-empty sequences
    def bad_seq(s):
        t = _norm_str(s)
        if not t:
            return False
        return bool(set(t.upper()) - AA_SET)
    bad = df["plain_peptide"].apply(bad_seq)
    if bad.any():
        n = bad.sum()
        issues.append(f"[{source_name}] plain_peptide has invalid characters in {n} row(s)")

    # charge: integer in [CHARGE_MIN, CHARGE_MAX]
    charge_vals = df["charge"].apply(_safe_int)
    bad_charge = charge_vals.isna() | (charge_vals < CHARGE_MIN) | (charge_vals > CHARGE_MAX)
    if bad_charge.any():
        n = bad_charge.sum()
        issues.append(f"[{source_name}] charge out of range [{CHARGE_MIN},{CHARGE_MAX}] or non-numeric in {n} row(s)")

    if require_modifications and "modifications" in df.columns:
        # modifications: can be '-', empty, or a string; warn if all null
        if df["modifications"].isna().all():
            issues.append(f"[{source_name}] All 'modifications' are null")

    return issues


def validate_numeric_ranges(
    df: pd.DataFrame,
    source_name: str = "CSV",
    checks: Optional[dict[str, tuple[Optional[float], Optional[float]]]] = None,
) -> list[str]:
    """Validate numeric columns are within (min, max) inclusive. None means no bound. Returns list of issues."""
    issues = []
    default_checks = {
        "mz": (MZ_MIN, MZ_MAX),
        "charge": (float(CHARGE_MIN), float(CHARGE_MAX)),
        "percolator_qvalue": (QVALUE_MIN, QVALUE_MAX),
        "q-value": (QVALUE_MIN, QVALUE_MAX),
        "qvalue": (QVALUE_MIN, QVALUE_MAX),
        "percolator_PEP": (QVALUE_MIN, QVALUE_MAX),
        "PEP": (QVALUE_MIN, QVALUE_MAX),
        "MS2_retention_time_sec": (RT_SEC_MIN, RT_SEC_MAX),
        "MS1_retention_time_sec": (RT_SEC_MIN, RT_SEC_MAX),
        "detected_peak_min_rt": (RT_SEC_MIN, RT_SEC_MAX),
        "detected_peak_max_rt": (RT_SEC_MIN, RT_SEC_MAX),
        "collection_min_rt": (RT_SEC_MIN, RT_SEC_MAX),
        "collection_max_rt": (RT_SEC_MIN, RT_SEC_MAX),
        "apex_rt": (RT_SEC_MIN, RT_SEC_MAX),
        "total_area": (TOTAL_AREA_MIN, None),
        "apex_intensity": (APEX_INTENSITY_MIN, None),
        "Selected": (0.0, 1.0),
    }
    to_check = dict(default_checks)
    if checks:
        to_check.update(checks)
    for col, (lo, hi) in to_check.items():
        if col not in df.columns:
            continue
        vals = df[col].apply(_safe_float)
        valid = vals.notna()
        if lo is not None:
            out = valid & (vals < lo)
            if out.any():
                n = out.sum()
                issues.append(f"[{source_name}] {col} < {lo} in {n} row(s)")
        if hi is not None:
            out = valid & (vals > hi)
            if out.any():
                n = out.sum()
                issues.append(f"[{source_name}] {col} > {hi} in {n} row(s)")
    return issues


def validate_no_duplicate_identifier_rows(
    df: pd.DataFrame,
    key_columns: Sequence[str] = ("plain_peptide", "charge", "modifications"),
    source_name: str = "CSV",
) -> list[str]:
    """Warn if key columns have duplicate rows (optional check for deduplicated CSVs)."""
    issues = []
    missing = [c for c in key_columns if c not in df.columns]
    if missing:
        return issues
    keys = list(key_columns)
    dupes = df.duplicated(subset=keys, keep=False)
    if dupes.any():
        n_dup = dupes.sum()
        n_unique = df.drop_duplicates(subset=keys).shape[0]
        issues.append(f"[{source_name}] Duplicate (plain_peptide, charge, mods) in {n_dup} row(s); {n_unique} unique peptides")
    return issues


def validate_peak_windows_columns(
    df: pd.DataFrame,
    source_name: str = "CSV",
) -> list[str]:
    """Validate a peak windows or windows_integration CSV has expected window columns and sensible RTs."""
    issues = []
    expected = [
        "detected_peak_min_rt",
        "detected_peak_max_rt",
        "collection_min_rt",
        "collection_max_rt",
    ]
    for c in expected:
        if c not in df.columns:
            issues.append(f"[{source_name}] Expected window column missing: {c}")
    if issues:
        return issues
    # min_rt should be <= max_rt where both present
    if "detected_peak_min_rt" in df.columns and "detected_peak_max_rt" in df.columns:
        lo = df["detected_peak_min_rt"].apply(_safe_float)
        hi = df["detected_peak_max_rt"].apply(_safe_float)
        both = lo.notna() & hi.notna()
        bad = both & (lo > hi)
        if bad.any():
            n = bad.sum()
            issues.append(f"[{source_name}] detected_peak_min_rt > detected_peak_max_rt in {n} row(s)")
    if "collection_min_rt" in df.columns and "collection_max_rt" in df.columns:
        lo = df["collection_min_rt"].apply(_safe_float)
        hi = df["collection_max_rt"].apply(_safe_float)
        both = lo.notna() & hi.notna()
        bad = both & (lo > hi)
        if bad.any():
            n = bad.sum()
            issues.append(f"[{source_name}] collection_min_rt > collection_max_rt in {n} row(s)")
    return issues


def validate_dataframe_sanity(
    df: pd.DataFrame,
    source_name: str = "CSV",
    min_rows: int = 0,
    max_rows: int = 10_000_000,
) -> list[str]:
    """Basic sanity: row count, no completely empty dataframe."""
    issues = []
    if df is None or not isinstance(df, pd.DataFrame):
        issues.append(f"[{source_name}] Not a DataFrame")
        return issues
    n = len(df)
    if n < min_rows:
        issues.append(f"[{source_name}] Row count {n} < minimum {min_rows}")
    if n > max_rows:
        issues.append(f"[{source_name}] Row count {n} > maximum {max_rows}")
    if n > 0 and df.shape[1] == 0:
        issues.append(f"[{source_name}] DataFrame has rows but no columns")
    return issues


def run_validation(
    df: pd.DataFrame,
    source_name: str = "CSV",
    *,
    required_columns: Optional[Sequence[str]] = None,
    validate_identifiers: bool = True,
    validate_numeric: bool = True,
    validate_windows: bool = False,
    validate_duplicates: bool = False,
    strict: bool = False,
) -> list[str]:
    """
    Run a standard set of validations and return all issue messages.
    If strict=True and there are issues, raises ValueError after collecting all issues.
    """
    all_issues = []

    all_issues.extend(validate_dataframe_sanity(df, source_name))

    if required_columns:
        all_issues.extend(validate_required_columns(df, required_columns, source_name))

    if validate_identifiers:
        all_issues.extend(
            validate_peptide_identifiers(df, source_name, require_modifications=False)
        )

    if validate_numeric:
        all_issues.extend(validate_numeric_ranges(df, source_name))

    if validate_windows:
        all_issues.extend(validate_peak_windows_columns(df, source_name))

    if validate_duplicates:
        all_issues.extend(validate_no_duplicate_identifier_rows(df, source_name=source_name))

    if strict and all_issues:
        raise ValueError("CSV validation failed:\n  " + "\n  ".join(all_issues))
    return all_issues


def log_validation_issues(
    issues: Sequence[str],
    source_name: str = "CSV",
    level: str = "warning",
) -> None:
    """Log each issue at the given level (warning, error, info). Record for end-of-run summary."""
    if not issues:
        return
    log_fn: Callable[[str], None] = getattr(logger, level, logger.warning)
    for msg in issues:
        log_fn(msg)
        record_run_warning(msg)
    # Also print so users see without configuring logging
    if issues:
        print(f"[validation] {source_name}: {len(issues)} check(s) reported")
        for msg in issues[:10]:
            print(f"  - {msg}")
        if len(issues) > 10:
            print(f"  - ... and {len(issues) - 10} more")


def validate_and_log(
    df: pd.DataFrame,
    source_name: str = "CSV",
    **kwargs: Any,
) -> list[str]:
    """Run run_validation and log any issues. Returns list of issues."""
    issues = run_validation(df, source_name, **kwargs)
    log_validation_issues(issues, source_name=source_name)
    return issues
