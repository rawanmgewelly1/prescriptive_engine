from __future__ import annotations

import json
import warnings
import os
import tempfile
import uuid
from importlib import metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd


# Expected project layout:
# project_root/
#   config.json
#   prescriptive_engine/engine.py
#   data/, models/, selected_features/, outputs/
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config.json"


class ConfigurationError(ValueError):
    """Raised when configuration is missing, contradictory, or unsafe to interpret."""


class StageBlocked(RuntimeError):
    """Raised when a stage cannot be run without making an unsupported assumption."""


@dataclass
class StageResult:
    name: str
    status: str
    details: str
    rows: int | None = None
    run_id: str = ""


def _require_object(cfg: dict[str, Any], key: str, *, default: dict[str, Any] | None = None) -> dict[str, Any]:
    value = cfg.get(key, default if default is not None else {})
    if not isinstance(value, dict):
        raise ConfigurationError(f"{key} must be a JSON object.")
    cfg[key] = value
    return value


def _require_plain_filename(value: Any, key: str, *, allow_empty: bool = False) -> str:
    if allow_empty and value in (None, ""):
        return ""
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{key} must be a non-empty filename string.")
    path = Path(value)
    if (path.is_absolute() or path.name != value or value in {".", "..", "_previous_runs"}
            or "/" in value or "\\" in value or "\x00" in value or value.strip() != value):
        raise ConfigurationError(f"{key} must be a plain, non-reserved filename inside its configured directory; got {value!r}.")
    return value


def _require_bool_map(cfg: dict[str, Any], key: str, names: tuple[str, ...]) -> dict[str, Any]:
    values = _require_object(cfg, key)
    unknown = set(values) - set(names)
    if unknown:
        raise ConfigurationError(f"{key} contains unsupported keys: {sorted(unknown)}")
    for name in names:
        if name in values and not isinstance(values[name], bool):
            raise ConfigurationError(f"{key}.{name} must be a boolean.")
    return values


def read_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load configuration, validate nested contracts, and resolve paths relative to config.json."""
    config_path = Path(path).resolve() if path else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid JSON in {config_path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ConfigurationError("Top-level config must be a JSON object.")

    root = config_path.parent
    for key in ("dataset_dir", "models_dir", "selected_features_dir", "output_dir"):
        if key not in cfg or not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ConfigurationError(f"Missing/non-string path setting: {key}")
        pth = Path(cfg[key])
        cfg[key] = str((root / pth).resolve() if not pth.is_absolute() else pth.resolve())
    output_path = Path(cfg["output_dir"]).resolve()
    if output_path == root.resolve() or output_path in root.resolve().parents:
        raise ConfigurationError("output_dir must not be the project root or an ancestor of the project root.")
    if output_path.exists() and not output_path.is_dir():
        raise ConfigurationError(f"output_dir exists but is not a directory: {output_path}")

    required = (
        "solar_dataset_file", "wind_dataset_file", "hydrogen_dataset_file", "fuel_cell_dataset_file",
        "solar_model_file", "wind_model_file", "hydrogen_model_file", "fuel_cell_model_file",
        "solar_target", "wind_target", "hydrogen_target", "solar_features", "wind_features",
        "hydrogen_features", "fuel_cell_features", "input_modes", "input_modes_verified",
        "candidate_generation", "decision_variables", "constraints", "outputs",
    )
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ConfigurationError(f"Missing required config keys: {missing}")

    components = ("solar", "wind", "hydrogen", "fuel_cell")
    # All file settings are confined to their configured directories; absolute and traversal paths are rejected.
    for key in (
        "solar_dataset_file", "wind_dataset_file", "hydrogen_dataset_file", "fuel_cell_dataset_file",
        "solar_model_file", "wind_model_file", "hydrogen_model_file", "fuel_cell_model_file",
    ):
        _require_plain_filename(cfg[key], key)
    for key in ("solar_scaler_file", "wind_scaler_file", "hydrogen_scaler_file", "fuel_cell_scaler_file"):
        if key in cfg:
            cfg[key] = _require_plain_filename(cfg[key], key, allow_empty=True)
    selected = _require_object(cfg, "selected_feature_files")
    unknown_selected = set(selected) - set(components)
    if unknown_selected:
        raise ConfigurationError(f"selected_feature_files contains unsupported keys: {sorted(unknown_selected)}")
    for key, value in selected.items():
        _require_plain_filename(value, f"selected_feature_files.{key}", allow_empty=True)

    feature_keys = ("solar_features", "wind_features", "hydrogen_features", "fuel_cell_features", "decision_variables")
    for key in ("solar_target", "wind_target", "hydrogen_target"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ConfigurationError(f"{key} must be a non-empty string.")
    for key in feature_keys:
        vals = cfg[key]
        if not isinstance(vals, list) or not vals or not all(isinstance(x, str) and x.strip() for x in vals):
            raise ConfigurationError(f"{key} must be a non-empty list of feature names.")
        if len(vals) != len(set(vals)):
            raise ConfigurationError(f"{key} contains duplicate feature names.")
        if any(x.strip() != x for x in vals):
            raise ConfigurationError(f"{key} feature names must not have leading/trailing whitespace.")

    modes = _require_object(cfg, "input_modes")
    unknown_modes = set(modes) - set(components)
    if unknown_modes:
        raise ConfigurationError(f"input_modes contains unsupported keys: {sorted(unknown_modes)}")
    for name in components:
        if modes.get(name) not in ("raw", "scaled"):
            raise ConfigurationError(f"input_modes.{name} must be 'raw' or 'scaled'.")
        scaler_key = f"{name}_scaler_file"
        if modes.get(name) == "scaled" and not _reference_present(cfg.get(scaler_key)):
            raise ConfigurationError(f"{scaler_key} must be configured when input_modes.{name} is 'scaled'.")
    input_verified = _require_bool_map(cfg, "input_modes_verified", components)
    input_refs = _require_object(cfg, "input_modes_verification_reference")
    if set(input_refs) - set(components):
        raise ConfigurationError(f"input_modes_verification_reference contains unsupported keys: {sorted(set(input_refs) - set(components))}")
    for ref_key, ref_value in input_refs.items():
        if ref_value is not None and not isinstance(ref_value, str):
            raise ConfigurationError(f"input_modes_verification_reference.{ref_key} must be a string or null.")
    for name in components:
        if name not in input_verified:
            raise ConfigurationError(f"input_modes_verified.{name} is required and must be a boolean.")
        if input_verified.get(name, False) and not _reference_present(input_refs.get(name, "")):
            raise ConfigurationError(f"input_modes_verification_reference.{name} is required when input_modes_verified.{name} is true.")

    target_verified = _require_bool_map(cfg, "target_units_verified", ("solar", "wind", "hydrogen"))
    target_refs = _require_object(cfg, "target_units_verification_reference")
    if set(target_refs) - {"solar", "wind", "hydrogen"}:
        raise ConfigurationError(f"target_units_verification_reference contains unsupported keys: {sorted(set(target_refs) - {'solar', 'wind', 'hydrogen'})}")
    for ref_key, ref_value in target_refs.items():
        if ref_value is not None and not isinstance(ref_value, str):
            raise ConfigurationError(f"target_units_verification_reference.{ref_key} must be a string or null.")
    for name in ("solar", "wind", "hydrogen"):
        if target_verified.get(name, False) and not _reference_present(target_refs.get(name, "")):
            raise ConfigurationError(f"target_units_verification_reference.{name} is required when target_units_verified.{name} is true.")

    observed_target_verified = _require_bool_map(cfg, "observed_target_units_verified", ("solar", "wind", "hydrogen"))
    observed_target_refs = _require_object(cfg, "observed_target_units_verification_reference")
    if set(observed_target_refs) - {"solar", "wind", "hydrogen"}:
        raise ConfigurationError(f"observed_target_units_verification_reference contains unsupported keys: {sorted(set(observed_target_refs) - {'solar', 'wind', 'hydrogen'})}")
    for ref_key, ref_value in observed_target_refs.items():
        if ref_value is not None and not isinstance(ref_value, str):
            raise ConfigurationError(f"observed_target_units_verification_reference.{ref_key} must be a string or null.")
    for name in ("solar", "wind", "hydrogen"):
        if observed_target_verified.get(name, False) and not _reference_present(observed_target_refs.get(name, "")):
            raise ConfigurationError(f"observed_target_units_verification_reference.{name} is required when observed_target_units_verified.{name} is true.")

    scaler_feature_orders = _require_object(cfg, "scaler_feature_order")
    if set(scaler_feature_orders) - set(components):
        raise ConfigurationError(
            f"scaler_feature_order contains unsupported keys: {sorted(set(scaler_feature_orders) - set(components))}"
        )
    for name, order in scaler_feature_orders.items():
        if not isinstance(order, list) or not all(isinstance(x, str) and x.strip() for x in order):
            raise ConfigurationError(f"scaler_feature_order.{name} must be a list of feature names.")
        if len(order) != len(set(order)):
            raise ConfigurationError(f"scaler_feature_order.{name} contains duplicate feature names.")
    model_order = _require_bool_map(cfg, "model_feature_order_verified", components)
    model_order_refs = _require_object(cfg, "model_feature_order_reference")
    scaler_order = _require_bool_map(cfg, "scaler_feature_order_verified", components)
    scaler_order_refs = _require_object(cfg, "scaler_feature_order_reference")
    for section_name, ref_map in (("model_feature_order_reference", model_order_refs), ("scaler_feature_order_reference", scaler_order_refs)):
        if set(ref_map) - set(components):
            raise ConfigurationError(f"{section_name} contains unsupported keys: {sorted(set(ref_map) - set(components))}")
        for ref_key, ref_value in ref_map.items():
            if ref_value is not None and not isinstance(ref_value, str):
                raise ConfigurationError(f"{section_name}.{ref_key} must be a string or null.")
    for name in components:
        if model_order.get(name, False) and not _reference_present(model_order_refs.get(name, "")):
            raise ConfigurationError(f"model_feature_order_reference.{name} is required when model_feature_order_verified.{name} is true.")
        if scaler_order.get(name, False) and not _reference_present(scaler_order_refs.get(name, "")):
            raise ConfigurationError(f"scaler_feature_order_reference.{name} is required when scaler_feature_order_verified.{name} is true.")

    for key in ("require_verified_preprocessing", "require_verified_target_units"):
        if key in cfg and not isinstance(cfg[key], bool):
            raise ConfigurationError(f"{key} must be a boolean.")
        if cfg.get(key, True) is not True:
            raise ConfigurationError(f"{key}=false is unsafe for unit-labelled/scientific inference and is not supported.")

    cg = _require_object(cfg, "candidate_generation")
    allowed_candidate_keys = {"method", "n_candidates", "random_seed", "top_n", "objective", "ranking_direction"}
    if set(cg) - allowed_candidate_keys:
        raise ConfigurationError(f"candidate_generation contains unsupported keys: {sorted(set(cg) - allowed_candidate_keys)}")
    if cg.get("method") not in ("observed_rows", "correlated_resample"):
        raise ConfigurationError("candidate_generation.method must be 'observed_rows' or 'correlated_resample'.")
    for key in ("n_candidates", "top_n"):
        value = cg.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigurationError(f"candidate_generation.{key} must be a positive integer.")
    seed = cg.get("random_seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigurationError("candidate_generation.random_seed must be an integer.")
    if cg.get("objective") != cfg["hydrogen_target"]:
        raise ConfigurationError("Only the configured hydrogen production target is currently supported as an optimization objective.")
    if cg.get("ranking_direction") not in ("maximize", "minimize"):
        raise ConfigurationError("candidate_generation.ranking_direction must be 'maximize' or 'minimize'.")

    renewable = _require_object(cfg, "renewable_combination", default={"method": "sum"})
    renewable.setdefault("method", "sum")
    if set(renewable) - {"method"}:
        raise ConfigurationError(f"renewable_combination contains unsupported keys: {sorted(set(renewable) - {'method'})}")
    if renewable["method"] not in ("sum", "solar_only", "wind_only"):
        raise ConfigurationError("renewable_combination.method must be sum, solar_only, or wind_only.")
    alignment = _require_object(cfg, "alignment")
    allowed_alignment_keys = {
        "solar_timestamp_column", "frequency", "timestamps_compatible_verified", "solar_power_semantics_verified",
        "wind_power_semantics_verified", "wind_aggregation", "wind_duplicate_timestamp_aggregation",
        "solar_duplicate_timestamp_aggregation", "wind_aggregation_semantics_verified", "wind_aggregation_interpretation",
        "wind_first_order_verified", "solar_first_order_verified", "wind_duplicate_timestamp_semantics_verified",
        "solar_duplicate_timestamp_semantics_verified", "solar_power_scope", "wind_power_scope", "timestamp_timezone_mode",
    }
    if set(alignment) - allowed_alignment_keys:
        raise ConfigurationError(f"alignment contains unsupported keys: {sorted(set(alignment) - allowed_alignment_keys)}")
    if "solar_timestamp_column" in alignment and (not isinstance(alignment["solar_timestamp_column"], str) or not alignment["solar_timestamp_column"].strip()):
        raise ConfigurationError("alignment.solar_timestamp_column must be a non-empty string.")
    for key in ("wind_aggregation", "wind_duplicate_timestamp_aggregation", "solar_duplicate_timestamp_aggregation"):
        value = alignment.get(key, "mean")
        if value not in ("mean", "sum", "first"):
            raise ConfigurationError(f"alignment.{key} must be mean, sum, or first.")
    alignment_bool_keys = (
        "timestamps_compatible_verified", "solar_power_semantics_verified", "wind_power_semantics_verified",
        "wind_aggregation_semantics_verified", "wind_first_order_verified", "solar_first_order_verified",
        "wind_duplicate_timestamp_semantics_verified", "solar_duplicate_timestamp_semantics_verified",
    )
    for key in alignment_bool_keys:
        if key in alignment and not isinstance(alignment[key], bool):
            raise ConfigurationError(f"alignment.{key} must be a boolean.")
    alignment_refs = _require_object(cfg, "alignment_verification_reference")
    allowed_alignment_reference_keys = {
        "timestamps_compatible_verified", "solar_power_semantics_verified", "wind_power_semantics_verified",
        "wind_aggregation_semantics_verified", "timestamp_timezone_mode", "wind_first_order_verified",
        "solar_first_order_verified", "wind_duplicate_timestamp_semantics_verified",
        "solar_duplicate_timestamp_semantics_verified", "solar_power_scope", "wind_power_scope",
    }
    if set(alignment_refs) - allowed_alignment_reference_keys:
        raise ConfigurationError(f"alignment_verification_reference contains unsupported keys: {sorted(set(alignment_refs) - allowed_alignment_reference_keys)}")
    for ref_key, ref_value in alignment_refs.items():
        if ref_value is not None and not isinstance(ref_value, str):
            raise ConfigurationError(f"alignment_verification_reference.{ref_key} must be a string or null.")
    for key in ("timestamps_compatible_verified", "solar_power_semantics_verified", "wind_power_semantics_verified"):
        if alignment.get(key, False) and not _reference_present(alignment_refs.get(key, "")):
            raise ConfigurationError(f"alignment_verification_reference.{key} is required when alignment.{key} is true.")
    for key in ("wind_aggregation_semantics_verified", "wind_first_order_verified", "solar_first_order_verified",
                "wind_duplicate_timestamp_semantics_verified", "solar_duplicate_timestamp_semantics_verified"):
        if alignment.get(key, False) and not _reference_present(alignment_refs.get(key, "")):
            raise ConfigurationError(f"alignment_verification_reference.{key} is required when alignment.{key} is true.")
    freq = alignment.get("frequency", "h")
    if not isinstance(freq, str) or not freq.strip():
        raise ConfigurationError("alignment.frequency must be a non-empty pandas frequency string.")
    try:
        frequency_offset = pd.tseries.frequencies.to_offset(freq)
        frequency_offset.nanos  # raises for calendar offsets that .floor() cannot apply
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"alignment.frequency must be a valid fixed pandas interval for floor-based aggregation: {freq!r}.") from exc
    if alignment.get("timestamp_timezone_mode") not in (None, "aware_utc", "naive_same_timezone"):
        raise ConfigurationError("alignment.timestamp_timezone_mode must be aware_utc or naive_same_timezone when specified.")
    if alignment.get("timestamps_compatible_verified", False) and not str(alignment.get("timestamp_timezone_mode", "")).strip():
        raise ConfigurationError("alignment.timestamp_timezone_mode is required when timestamp compatibility is verified.")
    if alignment.get("timestamp_timezone_mode") and not _reference_present(alignment_refs.get("timestamp_timezone_mode", "")):
        raise ConfigurationError("alignment_verification_reference.timestamp_timezone_mode is required when timestamp_timezone_mode is set.")
    allowed_solar_scopes = ("unknown", "unit_output", "farm_total")
    allowed_wind_scopes = ("unknown", "selected_turbine", "per_turbine_mean", "farm_total")
    alignment.setdefault("solar_power_scope", "unknown")
    alignment.setdefault("wind_power_scope", "unknown")
    if alignment["solar_power_scope"] not in allowed_solar_scopes:
        raise ConfigurationError(f"alignment.solar_power_scope must be one of {allowed_solar_scopes}.")
    if alignment["wind_power_scope"] not in allowed_wind_scopes:
        raise ConfigurationError(f"alignment.wind_power_scope must be one of {allowed_wind_scopes}.")
    for key in ("solar_power_scope", "wind_power_scope"):
        if alignment[key] != "unknown" and not _reference_present(alignment_refs.get(key, "")):
            raise ConfigurationError(f"alignment_verification_reference.{key} is required when alignment.{key} is not unknown.")

    wind_evidence = _require_object(cfg, "wind_aggregation_evidence")
    allowed_wind_evidence_keys = {"target_semantics", "turbine_id_column", "power_unit", "simultaneous_observations_verified", "turbine_coverage_verified", "reference"}
    if set(wind_evidence) - allowed_wind_evidence_keys:
        raise ConfigurationError(f"wind_aggregation_evidence contains unsupported keys: {sorted(set(wind_evidence) - allowed_wind_evidence_keys)}")
    if wind_evidence.get("target_semantics") not in (None, "per_turbine_power", "farm_total_power"):
        raise ConfigurationError("wind_aggregation_evidence.target_semantics must be per_turbine_power, farm_total_power, or null.")
    for key in ("simultaneous_observations_verified", "turbine_coverage_verified"):
        if key in wind_evidence and not isinstance(wind_evidence[key], bool):
            raise ConfigurationError(f"wind_aggregation_evidence.{key} must be a boolean.")
    if wind_evidence.get("reference") is not None and not isinstance(wind_evidence.get("reference"), str):
        raise ConfigurationError("wind_aggregation_evidence.reference must be a string or null.")
    if wind_evidence.get("target_semantics") and not _reference_present(wind_evidence.get("reference", "")):
        raise ConfigurationError("wind_aggregation_evidence.reference is required when target_semantics is configured.")
    if wind_evidence.get("target_semantics") and wind_evidence.get("power_unit") != "kW":
        raise ConfigurationError("wind_aggregation_evidence.power_unit must be 'kW' when target semantics are configured.")
    if wind_evidence.get("turbine_id_column") is not None and (
        not isinstance(wind_evidence.get("turbine_id_column"), str) or not wind_evidence.get("turbine_id_column", "").strip()
    ):
        raise ConfigurationError("wind_aggregation_evidence.turbine_id_column must be a non-empty string or null.")
    if wind_evidence.get("target_semantics") == "per_turbine_power" and not wind_evidence.get("turbine_id_column"):
        raise ConfigurationError("wind_aggregation_evidence.turbine_id_column is required for per_turbine_power semantics.")

    chunksize = cfg.get("wind_chunksize", 100_000)
    if isinstance(chunksize, bool) or not isinstance(chunksize, int) or chunksize < 1:
        raise ConfigurationError("wind_chunksize must be a positive integer.")
    if cfg.get("wind_time_numeric_unit") not in (None, "minutes", "seconds"):
        raise ConfigurationError("wind_time_numeric_unit must be 'minutes', 'seconds', or null.")
    wind_calendar = _require_object(cfg, "wind_calendar")
    allowed_wind_calendar_keys = {"day_index_zero_date", "day_index_zero_date_verified", "day_index_zero_date_reference", "allow_unused_month_relative_inference"}
    if set(wind_calendar) - allowed_wind_calendar_keys:
        raise ConfigurationError(f"wind_calendar contains unsupported keys: {sorted(set(wind_calendar) - allowed_wind_calendar_keys)}")
    if wind_calendar.get("day_index_zero_date") is not None:
        if not isinstance(wind_calendar["day_index_zero_date"], str) or not wind_calendar["day_index_zero_date"].strip():
            raise ConfigurationError("wind_calendar.day_index_zero_date must be a non-empty date string or null.")
        try:
            origin = pd.Timestamp(wind_calendar["day_index_zero_date"])
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("wind_calendar.day_index_zero_date must be a valid date or null.") from exc
        if pd.isna(origin) or origin.tzinfo is not None:
            raise ConfigurationError("wind_calendar.day_index_zero_date must be a valid timezone-naive date or null.")
        if wind_calendar.get("day_index_zero_date_verified", False) is not True or not _reference_present(wind_calendar.get("day_index_zero_date_reference", "")):
            # Keep unverified dates visible for diagnostics, but they cannot be used to assign calendar timestamps.
            pass
    if "day_index_zero_date_verified" in wind_calendar and not isinstance(wind_calendar["day_index_zero_date_verified"], bool):
        raise ConfigurationError("wind_calendar.day_index_zero_date_verified must be a boolean.")
    if wind_calendar.get("day_index_zero_date_verified", False) and wind_calendar.get("day_index_zero_date") is None:
        raise ConfigurationError("wind_calendar.day_index_zero_date is required when day_index_zero_date_verified is true.")
    if wind_calendar.get("day_index_zero_date_reference") is not None and not isinstance(wind_calendar.get("day_index_zero_date_reference"), str):
        raise ConfigurationError("wind_calendar.day_index_zero_date_reference must be a string or null.")
    if wind_calendar.get("day_index_zero_date_verified", False) and not _reference_present(wind_calendar.get("day_index_zero_date_reference", "")):
        raise ConfigurationError("wind_calendar.day_index_zero_date_reference is required when day_index_zero_date_verified is true.")

    if not isinstance(wind_calendar.get("allow_unused_month_relative_inference", False), bool):
        raise ConfigurationError("wind_calendar.allow_unused_month_relative_inference must be boolean.")

    power_validation = _require_object(cfg, "power_validation")
    if set(power_validation) - {"reject_negative_predictions"}:
        raise ConfigurationError(f"power_validation contains unsupported keys: {sorted(set(power_validation) - {'reject_negative_predictions'})}")
    if "reject_negative_predictions" in power_validation and not isinstance(power_validation["reject_negative_predictions"], bool):
        raise ConfigurationError("power_validation.reject_negative_predictions must be a boolean.")

    ranges = _require_object(cfg, "approved_ranges")
    allowed_range_keys = {"enabled", "min", "max", "units", "approved_by", "approval_reference", "allow_outside_observed", "extrapolation_reference"}
    for col, spec in ranges.items():
        if not isinstance(spec, dict):
            raise ConfigurationError(f"approved_ranges.{col} must be an object.")
        if set(spec) - allowed_range_keys:
            raise ConfigurationError(f"approved_ranges.{col} contains unsupported keys: {sorted(set(spec) - allowed_range_keys)}")
        if "enabled" in spec and not isinstance(spec["enabled"], bool):
            raise ConfigurationError(f"approved_ranges.{col}.enabled must be a boolean.")
        if "allow_outside_observed" in spec and not isinstance(spec["allow_outside_observed"], bool):
            raise ConfigurationError(f"approved_ranges.{col}.allow_outside_observed must be a boolean.")
        if spec.get("enabled", False):
            try:
                bounds = [float(spec["min"]), float(spec["max"])]
            except (KeyError, TypeError, ValueError) as exc:
                raise ConfigurationError(f"Enabled approved_ranges.{col} requires numeric min and max.") from exc
            if not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
                raise ConfigurationError(f"approved_ranges.{col} requires finite ordered min/max bounds.")
            if not all(isinstance(spec.get(k), str) and spec.get(k, "").strip() for k in ("approved_by", "approval_reference", "units")):
                raise ConfigurationError(f"Enabled approved_ranges.{col} requires non-empty approved_by, approval_reference, and units strings.")
            if spec.get("allow_outside_observed", False) and not (isinstance(spec.get("extrapolation_reference"), str) and spec.get("extrapolation_reference", "").strip()):
                raise ConfigurationError(f"Enabled approved_ranges.{col} with allow_outside_observed=true requires extrapolation_reference.")
    constraints = cfg["constraints"]
    if not isinstance(constraints, list) or any(not isinstance(rule, dict) for rule in constraints):
        raise ConfigurationError("constraints must be a list of JSON objects.")
    seen_constraint_names: set[str] = set()
    allowed_constraint_keys = {"name", "enabled", "verified", "verified_by", "verification_reference", "approval_reference", "source_reference", "units", "column", "operator", "value"}
    for i, rule in enumerate(constraints, 1):
        if set(rule) - allowed_constraint_keys:
            raise ConfigurationError(f"constraints[{i-1}] contains unsupported keys: {sorted(set(rule) - allowed_constraint_keys)}")
        name = rule.get("name", f"constraint_{i}")
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError(f"constraints[{i-1}].name must be a non-empty string.")
        output_col = f"constraint_{name}_pass"
        if output_col in seen_constraint_names:
            raise ConfigurationError(f"Duplicate constraint name would create conflicting output columns: {name!r}.")
        seen_constraint_names.add(output_col)
        if "enabled" in rule and not isinstance(rule["enabled"], bool):
            raise ConfigurationError(f"constraints[{i-1}].enabled must be a boolean.")
        if "verified" in rule and not isinstance(rule["verified"], bool):
            raise ConfigurationError(f"constraints[{i-1}].verified must be a boolean.")
        for metadata_key in ("verified_by", "verification_reference", "units"):
            if rule.get(metadata_key) is not None and not isinstance(rule.get(metadata_key), str):
                raise ConfigurationError(f"constraints[{i-1}].{metadata_key} must be a string or null.")
        if rule.get("enabled", False):
            if not isinstance(rule.get("column"), str) or not rule["column"].strip():
                raise ConfigurationError(f"constraints[{i-1}].column must be a non-empty string.")
            operator = rule.get("operator")
            if operator not in ("<=", "<", ">=", ">", "between", "==", "!="):
                raise ConfigurationError(f"constraints[{i-1}].operator is unsupported: {operator!r}.")
            value = rule.get("value")
            if operator in ("<=", "<", ">=", ">"):
                try:
                    if isinstance(value, bool):
                        raise ValueError("boolean threshold")
                    numeric_value = float(value)
                except (TypeError, ValueError) as exc:
                    raise ConfigurationError(f"constraints[{i-1}].value must be a finite numeric threshold.") from exc
                if not np.isfinite(numeric_value):
                    raise ConfigurationError(f"constraints[{i-1}].value must be finite.")
            elif operator == "between":
                if not isinstance(value, list) or len(value) != 2 or any(isinstance(v, bool) for v in value):
                    raise ConfigurationError(f"constraints[{i-1}].value must be a numeric [min, max] list.")
                try:
                    lo, hi = map(float, value)
                except (TypeError, ValueError) as exc:
                    raise ConfigurationError(f"constraints[{i-1}].value must be numeric.") from exc
                if not np.isfinite([lo, hi]).all() or lo > hi:
                    raise ConfigurationError(f"constraints[{i-1}].value must have finite ordered bounds.")
            elif not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"constraints[{i-1}].value must be a non-empty string for {operator}.")

    fc = _require_object(cfg, "fuel_cell_pass_constraint")
    allowed_fc_keys = {"enabled", "required_class_label", "verified", "input_mapping_verified", "mapping_reference", "class_label_mapping", "class_mapping_verified", "class_mapping_reference"}
    if set(fc) - allowed_fc_keys:
        raise ConfigurationError(f"fuel_cell_pass_constraint contains unsupported keys: {sorted(set(fc) - allowed_fc_keys)}")
    required_class_label = fc.get("required_class_label", "Pass")
    if not isinstance(required_class_label, str) or not required_class_label.strip():
        raise ConfigurationError("fuel_cell_pass_constraint.required_class_label must be a non-empty string.")
    mapping = fc.get("class_label_mapping", {})
    if not isinstance(mapping, dict):
        raise ConfigurationError("fuel_cell_pass_constraint.class_label_mapping must be an object.")
    for key in ("enabled", "verified", "input_mapping_verified", "class_mapping_verified"):
        if key in fc and not isinstance(fc[key], bool):
            raise ConfigurationError(f"fuel_cell_pass_constraint.{key} must be a boolean.")
    if fc.get("class_mapping_reference") is not None and not isinstance(fc.get("class_mapping_reference"), str):
        raise ConfigurationError("fuel_cell_pass_constraint.class_mapping_reference must be a string or null.")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in mapping.items()):
        raise ConfigurationError("fuel_cell_pass_constraint.class_label_mapping keys and values must be strings.")
    if fc.get("class_mapping_verified", False) and not _reference_present(fc.get("class_mapping_reference", "")):
        raise ConfigurationError("fuel_cell_pass_constraint.class_mapping_reference is required when class_mapping_verified is true.")
    if fc.get("input_mapping_verified", False):
        raise ConfigurationError(
            "fuel_cell_pass_constraint.input_mapping_verified cannot be enabled: candidate-level Hydrogen-to-Fuel-Cell input mapping is not implemented in this engine."
        )
    if fc.get("verified", False):
        raise ConfigurationError(
            "fuel_cell_pass_constraint.verified cannot enable candidate integration; that integration remains blocked until implemented and validated."
        )

    outputs = _require_object(cfg, "outputs")
    required_outputs = ("model_schema", "preprocessing", "bounds", "all_candidates", "feasible", "top", "best", "solar", "wind", "renewable", "fuel_cell_predictions", "fuel_cell_classes", "diagnostics")
    missing_outputs = [key for key in required_outputs if key not in outputs]
    if missing_outputs:
        raise ConfigurationError(f"outputs is missing required output keys: {missing_outputs}")
    unknown_outputs = set(outputs) - set(required_outputs)
    if unknown_outputs:
        raise ConfigurationError(f"outputs contains unsupported keys: {sorted(unknown_outputs)}")
    names: list[str] = []
    for key, value in outputs.items():
        name = _require_plain_filename(value, f"outputs.{key}")
        names.append(name.casefold())
    if len(names) != len(set(names)):
        raise ConfigurationError("outputs contains duplicate filenames; each output key must have a unique filename.")

    return cfg


def required_file(folder: str | Path, filename: str | None, *, required: bool = True) -> Path | None:
    if not filename:
        if required:
            raise ConfigurationError(f"Required filename is not configured under {folder}.")
        return None
    filename = _require_plain_filename(filename, "filename")
    p = Path(folder) / filename
    if not p.is_file():
        if required:
            raise FileNotFoundError(f"Required file not found: {p}")
        return None
    return p


def _safe_output_path(out_dir: Path, filename: str) -> Path:
    """Resolve a configured output filename without allowing directory traversal."""
    name = _require_plain_filename(filename, "output filename")
    base = out_dir.resolve()
    path = base / name
    if path.parent.resolve() != base:
        raise ConfigurationError(f"Output path escapes configured output directory: {filename!r}.")
    if path.is_symlink():
        raise ConfigurationError(f"Configured output path is a symbolic link and is not safe to overwrite: {filename!r}.")
    return path


def atomic_to_csv(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    """Write a CSV beside its destination and atomically replace the final file on success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", suffix=".tmp", prefix=f".{path.name}.",
        dir=str(path.parent), delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            frame.to_csv(handle, index=index)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _reference_present(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _pipeline_has_transformer(model: Any) -> bool:
    """Conservatively detect a transformer inside a model pipeline to avoid double preprocessing."""
    seen: set[int] = set()
    stack: list[Any] = [model]
    while stack:
        candidate = stack.pop()
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        steps = getattr(candidate, "steps", None)
        if isinstance(steps, (list, tuple)) and len(steps) > 1:
            for _, step in steps[:-1]:
                if callable(getattr(step, "transform", None)):
                    return True
        for attr in ("regressor_", "estimator_", "estimator", "regressor", "model"):
            nested = getattr(candidate, attr, None)
            if nested is not None and nested is not candidate:
                stack.append(nested)
    return False


def load_pickle(path: Path) -> Any:
    """Load a trusted local model/scaler artifact and surface compatibility warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj = joblib.load(path)
    for warning in caught:
        print(f"[LOAD WARNING] {path.name}: {warning.message}")
    return obj


def model_names(model: Any) -> list[str] | None:
    """Return artifact feature names only when they appear to identify real input columns."""
    def usable(values: Any) -> list[str] | None:
        if values is None:
            return None
        try:
            normalized = [str(x) for x in values]
        except TypeError:
            return None
        n = len(normalized)
        placeholders = (
            [str(i) for i in range(n)],
            [f"f{i}" for i in range(n)],
            [f"feature_{i}" for i in range(n)],
        )
        return None if normalized in placeholders else normalized

    seen: set[int] = set()
    stack: list[Any] = [model]
    while stack:
        candidate = stack.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        for attr in ("feature_names_in_", "feature_names_", "feature_names"):
            names = usable(getattr(candidate, attr, None))
            if names is not None:
                return names
        named_steps = getattr(candidate, "named_steps", None)
        if isinstance(named_steps, dict):
            stack.extend(named_steps.values())
        for attr in ("best_estimator_", "estimator_", "regressor_", "estimator", "regressor", "base_estimator", "classifier", "model"):
            nested = getattr(candidate, attr, None)
            if nested is not None and nested is not candidate:
                stack.append(nested)
    return None


def scaler_names(scaler: Any) -> list[str] | None:
    names = getattr(scaler, "feature_names_in_", None)
    return [str(x) for x in names] if names is not None else None


def load_feature_file(cfg: dict[str, Any], key: str, fallback: list[str]) -> list[str]:
    """Read a selected-feature CSV if configured; otherwise use the explicit config order."""
    filename = cfg.get("selected_feature_files", {}).get(key)
    if not filename:
        return list(fallback)
    p = required_file(cfg["selected_features_dir"], filename)
    assert p is not None
    df = pd.read_csv(p, encoding="utf-8-sig")
    if "feature" in df.columns:
        vals = df["feature"].dropna().astype(str).str.strip().tolist()
    elif df.shape[1] == 1:
        vals = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
    else:
        raise ConfigurationError(f"Cannot infer feature list from {p}; expected a 'feature' column or one column.")
    if not vals or any(not x for x in vals) or len(vals) != len(set(vals)):
        raise ConfigurationError(f"Selected-feature file is empty or contains duplicate/blank names: {p}")
    if vals != fallback:
        # Selected-feature order is a model contract; explicit config and CSV must not silently disagree.
        raise ConfigurationError(
            f"{key} selected-feature file order differs from config. "
            f"CSV={vals}; config={fallback}. Reconcile with training pipeline."
        )
    return vals


def prepare_input(
    frame: pd.DataFrame, model: Any, configured: list[str], scaler: Any | None,
    mode: str, label: str, scaler_order_verified: bool = False,
    model_order_verified: bool = False, scaler_order_reference: str = "",
    model_order_reference: str = "",
    scaler_feature_order: list[str] | None = None,
) -> pd.DataFrame | np.ndarray:
    """Validate model input schema and apply explicitly configured preprocessing."""
    artifact_names = model_names(model)
    if artifact_names is None and not (
        model_order_verified and str(model_order_reference).strip()
    ):
        raise StageBlocked(
            f"{label}: model artifact does not expose feature names. Configured feature count/order "
            "is not proof of feature identity; provide original training-pipeline evidence and "
            "verify model_feature_order_verified and provide its training-evidence reference."
        )

    expected = artifact_names if artifact_names is not None else list(configured)
    if len(expected) != len(set(expected)):
        raise ValueError(f"{label}: duplicate feature names in model schema: {expected}")
    if expected != list(configured):
        raise ValueError(
            f"{label}: model feature order differs from configured/training specification. "
            f"model={expected}; configured={configured}"
        )

    missing = [c for c in expected if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{label}: missing model input columns {missing}; available={list(frame.columns)}"
        )

    X = frame.loc[:, expected].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    values = X.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = [
            c for c in X.columns
            if not np.isfinite(X[c].to_numpy(dtype=float)).all()
        ]
        raise ValueError(
            f"{label}: input contains missing/non-finite values in columns {bad}."
        )

    n = getattr(model, "n_features_in_", None)
    # Some legacy serialized estimators expose 0 as an unset/unknown feature count.
    # Treat only a positive count as authoritative; feature names/order and the
    # training-evidence checks above remain in force.
    if n is not None and int(n) > 0 and int(n) != X.shape[1]:
        raise ValueError(
            f"{label}: model expects {n} features but prepared {X.shape[1]}."
        )

    if mode == "raw":
        return X
    if mode != "scaled":
        raise ConfigurationError(
            f"{label}: input mode must be 'raw' or 'scaled', got {mode!r}."
        )
    if scaler is None:
        raise ConfigurationError(
            f"{label}: scaled input requested but scaler file is not configured."
        )
    if _pipeline_has_transformer(model):
        raise StageBlocked(
            f"{label}: model pipeline contains an internal transformer and an external scaler is configured. "
            "Double preprocessing cannot be ruled out from the available artifacts; verify the original pipeline before inference."
        )

    scaler_feature_names = scaler_names(scaler)
    explicit_order = (
        [str(c) for c in scaler_feature_order]
        if scaler_feature_order is not None else None
    )
    evidence = (
        scaler_order_verified and bool(str(scaler_order_reference).strip())
    )

    if scaler_feature_names is None:
        if not evidence or explicit_order is None:
            raise StageBlocked(
                f"{label}: scaler has no feature_names_in_; feature identity/order cannot be verified. "
                "Provide scaler_feature_order, verification, and a training-evidence reference."
            )
        scaler_feature_names = explicit_order
    else:
        scaler_feature_names = [str(c) for c in scaler_feature_names]
        if explicit_order is not None and explicit_order != scaler_feature_names:
            raise ValueError(
                f"{label}: explicit scaler feature order differs from scaler metadata"
            )
        if scaler_feature_names != expected and not (
            evidence and explicit_order == scaler_feature_names
        ):
            raise ValueError(
                f"{label}: scaler feature order mismatch. "
                f"scaler={scaler_feature_names}; model={expected}"
            )

    if len(scaler_feature_names) != len(set(scaler_feature_names)):
        raise ValueError(
            f"{label}: duplicate feature names in scaler schema: {scaler_feature_names}"
        )
    if set(scaler_feature_names) != set(expected):
        raise ValueError(
            f"{label}: scaler/model feature sets differ. "
            f"scaler={scaler_feature_names}; model={expected}"
        )

    expected_n = getattr(scaler, "n_features_in_", None)
    if expected_n is not None and int(expected_n) != len(scaler_feature_names):
        raise ValueError(
            f"{label}: scaler expects {expected_n} features but its schema has "
            f"{len(scaler_feature_names)} names."
        )

    # Apply scaling in the exact order used when the scaler was fitted.
    scaler_input = X.loc[:, scaler_feature_names]
    transformed = np.asarray(scaler.transform(scaler_input), dtype=float)
    if transformed.shape != scaler_input.shape or not np.isfinite(transformed).all():
        raise ValueError(
            f"{label}: scaler returned invalid shape/values: {transformed.shape}."
        )

    # Reorder scaled values to the model's expected feature order.
    scaled = pd.DataFrame(
        transformed, columns=scaler_feature_names, index=X.index
    )
    return scaled.loc[:, expected]

def predict(
    model: Any, frame: pd.DataFrame, configured: list[str], scaler: Any | None,
    mode: str, label: str, scaler_order_verified: bool = False,
    model_order_verified: bool = False, scaler_order_reference: str = "",
    model_order_reference: str = "",
    scaler_feature_order: list[str] | None = None,
) -> np.ndarray:
    X = prepare_input(
        frame, model, configured, scaler, mode, label,
        scaler_order_verified=scaler_order_verified,
        model_order_verified=model_order_verified,
        scaler_order_reference=scaler_order_reference,
        model_order_reference=model_order_reference,
        scaler_feature_order=scaler_feature_order,
    )
    try:
        pred = np.asarray(model.predict(X)).reshape(-1)
    except Exception as exc:
        raise RuntimeError(
            f"{label} prediction failed using mode={mode}, shape={getattr(X, 'shape', None)}. "
            f"Check training preprocessing and package versions. Original error: {exc}"
        ) from exc
    if len(pred) != len(frame):
        raise RuntimeError(f"{label}: {len(pred)} predictions for {len(frame)} rows, expected {len(frame)}.")
    try:
        numeric_pred = pred.astype(float)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label}: regression model returned non-numeric predictions.") from exc
    if not np.isfinite(numeric_pred).all():
        raise RuntimeError(f"{label}: model returned non-finite numeric predictions.")
    return numeric_pred


def numeric_frame(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Copy, check required columns, and coerce selected columns to numeric."""
    out = df.copy()
    missing = [c for c in columns if c not in out.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns {missing}; available={list(out.columns)}")
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def has_expected_model_type(model: Any, component: str) -> bool:
    """Check estimator identity through common sklearn wrappers/pipelines without loading new models."""
    expected = {
        "solar": "CatBoostRegressor",
        "wind": "XGBRegressor",
        "hydrogen": "CatBoostRegressor",
        "fuel_cell": "XGBClassifier",
    }.get(component)
    if expected is None:
        return False

    seen: set[int] = set()
    stack: list[Any] = [model]
    while stack:
        candidate = stack.pop()
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if any(cls.__name__ == expected for cls in type(candidate).__mro__):
            return True
        # sklearn Pipelines, meta-estimators and common wrappers expose estimators here.
        named_steps = getattr(candidate, "named_steps", None)
        if isinstance(named_steps, dict):
            stack.extend(named_steps.values())
        get_params = getattr(candidate, "get_params", None)
        if callable(get_params):
            try:
                params = get_params(deep=True)
            except Exception:
                params = {}
            if isinstance(params, dict):
                stack.extend(value for value in params.values() if hasattr(value, "predict") or hasattr(value, "fit"))
        for attr in ("best_estimator_", "estimator_", "regressor_", "base_estimator_", "estimator", "regressor", "base_estimator", "classifier", "model"):
            value = getattr(candidate, attr, None)
            if value is not None and value is not candidate and (hasattr(value, "predict") or hasattr(value, "fit")):
                stack.append(value)
    return False


def artifact_library_version(obj: Any) -> str:
    """Report the underlying estimator library for common wrappers without recursing on cycles."""
    def find_package(candidate: Any, seen: set[int]) -> str:
        if candidate is None or id(candidate) in seen:
            return ""
        seen.add(id(candidate))
        module = type(candidate).__module__.split(".", 1)[0]
        if module == "sklearn":
            named_steps = getattr(candidate, "named_steps", None)
            if isinstance(named_steps, dict):
                for nested in reversed(list(named_steps.values())):
                    nested_module = type(nested).__module__.split(".", 1)[0]
                    if nested_module not in {"sklearn", "builtins"}:
                        found = find_package(nested, seen)
                        if found:
                            return found
            for attr in ("best_estimator_", "estimator_", "regressor_", "estimator", "regressor", "model"):
                nested = getattr(candidate, attr, None)
                if nested is not None and nested is not candidate:
                    found = find_package(nested, seen)
                    if found:
                        return found
        package_name = {"sklearn": "scikit-learn", "xgboost": "xgboost", "catboost": "catboost"}.get(module, module)
        try:
            return metadata.version(package_name)
        except metadata.PackageNotFoundError:
            return ""

    return find_package(obj, set())


def load_assets(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load model assets independently so one missing artifact does not hide other stage diagnostics."""
    specs = {
        "solar": ("solar_model_file", "solar_scaler_file"),
        "wind": ("wind_model_file", "wind_scaler_file"),
        "hydrogen": ("hydrogen_model_file", "hydrogen_scaler_file"),
        "fuel_cell": ("fuel_cell_model_file", "fuel_cell_scaler_file"),
    }
    assets: dict[str, Any] = {}
    errors: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for name, (model_key, scaler_key) in specs.items():
        try:
            model_path = required_file(cfg["models_dir"], cfg.get(model_key))
            scaler_path = required_file(
                cfg["models_dir"], cfg.get(scaler_key),
                required=(cfg.get("input_modes", {}).get(name) == "scaled"),
            )
            assert model_path is not None
            model = load_pickle(model_path)
            if not has_expected_model_type(model, name):
                raise ConfigurationError(
                    f"{name} model artifact type {type(model).__name__} does not expose the expected "
                    "estimator type (CatBoostRegressor/XGBRegressor/XGBClassifier as configured)."
                )
            scaler = None
            scaler_error = ""
            if cfg.get(scaler_key) and scaler_path is None and cfg.get("input_modes", {}).get(name) == "raw":
                scaler_error = f"Configured scaler file is missing but unused in raw input mode: {Path(cfg['models_dir']) / cfg[scaler_key]}"
            if scaler_path is not None:
                try:
                    scaler = load_pickle(scaler_path)
                    if not callable(getattr(scaler, "transform", None)):
                        raise ConfigurationError(f"Configured scaler {scaler_path} has no callable transform method.")
                except Exception as exc:
                    if cfg.get("input_modes", {}).get(name) == "scaled":
                        raise
                    scaler_error = f"{type(exc).__name__}: {exc}; scaler is unused in raw input mode"
                    scaler = None
            assets[name] = (model, scaler)
            rows.append({
                "component": name, "status": "LOADED" if not scaler_error else "MODEL_LOADED_UNUSED_SCALER_BLOCKED", "error": scaler_error,
                "model_file": str(model_path), "scaler_file": str(scaler_path or ""),
                "model_type": type(model).__name__, "scaler_type": type(scaler).__name__ if scaler is not None else "",
                "model_features": json.dumps(model_names(model)),
                "scaler_features": json.dumps(scaler_names(scaler) if scaler is not None else None),
                "model_n_features": getattr(model, "n_features_in_", None),
                "scaler_n_features": getattr(scaler, "n_features_in_", None),
                "classes": json.dumps([str(x) for x in getattr(model, "classes_", [])]),
                "model_version": getattr(model, "__version__", ""),
                "model_library_version": artifact_library_version(model),
                "scaler_library_version": artifact_library_version(scaler) if scaler is not None else "",
            })
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            rows.append({
                "component": name, "status": "BLOCKED_ASSET_LOAD", "error": errors[name],
                "model_file": str(Path(cfg["models_dir"]) / str(cfg.get(model_key, ""))),
                "scaler_file": str(Path(cfg["models_dir"]) / str(cfg.get(scaler_key, ""))),
                "model_type": "", "scaler_type": "", "model_features": "",
                "scaler_features": "", "model_n_features": "", "scaler_n_features": "",
                "classes": "", "model_version": "", "model_library_version": "", "scaler_library_version": "",
            })
    assets["__errors__"] = errors
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_to_csv(pd.DataFrame(rows), _safe_output_path(out_dir, cfg["outputs"]["model_schema"]))
    return assets


def require_verified_preprocessing(cfg: dict[str, Any], model_name: str) -> None:
    refs = cfg.get("input_modes_verification_reference", {})
    verified = cfg.get("input_modes_verified", {}).get(model_name) is True
    referenced = _reference_present(refs.get(model_name, ""))
    if not (verified and referenced):
        raise StageBlocked(
            f"{model_name} inference blocked: input mode/preprocessing lacks a verified flag and traceable reference. "
            f"Inspect the original training/preprocessing pipeline before enabling it."
        )


def require_verified_target_units(cfg: dict[str, Any], model_name: str) -> None:
    """Block unit-labelled power/production inference until target scale and units have evidence."""
    refs = cfg.get("target_units_verification_reference", {})
    verified = cfg.get("target_units_verified", {}).get(model_name) is True
    referenced = _reference_present(refs.get(model_name, ""))
    if not (verified and referenced):
        raise StageBlocked(
            f"{model_name} inference blocked: target units/output transform lack a verified flag and traceable reference. "
            "Inspect training target preprocessing and any inverse transform before enabling unit-labelled outputs."
        )


def solar_power(cfg: dict[str, Any], model: Any, scaler: Any | None, features: list[str]) -> pd.DataFrame:
    """Predict Solar power using calendar fields derived only from the configured source timestamp."""
    p = required_file(cfg["dataset_dir"], cfg["solar_dataset_file"])
    assert p is not None
    df = pd.read_csv(p, encoding="utf-8-sig")
    input_rows = len(df)
    timestamp_col = cfg["alignment"].get("solar_timestamp_column", "timestamp")
    if timestamp_col not in df:
        raise StageBlocked(
            f"Solar timestamp column {timestamp_col!r} is absent; calendar features cannot be derived."
        )
    source_timestamp = df[timestamp_col].copy()
    try:
        dt = pd.to_datetime(source_timestamp, errors="coerce")
    except (TypeError, ValueError, OverflowError) as exc:
        raise StageBlocked(f"Solar timestamps could not be parsed consistently: {exc}") from exc
    if dt.notna().any() and not pd.api.types.is_datetime64_any_dtype(dt.dtype):
        raise StageBlocked(
            "Solar timestamps produced mixed/ambiguous timezone or datetime types. "
            "Normalize the source timestamps according to documented training-time semantics first."
        )
    df["timestamp"] = dt
    derived = {
        "Hour": dt.dt.hour, "Day": dt.dt.day, "DayOfYear": dt.dt.day_of_year,
        "Minute": dt.dt.minute, "Month": dt.dt.month,
    }
    for col, values in derived.items():
        if col in features:
            df[col] = values
    valid_timestamp = dt.notna()
    timestamp_rejected = int((~valid_timestamp).sum())
    df = df.loc[valid_timestamp].copy()
    df["source_timestamp"] = source_timestamp.loc[df.index].astype(str)
    df = numeric_frame(df, features)
    finite_mask = np.isfinite(df[features].to_numpy(dtype=float)).all(axis=1)
    feature_rejected = int((~finite_mask).sum())
    df = df.loc[finite_mask].copy()
    if df.empty:
        raise StageBlocked("No complete finite Solar rows remain after timestamp/feature validation.")
    require_verified_preprocessing(cfg, "solar")
    require_verified_target_units(cfg, "solar")
    predictions = predict(
        model, df, features, scaler, cfg["input_modes"]["solar"], "Solar model",
        scaler_order_verified=cfg.get("scaler_feature_order_verified", {}).get("solar", False),
        model_order_verified=cfg.get("model_feature_order_verified", {}).get("solar", False),
        scaler_order_reference=cfg.get("scaler_feature_order_reference", {}).get("solar", ""),
        model_order_reference=cfg.get("model_feature_order_reference", {}).get("solar", ""),
        scaler_feature_order=cfg.get("scaler_feature_order", {}).get("solar"),
    ).astype(float)
    # Preserve raw model output for diagnosis. Never silently clip negatives, but do
    # write the stage so the user can inspect which rows/features produce them.
    # align_renewables() still rejects negative values, so invalid power cannot be
    # used in the integrated renewable series.
    negative_prediction_count = int((predictions < 0).sum())
    df["predicted_solar_power_kw"] = predictions
    df["prediction_validity_status"] = np.where(
        predictions < 0, "INVALID_NEGATIVE_POWER_PREDICTION", "NONNEGATIVE_PREDICTION"
    )
    target = cfg["solar_target"]
    observed_target_col = None
    if target in df:
        observed_verified = (
            cfg.get("observed_target_units_verified", {}).get("solar") is True
            and _reference_present(cfg.get("observed_target_units_verification_reference", {}).get("solar", ""))
        )
        observed_target_col = f"observed_{target}" if observed_verified else f"observed_{target}_value_unverified_units"
        df[observed_target_col] = pd.to_numeric(df[target], errors="coerce")
    cols = ["timestamp", "source_timestamp"] + ([observed_target_col] if observed_target_col else []) + ["predicted_solar_power_kw", "prediction_validity_status"]
    result = df[cols].copy()
    result["input_rows_seen"] = input_rows
    result["negative_prediction_rows"] = negative_prediction_count
    result["invalid_timestamp_rows_rejected"] = timestamp_rejected
    result["invalid_feature_rows_rejected"] = feature_rejected
    solar_semantics_verified = (
        cfg["alignment"].get("solar_power_semantics_verified") is True
        and _reference_present(cfg.get("alignment_verification_reference", {}).get("solar_power_semantics_verified", ""))
    )
    result["power_semantics_status"] = (
        "VERIFIED" if solar_semantics_verified else "UNVERIFIED_TURBINE_OR_FARM_SEMANTICS"
    )
    result["solar_power_scope"] = cfg["alignment"].get("solar_power_scope", "unknown")
    result["physical_safety_status"] = "NOT_ESTABLISHED"
    return result


def parse_clock(series: pd.Series, numeric_unit: str | None) -> pd.Series:
    """Parse HH:MM[:SS] text and documented numeric clock values row-by-row.

    Numeric values are never interpreted without an explicit minutes/seconds unit.
    Mixed string/numeric columns are supported only when that unit is configured.
    Returned values are minutes after midnight; invalid/out-of-range rows remain NaN.
    """
    raw = series.copy()
    text = raw.astype("string").str.strip()
    result = pd.Series(np.nan, index=series.index, dtype=float)

    parsed = pd.to_datetime(text, format="%H:%M:%S", errors="coerce")
    missing = parsed.isna()
    parsed2 = pd.to_datetime(text.loc[missing], format="%H:%M", errors="coerce")
    parsed.loc[missing] = parsed2
    valid_text = parsed.notna()
    result.loc[valid_text] = (
        parsed.loc[valid_text].dt.hour * 60
        + parsed.loc[valid_text].dt.minute
        + parsed.loc[valid_text].dt.second / 60.0
    )

    numeric = pd.to_numeric(raw, errors="coerce")
    numeric_rows = numeric.notna() & ~valid_text
    if numeric_rows.any():
        if numeric_unit not in ("minutes", "seconds"):
            raise StageBlocked(
                "Numeric wind time_of_day values require wind_time_numeric_unit='minutes' or 'seconds'."
            )
        values = numeric.loc[numeric_rows].astype(float)
        if numeric_unit == "seconds":
            values = values / 60.0
        result.loc[numeric_rows] = values

    # A clock value must represent a valid time of day; reject rather than wrap.
    result = result.where(result.ge(0) & result.lt(1440))
    return result


def _wind_month_unused_in_saved_xgboost(model: Any, features: list[str]) -> bool:
    """Fail-closed audit: native saved XGBoost trees never split on Month."""
    if "Month" not in features or not hasattr(model, "get_booster"):
        return False
    try:
        booster = model.get_booster()
        names = list(booster.feature_names or [])
        if names != list(features):
            return False
        # get_score('weight') reports each feature used in a tree split.
        # If Month has zero splits across the complete saved ensemble, its
        # supplied numeric value cannot change a tree's traversal.
        split_counts = booster.get_score(importance_type="weight")
        return bool(split_counts) and "Month" not in split_counts
    except (AttributeError, TypeError, ValueError, KeyError):
        return False


def wind_power(cfg: dict[str, Any], model: Any, scaler: Any | None, features: list[str]) -> pd.DataFrame:
    """Predict wind power in chunks; only assign calendar timestamps with a verified day-index origin."""
    p = required_file(cfg["dataset_dir"], cfg["wind_dataset_file"])
    assert p is not None
    chunk_size = int(cfg.get("wind_chunksize", 100_000))
    if chunk_size < 1:
        raise ConfigurationError("wind_chunksize must be positive.")
    wind_calendar = cfg.get("wind_calendar", {})
    origin = wind_calendar.get("day_index_zero_date")
    origin_verified = (
        origin is not None
        and wind_calendar.get("day_index_zero_date_verified") is True
        and _reference_present(wind_calendar.get("day_index_zero_date_reference", ""))
    )
    # The original training notebook derives Month/Hour/Minute directly from
    # pd.to_datetime(df['time_of_day']). If the source column contains full dates,
    # use those dates as the authoritative timestamps and do not invent a day_index
    # origin. Only clock-only data needs the separately documented origin.
    require_verified_preprocessing(cfg, "wind")
    require_verified_target_units(cfg, "wind")
    relative_month_mode = (
        wind_calendar.get("allow_unused_month_relative_inference") is True
        and not origin_verified
        and _wind_month_unused_in_saved_xgboost(model, features)
    )
    if wind_calendar.get("allow_unused_month_relative_inference") is True and not origin_verified and not relative_month_mode:
        raise StageBlocked("Wind relative inference requires exact saved XGBoost feature order and proof Month has zero tree splits.")
    # Streaming aggregate state avoids retaining every row's prediction for large CSVs.
    aggregation = cfg["alignment"].get("wind_aggregation", "mean")
    if aggregation not in ("mean", "sum", "first"):
        raise ConfigurationError("alignment.wind_aggregation must be 'mean', 'sum', or 'first'.")
    alignment_cfg = cfg["alignment"]
    alignment_refs = cfg.get("alignment_verification_reference", {})
    wind_evidence = cfg.get("wind_aggregation_evidence", {})
    if aggregation == "sum":
        if not (
            alignment_cfg.get("wind_aggregation_semantics_verified") is True
            and str(alignment_refs.get("wind_aggregation_semantics_verified", "")).strip()
        ):
            raise StageBlocked("Wind sum aggregation is blocked until aggregation semantics have a traceable verification reference.")
        if not (
            wind_evidence.get("target_semantics") == "per_turbine_power"
            and isinstance(wind_evidence.get("turbine_id_column"), str)
            and wind_evidence.get("turbine_id_column", "").strip()
            and wind_evidence.get("power_unit") == "kW"
            and wind_evidence.get("simultaneous_observations_verified") is True
            and wind_evidence.get("turbine_coverage_verified") is True
            and _reference_present(wind_evidence.get("reference", ""))
        ):
            raise StageBlocked(
                "Wind sum aggregation requires evidence of per-turbine kW targets, a turbine ID column, "
                "simultaneous observations, turbine coverage, and a traceable reference."
            )
    if aggregation == "first" and not (
        alignment_cfg.get("wind_first_order_verified") is True
        and _reference_present(alignment_refs.get("wind_first_order_verified", ""))
    ):
        raise StageBlocked("Wind first aggregation depends on file order; document and verify that order before using it.")
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    rejected_rows = 0
    input_rows = 0
    negative_count = 0
    predicted_count = 0
    negative_min = float("inf")
    negative_max = float("-inf")
    negative_by_day: dict[str, int] = {}
    turbine_stats: dict[str, dict[str, float | int]] = {}
    # Stream every numerically valid raw row prediction to a diagnostic-only CSV.
    # Never offer this file to renewable_alignment or Hydrogen integration.
    raw_path = _safe_output_path(Path(cfg.get("output_dir", cfg["dataset_dir"])), "wind_raw_predictions_diagnostic.csv")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", suffix=".tmp",
        prefix=".wind_raw_predictions_diagnostic.", dir=str(raw_path.parent), delete=False,
    )
    raw_temp = Path(raw_handle.name)
    raw_header = True
    try:
        with pd.read_csv(p, encoding="utf-8-sig", chunksize=chunk_size) as reader:
            for chunk in reader:
                input_rows += len(chunk)
                if "time_of_day" not in chunk.columns:
                    raise StageBlocked("Wind dataset needs 'time_of_day' for the training-time calendar features.")
                raw_time = chunk["time_of_day"]
                raw_text = raw_time.astype("string").str.strip()
                explicit_date_rows = raw_text.str.match(
                    r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[T\s].*)?$", na=False
                )
                clock_rows = raw_text.notna() & raw_text.ne("") & ~explicit_date_rows
                if explicit_date_rows.any() and clock_rows.any():
                    raise StageBlocked(
                        "Wind time_of_day mixes full datetimes and clock-only values; "
                        "a mixed calendar representation cannot be interpreted safely."
                    )
                # Do not pass clock-only strings through pd.to_datetime: pandas may
                # silently attach the current execution date, creating a false Month.
                full_datetime_mode = bool(explicit_date_rows.any())
                parsed_full = pd.to_datetime(raw_time, errors="coerce") if full_datetime_mode else None
                if full_datetime_mode:
                    exact_datetime = parsed_full
                    valid_clock = exact_datetime.notna()
                    chunk["Hour"] = exact_datetime.dt.hour
                    chunk["Minute"] = exact_datetime.dt.minute
                    chunk["Month"] = exact_datetime.dt.month
                    minutes = (exact_datetime.dt.hour * 60 + exact_datetime.dt.minute + exact_datetime.dt.second / 60.0)
                    day_index = pd.Series(np.nan, index=chunk.index, dtype=float)
                    day_valid = valid_clock
                else:
                    if "day_index" not in chunk.columns:
                        raise StageBlocked("Wind time_of_day contains clock-only values; dataset also needs day_index.")
                    if "Month" in features and not origin_verified and not relative_month_mode:
                        raise StageBlocked(
                            "Wind time_of_day contains clock-only values; day_index_zero_date lacks documentary verification, "
                            "so historical Month and timestamps cannot be derived. The training notebook's "
                            "pd.to_datetime(clock-only values) attached a processing date, not a verified measurement date. "
                            "Training-time Month semantics also require validation against the saved model."
                        )
                    minutes = parse_clock(raw_time, cfg.get("wind_time_numeric_unit"))
                    valid_clock = minutes.notna() & minutes.ge(0) & minutes.lt(1440)
                    chunk["Hour"] = np.floor(minutes / 60).where(valid_clock)
                    chunk["Minute"] = np.floor(minutes % 60).where(valid_clock)
                    day_index = pd.to_numeric(chunk["day_index"], errors="coerce")
                    day_valid = day_index.notna() & day_index.ge(0) & np.isclose(day_index.fillna(-1), np.round(day_index.fillna(-1)))
                    if relative_month_mode:
                        # Structural placeholder only, not a claimed calendar month.
                        # Verified unused in every split of this exact saved model.
                        chunk["Month"] = 1
                    if origin_verified:
                        date_origin = pd.Timestamp(origin)
                        chunk["Month"] = (date_origin + pd.to_timedelta(day_index, unit="D")).dt.month
                required_wind_columns = features + [cfg["wind_target"]]
                if aggregation == "sum":
                    required_wind_columns.append(str(wind_evidence["turbine_id_column"]))
                missing = [c for c in required_wind_columns if c not in chunk.columns]
                if missing:
                    raise ValueError(f"Wind dataset missing columns {missing}")
                chunk = numeric_frame(chunk, features)
                valid = valid_clock & day_valid & np.isfinite(chunk[features].to_numpy(dtype=float)).all(axis=1)
                rejected_rows += int((~valid).sum())
                chunk = chunk.loc[valid].copy()
                if chunk.empty:
                    continue
                pred = predict(
                    model, chunk, features, scaler, cfg["input_modes"]["wind"], "Wind model",
                    scaler_order_verified=cfg.get("scaler_feature_order_verified", {}).get("wind", False),
                    model_order_verified=cfg.get("model_feature_order_verified", {}).get("wind", False),
                    scaler_order_reference=cfg.get("scaler_feature_order_reference", {}).get("wind", ""),
                    model_order_reference=cfg.get("model_feature_order_reference", {}).get("wind", ""),
                    scaler_feature_order=cfg.get("scaler_feature_order", {}).get("wind"),
                ).astype(float)
                predicted_count += len(pred)
                # Preserve original model output, including negative values, with no clipping.
                diagnostic_rows = pd.DataFrame({
                    "source_row_index": chunk.index.to_numpy(),
                    "day_index": chunk["day_index"].to_numpy() if "day_index" in chunk else np.nan,
                    "time_of_day_source": raw_time.loc[chunk.index].to_numpy(),
                    "wind_speed_m_s": chunk["wind_speed_m_s"].to_numpy() if "wind_speed_m_s" in chunk else np.nan,
                    "raw_model_prediction_kw": pred,
                    "negative_prediction": pred < 0,
                    "source_target_unverified_units": pd.to_numeric(
                        chunk[cfg["wind_target"]], errors="coerce"
                    ).to_numpy(),
                    "approval_status": "DIAGNOSTIC_ONLY_NOT_APPROVED",
                })
                # Preserve the source turbine identity; do not infer turbine power scope.
                if "turbine_id" in chunk.columns:
                    diagnostic_rows.insert(1, "turbine_id", chunk["turbine_id"].astype("string").to_numpy())
                    for turbine_id, row_ids in chunk.groupby("turbine_id", dropna=False, sort=False).groups.items():
                        key = str(turbine_id)
                        values = pred[chunk.index.get_indexer(row_ids)]
                        if not len(values):
                            continue
                        state = turbine_stats.setdefault(key, {
                            "rows": 0, "negative": 0, "sum": 0.0,
                            "min": float("inf"), "max": float("-inf")
                        })
                        state["rows"] += len(values)
                        state["negative"] += int((values < 0).sum())
                        state["sum"] += float(values.sum())
                        state["min"] = min(state["min"], float(values.min()))
                        state["max"] = max(state["max"], float(values.max()))
                diagnostic_rows.to_csv(raw_handle, index=False, header=raw_header)
                raw_header = False
                negative_mask = pred < 0
                if negative_mask.any() and cfg.get("power_validation", {}).get("reject_negative_predictions", True):
                    negative_values = pred[negative_mask]
                    negative_count += int(negative_mask.sum())
                    negative_min = min(negative_min, float(negative_values.min()))
                    negative_max = max(negative_max, float(negative_values.max()))
                    if "day_index" in chunk.columns:
                        for day, count in chunk.loc[negative_mask, "day_index"].astype(str).value_counts().items():
                            negative_by_day[day] = negative_by_day.get(day, 0) + int(count)
                    # Never aggregate invalid negative predictions into the wind series.
                    chunk = chunk.loc[~negative_mask].copy()
                    pred = pred[~negative_mask]
                    if chunk.empty:
                        continue
                chunk["predicted_wind_power_kw"] = pred
                exact_minutes = minutes.loc[chunk.index].astype(float)
                if full_datetime_mode:
                    exact_ts = exact_datetime.loc[chunk.index]
                    bucket_ts = exact_ts.dt.floor(cfg["alignment"].get("frequency", "h"))
                    chunk["_key"] = bucket_ts.astype(str)
                    chunk["_exact_key"] = exact_ts.astype(str)
                    key_kind = "timestamp"
                elif origin_verified:
                    exact_ts = (pd.Timestamp(origin) + pd.to_timedelta(day_index.loc[chunk.index], unit="D")
                                + pd.to_timedelta(exact_minutes, unit="m"))
                    bucket_ts = exact_ts.dt.floor(cfg["alignment"].get("frequency", "h"))
                    chunk["_key"] = bucket_ts.astype(str)
                    chunk["_exact_key"] = exact_ts.astype(str)
                    key_kind = "timestamp"
                else:
                    freq_offset = pd.tseries.frequencies.to_offset(cfg["alignment"].get("frequency", "h"))
                    try:
                        bucket_minutes = freq_offset.nanos / 60_000_000_000
                    except ValueError as exc:
                        raise StageBlocked("A verified calendar origin is required for non-fixed wind output frequencies.") from exc
                    if bucket_minutes <= 0:
                        raise ConfigurationError("alignment.frequency must represent a positive fixed interval for relative wind timestamps.")
                    elapsed_minutes = day_index.loc[chunk.index].astype(float) * 1440.0 + exact_minutes
                    bucket_index = np.floor(elapsed_minutes / bucket_minutes).astype("int64")
                    chunk["_key"] = bucket_index.astype(str)
                    chunk["_exact_key"] = (
                        day_index.loc[chunk.index].astype(int).astype(str) + "_" + exact_minutes.map(lambda v: format(v, ".9f")).astype(str)
                    )
                    key_kind = "relative"
                for key, values in chunk.groupby("_key")["predicted_wind_power_kw"]:
                    state_key = (key_kind, str(key))
                    vals = values.to_numpy(dtype=float)
                    group_rows = chunk.loc[values.index]
                    exact_keys = group_rows["_exact_key"].astype(str).unique().tolist()
                    if aggregation == "sum":
                        if len(exact_keys) != 1:
                            raise StageBlocked(
                                "Wind sum aggregation would add observations from different times inside one output-frequency bucket. "
                                "Use a frequency matching simultaneous observations or a separately validated temporal aggregation."
                            )
                        id_column = str(wind_evidence["turbine_id_column"])
                        ids = group_rows[id_column]
                        if ids.isna().any():
                            raise StageBlocked(f"Wind sum aggregation requires non-missing turbine IDs in {id_column!r}.")
                        ids = ids.astype(str)
                        if ids.duplicated().any():
                            raise StageBlocked("Wind sum aggregation found duplicate rows for the same turbine at one timestamp.")
                        if state_key in groups and groups[state_key].get("exact_key") != exact_keys[0]:
                            raise StageBlocked("Wind sum aggregation encountered different exact timestamps in the same output bucket.")
                        seen_ids = groups[state_key].setdefault("turbine_ids", set()) if state_key in groups else set()
                        if set(ids).intersection(seen_ids):
                            raise StageBlocked("Wind sum aggregation found a turbine repeated across chunks at the same timestamp.")
                        seen_ids.update(ids)
                    if state_key not in groups:
                        groups[state_key] = {
                            "count": int(len(vals)), "sum": float(vals.sum()), "first": float(vals[0]),
                            "exact_key": exact_keys[0] if len(exact_keys) == 1 else "MULTIPLE_EXACT_TIMES",
                            "turbine_ids": set(group_rows[str(wind_evidence.get("turbine_id_column", "__no_turbine_id__"))].astype(str)) if aggregation == "sum" else set(),
                        }
                    else:
                        groups[state_key]["count"] = int(groups[state_key]["count"]) + int(len(vals))
                        groups[state_key]["sum"] = float(groups[state_key]["sum"]) + float(vals.sum())
        if raw_header:
            pd.DataFrame(columns=[
                "source_row_index", "turbine_id", "day_index", "time_of_day_source", "wind_speed_m_s",
                "raw_model_prediction_kw", "negative_prediction", "source_target_unverified_units",
                "approval_status",
            ]).to_csv(raw_handle, index=False)
        raw_handle.flush()
        os.fsync(raw_handle.fileno())
        raw_handle.close()
        os.replace(raw_temp, raw_path)
    except BaseException:
        raw_handle.close()
        raw_temp.unlink(missing_ok=True)
        raise
    # Independent, non-approved per-turbine statistics. No aggregation of turbine power.
    turbine_summary = pd.DataFrame([{
        "turbine_id": key,
        "valid_prediction_rows": int(s["rows"]),
        "negative_prediction_rows": int(s["negative"]),
        "negative_pct": 100 * int(s["negative"]) / int(s["rows"]),
        "raw_prediction_min_kw": float(s["min"]),
        "raw_prediction_max_kw": float(s["max"]),
        "raw_prediction_mean_kw": float(s["sum"]) / int(s["rows"]),
        "approval_status": "DIAGNOSTIC_ONLY_NOT_APPROVED",
        "power_scope_status": "UNVERIFIED",
    } for key, s in sorted(turbine_stats.items())], columns=[
        "turbine_id", "valid_prediction_rows", "negative_prediction_rows",
        "negative_pct", "raw_prediction_min_kw", "raw_prediction_max_kw",
        "raw_prediction_mean_kw", "approval_status", "power_scope_status"
    ])
    atomic_to_csv(turbine_summary, _safe_output_path(
        Path(cfg.get("output_dir", cfg["dataset_dir"])),
        "wind_per_turbine_diagnostic.csv"
    ))
    if negative_count and cfg.get("power_validation", {}).get("reject_negative_predictions", True):
        affected_days = sorted(negative_by_day.items(), key=lambda item: -item[1])[:10]
        raise StageBlocked(
            f"Wind model produced {negative_count} negative predictions among {predicted_count} "
            f"valid-row predictions ({100 * negative_count / predicted_count:.4f}%); "
            f"minimum={negative_min:.8g} kW, largest_negative={negative_max:.8g} kW; "
            f"top_negative_days={affected_days}. No negative predictions were clipped, "
            "and no Wind output was approved. Raw predictions saved to "
            f"{raw_path} as DIAGNOSTIC_ONLY_NOT_APPROVED."
        )
    if not groups:
        raise StageBlocked(
            "No valid wind predictions after row validation; "
            f"input_rows_seen={input_rows}, input_rows_rejected={rejected_rows}. "
            "Check time format, integer/non-negative day_index, input columns, and preprocessing."
        )
    rows = []
    for (kind, key), state in groups.items():
        count = int(state["count"])
        if aggregation == "mean":
            power = float(state["sum"]) / count
        elif aggregation == "sum":
            power = float(state["sum"])
        else:
            # "first" means first valid observation in file/chunk order; this is not
            # meaningful until that ordering has a documented physical interpretation.
            power = float(state["first"])
        row: dict[str, Any] = {
            "predicted_wind_power_kw": power,
            "observations_aggregated": count,
            "aggregation_method": aggregation,
            "aggregation_interpretation": (
                cfg["alignment"].get("wind_aggregation_interpretation")
                or "UNDOCUMENTED: configured aggregation applied; interpretation requires dataset evidence"
            ),
            "aggregation_semantics_verified": (
                cfg["alignment"].get("wind_aggregation_semantics_verified") is True
                and _reference_present(cfg.get("alignment_verification_reference", {}).get("wind_aggregation_semantics_verified", ""))
            ),
            "calendar_status": (
                "RELATIVE_ONLY_MONTH_UNUSED_VERIFIED_NO_HISTORICAL_DATE"
                if kind == "relative" and relative_month_mode
                else ("RELATIVE_ONLY_CALENDAR_UNVERIFIED" if kind == "relative" else "DATED")
            ),
            "input_rows_seen": input_rows,
            "input_rows_rejected": rejected_rows,
            "power_semantics_status": (
                "VERIFIED" if cfg["alignment"].get("wind_power_semantics_verified") is True
                and _reference_present(cfg.get("alignment_verification_reference", {}).get("wind_power_semantics_verified", ""))
                else "UNVERIFIED_TURBINE_OR_FARM_SEMANTICS"
            ),
        }
        if kind == "timestamp":
            row["timestamp"] = pd.Timestamp(key)
        else:
            bucket_index = int(key)
            offset = pd.tseries.frequencies.to_offset(cfg["alignment"].get("frequency", "h"))
            bucket_minutes = offset.nanos / 60_000_000_000
            start_minute = bucket_index * bucket_minutes
            day = int(start_minute // 1440)
            minute_of_day = start_minute % 1440
            row.update(day_index=day, Hour=int(minute_of_day // 60), Minute=int(minute_of_day % 60), time_key=key)
        target_semantics = wind_evidence.get("target_semantics")
        if target_semantics == "farm_total_power":
            output_scope = "farm_total" if aggregation in ("mean", "first") else "UNKNOWN"
        elif target_semantics == "per_turbine_power":
            output_scope = "farm_total" if aggregation == "sum" else ("per_turbine_mean" if aggregation == "mean" else "selected_turbine")
        else:
            output_scope = "unknown"
        row["wind_target_semantics"] = target_semantics or "UNVERIFIED"
        row["wind_output_scope"] = output_scope
        row["wind_aggregation_evidence_reference"] = str(wind_evidence.get("reference", ""))
        rows.append(row)
    result = pd.DataFrame(rows)
    sort_cols = ["timestamp"] if "timestamp" in result else ["day_index", "Hour"]
    return result.sort_values(sort_cols).reset_index(drop=True)


def _normalize_alignment_timestamps(series: pd.Series, label: str, mode: str) -> pd.Series:
    """Parse timestamps without silently treating naive local time as UTC."""
    parsed_values: list[pd.Timestamp | Any] = []
    for value in series:
        if pd.isna(value):
            parsed_values.append(pd.NaT)
            continue
        try:
            parsed_values.append(pd.Timestamp(value))
        except (TypeError, ValueError, OverflowError):
            parsed_values.append(pd.NaT)
    aware_flags = {value.tzinfo is not None for value in parsed_values if not pd.isna(value)}
    if len(aware_flags) > 1:
        raise StageBlocked(f"{label} timestamps mix timezone-aware and timezone-naive values.")
    is_aware = next(iter(aware_flags), None)
    if is_aware is None:
        return pd.Series(pd.to_datetime(parsed_values, errors="coerce"), index=series.index)
    if is_aware:
        if mode != "aware_utc":
            raise StageBlocked(f"{label} timestamps are timezone-aware but alignment.timestamp_timezone_mode is not 'aware_utc'.")
        utc_values = [value.tz_convert("UTC") if not pd.isna(value) else pd.NaT for value in parsed_values]
        return pd.Series(pd.to_datetime(utc_values, errors="coerce", utc=True), index=series.index)
    if mode != "naive_same_timezone":
        raise StageBlocked(f"{label} timestamps are timezone-naive but alignment.timestamp_timezone_mode is not 'naive_same_timezone'.")
    return pd.Series(pd.to_datetime(parsed_values, errors="coerce"), index=series.index)


def align_renewables(cfg: dict[str, Any], solar: pd.DataFrame, wind: pd.DataFrame) -> pd.DataFrame:
    """Combine only time- and semantically-compatible power series; diagnostics retain coverage losses."""
    method = cfg["renewable_combination"].get("method", "sum")
    if method == "solar_only":
        if solar.empty or "timestamp" not in solar:
            raise StageBlocked("Solar-only renewable series unavailable.")
        result = solar[["timestamp", "predicted_solar_power_kw"]].copy()
        result["predicted_total_renewable_power_kw"] = result["predicted_solar_power_kw"]
        result["integration_status"] = "SOLAR_ONLY_NOT_FULL_RENEWABLE_INTEGRATION"
        result["physical_safety_status"] = "NOT_ESTABLISHED"
        return result
    if method == "wind_only":
        if wind.empty or "timestamp" not in wind:
            raise StageBlocked("Wind-only mode requires a verified Wind calendar timestamp origin.")
        result = wind[["timestamp", "predicted_wind_power_kw"]].copy()
        result["predicted_total_renewable_power_kw"] = result["predicted_wind_power_kw"]
        result["integration_status"] = "WIND_ONLY_NOT_FULL_RENEWABLE_INTEGRATION"
        result["physical_safety_status"] = "NOT_ESTABLISHED"
        return result
    if method != "sum":
        raise ConfigurationError(f"Unsupported renewable combination method: {method}")
    if solar.empty or wind.empty or "timestamp" not in wind or "timestamp" not in solar:
        raise StageBlocked("Cannot combine Solar/Wind: verified timestamps are unavailable.")

    alignment = cfg["alignment"]
    refs = cfg.get("alignment_verification_reference", {})
    for flag in ("timestamps_compatible_verified", "solar_power_semantics_verified", "wind_power_semantics_verified"):
        if alignment.get(flag) is not True or not _reference_present(refs.get(flag, "")):
            raise StageBlocked(f"Solar/Wind combination blocked: {flag} lacks traceable documentary evidence.")
    if alignment.get("wind_aggregation_semantics_verified") is not True or not _reference_present(refs.get("wind_aggregation_semantics_verified", "")):
        raise StageBlocked("Solar/Wind combination blocked: Wind aggregation semantics lack traceable evidence.")
    if alignment.get("solar_power_scope") != "farm_total" or alignment.get("wind_power_scope") != "farm_total":
        raise StageBlocked("Solar/Wind addition requires documented farm-total power scopes for both series.")
    if not _reference_present(refs.get("solar_power_scope", "")) or not _reference_present(refs.get("wind_power_scope", "")):
        raise StageBlocked("Solar/Wind farm-total scope claims require traceable supporting references.")
    if "wind_output_scope" not in wind.columns or set(wind["wind_output_scope"].dropna().astype(str)) != {"farm_total"}:
        raise StageBlocked("Wind output scope is not verified as farm-total power; mean per-turbine output cannot be added to Solar farm power.")
    mode = alignment.get("timestamp_timezone_mode")
    if mode not in ("aware_utc", "naive_same_timezone") or not _reference_present(refs.get("timestamp_timezone_mode", "")):
        raise StageBlocked("Solar/Wind combination blocked: timestamp timezone mode and its evidence reference are required.")

    s = solar.copy()
    w = wind.copy()
    for frame, col, label in (
        (s, "predicted_solar_power_kw", "Solar"),
        (w, "predicted_wind_power_kw", "Wind"),
    ):
        if col not in frame.columns:
            raise StageBlocked(f"{label} output lacks required power column {col!r}.")
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise StageBlocked(f"{label} power series contains non-finite or negative values; renewable addition is blocked.")
    freq = alignment.get("frequency", "h")
    try:
        frequency_offset = pd.tseries.frequencies.to_offset(freq)
        frequency_offset.nanos
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"alignment.frequency must be a valid fixed interval: {freq!r}.") from exc
    solar_before, wind_before = len(s), len(w)
    s["timestamp"] = _normalize_alignment_timestamps(s["timestamp"], "Solar", mode).dt.floor(freq)
    w["timestamp"] = _normalize_alignment_timestamps(w["timestamp"], "Wind", mode).dt.floor(freq)
    solar_has_aware = isinstance(s["timestamp"].dtype, pd.DatetimeTZDtype)
    wind_has_aware = isinstance(w["timestamp"].dtype, pd.DatetimeTZDtype)
    if solar_has_aware != wind_has_aware:
        raise StageBlocked("Solar/Wind timestamp timezone awareness differs; timestamps cannot be aligned safely.")
    s = s.loc[s["timestamp"].notna()].copy()
    w = w.loc[w["timestamp"].notna()].copy()
    solar_invalid_ts, wind_invalid_ts = solar_before - len(s), wind_before - len(w)
    solar_duplicate_rows = int(s.duplicated("timestamp", keep=False).sum())
    wind_duplicate_rows = int(w.duplicated("timestamp", keep=False).sum())
    solar_agg = alignment.get("solar_duplicate_timestamp_aggregation", "mean")
    wind_agg = alignment.get("wind_duplicate_timestamp_aggregation", "mean")
    allowed = {"mean", "sum", "first"}
    if solar_agg not in allowed or wind_agg not in allowed:
        raise ConfigurationError("Duplicate timestamp aggregation must be mean, sum, or first.")
    if solar_duplicate_rows and not (alignment.get("solar_duplicate_timestamp_semantics_verified") is True and _reference_present(refs.get("solar_duplicate_timestamp_semantics_verified", ""))):
        raise StageBlocked("Solar timestamps require aggregation, but its temporal aggregation semantics are unverified.")
    if wind_duplicate_rows and not (alignment.get("wind_duplicate_timestamp_semantics_verified") is True and _reference_present(refs.get("wind_duplicate_timestamp_semantics_verified", ""))):
        raise StageBlocked("Wind timestamps require aggregation, but its temporal aggregation semantics are unverified.")
    if solar_duplicate_rows and solar_agg == "first" and not (alignment.get("solar_first_order_verified") is True and _reference_present(refs.get("solar_first_order_verified", ""))):
        raise StageBlocked("Solar first aggregation depends on file order, which is not documented and verified.")
    if wind_duplicate_rows and wind_agg == "first" and not (alignment.get("wind_first_order_verified") is True and _reference_present(refs.get("wind_first_order_verified", ""))):
        raise StageBlocked("Wind first aggregation depends on file order, which is not documented and verified.")
    if solar_duplicate_rows and solar_agg == "sum" and not (alignment.get("solar_duplicate_timestamp_semantics_verified") is True and _reference_present(refs.get("solar_duplicate_timestamp_semantics_verified", ""))):
        raise StageBlocked("Solar sum aggregation requires documented duplicate/resampling semantics.")
    if wind_duplicate_rows and wind_agg == "sum" and not (alignment.get("wind_duplicate_timestamp_semantics_verified") is True and _reference_present(refs.get("wind_duplicate_timestamp_semantics_verified", ""))):
        raise StageBlocked("Wind sum aggregation requires documented duplicate/resampling semantics.")

    s = s.groupby("timestamp", as_index=False)["predicted_solar_power_kw"].agg(solar_agg)
    w = w.groupby("timestamp", as_index=False)["predicted_wind_power_kw"].agg(wind_agg)
    solar_ts, wind_ts = set(s["timestamp"]), set(w["timestamp"])
    overlap = solar_ts & wind_ts
    dropped_solar, dropped_wind = len(solar_ts - overlap), len(wind_ts - overlap)
    if not overlap:
        raise StageBlocked(
            "No overlapping Solar/Wind timestamps after frequency normalization; "
            f"solar_rows_before={solar_before}, wind_rows_before={wind_before}, "
            f"solar_unique={len(solar_ts)}, wind_unique={len(wind_ts)}, "
            f"solar_dropped={dropped_solar}, wind_dropped={dropped_wind}, "
            f"invalid_timestamps_solar={solar_invalid_ts}, invalid_timestamps_wind={wind_invalid_ts}."
        )
    result = s.merge(w, on="timestamp", how="inner", validate="one_to_one").sort_values("timestamp").reset_index(drop=True)
    result["predicted_total_renewable_power_kw"] = result["predicted_solar_power_kw"] + result["predicted_wind_power_kw"]
    result["solar_rows_before_alignment"] = solar_before
    result["wind_rows_before_alignment"] = wind_before
    result["solar_duplicate_timestamp_rows"] = solar_duplicate_rows
    result["wind_duplicate_timestamp_rows"] = wind_duplicate_rows
    result["solar_invalid_timestamp_rows"] = solar_invalid_ts
    result["wind_invalid_timestamp_rows"] = wind_invalid_ts
    result["solar_timestamps_dropped_no_overlap"] = dropped_solar
    result["wind_timestamps_dropped_no_overlap"] = dropped_wind
    result["overlap_start"] = result["timestamp"].min()
    result["overlap_end"] = result["timestamp"].max()
    result["integration_status"] = "VERIFIED_TIMESTAMP_AND_POWER_SEMANTICS; PHYSICAL_SAFETY_NOT_ESTABLISHED"
    result["physical_safety_status"] = "NOT_ESTABLISHED"
    result["timestamp_timezone_mode"] = mode
    result["alignment_frequency"] = freq
    return result


def generate_candidates(data: pd.DataFrame, cfg: dict[str, Any], features: list[str]) -> pd.DataFrame:
    """Generate candidates from observed multivariate rows or row-resampling preserving joint structure."""
    target = cfg["hydrogen_target"]
    required = list(dict.fromkeys(features + [target]))
    clean = numeric_frame(data, required)
    finite = np.isfinite(clean[features].to_numpy(dtype=float)).all(axis=1)
    finite &= np.isfinite(clean[target].to_numpy(dtype=float))
    clean = clean.loc[finite].copy()
    if clean.empty:
        raise StageBlocked("No complete finite Green Hydrogen rows are available for candidate generation.")
    duplicate_target_counts = clean.groupby(features, dropna=False)[target].nunique(dropna=False)
    conflicting_points = int((duplicate_target_counts > 1).sum())
    if conflicting_points:
        raise StageBlocked(
            f"{conflicting_points} operating-point feature combination(s) have conflicting observed target values. "
            "Do not silently select/average repeated measurements; document their semantics or resolve upstream."
        )
    clean = clean.drop_duplicates(subset=features + [target]).reset_index(drop=True)
    candidate_cfg = cfg["candidate_generation"]
    n = int(candidate_cfg["n_candidates"])
    method = candidate_cfg["method"]
    rng = np.random.default_rng(int(candidate_cfg.get("random_seed", 42)))
    decision = cfg["decision_variables"]
    missing_decisions = [c for c in decision if c not in features]
    if missing_decisions:
        raise ConfigurationError(f"Decision variables must be Green Hydrogen model features: {missing_decisions}")
    # Any configured bounds must be approved, explicitly enabled, finite, ordered, and within observed support
    # unless an approved extrapolation has been documented.
    ranges = cfg.get("approved_ranges", {})
    for col, spec in ranges.items():
        if col not in features:
            raise ConfigurationError(f"approved_ranges references non-feature {col!r}.")
        if spec.get("enabled", False):
            if not all(isinstance(spec.get(k), str) and spec.get(k, "").strip() for k in ("approved_by", "approval_reference", "units")):
                raise ConfigurationError(
                    f"Enabled approved range for {col} needs non-empty approved_by, approval_reference, and units strings."
                )
            lo, hi = float(spec["min"]), float(spec["max"])
            if not np.isfinite([lo, hi]).all() or lo > hi:
                raise ConfigurationError(f"Invalid finite range for {col}: min={lo}, max={hi}.")
            observed_lo, observed_hi = float(clean[col].min()), float(clean[col].max())
            extends_support = lo < observed_lo or hi > observed_hi
            if extends_support and not spec.get("allow_outside_observed", False):
                raise ConfigurationError(
                    f"Approved range for {col} extends outside observed data support; explicitly document "
                    "extrapolation before enabling."
                )
            if extends_support and spec.get("allow_outside_observed", False) and not str(spec.get("extrapolation_reference", "")).strip():
                raise ConfigurationError(
                    f"Approved range for {col} extends outside observed support and requires extrapolation_reference evidence."
                )
    observed_target_verified = (
        cfg.get("observed_target_units_verified", {}).get("hydrogen") is True
        and _reference_present(cfg.get("observed_target_units_verification_reference", {}).get("hydrogen", ""))
    )
    observed_target_column = (
        f"observed_{target}" if observed_target_verified
        else "observed_hydrogen_target_value_unverified_units"
    )
    if method == "observed_rows":
        # Rank actual observed rows only. This does not explore new operating conditions.
        if len(clean) > n:
            indices = np.sort(rng.choice(len(clean), size=n, replace=False))
            selected = clean.iloc[indices][features + [target]].copy()
        else:
            selected = clean[features + [target]].copy()
        candidates = selected[features].copy()
        candidates[observed_target_column] = selected[target].to_numpy()
        candidates["candidate_source"] = "observed_row"
    elif method == "correlated_resample":
        # Bootstrap complete observed rows (legacy method semantics), then deduplicate.
        # Repeated draws are not distinct operating conditions and cannot create an improved optimum.
        indices = rng.choice(len(clean), size=n, replace=True)
        selected = clean.iloc[indices][features + [target]].reset_index(drop=True).copy()
        candidates = selected[features].copy()
        candidates[observed_target_column] = selected[target].to_numpy()
        candidates["candidate_source"] = "observed_row_bootstrap_deduplicated"
    else:
        raise ConfigurationError(f"Unsupported candidate generation method {method!r}.")
    candidates = candidates.drop_duplicates(subset=features).reset_index(drop=True)
    candidates.insert(0, "candidate_id", np.arange(1, len(candidates) + 1))
    candidates["candidate_generation_method"] = method
    candidates["candidate_basis"] = "Observed multivariate operating rows; data support is not a safety envelope"
    candidates["observed_row_support"] = True
    candidates["candidate_is_synthetic"] = False
    candidates["search_interpretation"] = "RANKING_OBSERVED_OPERATING_POINTS_ONLY"
    candidates["physical_safety_status"] = "NOT_ESTABLISHED"
    candidates["feasibility_scope"] = "Only explicitly configured, evidence-referenced constraints are evaluated; this is not a complete process-safety assessment."
    candidates["requested_candidate_draws"] = n
    candidates["unique_candidate_count_after_deduplication"] = len(candidates)
    candidates["target_units_status"] = (
        "VERIFIED" if cfg.get("target_units_verified", {}).get("hydrogen", False)
        else "UNVERIFIED_TARGET_SCALE_OR_UNITS"
    )
    candidates["observed_target_units_status"] = (
        "VERIFIED" if observed_target_verified else "UNVERIFIED_SOURCE_TARGET_SCALE_OR_UNITS"
    )
    # Explicit configured bounds are checks only; they do not modify candidate values.
    for col, spec in ranges.items():
        if spec.get("enabled", False):
            lo, hi = float(spec["min"]), float(spec["max"])
            candidates[f"approved_range_{col}_pass"] = candidates[col].between(lo, hi, inclusive="both")
    return candidates


def evaluate_constraints(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[str], bool]:
    """Evaluate configured scalar constraints and return per-row status, used names, and verification state."""
    out = df.copy()
    used: list[str] = []
    all_verified = True
    pass_masks: list[pd.Series] = []
    for index, rule in enumerate(cfg.get("constraints", []), 1):
        name = str(rule.get("name", f"constraint_{index}"))
        if not name.strip():
            raise ConfigurationError(f"Constraint at index {index-1} has an empty name.")
        if not rule.get("enabled", False):
            continue
        used.append(name)
        # A JSON boolean is not evidence by itself. Require a traceable source/approval reference
        # and the identity of the person/body that verified the rule.
        evidence_ref = rule.get("verification_reference") or rule.get("approval_reference") or rule.get("source_reference")
        verified_by = rule.get("verified_by") or rule.get("approved_by")
        declared_verified = rule.get("verified", False) is True
        verified = bool(
            declared_verified
            and isinstance(evidence_ref, str) and evidence_ref.strip()
            and isinstance(verified_by, str) and verified_by.strip()
            and isinstance(rule.get("units"), str) and rule.get("units", "").strip()
        )
        all_verified = all_verified and verified
        column, operator, value = rule.get("column"), rule.get("operator"), rule.get("value")
        if not isinstance(column, str) or column not in out.columns:
            raise ConfigurationError(f"Constraint {name!r} references missing column {column!r}.")
        series = out[column]
        if operator in ("<=", "<", ">=", ">"):
            numeric = pd.to_numeric(series, errors="coerce")
            try:
                if isinstance(value, bool):
                    raise ValueError("boolean is not a numeric threshold")
                threshold = float(value)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(f"Constraint {name!r} needs a numeric value.") from exc
            if not np.isfinite(threshold):
                raise ConfigurationError(f"Constraint {name!r} value must be finite.")
            if operator == "<=": passed = numeric <= threshold
            elif operator == "<": passed = numeric < threshold
            elif operator == ">=": passed = numeric >= threshold
            else: passed = numeric > threshold
        elif operator == "between":
            if not isinstance(value, list) or len(value) != 2:
                raise ConfigurationError(f"Constraint {name!r} between value must be [min, max].")
            try:
                if any(isinstance(bound, bool) for bound in value):
                    raise ValueError("boolean is not a numeric bound")
                lo, hi = map(float, value)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(f"Constraint {name!r} between bounds must be numeric.") from exc
            if not np.isfinite([lo, hi]).all() or lo > hi:
                raise ConfigurationError(f"Constraint {name!r} has invalid between bounds.")
            numeric = pd.to_numeric(series, errors="coerce")
            passed = numeric.between(lo, hi, inclusive="both")
        elif operator in ("==", "!="):
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"Constraint {name!r} categorical comparison requires a non-empty string value.")
            if pd.api.types.is_numeric_dtype(series.dtype):
                raise ConfigurationError(f"Constraint {name!r} uses categorical comparison on a numeric column; use a numeric operator.")
            eq = series.astype("string").str.casefold() == value.strip().casefold()
            passed = (series.notna() & eq) if operator == "==" else (series.notna() & ~eq)
        else:
            raise ConfigurationError(f"Constraint {name!r} uses unsupported operator {operator!r}.")
        passed = passed.fillna(False).astype(bool)
        pass_col = f"constraint_{name}_pass"
        if pass_col in out.columns:
            raise ConfigurationError(f"Constraint {name!r} would overwrite existing output column {pass_col!r}.")
        out[pass_col] = passed
        out[f"constraint_{name}_verified"] = verified
        out[f"constraint_{name}_verification_reference"] = str(evidence_ref or "")
        out[f"constraint_{name}_units"] = str(rule.get("units") or "")
        pass_masks.append(passed)
    if not used:
        out["feasibility_status"] = "UNVERIFIED_NO_ENABLED_CONSTRAINTS"
        return out, used, False
    mask = pd.Series(True, index=out.index)
    for passed in pass_masks:
        mask &= passed
    if not all_verified:
        out["feasibility_status"] = np.where(mask, "UNVERIFIED", "INFEASIBLE")
    else:
        out["feasibility_status"] = np.where(mask, "FEASIBLE", "INFEASIBLE")
    return out, used, all_verified


def apply_fuel_cell_candidate_gate(
    candidates: pd.DataFrame, cfg: dict[str, Any], used_constraints: list[str], constraints_verified: bool
) -> tuple[pd.DataFrame, list[str], bool]:
    """Prevent FEASIBLE ranking when Fuel Cell Pass is mandatory but no validated candidate mapping exists."""
    fc = cfg.get("fuel_cell_pass_constraint", {})
    if not fc.get("enabled", False):
        return candidates, used_constraints, constraints_verified
    out = candidates.copy()
    out["fuel_cell_candidate_constraint_status"] = "BLOCKED_MISSING_VALIDATED_CANDIDATE_INPUT_MAPPING"
    if "feasibility_status" not in out:
        out["feasibility_status"] = "UNVERIFIED"
    else:
        out.loc[out["feasibility_status"] != "INFEASIBLE", "feasibility_status"] = "UNVERIFIED"
    if "fuel_cell_pass_constraint" not in used_constraints:
        used_constraints = list(used_constraints) + ["fuel_cell_pass_constraint"]
    return out, used_constraints, False



def fuel_cell_diagnostics(
    cfg: dict[str, Any],
    model: Any,
    scaler: Any | None,
    features: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Diagnose Fuel Cell predictions without inventing candidate-level input mappings."""
    path = required_file(
        cfg["dataset_dir"], cfg["fuel_cell_dataset_file"], required=False
    )
    classes = list(getattr(model, "classes_", []))
    label_cfg = cfg.get("fuel_cell_pass_constraint", {})
    required_label = str(label_cfg.get("required_class_label", "Pass")).strip()

    evidence = {
        "model_classes": [str(c) for c in classes],
        "configured_required_label": required_label,
        "candidate_level_integration_status":
            "BLOCKED_MISSING_VALIDATED_FUEL_CELL_MAPPING",
    }

    if path is None:
        return pd.DataFrame(), {
            **evidence, "status": "BLOCKED_MISSING_FUEL_CELL_DATASET"
        }

    data = pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in features if c not in data.columns]
    if missing:
        return pd.DataFrame(), {
            **evidence,
            "status": "BLOCKED_MISSING_FUEL_CELL_INPUTS",
            "details": str(missing),
        }

    data = numeric_frame(data, features)
    valid = np.isfinite(data[features].to_numpy(dtype=float)).all(axis=1)
    clean = data.loc[valid].copy()

    if clean.empty:
        return pd.DataFrame(), {
            **evidence, "status": "BLOCKED_NO_VALID_FUEL_CELL_ROWS"
        }

    try:
        require_verified_preprocessing(cfg, "fuel_cell")
        X = prepare_input(
            clean,
            model,
            features,
            scaler,
            cfg["input_modes"]["fuel_cell"],
            "Fuel Cell classifier",
            scaler_order_verified=cfg.get(
                "scaler_feature_order_verified", {}
            ).get("fuel_cell", False),
            model_order_verified=cfg.get(
                "model_feature_order_verified", {}
            ).get("fuel_cell", False),
            scaler_order_reference=cfg.get("scaler_feature_order_reference", {}).get("fuel_cell", ""),
            model_order_reference=cfg.get("model_feature_order_reference", {}).get("fuel_cell", ""),
            scaler_feature_order=cfg.get("scaler_feature_order", {}).get("fuel_cell"),
        )

        predictions = np.asarray(model.predict(X)).reshape(-1)
        if len(predictions) != len(clean):
            raise RuntimeError(
                "Fuel Cell classifier returned a mismatched prediction count."
            )

        if not classes:
            raise StageBlocked(
                "Fuel Cell classifier does not expose classes_; "
                "class semantics and probability order cannot be verified."
            )

        result = clean.copy()
        result["predicted_class_raw"] = [str(v) for v in predictions]

        mapping = label_cfg.get("class_label_mapping", {})
        mapping_reference = label_cfg.get("class_mapping_reference")
        mapping_verified = (
            label_cfg.get("class_mapping_verified") is True
            and _reference_present(mapping_reference)
            and isinstance(mapping, dict)
        )
        normalized_classes = [str(c).strip().casefold() for c in classes]
        native_pass_fail = (
            all(isinstance(c, (str, np.str_)) for c in classes)
            and all(c in {"pass", "fail"} for c in normalized_classes)
            and len(set(normalized_classes)) == len(normalized_classes)
        )

        # Even native-looking Pass/Fail labels need training-target/source evidence.
        # A label's spelling is not proof of what the model was trained to predict.
        semantic_classes: dict[str, str] = {}
        class_semantics_verified = False
        if mapping_verified:
            normalized_mapping: dict[str, str] = {}
            duplicate_keys = False
            for key, value in mapping.items():
                skey = str(key)
                if skey in normalized_mapping:
                    duplicate_keys = True
                normalized_mapping[skey] = str(value).strip()
            mapped_values = [normalized_mapping.get(str(c)) for c in classes]
            expected_keys = {str(c) for c in classes}
            valid_mapping = (
                not duplicate_keys
                and set(normalized_mapping) == expected_keys
                and all(value is not None and value != "" for value in mapped_values)
                and len({value.casefold() for value in mapped_values}) == len(mapped_values)
            )
            if valid_mapping:
                semantic_classes = normalized_mapping
                class_semantics_verified = True

        # Predictions must be actual model classes; otherwise class association is untrustworthy.
        for value in predictions:
            if not any(value == model_class for model_class in classes):
                raise RuntimeError(f"Classifier predicted label {value!r} that is absent from model.classes_.")

        result["predicted_outcome"] = [
            semantic_classes.get(str(value), "UNMAPPED_CLASS:" + str(value))
            for value in predictions
        ]
        result["required_label_configured"] = required_label
        required_label_exists = (
            class_semantics_verified
            and sum(
                str(semantic_classes.get(str(c), "")).casefold() == required_label.casefold()
                for c in classes
            ) == 1
        )
        result["required_label_exists_in_model_classes"] = required_label_exists
        result["class_mapping_verified"] = class_semantics_verified

        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(X), dtype=float)
            if probabilities.shape != (len(clean), len(classes)):
                raise RuntimeError(
                    "Classifier predict_proba dimensions do not match the number of input rows and classes_."
                )
            if (
                not np.isfinite(probabilities).all()
                or (probabilities < 0).any()
                or (probabilities > 1).any()
            ):
                raise RuntimeError("Classifier returned non-finite probabilities or values outside [0, 1].")
            row_sums = probabilities.sum(axis=1)
            if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-6):
                raise RuntimeError(
                    "Classifier probability rows must sum to 1 within absolute tolerance 1e-6; "
                    f"observed range={float(row_sums.min())}..{float(row_sums.max())}."
                )
            result["probability_row_sum"] = row_sums
            result["probabilities_validated"] = True
            matching = [
                i for i, c in enumerate(classes)
                if class_semantics_verified
                and str(semantic_classes.get(str(c), "")).casefold() == required_label.casefold()
            ]
            if required_label_exists and len(matching) == 1:
                # Probability columns follow model.classes_ order, not a presumed Pass/Fail order.
                result["required_label_probability"] = probabilities[:, matching[0]]
        else:
            result["probabilities_validated"] = False

        if not class_semantics_verified:
            if not all(isinstance(c, (str, np.str_)) for c in classes):
                status = "BLOCKED_NUMERIC_CLASS_MAPPING_UNVERIFIED"
            else:
                status = "BLOCKED_CLASS_MAPPING_UNVERIFIED"
        elif not required_label_exists:
            status = "BLOCKED_REQUIRED_LABEL_NOT_IN_CLASSES"
        else:
            status = "PASS"

        return result, {
            **evidence,
            "status": status,
            "valid_rows": len(result),
            "class_mapping_verified": class_semantics_verified,
            "class_mapping_reference": mapping_reference,
            "semantic_class_mapping": semantic_classes,
            "native_pass_fail_labels_observed": native_pass_fail,
            "probability_row_sum_absolute_tolerance": 1e-6,
        }

    except Exception as exc:
        return pd.DataFrame(), {
            **evidence,
            "status": "BLOCKED_CLASSIFIER_INFERENCE",
            "details": f"{type(exc).__name__}: {exc}",
        }


def write_diagnostics(
    out_dir: Path, filename: str, rows: list[StageResult | dict[str, Any]], run_id: str = ""
) -> None:
    normalized = []
    for row in rows:
        if isinstance(row, StageResult):
            normalized.append({"run_id": run_id or row.run_id, "stage": row.name, "status": row.status, "details": row.details, "rows": row.rows})
        else:
            normalized.append({"run_id": run_id, **row})
    atomic_to_csv(pd.DataFrame(normalized), _safe_output_path(out_dir, filename))


def invalidate_stale_outputs(out_dir: Path, outputs: dict[str, Any], run_id: str | None = None) -> list[str]:
    """Invalidate configured outputs; archive prior results when a run ID is available."""
    invalidated: list[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_root = out_dir.resolve() / "_previous_runs"
    if run_id and archive_root.is_symlink():
        raise ConfigurationError("outputs/_previous_runs must not be a symbolic link.")
    archive_dir = archive_root / run_id if run_id else None
    if archive_dir is not None and archive_dir.is_symlink():
        raise ConfigurationError(f"Run archive path is a symbolic link: {archive_dir}")
    for output_name in outputs.values():
        if isinstance(output_name, str) and output_name:
            stale_path = _safe_output_path(out_dir, output_name)
            if stale_path.is_file():
                if archive_dir is not None:
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    archive_path = archive_dir / stale_path.name
                    os.replace(stale_path, archive_path)
                else:
                    stale_path.unlink()
                invalidated.append(output_name)
    return invalidated


def write_hydrogen_exploratory_reports(
    cfg: dict[str, Any], data: pd.DataFrame, candidates: pd.DataFrame, out: Path
) -> dict[str, int]:
    """Diagnostic-only: observed data support and ranked model predictions, never approved optima."""
    features = list(cfg["hydrogen_features"])
    decision = list(cfg["decision_variables"])
    prediction = "predicted_hydrogen_production_kg_h"
    summary_rows = []
    for feature in features:
        values = pd.to_numeric(data[feature], errors="coerce") if feature in data else pd.Series(dtype=float)
        values = values[np.isfinite(values)]
        summary_rows.append({
            "variable": feature,
            "is_decision_variable": feature in decision,
            "observed_count": int(len(values)),
            "missing_or_nonfinite_count": int(len(data) - len(values)),
            "observed_min": float(values.min()) if len(values) else np.nan,
            "observed_p05": float(values.quantile(0.05)) if len(values) else np.nan,
            "observed_median": float(values.median()) if len(values) else np.nan,
            "observed_p95": float(values.quantile(0.95)) if len(values) else np.nan,
            "observed_max": float(values.max()) if len(values) else np.nan,
            "operating_limit_status": "NOT_APPROVED_OBSERVED_VALUES_ONLY",
            "physical_safety_status": "NOT_ESTABLISHED",
        })
    atomic_to_csv(pd.DataFrame(summary_rows), out / "hydrogen_observed_support_diagnostic.csv")
    cols = ["candidate_id"] + [c for c in decision if c in candidates.columns]
    cols += [c for c in ("renewable_power_kw", prediction, "candidate_source", "feasibility_status")
             if c in candidates.columns and c not in cols]
    valid = candidates.copy()
    if prediction in valid:
        predicted = pd.to_numeric(valid[prediction], errors="coerce")
        valid = valid.loc[np.isfinite(predicted)].copy()
        valid = valid.sort_values(prediction, ascending=False, kind="stable")
    else:
        valid = valid.iloc[0:0].copy()
    result = valid.loc[:, [c for c in cols if c in valid.columns]].head(int(cfg["candidate_generation"].get("top_n", 100))).copy()
    result.insert(0, "exploratory_rank", range(1, len(result) + 1))
    result["decision_status"] = "EXPLORATORY_ONLY_NOT_AN_APPROVED_OPERATING_RECOMMENDATION"
    result["physical_safety_status"] = "NOT_ESTABLISHED"
    result["optimality_status"] = "OBSERVED_CANDIDATES_ONLY_NOT_GLOBAL_OPTIMUM"
    result["renewable_integration_status"] = "NOT_VALIDATED_WITH_SOLAR_AND_WIND"
    atomic_to_csv(result, out / "hydrogen_exploratory_ranking.csv")
    return {"observed_feature_rows": len(summary_rows), "exploratory_ranked_rows": len(result)}


def write_integration_readiness(out: Path, diagnostics: list[StageResult | dict[str, Any]], run_id: str) -> pd.DataFrame:
    """Summarize current-run stage evidence; never infer feasibility from stage execution."""
    observed: dict[str, dict[str, Any]] = {}
    for item in diagnostics:
        if isinstance(item, StageResult):
            observed[item.name] = {"status": item.status, "details": item.details, "rows": item.rows}
        else:
            observed[str(item.get("stage", ""))] = item
    specs = (
        ("solar_prediction", "MODEL_INFERENCE", "Power scope/unit must be verified before integration"),
        ("wind_prediction", "MODEL_INFERENCE", "Raw diagnostics do not establish usable generation or historical timestamps"),
        ("hydrogen_prescriptive_search", "CANDIDATE_SEARCH", "Approved operating constraints and safety case required"),
        ("hydrogen_exploratory_reporting", "DIAGNOSTIC_REPORT", "Observed ranges are not safe equipment limits"),
        ("fuel_cell_diagnostics", "HISTORICAL_CLASSIFICATION", "Diagnostic PASS does not verify candidate input mapping"),
        ("renewable_alignment", "INTEGRATION", "Requires compatible timestamps, power units and scopes"),
        ("fuel_cell_candidate_integration", "INTEGRATION", "Requires validated kg/h to L/min process mapping"),
    )
    rows = []
    for name, kind, requirement in specs:
        record = observed.get(name, {})
        status = str(record.get("status", "NOT_RECORDED"))
        rows.append({
            "run_id": run_id, "stage": name, "stage_type": kind,
            "execution_status": status, "reported_rows": record.get("rows", ""),
            "execution_detail": str(record.get("details", "") or ""),
            "scientific_approval": "NOT_ESTABLISHED",
            "outstanding_evidence": requirement,
        })
    report = pd.DataFrame(rows)
    atomic_to_csv(report, _safe_output_path(out, "integration_readiness.csv"))
    return report


def run_engine(config_path: Path | str | None = None) -> dict[str, Any]:
    """Run available stages independently; blocked stages produce explicit diagnostics."""
    cfg = read_config(config_path)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    # Move configured prior outputs to an archive before execution so failed stages cannot expose stale results.
    removed_outputs = invalidate_stale_outputs(out, cfg.get("outputs", {}), run_id=run_id)
    diagnostics: list[StageResult | dict[str, Any]] = []
    diagnostics.append(StageResult(
        "stale_output_invalidation", "PASS",
        f"Archived/invalidated {len(removed_outputs)} configured prior-run output file(s): {removed_outputs}",
        len(removed_outputs),
        run_id,
    ))
    print(f"Project root: {Path(config_path).resolve().parent if config_path else ROOT}")

    feature_lists: dict[str, list[str]] = {}
    feature_errors: dict[str, str] = {}
    for name in ("solar", "wind", "hydrogen", "fuel_cell"):
        try:
            feature_lists[name] = load_feature_file(cfg, name, cfg[f"{name}_features"])
        except Exception as exc:
            # Keep the configured order visible for diagnostics, but never use it to
            # bypass a configured selected-feature file that is missing or inconsistent.
            feature_lists[name] = list(cfg[f"{name}_features"])
            feature_errors[name] = f"{type(exc).__name__}: {exc}"
            diagnostics.append(StageResult(
                f"{name}_feature_contract", "BLOCKED", feature_errors[name]
            ))
    assets = load_assets(cfg)
    preprocessing_rows = []
    for name in ("solar", "wind", "hydrogen", "fuel_cell"):
        if name not in assets:
            preprocessing_rows.append({
                "component": name, "configured_input_mode": cfg["input_modes"][name],
                "preprocessing_verified": cfg["input_modes_verified"][name],
                "target_units_verified": cfg.get("target_units_verified", {}).get(name, False),
                "inference_allowed_by_config": False, "model_type": "", "scaler_type": "",
                "model_feature_names": "", "configured_feature_order": json.dumps(feature_lists[name]),
                "note": assets.get("__errors__", {}).get(name, "Model asset unavailable"),
            })
            continue
        model, scaler = assets[name]
        preprocessing_rows.append({
            "component": name,
            "configured_input_mode": cfg["input_modes"][name],
            "preprocessing_verified": cfg["input_modes_verified"][name],
            "target_units_verified": cfg.get("target_units_verified", {}).get(name, False),
            "inference_allowed_by_config": (
                cfg["input_modes_verified"].get(name) is True
                and bool(str(cfg.get("input_modes_verification_reference", {}).get(name, "")).strip())
                and (name == "fuel_cell" or (
                    cfg.get("target_units_verified", {}).get(name) is True
                    and bool(str(cfg.get("target_units_verification_reference", {}).get(name, "")).strip())
                ))
                and (model_names(model) is not None or (
                    cfg.get("model_feature_order_verified", {}).get(name) is True
                    and bool(str(cfg.get("model_feature_order_reference", {}).get(name, "")).strip())
                ))
                and (cfg["input_modes"].get(name) != "scaled" or scaler_names(scaler) is not None or (
                    cfg.get("scaler_feature_order_verified", {}).get(name) is True
                    and bool(str(cfg.get("scaler_feature_order_reference", {}).get(name, "")).strip())
                ))
                and name not in feature_errors
            ),
            "model_type": type(model).__name__,
            "scaler_type": type(scaler).__name__ if scaler is not None else "",
            "model_feature_names": json.dumps(model_names(model)),
            "configured_feature_order": json.dumps(feature_lists[name]),
            "model_feature_order_verified": cfg.get("model_feature_order_verified", {}).get(name, False),
            "scaler_feature_order_verified": cfg.get("scaler_feature_order_verified", {}).get(name, False),
            "note": "Every true verification flag must be backed by the corresponding traceable reference and original training evidence",
        })
        if cfg["input_modes"][name] == "scaled" and scaler is None:
            preprocessing_rows[-1]["note"] = "BLOCKED: scaled mode configured but scaler is unavailable"
    atomic_to_csv(pd.DataFrame(preprocessing_rows), _safe_output_path(out, cfg["outputs"]["preprocessing"]))

    # Hydrogen candidate search uses only the Hydrogen dataset. It does not mix in unrelated renewable rows.
    hydrogen_data = pd.DataFrame()
    candidates_path = out / cfg["outputs"]["all_candidates"]
    hydrogen_preblocked = feature_errors.get("hydrogen")
    try:
        if hydrogen_preblocked:
            raise StageBlocked(hydrogen_preblocked)
        hydrogen_path = required_file(cfg["dataset_dir"], cfg["hydrogen_dataset_file"])
        assert hydrogen_path is not None
        hydrogen_data = pd.read_csv(hydrogen_path, encoding="utf-8-sig")
        candidates = generate_candidates(hydrogen_data, cfg, feature_lists["hydrogen"])
    except Exception as exc:
        hydrogen_preblocked = f"{type(exc).__name__}: {exc}"
        empty_observed_target = (
            "observed_hydrogen_production_kg_h"
            if cfg.get("observed_target_units_verified", {}).get("hydrogen") is True
            and _reference_present(cfg.get("observed_target_units_verification_reference", {}).get("hydrogen", ""))
            else "observed_hydrogen_target_value_unverified_units"
        )
        candidates = pd.DataFrame(columns=feature_lists["hydrogen"] + [
            empty_observed_target, "candidate_id", "candidate_source"
        ])
    # Document observed support separately from approved operating bounds.
    support_rows = []
    for feature in feature_lists["hydrogen"]:
        if feature in hydrogen_data:
            vals = pd.to_numeric(hydrogen_data[feature], errors="coerce")
            vals = vals[np.isfinite(vals)]
        else:
            vals = pd.Series(dtype=float)
        support_rows.append({
            "feature": feature,
            "observed_min": float(vals.min()) if len(vals) else np.nan,
            "observed_max": float(vals.max()) if len(vals) else np.nan,
            "interpretation": "Observed data support only; not a safety limit",
        })
    atomic_to_csv(pd.DataFrame(support_rows), _safe_output_path(out, cfg["outputs"]["bounds"]))
    candidates["predicted_hydrogen_production_kg_h"] = np.nan
    hydrogen_status = "BLOCKED"
    try:
        if hydrogen_preblocked:
            raise StageBlocked(hydrogen_preblocked)
        if "hydrogen" not in assets:
            raise StageBlocked(assets.get("__errors__", {}).get("hydrogen", "Hydrogen model unavailable"))
        require_verified_preprocessing(cfg, "hydrogen")
        require_verified_target_units(cfg, "hydrogen")
        model, scaler = assets["hydrogen"]
        candidates[cfg["hydrogen_target"]] = predict(
            model, candidates, feature_lists["hydrogen"], scaler,
            cfg["input_modes"]["hydrogen"], "Green Hydrogen model",
            scaler_order_verified=cfg.get("scaler_feature_order_verified", {}).get("hydrogen", False),
            model_order_verified=cfg.get("model_feature_order_verified", {}).get("hydrogen", False),
            scaler_order_reference=cfg.get("scaler_feature_order_reference", {}).get("hydrogen", ""),
            model_order_reference=cfg.get("model_feature_order_reference", {}).get("hydrogen", ""),
            scaler_feature_order=cfg.get("scaler_feature_order", {}).get("hydrogen"),
        ).astype(float)
        # Avoid overwriting a measured target in candidate input; preserve prediction under explicit name.
        candidates["predicted_hydrogen_production_kg_h"] = candidates[cfg["hydrogen_target"]]
        candidates, used_constraints, constraints_verified = evaluate_constraints(candidates, cfg)
        # Explicitly approved operating bounds are hard constraints only when enabled and backed by approval evidence.
        enabled_ranges = [(col, spec) for col, spec in cfg.get("approved_ranges", {}).items()
                          if spec.get("enabled", False)]
        if enabled_ranges:
            range_mask = pd.Series(True, index=candidates.index)
            for col, spec in enabled_ranges:
                range_mask &= candidates[f"approved_range_{col}_pass"].fillna(False).astype(bool)
            candidates["approved_ranges_pass"] = range_mask
            candidates["approved_ranges_verified"] = all(
                bool(spec.get("approved_by") and spec.get("approval_reference") and spec.get("units"))
                for _, spec in enabled_ranges
            )
            if used_constraints:
                candidates.loc[~range_mask, "feasibility_status"] = "INFEASIBLE"
                if not candidates["approved_ranges_verified"].all():
                    candidates.loc[range_mask, "feasibility_status"] = "UNVERIFIED"
                    constraints_verified = False
            else:
                # Explicitly approved bounds may establish feasibility only with traceable approval metadata.
                if candidates["approved_ranges_verified"].all():
                    candidates["feasibility_status"] = np.where(range_mask, "FEASIBLE", "INFEASIBLE")
                    used_constraints = ["approved_ranges"]
                    constraints_verified = True
                else:
                    candidates["feasibility_status"] = np.where(range_mask, "UNVERIFIED", "INFEASIBLE")
                    used_constraints = ["approved_ranges"]
                    constraints_verified = False
        candidates, used_constraints, constraints_verified = apply_fuel_cell_candidate_gate(
            candidates, cfg, used_constraints, constraints_verified
        )
        objective = cfg["candidate_generation"]["objective"]
        direction = cfg["candidate_generation"]["ranking_direction"]
        if objective not in candidates:
            raise ConfigurationError(f"Objective column {objective!r} does not exist in candidate results.")
        if not used_constraints:
            candidates["feasibility_status"] = "UNVERIFIED_NO_ENABLED_CONSTRAINTS"
            ranked = candidates.iloc[0:0].copy()
            hydrogen_status = "PREDICTIONS_ONLY_NO_VERIFIED_FEASIBILITY"
        elif not constraints_verified:
            candidates.loc[candidates["feasibility_status"] == "UNVERIFIED_CONSTRAINTS", "feasibility_status"] = "UNVERIFIED"
            ranked = candidates.iloc[0:0].copy()
            hydrogen_status = "PREDICTIONS_ONLY_CONSTRAINTS_UNVERIFIED"
        else:
            ranked = candidates.loc[candidates["feasibility_status"] == "FEASIBLE"].copy()
            hydrogen_status = "OBSERVED_CANDIDATE_RANKING_COMPLETED" if not ranked.empty else "NO_FEASIBLE_OBSERVED_CANDIDATES"
        candidates["objective_name"] = objective
        candidates["ranking_direction"] = direction
        candidates["optimality_claim"] = "Best candidate found in generated candidate set; not proof of global optimum or safety"
        atomic_to_csv(candidates, candidates_path)
        ranked = ranked.sort_values(objective, ascending=(direction == "minimize")).reset_index(drop=True)
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        atomic_to_csv(ranked, _safe_output_path(out, cfg["outputs"]["feasible"]))
        atomic_to_csv(ranked.head(int(cfg["candidate_generation"]["top_n"])), _safe_output_path(out, cfg["outputs"]["top"]))
        best_path = out / cfg["outputs"]["best"]
        if hydrogen_status == "OBSERVED_CANDIDATE_RANKING_COMPLETED" and not ranked.empty:
            atomic_to_csv(ranked.head(1), best_path)
        elif best_path.exists():
            best_path.unlink()
        diagnostics.append(StageResult("hydrogen_prescriptive_search", hydrogen_status,
                                       f"method={cfg['candidate_generation']['method']}; constraints={used_constraints}; "
                                       f"verified={constraints_verified}", len(candidates)))
    except Exception as exc:
        atomic_to_csv(candidates, candidates_path)
        # Do not create a best-candidate artifact from unverified predictions/constraints.
        for output_key in ("feasible", "top", "best"):
            path = out / cfg["outputs"][output_key]
            if path.exists():
                path.unlink()
        diagnostics.append(StageResult("hydrogen_prescriptive_search", "BLOCKED", f"{type(exc).__name__}: {exc}", len(candidates)))
        print(f"[BLOCKED] Hydrogen prescriptive search: {exc}")

    # Independent hydrogen evidence summaries; no constraint or safety gate is changed.
    try:
        audit_counts = write_hydrogen_exploratory_reports(cfg, hydrogen_data, candidates, out)
        diagnostics.append(StageResult(
            "hydrogen_exploratory_reporting", "DIAGNOSTIC_ONLY_NOT_APPROVED",
            f"Observed feature rows={audit_counts['observed_feature_rows']}; "
            f"exploratory candidates={audit_counts['exploratory_ranked_rows']}; "
            "neither are safe operating limits or validated optimums",
            audit_counts["exploratory_ranked_rows"],
        ))
    except Exception as exc:
        diagnostics.append(StageResult(
            "hydrogen_exploratory_reporting", "BLOCKED", f"{type(exc).__name__}: {exc}"
        ))
        print(f"[BLOCKED] Hydrogen exploratory reporting: {exc}")

    # Solar and Wind predictions are independent stages; alignment has additional gates.
    solar = pd.DataFrame()
    wind = pd.DataFrame()
    for name, fn, model_key, output_key in (
        ("solar", solar_power, "solar", "solar"),
        ("wind", wind_power, "wind", "wind"),
    ):
        try:
            if model_key in feature_errors:
                raise StageBlocked(feature_errors[model_key])
            if model_key not in assets:
                raise StageBlocked(assets.get("__errors__", {}).get(model_key, f"{model_key} model unavailable"))
            model, scaler = assets[model_key]
            result = fn(cfg, model, scaler, feature_lists[model_key])
            atomic_to_csv(result, _safe_output_path(out, cfg["outputs"][output_key]))
            if name == "solar":
                solar = result
            else:
                wind = result
            diagnostics.append(StageResult(f"{name}_prediction", "PASS", "Prediction output created", len(result)))
        except Exception as exc:
            stale = out / cfg["outputs"][output_key]
            if stale.exists():
                stale.unlink()
            diagnostics.append(StageResult(f"{name}_prediction", "BLOCKED", f"{type(exc).__name__}: {exc}"))
            print(f"[BLOCKED] {name.title()} prediction: {exc}")

    try:
        combined = align_renewables(cfg, solar, wind)
        atomic_to_csv(combined, _safe_output_path(out, cfg["outputs"]["renewable"]))
        diagnostics.append(StageResult("renewable_alignment", "PASS", str(combined.get("integration_status", pd.Series([""])).iloc[0]), len(combined)))
    except Exception as exc:
        existing = out / cfg["outputs"]["renewable"]
        if existing.exists():
            existing.unlink()
        diagnostics.append(StageResult("renewable_alignment", "BLOCKED", f"{type(exc).__name__}: {exc}"))
        print(f"[BLOCKED] Renewable alignment: {exc}")

    try:
        if "fuel_cell" in assets and "fuel_cell" not in feature_errors:
            fuel_model, fuel_scaler = assets["fuel_cell"]
            fuel_result, fuel_evidence = fuel_cell_diagnostics(cfg, fuel_model, fuel_scaler, feature_lists["fuel_cell"])
            if not fuel_result.empty:
                atomic_to_csv(fuel_result, _safe_output_path(out, cfg["outputs"]["fuel_cell_predictions"]))
            atomic_to_csv(
                pd.DataFrame([{"class_index": i, "class_label": str(c)} for i, c in
                               enumerate(getattr(fuel_model, "classes_", []))]),
                _safe_output_path(out, cfg["outputs"]["fuel_cell_classes"]),
            )
            diagnostics.append({"stage": "fuel_cell_diagnostics", **fuel_evidence})
        else:
            fuel_evidence = {
                "status": "BLOCKED_ASSET_LOAD",
                "details": feature_errors.get("fuel_cell") or assets.get("__errors__", {}).get("fuel_cell", "Fuel Cell model unavailable"),
                "candidate_level_integration_status": "BLOCKED_MISSING_VALIDATED_FUEL_CELL_MAPPING",
            }
            diagnostics.append({"stage": "fuel_cell_diagnostics", **fuel_evidence})
    except Exception as exc:
        for output_key in ("fuel_cell_predictions", "fuel_cell_classes"):
            try:
                _safe_output_path(out, cfg["outputs"][output_key]).unlink(missing_ok=True)
            except OSError:
                pass
        diagnostics.append({
            "stage": "fuel_cell_diagnostics",
            "status": "BLOCKED_STAGE_ERROR",
            "details": f"{type(exc).__name__}: {exc}",
            "candidate_level_integration_status": "BLOCKED_MISSING_VALIDATED_FUEL_CELL_MAPPING",
        })
    diagnostics.append(StageResult("fuel_cell_candidate_integration", "BLOCKED_MISSING_VALIDATED_FUEL_CELL_MAPPING",
                                   "Hydrogen kg/h is not converted to Fuel Cell hydrogen_flow_l_min; no validated candidate-input mapping is implemented."))
    write_diagnostics(out, cfg["outputs"]["diagnostics"], diagnostics, run_id=run_id)
    try:
        write_integration_readiness(out, diagnostics, run_id)
    except Exception as exc:
        print(f"[REPORT WARNING] Could not write integration readiness: {type(exc).__name__}: {exc}")
    print(f"Run finished. Outputs: {out}")
    print("Execution success does not establish model validity, physical feasibility, or scientific validation.")
    return {"output_dir": str(out), "run_id": run_id, "diagnostics": diagnostics, "hydrogen_status": hydrogen_status}



def research_power_dispatch(
    cfg: dict[str, Any], candidates: pd.DataFrame, available_power_kw: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Research-only filtering of existing candidates, never a process dispatch or safety approval.

    The `renewable_power_kw` column is available supply context, *not* verified
    electrolyzer consumption. The <= comparison is a scenario-data filter only.
    """
    if isinstance(available_power_kw, bool) or not np.isfinite(available_power_kw) or available_power_kw <= 0:
        raise ValueError("Research available_power_kw must be a finite positive number.")
    power_column = "renewable_power_kw"
    prediction = "predicted_hydrogen_production_kg_h"
    required = ["candidate_id", power_column, prediction] + list(cfg["decision_variables"])
    missing = [col for col in required if col not in candidates.columns]
    if missing:
        raise ValueError(f"Candidate table missing required columns: {missing}")
    if candidates["candidate_id"].isna().any() or candidates["candidate_id"].duplicated().any():
        raise ValueError("Candidate identifiers must be unique and non-missing.")
    data = candidates.copy()
    for col in dict.fromkeys([power_column, prediction] + list(cfg["decision_variables"])):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    numeric_cols = list(dict.fromkeys([power_column, prediction] + list(cfg["decision_variables"])))
    values = data[numeric_cols].to_numpy(dtype=float)
    valid = np.isfinite(values).all(axis=1)
    valid &= data[power_column].to_numpy() > 0
    valid &= data[prediction].to_numpy() >= 0
    valid &= data[power_column].to_numpy() <= available_power_kw
    result = data.loc[valid].copy()
    result["available_supply_proxy_kg_per_kwh"] = result[prediction] / result[power_column]
    # Robust O(n log n) two-objective non-dominated frontier; ties share frontier status.
    ordered = result.sort_values([prediction, "available_supply_proxy_kg_per_kwh"], ascending=[False, False], kind="mergesort")
    best_ratio = -np.inf
    best_production = np.inf
    frontier_ids = set()
    for row in ordered[["candidate_id", prediction, "available_supply_proxy_kg_per_kwh"]].itertuples(index=False, name=None):
        cid, production_value, proxy_value = row
        is_dominated = (best_ratio >= proxy_value and best_production > production_value) or (best_ratio > proxy_value)
        if not is_dominated:
            frontier_ids.add(cid)
        if proxy_value > best_ratio:
            best_ratio = proxy_value
            best_production = production_value
    result["pareto_frontier"] = result["candidate_id"].isin(frontier_ids)
    result["research_status"] = "EXPLORATORY_ONLY_NOT_APPROVED"
    result["available_power_interpretation"] = "AVAILABLE_SUPPLY_NOT_VERIFIED_ELECTROLYZER_DRAW"
    result["power_filter_interpretation"] = "OBSERVED_POWER_CONTEXT_LEQ_INPUT_NOT_PHYSICAL_FEASIBILITY"
    result["operating_safety"] = "NOT_VERIFIED"
    result["fuel_cell_integration"] = "NOT_VALIDATED"
    result = result.sort_values([prediction, "available_supply_proxy_kg_per_kwh", "candidate_id"], ascending=[False, False, True], kind="stable").reset_index(drop=True)
    summary = pd.DataFrame([{
        "research_available_power_kw": float(available_power_kw),
        "source_candidate_count": int(len(data)),
        "matching_candidate_count": int(len(result)),
        "pareto_candidate_count": int(result["pareto_frontier"].sum()),
        "excluded_nonfinite_or_outside_power_context_count": int(len(data)-len(result)),
        "best_predicted_hydrogen_kg_h": float(result[prediction].max()) if len(result) else np.nan,
        "status": "RESEARCH_SCENARIO_NOT_AN_OPERATING_COMMAND",
        "power_semantics": "RENEWABLE_POWER_AVAILABLE_NOT_CONSUMED",
        "physical_feasibility": "NOT_VERIFIED",
        "solar_wind_alignment": "NOT_VERIFIED",
        "fuel_cell_mapping": "NOT_VERIFIED",
        "note": "Candidate power <= user-entered available supply is an exploratory data filter, not a dispatch or load-flow constraint. Pareto uses predicted production and production/available supply; not SEC.",
    }])
    return result, summary


def run_research_power_dispatch(config_path: Path | str | None, *, power_kw: float, candidate_file: Path | str | None = None, output_dir: Path | str | None = None) -> dict[str, Any]:
    """Run an opt-in research scenario without loading or changing saved models."""
    cfg = read_config(config_path)
    candidate_path = Path(candidate_file) if candidate_file else Path(cfg["output_dir"]) / cfg["outputs"]["all_candidates"]
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Candidate CSV missing: {candidate_path}. Run engine.py normally first.")
    candidates = pd.read_csv(candidate_path)
    selected, summary = research_power_dispatch(cfg, candidates, power_kw)
    out = Path(output_dir) if output_dir else Path(cfg["output_dir"]).parent / "research_dispatch"
    out.mkdir(parents=True, exist_ok=True)
    atomic_to_csv(selected, _safe_output_path(out, "research_power_candidates.csv"))
    atomic_to_csv(selected.loc[selected["pareto_frontier"]].copy(), _safe_output_path(out, "research_power_pareto.csv"))
    atomic_to_csv(summary, _safe_output_path(out, "research_power_summary.csv"))
    print(f"[RESEARCH ONLY] {len(selected)} matching observed candidates; {int(selected['pareto_frontier'].sum())} Pareto candidates. Outputs: {out}")
    print("No electrical load dispatch, safe operating recommendation, or Solar/Wind/Fuel Cell integration has been approved.")
    return {"output_dir": str(out), "summary": summary, "candidates": selected}


def run_power_contract_research(config_path, contract_file, candidate_file=None, output_dir=None):
    """Research scenario with explicit renewable supply provenance and no physical dispatch."""
    from power_contract import read_power_contract
    contract = read_power_contract(contract_file)
    outcome = run_research_power_dispatch(config_path, power_kw=contract["available_power_kw"], candidate_file=candidate_file, output_dir=output_dir)
    target = Path(outcome["output_dir"]) / "power_contract_status.json"
    fd, temp = tempfile.mkstemp(prefix=".power_contract.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as stream:
            json.dump(contract,stream,ensure_ascii=False,indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp,target)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    print(f"[POWER CONTRACT] Research-only provenance saved: {target}")
    return outcome


def run_electrolyzer_contract(contract_file: Path | str, output_dir: Path | str | None = None) -> dict[str, Any]:
    """Standalone observational input gate; cannot authorize dispatch or change model behavior."""
    try:
        from .electrolyzer_contract import read_electrolyzer_contract
    except ImportError:
        from electrolyzer_contract import read_electrolyzer_contract
    assessment = read_electrolyzer_contract(contract_file)
    out = Path(output_dir) if output_dir else ROOT / "research_dispatch"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "electrolyzer_consumption_status.json"
    if target.is_symlink():
        raise ConfigurationError("Refusing to overwrite a symlink output")
    fd, temp = tempfile.mkstemp(prefix=".electrolyzer_status.", suffix=".tmp", dir=str(out))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(assessment, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    print(f"[ELECTROLYZER CONTRACT] {assessment['sec_status']}. Output: {target}")
    print("No physical dispatch, safe operation, or integration has been approved.")
    return assessment


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Prescriptive Engine with optional safe research-only power scenario")
    parser.add_argument("--config", default=None, help="Configuration file; defaults to project-root config.json")
    parser.add_argument("--research-power-kw", type=float, default=None, help="Available renewable supply (kW) for research-only scenario")
    parser.add_argument("--candidate-file", default=None, help="Existing hydrogen_candidates.csv; only with --research-power-kw")
    parser.add_argument("--research-out", default=None, help="Research outputs folder; only with --research-power-kw")
    parser.add_argument("--power-contract", default=None, help="JSON available supply provenance for research-only decision")
    parser.add_argument("--electrolyzer-contract", default=None, help="Evidence-gated observed consumption/production assessment; no dispatch")
    parser.add_argument("--electrolyzer-simulation", default=None, help="Research-only documented electricity-consumption simulation JSON")
    parser.add_argument("--phase3-rules", default=None, help="Research-only referenced constraint screening and post-screening Pareto")
    parser.add_argument("--phase4-contract", default=None, help="Evidence-gated four-model research integration (no physical dispatch)")
    args = parser.parse_args()
    if args.phase4_contract:
        if (args.phase3_rules or args.electrolyzer_contract or args.electrolyzer_simulation
                or args.power_contract or args.research_power_kw is not None
                or args.candidate_file):
            parser.error("--phase4-contract is standalone; --research-out is allowed")
        try:
            from .phase4_integration import run as run_phase4
        except ImportError:
            from phase4_integration import run as run_phase4
        run_phase4(args.phase4_contract, config_path=args.config, output_dir=args.research_out)
    elif args.phase3_rules:
        if args.electrolyzer_simulation or args.electrolyzer_contract or args.power_contract or args.research_power_kw is not None:
            parser.error("--phase3-rules cannot be combined with other modes")
        try:
            from .phase3_constraints import run as run_phase3
        except ImportError:
            from phase3_constraints import run as run_phase3
        cfg = read_config(args.config)
        candidates = Path(args.candidate_file) if args.candidate_file else Path(cfg["output_dir"]) / cfg["outputs"]["all_candidates"]
        destination = Path(args.research_out) if args.research_out else ROOT / "phase3_results"
        result = run_phase3(candidates, args.phase3_rules, destination)
        print("[PHASE3] " + result["research_status"] + "; passed=" + str(result["passed_rows"]) + "; Pareto=" + str(result["pareto_after_screening"]))
        print("Research screening only. No safe operating or physical approval.")
    elif args.electrolyzer_simulation:
        if args.electrolyzer_contract or args.power_contract or args.research_power_kw is not None:
            parser.error("--electrolyzer-simulation cannot be combined with other execution modes")
        try:
            from .electrolyzer_simulation import run_simulation
        except ImportError:
            from electrolyzer_simulation import run_simulation
        cfg=read_config(args.config)
        candidate_path=Path(args.candidate_file) if args.candidate_file else Path(cfg["output_dir"])/cfg["outputs"]["all_candidates"]
        out_path=Path(args.research_out) if args.research_out else ROOT/"research_dispatch"
        run_simulation(args.electrolyzer_simulation,candidate_path,out_path)
    elif args.electrolyzer_contract:
        if args.power_contract or args.research_power_kw is not None or args.candidate_file:
            parser.error("--electrolyzer-contract is standalone and cannot be combined with scenario/dispatch arguments")
        run_electrolyzer_contract(args.electrolyzer_contract, output_dir=args.research_out)
    elif args.power_contract:
        if args.research_power_kw is not None:
            parser.error("--power-contract cannot be combined with --research-power-kw")
        run_power_contract_research(args.config,args.power_contract,candidate_file=args.candidate_file,output_dir=args.research_out)
    elif args.research_power_kw is None:
        if args.candidate_file or args.research_out:
            parser.error("--candidate-file and --research-out require --research-power-kw")
        run_engine(args.config)
    else:
        run_research_power_dispatch(args.config, power_kw=args.research_power_kw, candidate_file=args.candidate_file, output_dir=args.research_out)


if __name__ == "__main__":
    main()

