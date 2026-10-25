"""Phase 5 independent, fail-closed final audit of the Phase 4 *research* scenario.

Does not execute/unpickle model files, modify model/config/engine, couple calendars,
claim candidate-specific Fuel Cell prediction, or approve industrial operation.
Uses Python standard library and independently recomputes every candidate row.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

R_GAS = 8.31446261815324
EXPECTED_BENCHMARK = "https://www.energy.gov/cmei/fuels/technical-targets-liquid-alkaline-electrolysis"
MODEL_KEYS = (
    ("solar", "solar_model_file", "solar_scaler_file"),
    ("wind", "wind_model_file", "wind_scaler_file"),
    ("hydrogen", "hydrogen_model_file", "hydrogen_scaler_file"),
    ("fuel_cell", "fuel_cell_model_file", "fuel_cell_scaler_file"),
)

class AuditError(Exception):
    pass


def _read_json(path):
    try:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise AuditError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise AuditError(f"Expected JSON object in {path}")
    return obj


def _atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", suffix=".tmp", dir=path.parent, delete=False) as tmp:
        temp = Path(tmp.name)
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(temp, path)


def _json_out(path, doc):
    _atomic_write(path, json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _csv_out(path, rows, fields):
    from io import StringIO
    buf = StringIO()
    wr = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    wr.writeheader()
    wr.writerows(rows)
    _atomic_write(path, buf.getvalue())


def _safe(root, rel):
    if not isinstance(rel, str) or not rel or "\x00" in rel:
        raise AuditError("Missing/invalid project-relative path")
    p = Path(rel)
    if p.is_absolute() or (".." in p.parts):
        raise AuditError(f"Path must be inside project: {rel}")
    dest = (root / p).resolve()
    if not dest.is_relative_to(root.resolve()):
        raise AuditError(f"Path escapes project root: {rel}")
    return dest


def _csv_in(path):
    if not path.is_file():
        raise AuditError(f"Missing required CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise AuditError(f"CSV has no header: {path}")
        result = list(reader)
    if not result:
        raise AuditError(f"CSV empty: {path}")
    return result


def _finite(row, field):
    try:
        val = float(row[field])
    except (KeyError, ValueError, TypeError, OverflowError) as exc:
        raise AuditError(f"Missing or invalid {field}") from exc
    if not math.isfinite(val):
        raise AuditError(f"Nonfinite {field}")
    return val


def _boolean(row, field):
    val = row.get(field, "")
    if val in ("True", "true", "1") or val is True:
        return True
    if val in ("False", "false", "0") or val is False:
        return False
    raise AuditError(f"Unknown boolean {field}={val!r}")


def _close(a, b, tol):
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def _digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sklearn_embedded_version(path):
    # Safely scan pickle BYTES only, without unpickling any untrusted executable object.
    try:
        data = path.read_bytes()
    except OSError:
        return None
    found = re.search(rb"_sklearn_version.{0,120}?(\d+\.\d+(?:\.\d+)?)", data, re.DOTALL)
    return found.group(1).decode("ascii") if found else None


def _pkg_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def _check(checks, label, condition, detail, *, severity="FAIL"):
    ok = bool(condition)
    checks.append({"check": label, "result": "PASS" if ok else severity,
                   "detail": str(detail)})
    return ok


def _model_inventory(root, cfg, checks):
    models_dir = _safe(root, cfg.get("models_dir", "models"))
    runtime_sklearn = _pkg_version("scikit-learn")
    model_files = []
    for label, model_key, scaler_key in MODEL_KEYS:
        for kind, key in (("model", model_key), ("scaler", scaler_key)):
            filename = cfg.get(key)
            if not filename:
                model_files.append({"component": label, "kind": kind, "path": "", "exists": False,
                                    "size_bytes": "", "sha256": "", "training_sklearn_version_hint": "",
                                    "runtime_sklearn_version": runtime_sklearn, "status": "NO_CONFIG_ENTRY"})
                continue
            path = _safe(models_dir, filename) if not Path(filename).is_absolute() else None
            if path is None:
                raise AuditError(f"Model absolute paths disallowed: {filename}")
            exists = path.is_file()
            v = _sklearn_embedded_version(path) if exists else None
            mismatch = v is not None and runtime_sklearn != "NOT_INSTALLED" and v != runtime_sklearn
            status = ("MISSING_ARTIFACT" if not exists else
                      "SKLEARN_VERSION_MISMATCH_UNVERIFIED" if mismatch else
                      "ARTIFACT_PRESENT_RUNTIME_NOT_REPRODUCIBILITY_VALIDATED")
            model_files.append({"component": label, "kind": kind,
                                "path": str(path.relative_to(root)), "exists": exists,
                                "size_bytes": path.stat().st_size if exists else "",
                                "sha256": _digest(path) if exists else "",
                                "training_sklearn_version_hint": v or "NOT_EXTRACTED",
                                "runtime_sklearn_version": runtime_sklearn,
                                "status": status})
    present = sum(int(x["exists"]) for x in model_files)
    _check(checks, "model_artifact_inventory", present == len(model_files),
           f"{present}/{len(model_files)} artifacts located; presence is not compatibility validation", severity="WARN")
    if any("MISMATCH" in x["status"] for x in model_files):
        _check(checks, "model_version_consistency", False,
               "Saved sklearn version differs from runtime; XGBoost legacy pickle compatibility is not independently proven", severity="WARN")
    else:
        _check(checks, "model_version_consistency", False,
               "Successful pickle loading does not independently establish XGBoost/sklearn version compatibility or prediction invariance", severity="WARN")
    return model_files


def _validate_rows(p3, all_rows, elig, ref, scenario, status, tol, checks):
    if scenario.get("schema_version") != 1 or scenario.get("mode") != "explicit_hypothetical_independent_research_scenario" or scenario.get("assumptions_acknowledged") is not True:
        raise AuditError("Phase4 scenario contract is not acknowledged hypothetical schema 1")
    supply = _finite(scenario, "available_power_kw")
    solar = _finite(scenario, "solar_assumed_kw")
    wind = _finite(scenario, "wind_assumed_kw")
    sec = _finite(scenario, "specific_energy_kwh_per_kg_h2")
    gas_t = _finite(scenario, "reference_gas_temperature_k")
    gas_p = _finite(scenario, "reference_gas_pressure_pa")
    mol_mass = _finite(scenario, "hydrogen_molar_mass_kg_per_mol")
    demand = _finite(ref[0], "hydrogen_flow_l_min")
    if not all(x > 0 for x in (supply, sec, gas_t, gas_p, mol_mass, demand)) or solar < 0 or wind < 0:
        raise AuditError("Nonpositive benchmark, supply, gas properties or historical flow")
    _check(checks, "renewable_assumed_power_balance", _close(solar + wind, supply, tol),
           f"Illustrative solar {solar} + wind {wind} = total {supply} kW")
    _check(checks, "literature_reference_and_scope", scenario.get("specific_energy_reference") == EXPECTED_BENCHMARK and _close(sec, 55.0, tol),
           f"Assumed SEC = {sec} kWh/kg; DOE 2022 liquid alkaline *system* status; not actual measured device SEC")
    _check(checks, "phase4_success_status", status.get("status") == "RESEARCH_SCENARIO_COMPLETE_NOT_PHYSICAL_INTEGRATION" and status.get("research_simulation_scope_closed") is True,
           str(status.get("status")))
    ids_p3 = {}
    for row in p3:
        cid = row.get("candidate_id")
        if not cid or cid in ids_p3:
            raise AuditError("Phase3 candidate IDs missing or duplicated")
        if not _boolean(row, "passes_research_screening"):
            raise AuditError(f"Phase3 input includes unscreened candidate {cid}")
        ids_p3[cid] = row
    # Independent two-objective Pareto audit of Phase3 frontier, both MAXIMIZED.
    # This repeats the mathematical test, not the Phase3 imported implementation.
    frontier_required = ('predicted_kg_per_available_kwh_proxy', 'pareto_nondominated_after_screening')
    pareto_points = None
    if all(all(k in row for k in frontier_required) for row in p3):
        vectors = [(cid, _finite(row, 'predicted_hydrogen_production_kg_h'),
                    _finite(row, 'predicted_kg_per_available_kwh_proxy'))
                   for cid, row in ids_p3.items()]
        vectors.sort(key=lambda x: (-x[1], -x[2]))
        best_proxy_for_strictly_better_production = float('-inf')
        predicted_frontier = set()
        i = 0
        while i < len(vectors):
            j = i + 1
            while j < len(vectors) and vectors[j][1] == vectors[i][1]:
                j += 1
            group = vectors[i:j]
            max_q = max(v[2] for v in group)
            if max_q > best_proxy_for_strictly_better_production:
                predicted_frontier.update(v[0] for v in group if v[2] == max_q)
            best_proxy_for_strictly_better_production = max(best_proxy_for_strictly_better_production, max_q)
            i = j
        stored_frontier = {cid for cid, row in ids_p3.items()
                           if _boolean(row, 'pareto_nondominated_after_screening')}
        pareto_points = len(predicted_frontier)
        _check(checks, 'phase3_pareto_independent_recalculation',
               stored_frontier == predicted_frontier,
               f'calculated={len(predicted_frontier)}; stored={len(stored_frontier)}; objectives: predicted H2 and available-power productivity proxy')
    else:
        _check(checks, 'phase3_pareto_independent_recalculation', False,
               'Missing Pareto frontier flags or objective proxy in Phase3 CSV', severity='FAIL')
    ids_all, ids_eligible, elig_rows = {}, set(), {}
    for row in all_rows:
        cid = row.get("candidate_id")
        if not cid or cid in ids_all:
            raise AuditError("Phase4 candidate IDs missing or duplicated")
        ids_all[cid] = row
    for row in elig:
        cid = row.get("candidate_id")
        if not cid or cid in ids_eligible:
            raise AuditError("Phase4 eligible IDs missing or duplicated")
        ids_eligible.add(cid)
        elig_rows[cid] = row
    _check(checks, "candidate_identity_reconciliation", set(ids_all) == set(ids_p3),
           f"Phase3={len(ids_p3)} vs Phase4={len(ids_all)}; exact candidate ID set")
    eligible_differences = []
    for cid, row in elig_rows.items():
        master = ids_all.get(cid)
        if master is None:
            eligible_differences.append(f"{cid}:not_in_all_candidates")
            continue
        if set(row) != set(master):
            eligible_differences.append(f"{cid}:eligible_column_set_differs")
            continue
        for key in row:
            if row[key] != master[key]:
                eligible_differences.append(f"{cid}:eligible_value_differs:{key}")
                break
    _check(checks, "eligible_csv_exact_original_rows", not eligible_differences,
           "Eligible CSV identical to corresponding source rows" if not eligible_differences else "; ".join(eligible_differences[:12]))
    if len(ref) != 1:
        raise AuditError("Historical FC reference should contain exactly one observed row")
    fc_label = ref[0].get("predicted_fc_label_at_unchanged_historical_point")
    _check(checks, "fuel_cell_reference_separate", fc_label in ("Pass", "Fail") and not _boolean(ref[0], "fc_candidate_specific_inference") and
           status.get("fuel_cell", {}).get("candidate_specific_fc_prediction") is False and
           status.get("fuel_cell", {}).get("saved_model_prediction_at_historic_reference") == fc_label and
           _close(status.get("fuel_cell", {}).get("historic_reference_observed_flow_l_min", float('nan')), demand, tol),
           f"Historical point class={fc_label!r}; not computed on Phase3 candidates")
    counts = {"proxy_within_assumed_supply": 0,
              "benchmark_within_assumed_supply": 0,
              "combined_research_screen_pass": 0}
    mismatches = []
    for cid, row in ids_all.items():
        source = ids_p3.get(cid)
        if source is None:
            mismatches.append(f"{cid}:not_in_phase3")
            continue
        h = _finite(row, "predicted_hydrogen_production_kg_h")
        available = _finite(row, "renewable_power_kw")
        if h < 0 or available < 0:
            mismatches.append(f"{cid}:negative_hydrogen_or_power")
            continue
        if not (_close(h, _finite(source, "predicted_hydrogen_production_kg_h"), tol) and
                _close(available, _finite(source, "renewable_power_kw"), tol)):
            mismatches.append(f"{cid}:source_mismatch")
        benchmark = h * sec
        ideal_cap = h * (R_GAS * gas_t / gas_p) * 1000. / (60. * mol_mass)
        proxy_ok = available <= supply
        benchmark_ok = benchmark <= supply
        fc_ok = ideal_cap >= demand
        both = proxy_ok and benchmark_ok
        expected_pass = both and fc_ok
        expected_values = (
            ("assumed_total_supply_kw", supply),
            ("literature_benchmark_required_kw", benchmark),
            ("ideal_gas_h2_capacity_l_min_reference_conditions", ideal_cap),
            ("illustrative_fc_reference_flow_l_min", demand),
        )
        for key, expected in expected_values:
            if not _close(_finite(row, key), expected, tol):
                mismatches.append(f"{cid}:{key}")
        expected_flags = (
            ("power_proxy_within_assumed_supply", proxy_ok),
            ("literature_benchmark_within_assumed_supply", benchmark_ok),
            ("research_scenario_candidate_supported", both),
            ("illustrative_fc_flow_capacity_sufficient", fc_ok),
            ("simulated_combined_screen_pass", expected_pass),
        )
        for key, expected in expected_flags:
            if _boolean(row, key) != expected:
                mismatches.append(f"{cid}:{key}")
        for key in ("physical_dispatch_approved", "same_calendar_verified", "same_device_chain_verified"):
            if _boolean(row, key):
                mismatches.append(f"{cid}:false_physical_claim_{key}")
        if not _boolean(row, "simulated_fc_usable_flow_not_measured"):
            mismatches.append(f"{cid}:missing_fc_flow_disclaimer")
        for k, v in (("proxy_within_assumed_supply", proxy_ok),
                     ("benchmark_within_assumed_supply", benchmark_ok),
                     ("combined_research_screen_pass", expected_pass)):
            counts[k] += int(v)
        if (cid in ids_eligible) != expected_pass:
            mismatches.append(f"{cid}:eligible_membership")
    _check(checks, "independent_1622_row_recomputation", not mismatches,
           "every Phase4 value, Boolean and Phase3 source independently recomputed" if not mismatches else "; ".join(mismatches[:12]))
    _check(checks, "eligible_membership_exact", len(ids_eligible) == counts["combined_research_screen_pass"],
           f"eligible={len(ids_eligible)}; recomputed={counts['combined_research_screen_pass']}")
    saved = status.get("counts", {})
    expected_count_names = dict(counts, all_phase3_screened=len(p3))
    _check(checks, "summary_counts_match", all(saved.get(k) == v for k, v in expected_count_names.items()),
           f"recomputed={expected_count_names}; saved={saved}")
    assumption_scope = status.get("scenario", {})
    _check(checks, "scenario_summary_match", all((
           _close(assumption_scope.get("assumed_total_kw", float("nan")), supply, tol),
           _close(assumption_scope.get("assumed_solar_kw", float("nan")), solar, tol),
           _close(assumption_scope.get("assumed_wind_kw", float("nan")), wind, tol),
           _close(assumption_scope.get("electrolyzer_benchmark_kwh_per_kg", float("nan")), sec, tol),
           assumption_scope.get("renewables_are_model_forecasts") is False,
           assumption_scope.get("actual_solar_wind_time_alignment") is False,
           )), "saved Phase4 assumptions match scenario input")
    limits = status.get("integration_limits", {})
    required_false = ("physical_safety_approved", "hydrogen_to_fc_flow_measured", "same_device_chain_verified",
                      "solar_wind_hydrogen_calendar_aligned", "electrolyzer_consumed_power_verified",
                      "complete_four_model_physical_integration")
    _check(checks, "no_false_physical_approval", all(limits.get(k) is False for k in required_false),
           "All physical/time/consumed power/device-chain claims remain explicitly false")
    best = max((ids_all[cid] for cid in ids_eligible),
               key=lambda r: _finite(r, "predicted_hydrogen_production_kg_h"), default=None)
    selected = status.get("selected_research_candidate")
    if best is None:
        _check(checks, "selected_candidate_match", selected is None,
               "No admissible candidate: no selected candidate")
    else:
        _check(checks, "selected_candidate_match", isinstance(selected, dict)
               and str(selected.get("candidate_id")) == str(best["candidate_id"])
               and _close(selected.get("predicted_hydrogen_production_kg_h", float('nan')),
                          _finite(best,"predicted_hydrogen_production_kg_h"), tol)
               and _close(selected.get("literature_benchmark_required_kw", float('nan')),
                          _finite(best,"literature_benchmark_required_kw"), tol)
               and _close(selected.get("ideal_gas_h2_capacity_l_min", float('nan')),
                          _finite(best,"ideal_gas_h2_capacity_l_min_reference_conditions"), tol),
               "Saved best candidate matches maximal predicted H2 among simulated eligible")
    return {"candidate_rows": len(ids_all), "eligible_rows": len(ids_eligible), "phase3_pareto_points": pareto_points,
            "recomputed_counts": expected_count_names,
            "best_candidate_id": best["candidate_id"] if best else None,
            "historical_fc_prediction": fc_label,
            "hypothetical_solar_kw": solar, "hypothetical_wind_kw": wind,
            "hypothetical_supply_kw": supply, "benchmark_kwh_per_kg": sec,
            "row_mismatch_samples": mismatches[:12]}


def _run_tests(root, out, enabled):
    if not enabled:
        return {"status": "NOT_RUN_BY_EXPLICIT_OPTION", "count": None, "returncode": None}
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", str(root / "tests"), "-q"]
    try:
        p = subprocess.run(cmd, cwd=root, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=240, check=False)
        log = p.stdout + "\n" + p.stderr
        _atomic_write(out / "phase5_unit_tests.log", log)
        m = re.search(r"Ran\s+(\d+)\s+tests", log)
        ok = p.returncode == 0 and m is not None and int(m.group(1)) >= 1 and re.search(r"(?:^|\n)OK\s*(?:\n|$)", log) is not None
        return {"status": "PASS" if ok else "FAIL", "count": int(m.group(1)) if m else None,
                "returncode": p.returncode, "log": "phase5_unit_tests.log"}
    except (OSError, subprocess.TimeoutExpired) as exc:
        _atomic_write(out / "phase5_unit_tests.log", str(exc))
        return {"status": "FAIL", "count": None, "returncode": None, "detail": str(exc), "log": "phase5_unit_tests.log"}


def _manifest(root, cfg, contract, model_files):
    rels = [contract.get(k) for k in (
        "phase3_passed", "phase4_scenario", "phase4_simulation_status",
        "phase4_simulated_candidates", "phase4_simulated_eligible",
        "phase4_fc_historical_reference", "engine_config", "engine_source")]
    rels += ["phase5_final_audit.py", "phase5_audit_contract.json",
             "phase4_research_simulation.py", "prescriptive_engine/phase4_research_simulation.py"]
    rels += [m["path"] for m in model_files if m["path"]]
    seen, rows = set(), []
    for rel in rels:
        if not rel or rel in seen:
            continue
        seen.add(rel)
        p = _safe(root, rel)
        rows.append({"relative_path": rel, "exists": p.is_file(),
                     "bytes": p.stat().st_size if p.is_file() else "",
                     "sha256": _digest(p) if p.is_file() else ""})
    return rows


def _report(result, checks, models, tests, manifest, links):
    passed = sum(c['result'] == 'PASS' for c in checks)
    failures = [c for c in checks if c['result'] == 'FAIL']
    warnings = [c for c in checks if c['result'] == 'WARN']
    lines = [
        '# Phase 5 - Final Research Audit and Acceptance Report', '',
        f'**Audit timestamp (UTC):** {result["created_utc"]}',
        f'**Audit status:** `{result["status"]}`', '',
        '## Acceptance Scope',
        'This audit applies only to an **independent hypothetical research simulation**. '
        'It does not establish shared time alignment or physical power coupling among Solar, Wind, '
        'Hydrogen and Fuel Cell, nor does it certify real equipment safety.',
        '', '## Independently Recalculated Results',
        f'- Phase 3 candidates independently reconciled: **{result["candidate_rows"]}**.',
        f'- Recalculated Phase 3 Pareto frontier: **{result["phase3_pareto_points"]}** points.',
        f'- Candidates meeting the Phase 4 hypothetical scenario: **{result["eligible_rows"]}**.',
        f'- Assumed power supply: **{result["hypothetical_supply_kw"]} kW**, '
        f'comprising hypothetical Solar = **{result["hypothetical_solar_kw"]} kW** '
        f'and Wind = **{result["hypothetical_wind_kw"]} kW**. '
        'These figures are not measured or model-aligned Solar/Wind outputs.',
        f'- Electrolyzer reference: **{result["benchmark_kwh_per_kg"]} kWh/kg-H2**, '
        'from the 2022 liquid alkaline system benchmark published by the US DOE. '
        'This is **not measured consumption for the studied device**.',
        f'- Fuel Cell historical reference outcome: **{result["historical_fc_prediction"]}**. '
        'This is a separate historical point, **not a candidate-specific Fuel Cell inference**.',
        f'- Checks: **{passed} passed**, **{len(failures)} failed**, **{len(warnings)} warnings**.',
        f'- Project unit tests: **{tests["status"]}**, '
        f'count **{tests["count"] if tests["count"] is not None else "not established"}**.',
        '', '## Independent Acceptance Checks',
        '| Check | Result | Detail |', '|---|---|---|'
    ]
    for c in checks:
        detail = c['detail'].replace('|', '\\|').replace('\n', ' ')[:350]
        lines.append(f'| `{c["check"]}` | {c["result"]} | {detail} |')
    lines += [
        '', '## Model Artifact Inventory and Compatibility',
        'The audit checks artifact existence, SHA-256 fingerprints and embedded version hints '
        '**without deserializing pickle files**. A previous successful load does not demonstrate '
        'numerical equivalence across library versions.',
        '| Component | Artifact Type | Status |', '|---|---|---|'
    ]
    for m in models:
        lines.append(f'| {m["component"]} | {m["kind"]} | `{m["status"]}` |')
    lines += [
        '', '## Known Engineering and Scientific Limitations',
        '- Solar, Wind and Hydrogen historical calendars are not verified as aligned.',
        '- Electrolyzer consumed electrical power has not been measured or validated as a dispatch model.',
        '- No evidenced common device chain or measured candidate Hydrogen flow into Fuel Cell is available.',
        '- Phase 3 observed-support screening is not an approved hardware-safe operating envelope.',
        '- No new independent external model holdout or production-model retraining validation was completed.',
        '- pickle, scikit-learn and XGBoost version warnings require a reproducible environment '
        'and separate inference-stability checks before production use.',
        '', '## Separate Phase 4 Physical Evidence Contract',
        f'**Physical-link status:** `{links}`. A BLOCKED physical-integration contract '
        'does not negate successful auditing of the hypothetical simulation.',
        '', '## Research Acceptance Decision',
        'The **hypothetical research simulation** may be accepted for its audited computational scope '
        'only when status is `RESEARCH_SCENARIO_AUDIT_PASSED_WITH_DISCLOSED_LIMITATIONS` '
        'and all project unit tests pass. It is **not certified** for industrial control, '
        'equipment safety or fully integrated physical operation.',
        '', '## Deliverables and Reproduction',
        '- `phase5_audit_status.json` - machine-readable audit outcome.',
        '- `phase5_acceptance_checks.csv` - independently evaluated checks.',
        '- `phase5_model_compatibility.csv` - model artifact inventory and version hints.',
        '- `phase5_sha256_manifest.csv` - input and artifact SHA-256 fingerprints.',
        '- `phase5_unit_tests.log` - complete unit test output.',
        '- `phase5_reproduction.txt` - reproducibility instructions.',
        '', '## Reference',
        EXPECTED_BENCHMARK,
        'The 55 kWh/kg-H2 reference is the **2022 Status - System Energy Efficiency** '
        'figure for **Liquid Alkaline Electrolysis** in the DOE table. '
        'It must not be represented as measured device consumption.',
    ]
    return '\n'.join(lines) + '\n'


def _html_report(result, checks, models, tests):
    esc = lambda x: html.escape(str(x), quote=True)
    rows = ''.join(
        '<tr><td><code>' + esc(c['check']) + '</code></td><td>' +
        esc(c['result']) + '</td><td>' + esc(c['detail']) + '</td></tr>'
        for c in checks)
    inventory = ''.join(
        '<tr><td>' + esc(m['component']) + '</td><td>' +
        esc(m['kind']) + '</td><td>' + esc(m['status']) + '</td></tr>'
        for m in models)
    return f'''<!doctype html>
<html lang="en" dir="ltr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase 5 - Final Research Audit Report</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0 auto;padding:28px 20px;max-width:1100px;color:#182333;background:#f8fafc;line-height:1.6}}
h1,h2{{color:#172b46}}h1{{font-size:26px}}h2{{font-size:19px;margin-top:36px}}
.summary{{display:flex;flex-wrap:wrap;gap:12px;margin:22px 0}}.tile{{flex:1;min-width:145px;background:white;border:1px solid #e0e6ed;border-radius:12px;padding:16px}}
.value{{font-size:24px;font-weight:750}}small{{color:#5b6778}}table{{border-collapse:collapse;width:100%;background:#fff;font-size:14px;text-align:left}}
th,td{{padding:9px 11px;border-bottom:1px solid #e6e9ed;vertical-align:top;overflow-wrap:anywhere}}th{{background:#eef2f7}}
.note{{padding:15px;background:#fff;border:1px solid #cbd5e1;border-radius:12px}}code{{font-family:Consolas,monospace}}
</style></head><body>
<h1>Phase 5 - Final Research Audit Report</h1>
<p><strong>Audit status:</strong> <code>{esc(result['status'])}</code></p>
<div class="summary">
<div class="tile"><div class="value">{esc(result['candidate_rows'])}</div><small>Candidate rows audited</small></div>
<div class="tile"><div class="value">{esc(result['eligible_rows'])}</div><small>Phase 4 hypothetical scenario eligible</small></div>
<div class="tile"><div class="value">{esc(result['phase3_pareto_points'])}</div><small>Recalculated Phase 3 Pareto points</small></div>
<div class="tile"><div class="value">{esc(tests['count'])}</div><small>Unit tests; status: {esc(tests['status'])}</small></div>
</div>
<div class="note">This report validates an <strong>independent hypothetical research scenario</strong> only. It does not confirm hardware safety or physical Solar/Wind/Hydrogen/Fuel Cell integration. The 100 kW + 100 kW renewable split is hypothetical; the Fuel Cell result is from a separate historical reference point.</div>
<h2>Research Reference Inputs</h2>
<p>Assumed supply: {esc(result['hypothetical_supply_kw'])} kW. Electrolyzer literature benchmark: {esc(result['benchmark_kwh_per_kg'])} kWh/kg-H2, based on the US DOE <a href="{esc(EXPECTED_BENCHMARK)}">2022 liquid alkaline system reference</a>; not a device power measurement.</p>
<h2>Independent Audit Checks</h2>
<div style="overflow-x:auto"><table><thead><tr><th>Check</th><th>Result</th><th>Detail</th></tr></thead><tbody>{rows}</tbody></table></div>
<h2>Model Inventory and Compatibility</h2>
<p>This audit does not deserialize model pickle files. Library version compatibility and inference stability require separate verification.</p>
<div style="overflow-x:auto"><table><thead><tr><th>Component</th><th>Artifact</th><th>Status</th></tr></thead><tbody>{inventory}</tbody></table></div>
<h2>Known Limitations</h2>
<p>There is no verified shared calendar, measured electrolyzer consumption, measured candidate Hydrogen flow into Fuel Cell, independent hardware safety assessment, or newly validated external test set. Phase 3 observed-support screening is not an approved safe operating range.</p>
<p><small>Generated at UTC {esc(result['created_utc'])}. For full data provenance, test output and fingerprints see the adjacent CSV and JSON files.</small></p>
</body></html>'''


def audit(contract_file, *, run_tests_override=None, output_override=None):
    contract_file = Path(contract_file).resolve()
    root = contract_file.parent
    doc = _read_json(contract_file)
    if doc.get("schema_version") != 1 or doc.get("purpose") != "FINAL_RESEARCH_AUDIT_NOT_HARDWARE_CERTIFICATION":
        raise AuditError("Unsupported Phase5 audit contract")
    tolerance = float(doc.get("audit_tolerance", 1e-7))
    if not math.isfinite(tolerance) or tolerance <= 0 or tolerance > 1e-4:
        raise AuditError("Bad audit tolerance")
    out = _safe(root, doc.get("output_dir", "phase5_results")) if output_override is None else Path(output_override).resolve()
    if out == root or not out.is_relative_to(root):
        raise AuditError("Phase5 output must be a separate subdirectory inside project")
    out.mkdir(parents=True, exist_ok=True)
    required = ("phase3_passed", "phase4_scenario", "phase4_simulation_status",
                "phase4_simulated_candidates", "phase4_simulated_eligible", "phase4_fc_historical_reference", "engine_config")
    paths = {key: _safe(root, doc[key]) for key in required}
    checks = []
    result = {"phase": 5, "created_utc": datetime.now(timezone.utc).isoformat(),
              "research_scope_only": True, "physical_safety_approved": False,
              "all_four_models_physically_integrated": False,
              "new_external_holdout_validation_completed": False}
    try:
        p3 = _csv_in(paths["phase3_passed"])
        all_rows = _csv_in(paths["phase4_simulated_candidates"])
        eligible = _csv_in(paths["phase4_simulated_eligible"])
        fc = _csv_in(paths["phase4_fc_historical_reference"])
        scenario = _read_json(paths["phase4_scenario"])
        status = _read_json(paths["phase4_simulation_status"])
        cfg = _read_json(paths["engine_config"])
        recalculated = _validate_rows(p3, all_rows, eligible, fc, scenario, status, tolerance, checks)
        result.update(recalculated)
        models = _model_inventory(root, cfg, checks)
        physical_path = _safe(root, doc.get("phase4_physical_link_status_optional", "phase4_results/phase4_status.json"))
        links = "PHYSICAL_INTEGRATION_EVIDENCE_REPORT_NOT_PROVIDED"
        if physical_path.exists():
            physical = _read_json(physical_path)
            links = physical.get("overall_status", "UNKNOWN")
            _check(checks, "physical_vs_research_scope_separation",
                   physical.get("solar_wind_hydrogen_fuel_cell_physical_integration_validated") is False,
                   f"Phase4 evidence status={links}; simulation does not override physical evidence")
        else:
            _check(checks, "physical_link_evidence_inventory", False,
                   "Physical contract report optional and not available: cannot validate physical coupling", severity="WARN")
        tests = _run_tests(root, out, doc.get("run_project_tests", True) if run_tests_override is None else run_tests_override)
        _check(checks, "complete_project_unit_tests", tests["status"] == "PASS",
               f"tests={tests['count']}; status={tests['status']}; see phase5_unit_tests.log",
               severity="FAIL")
        result["checks"] = checks
        result["unit_tests"] = tests
        result["runtime"] = {"python": platform.python_version(),
                             "platform": platform.platform(),
                             "scikit_learn": _pkg_version("scikit-learn"),
                             "xgboost": _pkg_version("xgboost"),
                             "catboost": _pkg_version("catboost"),
                             "numpy": _pkg_version("numpy"),
                             "pandas": _pkg_version("pandas")}
        failure = any(c["result"] == "FAIL" for c in checks)
        result["status"] = ("RESEARCH_SCENARIO_AUDIT_FAILED" if failure else
                            "RESEARCH_SCENARIO_AUDIT_PASSED_WITH_DISCLOSED_LIMITATIONS")
        result["scope_accepted"] = not failure
        result["model_reproducibility_independently_verified"] = False
        manifest = _manifest(root, cfg, doc, models)
        result["file_manifest_sha256"] = "phase5_sha256_manifest.csv"
        result["model_compatibility"] = "phase5_model_compatibility.csv"
        _csv_out(out / "phase5_sha256_manifest.csv", manifest, ["relative_path", "exists", "bytes", "sha256"])
        _csv_out(out / "phase5_model_compatibility.csv", models,
                 ["component", "kind", "path", "exists", "size_bytes", "sha256", "training_sklearn_version_hint", "runtime_sklearn_version", "status"])
        _csv_out(out / "phase5_acceptance_checks.csv", checks, ["check", "result", "detail"])
        _json_out(out / "phase5_audit_status.json", result)
        _atomic_write(out / "phase5_scientific_report_EN.md", _report(result, checks, models, tests, manifest, links))
        _atomic_write(out / "phase5_scientific_report_EN.html", _html_report(result, checks, models, tests))
        _atomic_write(out / "phase5_reproduction.txt", '\n'.join([
            'RESEARCH AUDIT REPRODUCTION — run from D:\\prescriptive_engine',
            '1. python -m unittest discover -s .\\tests -q',
            '2. python .\\phase4_research_simulation.py --scenario .\\phase4_research_scenario.json',
            '3. python .\\phase5_final_audit.py --contract .\\phase5_audit_contract.json',
            '', 'Do not modify data, models, or scenarios between step 2 and 3.',
            'Use phase5_sha256_manifest.csv to compare source hashes.',
            'Research-only. DOE system benchmark and fictional renewable split are not measured dispatch.',
            'A physical-integration BLOCKED status remains separate from successful scenario auditing.',
        ]) + '\n')
    except (AuditError, ValueError, KeyError, OSError, TypeError) as exc:
        result.update({"status": "RESEARCH_SCENARIO_AUDIT_FAILED", "scope_accepted": False, "error": str(exc)})
        _json_out(out / "phase5_audit_status.json", result)
        _atomic_write(out / "phase5_scientific_report_EN.html", '<html lang="en" dir="ltr"><meta charset="utf-8"><h1>Phase 5 Audit Failed</h1><p>Review phase5_audit_status.json. Prior reports must not be treated as accepted.</p></html>')
        _atomic_write(out / "phase5_scientific_report_EN.md", "# Phase 5 Audit Failed\n\n**Status: RESEARCH_SCENARIO_AUDIT_FAILED**\n\nReason: `" + str(exc) + "`\n\nDo not rely on an earlier Phase 5 report. Review the current audit status file.\n")
        print(f"[PHASE5] RESEARCH_SCENARIO_AUDIT_FAILED: {exc}")
        print(f"Report: {out / 'phase5_audit_status.json'}")
        return result
    print(f"[PHASE5] {result['status']}; audited={result['candidate_rows']}; eligible={result['eligible_rows']}; tests={tests['status']} ({tests['count']})")
    print(f"Report: {out / 'phase5_scientific_report_EN.md'}")
    print('Research audit only. No physical system safety, cross-model device coupling, or external model validity certification.')
    return result


def main():
    ap = argparse.ArgumentParser(description="Independent final research validation and reporting; no model modifications")
    ap.add_argument("--contract", default="phase5_audit_contract.json")
    ap.add_argument("--no-tests", action="store_true", help="Diagnostic only; does not qualify for acceptance")
    args = ap.parse_args()
    result = audit(args.contract, run_tests_override=False if args.no_tests else None)
    return 0 if result['status'] == "RESEARCH_SCENARIO_AUDIT_PASSED_WITH_DISCLOSED_LIMITATIONS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
