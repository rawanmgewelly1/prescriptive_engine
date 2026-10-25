"""Research-only Phase 3 screening and Pareto optimization.

Safety boundary: empirical observed-data ranges are NOT engineering operating limits.
An engineering approval requires independently verified device/process-specific evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_UNITS = {
    'voltage_v': 'V',
    'temperature_c': 'degC',
    'pressure_bar': 'bar',
    'current_density_a_cm2': 'A/cm2',
    'electrolyte_concentration_wt_pct': 'wt%',
    'water_flow_l_min': 'L/min',
    'renewable_power_kw': 'kW',
}
DEFAULT_SUPPORT_FEATURES = (
    'voltage_v', 'temperature_c', 'pressure_bar',
    'current_density_a_cm2', 'electrolyte_concentration_wt_pct',
    'water_flow_l_min',
)
OUTPUT_PRODUCTION = 'predicted_hydrogen_production_kg_h'
OUTPUT_PROXY = 'predicted_kg_per_available_kwh_proxy'

class ContractError(ValueError):
    pass


def validated_rules(config):
    """Backward-compatible referenced research rules; references require review."""
    if not isinstance(config, dict):
        raise ContractError('Rules file must be a JSON object')
    rules = config.get('research_constraints', [])
    if not isinstance(rules, list):
        raise ContractError('research_constraints must be a list')
    valid, names = [], set()
    for rule in rules:
        if not isinstance(rule, dict):
            raise ContractError('Each rule must be an object')
        if rule.get('enabled') is not True:
            continue
        name, col = rule.get('name'), rule.get('column')
        if not isinstance(name, str) or not name or name in names:
            raise ContractError('Rule names must be unique and nonempty')
        names.add(name)
        if col not in FEATURE_UNITS:
            raise ContractError(f'{name}: unsupported variable {col!r}')
        if rule.get('units') != FEATURE_UNITS[col]:
            raise ContractError(f'{name}: units must be {FEATURE_UNITS[col]}')
        if not all(isinstance(rule.get(k), str) and rule[k].strip()
                   for k in ('source_reference', 'scope', 'evidence_type')):
            raise ContractError(f'{name}: source_reference, scope, evidence_type required')
        if rule['evidence_type'] not in ('published_research_range', 'manufacturer_limit'):
            raise ContractError(f'{name}: unsupported evidence_type')
        try:
            lo, hi = float(rule['min']), float(rule['max'])
        except (ValueError, TypeError, KeyError) as exc:
            raise ContractError(f'{name}: min/max required') from exc
        if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
            raise ContractError(f'{name}: invalid limits')
        valid.append((name, col, lo, hi, rule['evidence_type']))
    return valid


def _observed_support(spec, rules_path, candidates_path):
    if spec is None:
        return [], None
    if not isinstance(spec, dict) or spec.get('enabled') is not True:
        raise ContractError('observed_data_screening must be an enabled JSON object')
    if spec.get('mode') != 'empirical_quantiles':
        raise ContractError('Only empirical_quantiles mode is supported')
    lo_q, hi_q = spec.get('lower_quantile'), spec.get('upper_quantile')
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) for x in (lo_q, hi_q)):
        raise ContractError('Quantile bounds must be numbers')
    if not (0 <= lo_q < hi_q <= 1):
        raise ContractError('Quantiles must satisfy 0 <= lower < upper <= 1')
    cols = spec.get('columns')
    if not isinstance(cols, list) or not cols or len(cols) != len(set(cols)):
        raise ContractError('Specify distinct screening columns')
    if any(col not in DEFAULT_SUPPORT_FEATURES for col in cols):
        raise ContractError('Observed support columns must be original Hydrogen operating variables')
    if spec.get('scientific_claim') != 'OBSERVED_DATA_SUPPORT_ONLY_NOT_SAFETY':
        raise ContractError('Empirical bounds must be marked NOT_SAFETY')
    reference = spec.get('dataset_file')
    if not isinstance(reference, str) or not reference.strip():
        raise ContractError('Reference dataset path required')
    reference_path = (Path(rules_path).resolve().parent / reference).resolve()
    if reference_path == Path(candidates_path).resolve():
        raise ContractError('Use original dataset, not candidate predictions, for empirical support')
    if not reference_path.is_file():
        raise ContractError(f'Observed reference dataset not found: {reference_path}')
    observed = pd.read_csv(reference_path, usecols=cols)
    rules = []
    for col in cols:
        values = pd.to_numeric(observed[col], errors='coerce')
        if values.notna().sum() < 20 or not np.isfinite(values.dropna().to_numpy(dtype=float)).all():
            raise ContractError(f'{col}: requires 20+ finite observed values')
        low, high = (float(values.quantile(q)) for q in (lo_q, hi_q))
        if not (np.isfinite(low) and np.isfinite(high) and low <= high):
            raise ContractError(f'{col}: cannot calculate finite observed support')
        rules.append((f'OBSERVED_SUPPORT_{col}', col, low, high, 'empirical_support_not_engineering'))
    metadata = {
        'reference_dataset': str(reference_path),
        'lower_quantile': lo_q,
        'upper_quantile': hi_q,
        'sample_rows': len(observed),
        'units': {col: FEATURE_UNITS[col] for col in cols},
        'derived_ranges': {col: {'min': rule[2], 'max': rule[3]} for col, rule in zip(cols, rules)},
        'limitations': (
            'Exploratory percentile bounds selected for sensitivity analysis; NOT physical safety, '
            'manufacturer limits, or independent scientific validation. Dataset used in model '
            'development; screened candidates are not independently validated.'
        ),
    }
    return rules, metadata


def non_dominated(df):
    """Both objectives maximized. Equal objective pairs remain; strict domination removes points."""
    if df.empty:
        return np.zeros(0, dtype=bool)
    p = df[OUTPUT_PRODUCTION].to_numpy(dtype=float)
    q = df[OUTPUT_PROXY].to_numpy(dtype=float)
    order = np.argsort(-p, kind='mergesort')
    keep = np.zeros(len(df), dtype=bool)
    best_q = -np.inf
    n = 0
    while n < len(order):
        m = n + 1
        while m < len(order) and p[order[m]] == p[order[n]]:
            m += 1
        group = order[n:m]
        max_q = float(np.max(q[group]))
        if max_q > best_q:  # equal proxy with strictly higher production is dominated
            keep[group[q[group] == max_q]] = True
        best_q = max(best_q, max_q)
        n = m
    return keep


def run(candidates_path, rules_path, output_dir):
    rules_path = Path(rules_path)
    config = json.loads(rules_path.read_text(encoding='utf-8'))
    referenced = validated_rules(config)
    empirical, empirical_metadata = _observed_support(
        config.get('observed_data_screening'), rules_path, candidates_path
    )
    rules = referenced + empirical
    df = pd.read_csv(candidates_path)
    needed = ['candidate_id', OUTPUT_PRODUCTION, OUTPUT_PROXY] + [col for _, col, _, _, _ in rules]
    missing = [name for name in needed if name not in df]
    if missing:
        raise ContractError(f'Missing columns: {missing}')
    if df['candidate_id'].isna().any() or df['candidate_id'].duplicated().any():
        raise ContractError('Duplicate/missing candidate IDs')
    for col in set(needed) - {'candidate_id'}:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ContractError(f'Invalid candidate values in {col}')
    if (df[[OUTPUT_PRODUCTION, OUTPUT_PROXY]] < 0).any().any():
        raise ContractError('Negative objectives are not supported')
    mask = pd.Series(True, index=df.index)
    reasons = [[] for _ in range(len(df))]
    for name, col, lo, hi, _ in rules:
        passed = df[col].between(lo, hi, inclusive='both')
        # Legacy output column convention is preserved for referenced literature rules.
        field = f'rule_{name}_pass'
        df[field] = passed
        mask &= passed
        for i in np.flatnonzero(~passed.to_numpy()):
            reasons[i].append(name)
    has_screening = bool(rules)
    df['passes_referenced_research_ranges'] = (mask if referenced else False)
    df['passes_research_screening'] = mask if has_screening else False
    df['screening_reasons'] = [
        ';'.join(row_reasons) if row_reasons else ('NONE' if has_screening else 'NO_SCREENING_RULES')
        for row_reasons in reasons
    ]
    df['approved_for_physical_operation'] = False
    df['screening_status'] = ('SCREENED_OBSERVED_SUPPORT_ONLY' if empirical and not referenced
                              else 'SCREENED_RESEARCH_ONLY' if has_screening
                              else 'BLOCKED_NO_REFERENCED_CONSTRAINTS')
    allowed = df.loc[mask].copy() if has_screening else df.iloc[:0].copy()
    allowed['pareto_nondominated_after_screening'] = non_dominated(allowed)
    front = allowed.loc[allowed['pareto_nondominated_after_screening']].copy()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'phase3_screening_all.csv', index=False)
    allowed.to_csv(out / 'phase3_research_passed.csv', index=False)
    front.to_csv(out / 'phase3_pareto_after_screening.csv', index=False)
    status = ('SCREENED_OBSERVED_SUPPORT_ONLY' if empirical and not referenced
              else 'SCREENED_RESEARCH_ONLY' if has_screening
              else 'BLOCKED_NO_REFERENCED_CONSTRAINTS')
    summary = {
        'input_rows': len(df),
        'enabled_referenced_rules': len(referenced),
        'empirical_support_rules': len(empirical),
        'passed_rows': len(allowed),
        'rejected_rows': int((~mask).sum()) if has_screening else None,
        'pareto_after_screening': len(front),
        'research_status': status,
        'research_screening_completed': has_screening,
        'engineering_limits_verified': False,
        'engineering_approval_status': 'BLOCKED_UNVERIFIED_ELECTROLYZER_TYPE_AND_LIMITS',
        'physical_operating_approval': False,
        'empirical_support': empirical_metadata,
        'warning': (
            'Research screening only. Quantile cuts are analyst-selected and not manufacturer '
            'or engineering safety limits. Scientific process compatibility, power consumption '
            'and equipment safety have not been established.'
        ),
    }
    (out / 'phase3_status.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--rules', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.candidates, args.rules, args.out), indent=2))
