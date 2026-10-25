"""Phase 4: evidence-gated four-model RESEARCH integration.

This module NEVER manufactures clocks, power, hydrogen-flow or FC operating points.
A failed link is a recorded engineering/data blocker, not a simulated success.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
DERIVED = (
    'phase4_renewable_aligned.csv',
    'phase4_hydrogen_power_context.csv',
    'phase4_fuel_cell_candidate_diagnostics.csv',
    'phase4_component_diagnostics.csv',
    'phase4_candidate_provenance.csv',
    'phase4_status.json',
)


class Phase4ContractError(ValueError):
    pass


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                 prefix='.phase4.', suffix='.tmp', delete=False) as f:
        tmp = Path(f.name)
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                 prefix='.phase4.', suffix='.tmp', newline='', delete=False) as f:
        tmp = Path(f.name)
        frame.to_csv(f, index=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _path(root: Path, relative: str | None) -> Path | None:
    if relative is None or relative == '':
        return None
    if not isinstance(relative, str) or '\x00' in relative:
        raise Phase4ContractError('Contract paths must be strings')
    p = Path(relative)
    if p.is_absolute():
        raise Phase4ContractError(f'Use project-root relative input paths: {relative}')
    resolved = (root / p).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise Phase4ContractError('Input path escapes project root')
    return resolved


def _attested(assurance: dict, *fields: str) -> bool:
    return all(isinstance(assurance.get(k), str) and len(assurance[k].strip()) >= 8
               for k in fields)


def _load_json(path: Path) -> dict:
    try:
        doc = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase4ContractError(f'Cannot read Phase 4 contract: {exc}') from exc
    if not isinstance(doc, dict) or doc.get('schema_version') != SCHEMA_VERSION:
        raise Phase4ContractError('Expected Phase 4 contract schema_version=1')
    for field in ('input_files', 'link_evidence'):
        if not isinstance(doc.get(field), dict):
            raise Phase4ContractError(f'{field} must be an object')
    return doc


def _stage(status: str, detail: str, rows: int | None = None, **extras) -> dict:
    return {'status': status, 'detail': detail, 'rows': rows, **extras}


def _source_diagnostic(root: Path, name: str, raw_path: Path | None,
                       expected: tuple[str, ...]) -> dict:
    if raw_path is None or not raw_path.is_file():
        return _stage('BLOCKED_MISSING_SOURCE', f'{name} dataset not found', path=str(raw_path or ''))
    try:
        sample = pd.read_csv(raw_path, nrows=32, encoding='utf-8-sig')
    except Exception as exc:
        return _stage('BLOCKED_SOURCE_READ', f'{type(exc).__name__}: {exc}', path=str(raw_path))
    missing = [c for c in expected if c not in sample.columns]
    if missing:
        return _stage('BLOCKED_MISSING_COLUMNS', f'Missing {missing}', path=str(raw_path))
    diag = _stage('SOURCE_SCHEMA_READABLE_NOT_INTEGRATION_APPROVED',
                  'Source schema inspected; training validity and cross-model coupling are separate questions',
                  path=str(raw_path), sample_rows_checked=len(sample))
    if name == 'wind':
        t = sample['time_of_day'].astype('string').str.strip()
        full = t.str.match(r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[T\s].*)?$', na=False)
        if not full.all():
            diag['clock_status'] = 'RELATIVE_TIME_NO_DOCUMENTED_CALENDAR_ORIGIN'
            diag['detail'] = 'Wind time_of_day has clock-only or relative rows; cannot align with dated Solar by assumption'
    return diag


def _predictions_file(root: Path, name: str, path: Path | None, must: tuple[str, ...]) -> tuple[pd.DataFrame | None, dict]:
    if path is None or not path.is_file():
        return None, _stage('BLOCKED_MISSING_PREDICTIONS', f'{name} prediction output not found; run the existing model engine first', path=str(path or ''))
    try:
        frame = pd.read_csv(path, encoding='utf-8-sig')
    except Exception as exc:
        return None, _stage('BLOCKED_PREDICTION_READ', f'{type(exc).__name__}: {exc}', path=str(path))
    missing = [c for c in must if c not in frame.columns]
    if missing:
        return None, _stage('BLOCKED_PREDICTION_SCHEMA', f'{name} missing {missing}', len(frame), path=str(path))
    if frame.empty:
        return None, _stage('BLOCKED_EMPTY_PREDICTIONS', f'{name} output empty', 0)
    return frame, _stage('PREDICTIONS_AVAILABLE_NOT_CROSS_VALIDATED', f'{name} predictions readable', len(frame), path=str(path))


def _renewables(cfg, solar: pd.DataFrame | None, wind: pd.DataFrame | None):
    if solar is None or wind is None:
        return None, _stage('BLOCKED_MISSING_SOLAR_OR_WIND_OUTPUT', 'Solar/Wind both require existing model predictions')
    try:
        from .engine import align_renewables
    except ImportError:
        from engine import align_renewables
    try:
        # The established engine checks calendar, UTC/zone, scope, aggregation and negativity.
        joined = align_renewables(cfg, solar, wind)
    except Exception as exc:
        return None, _stage('BLOCKED_RENEWABLE_ALIGNMENT', f'{type(exc).__name__}: {exc}')
    if (cfg.get('renewable_combination', {}).get('method') != 'sum'
            or joined.empty or 'predicted_wind_power_kw' not in joined.columns
            or 'predicted_solar_power_kw' not in joined.columns):
        return None, _stage('BLOCKED_NOT_TWO_SOURCE_INTEGRATION', 'Solar + Wind both required')
    return joined, _stage('RESEARCH_SOLAR_WIND_ALIGNED_NOT_PHYSICAL_APPROVAL',
                          'Existing alignment engine accepted data and documented config', len(joined))


def _timestamp(series: pd.Series, fmt: str, *, label: str):
    if not isinstance(fmt, str) or not fmt.strip():
        raise Phase4ContractError(f'{label}: explicit source timestamp format required')
    out = pd.to_datetime(series, format=fmt, errors='coerce')
    if out.isna().any() or not pd.api.types.is_datetime64_any_dtype(out.dtype):
        raise Phase4ContractError(f'{label}: source timestamps missing or incompatible with documented format')
    return out


def _hydrogen_row_map(candidates: pd.DataFrame, source: pd.DataFrame):
    """Recover original timestamps from *unique seven-variable equality*.

    A legacy candidate source_row_number may be a shuffled row index, not source
    sample_id. Never silently use that column as provenance; record its mismatch.
    Duplicate observed operating vectors are ambiguous: refuse timestamp assignment.
    """
    if candidates.empty or 'candidate_id' not in candidates:
        raise Phase4ContractError('Phase3 candidates require unique candidate_id')
    if candidates['candidate_id'].isna().any() or candidates['candidate_id'].duplicated().any():
        raise Phase4ContractError('candidate_id must be unique and non-null')
    if 'sample_id' not in source or 'timestamp' not in source:
        raise Phase4ContractError('Original Hydrogen source requires sample_id and timestamp')
    if source['sample_id'].isna().any() or source['sample_id'].duplicated().any():
        raise Phase4ContractError('Hydrogen source sample_id must be unique and non-null')
    required = ('renewable_power_kw', 'voltage_v', 'temperature_c',
                'current_density_a_cm2', 'electrolyte_concentration_wt_pct',
                'water_flow_l_min', 'pressure_bar')
    for f in required:
        if f not in source or f not in candidates:
            raise Phase4ContractError(f'Missing original feature {f}')
        for frame in (source, candidates):
            x = pd.to_numeric(frame[f], errors='coerce').to_numpy(dtype=float)
            if not np.isfinite(x).all():
                raise Phase4ContractError(f'Non-finite source/candidate feature {f}')
    if source.duplicated(list(required)).any():
        raise Phase4ContractError('Identical Hydrogen operating points have multiple source timestamps; mapping is ambiguous')
    src = source[['sample_id', 'timestamp', *required]].copy()
    # Categorical float equality is intentionally exact: rounding or nearest matching
    # would create an invented observation-to-time relationship.
    linked = candidates.merge(src, on=list(required), how='left', validate='many_to_one', indicator=True)
    if (linked['_merge'] != 'both').any() or linked['sample_id'].isna().any():
        raise Phase4ContractError('Some Phase3 candidates have no exact seven-variable match in original Hydrogen data')
    linked['original_sample_id_matched_by_features'] = linked['sample_id']
    if 'source_row_number' in linked.columns:
        linked['legacy_source_row_matches_original'] = (
            pd.to_numeric(linked['source_row_number'], errors='coerce')
            == pd.to_numeric(linked['sample_id'], errors='coerce'))
    else:
        linked['source_row_number'] = pd.NA
        linked['legacy_source_row_matches_original'] = False
    cols = ['candidate_id', 'timestamp', 'renewable_power_kw',
            'predicted_hydrogen_production_kg_h', 'source_row_number',
            'original_sample_id_matched_by_features', 'legacy_source_row_matches_original']
    if 'predicted_hydrogen_production_kg_h' not in linked.columns:
        raise Phase4ContractError('Predicted hydrogen production is missing')
    return linked[cols].copy()


def _hydrogen_power(root, files, evidence, aligned, candidate, source):
    if aligned is None:
        return None, _stage('BLOCKED_NO_VERIFIED_RENEWABLE_ALIGNMENT', 'Renewable timeseries unavailable')
    required_refs = ('common_clock_reference', 'hydrogen_timestamp_reference')
    if not isinstance(evidence, dict) or not _attested(evidence, *required_refs):
        return None, _stage('BLOCKED_NO_VERIFIED_HYDROGEN_CLOCK_MAPPING',
                             'Evidence for shared clock and Hydrogen timestamp is missing')
    if candidate is None or source is None:
        return None, _stage('BLOCKED_MISSING_HYDROGEN_INPUT', 'Missing Phase3 candidates or source Hydrogen dataset')
    try:
        joined = _hydrogen_row_map(candidate, source)
        fmt = evidence.get('hydrogen_timestamp_format')
        joined['timestamp'] = _timestamp(joined['timestamp'], fmt, label='Hydrogen')
        renew = aligned[['timestamp', 'predicted_total_renewable_power_kw']].copy()
        # No calendar shifts, offsets, nearest neighbour, rescaling or arbitrary joins.
        if renew['timestamp'].duplicated().any():
            raise Phase4ContractError('Aligned renewable timestamps are not unique')
        if isinstance(renew['timestamp'].dtype, pd.DatetimeTZDtype) != isinstance(joined['timestamp'].dtype, pd.DatetimeTZDtype):
            raise Phase4ContractError('Hydrogen/Renewable timezone-awareness mismatch')
        result = joined.merge(renew, on='timestamp', how='inner', validate='many_to_one')
        if result.empty:
            return None, _stage('BLOCKED_ZERO_TIMESTAMP_OVERLAP',
                                'Solar/Wind and Hydrogen historical timestamps have zero exact overlap; no fabricated date offsets', 0)
        for f in ('predicted_total_renewable_power_kw', 'renewable_power_kw'):
            a = pd.to_numeric(result[f], errors='coerce').to_numpy(dtype=float)
            if not np.isfinite(a).all() or (a < 0).any():
                raise Phase4ContractError(f'Invalid {f} in dated power comparison')
        result['supply_proxy_supports_scenario'] = result['predicted_total_renewable_power_kw'] >= result['renewable_power_kw']
        result['consumed_electrolyzer_power_verified'] = False
        result['physical_dispatch_approved'] = False
        result['integration_claim'] = 'MATCHED_RESEARCH_SUPPLY_CONTEXT_NOT_CONSUMED_POWER'
        return result, _stage('RESEARCH_CLOCK_MATCHED_POWER_PROXY_ONLY',
                              'Exact shared-clock match; available supply is not electrolyzer power consumption', len(result),
                              supporting_rows=int(result['supply_proxy_supports_scenario'].sum()))
    except Exception as exc:
        return None, _stage('BLOCKED_HYDROGEN_MAPPING', f'{type(exc).__name__}: {exc}')


def _fuel_cell_link(root, cfg, files, evidence, candidate, source, *, power_context=None):
    if candidate is None or source is None:
        return None, _stage('BLOCKED_MISSING_HYDROGEN_INPUT', 'Phase3 candidates and source Hydrogen required')
    required_refs = ('same_device_chain_reference', 'measured_flow_reference',
                     'fuel_cell_operating_point_reference', 'shared_clock_reference')
    if not isinstance(evidence, dict) or not _attested(evidence, *required_refs):
        return None, _stage('BLOCKED_NO_VERIFIED_HYDROGEN_TO_FC_MAPPING',
                            'Requires device-chain, observed hydrogen flow, FC operating point and time evidence')
    bridge_file = _path(root, files.get('fuel_cell_bridge'))
    if bridge_file is None or not bridge_file.is_file():
        return None, _stage('BLOCKED_MISSING_FC_BRIDGE', 'No measured candidate-to-FC bridge file')
    try:
        bridge = pd.read_csv(bridge_file, encoding='utf-8-sig')
        required = ['candidate_id', 'timestamp', 'measurement_reference'] + list(cfg['fuel_cell_features'])
        if set(required) - set(bridge):
            raise Phase4ContractError(f'Missing FC bridge columns {sorted(set(required)-set(bridge))}')
        if bridge.empty or bridge['candidate_id'].isna().any() or bridge['candidate_id'].duplicated().any():
            raise Phase4ContractError('FC bridge must contain unique candidate IDs')
        if bridge['measurement_reference'].isna().any() or bridge['measurement_reference'].astype(str).str.strip().eq('').any():
            raise Phase4ContractError('Each FC bridge observation needs its own provenance reference')
        mapped = _hydrogen_row_map(candidate, source)
        if not set(bridge['candidate_id']).issubset(set(mapped['candidate_id'])):
            raise Phase4ContractError('FC bridge references unknown Phase3 candidate IDs')
        if power_context is not None:
            power_ids = set(power_context['candidate_id'])
            if not set(bridge['candidate_id']).issubset(power_ids):
                raise Phase4ContractError('FC bridge refers to candidates with no matched renewable-power context')
        # Explicitly require the same documentary timestamp representation for each source.
        bridge['timestamp'] = _timestamp(bridge['timestamp'], evidence.get('fuel_cell_timestamp_format'), label='FC bridge')
        mapped['timestamp'] = _timestamp(mapped['timestamp'], evidence.get('hydrogen_timestamp_format'), label='Hydrogen')
        joint = bridge.merge(mapped[['candidate_id', 'timestamp']], on=['candidate_id', 'timestamp'],
                             how='left', validate='one_to_one', indicator=True)
        if (joint['_merge'] != 'both').any():
            raise Phase4ContractError('Measured FC flow and candidate Hydrogen timestamps do not match exactly')
        joint = joint.drop(columns=['_merge'])
        if not cfg.get('fuel_cell_pass_constraint', {}).get('class_mapping_verified') is True or not cfg.get('fuel_cell_pass_constraint', {}).get('class_mapping_reference'):
            raise Phase4ContractError('Fuel Cell Pass/Fail class mapping is unverified')
        try:
            from . import engine
        except ImportError:
            import engine
        engine.require_verified_preprocessing(cfg, 'fuel_cell')
        features = engine.load_feature_file(cfg, 'fuel_cell', cfg['fuel_cell_features'])
        model_file = (Path(cfg['models_dir']) / cfg['fuel_cell_model_file'])
        scaler_file = (Path(cfg['models_dir']) / cfg['fuel_cell_scaler_file'])
        model = engine.load_pickle(model_file)
        scaler = engine.load_pickle(scaler_file) if cfg['input_modes']['fuel_cell'] == 'scaled' else None
        X = engine.prepare_input(
            joint, model, features, scaler, cfg['input_modes']['fuel_cell'], 'Phase4 Fuel Cell',
            scaler_order_verified=cfg.get('scaler_feature_order_verified', {}).get('fuel_cell', False),
            model_order_verified=cfg.get('model_feature_order_verified', {}).get('fuel_cell', False),
            scaler_order_reference=cfg.get('scaler_feature_order_reference', {}).get('fuel_cell', ''),
            model_order_reference=cfg.get('model_feature_order_reference', {}).get('fuel_cell', ''),
            scaler_feature_order=cfg.get('scaler_feature_order', {}).get('fuel_cell'))
        predicted = model.predict(X)
        if len(predicted) != len(joint):
            raise Phase4ContractError('Fuel Cell model output count mismatch')
        label_map = cfg['fuel_cell_pass_constraint']['class_label_mapping']
        if any(str(x) not in label_map for x in predicted):
            raise Phase4ContractError('Unmapped Fuel Cell class labels')
        joint['predicted_fc_class'] = [label_map[str(x)] for x in predicted]
        joint['fuel_cell_inference_status'] = 'RESEARCH_LINKED_TO_EXPLICIT_MEASUREMENTS_NOT_SAFETY_APPROVAL'
        joint['physical_safety_approved'] = False
        return joint, _stage('RESEARCH_FC_CLASSIFIED_WITH_EXPLICIT_BRIDGE',
                             'FC model used separate measured flow and operating-point features; not physical approval', len(joint))
    except Exception as exc:
        return None, _stage('BLOCKED_FC_BRIDGE_VALIDATION', f'{type(exc).__name__}: {exc}')


def run(contract_path: str | Path, *, config_path: str | Path | None = None,
        output_dir: str | Path | None = None) -> dict:
    contract_path = Path(contract_path).resolve()
    contract = _load_json(contract_path)
    root = contract_path.parent
    # Contract resides at project root, so relative paths are reproducible and cannot silently cross projects.
    if config_path is None:
        config_path = root / 'config.json'
    try:
        from . import engine
    except ImportError:
        import engine
    cfg = engine.read_config(config_path)
    files = contract['input_files']
    evidence = contract['link_evidence']
    out = Path(output_dir).resolve() if output_dir is not None else root / 'phase4_results'
    out.mkdir(parents=True, exist_ok=True)
    if any((out / filename).is_symlink() for filename in DERIVED):
        raise Phase4ContractError('Refusing to overwrite symlink output')
    prior = [out / filename for filename in DERIVED if (out / filename).exists()]
    run_id = uuid.uuid4().hex
    if prior:
        archive = out / '_previous_runs' / run_id
        archive.mkdir(parents=True, exist_ok=False)
        for p in prior:
            p.replace(archive / p.name)

    resolved = {key: _path(root, value) for key, value in files.items()}
    status = {
        'schema_version': 1, 'run_id': run_id,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'phase': '4', 'components': {}, 'links': {},
        'physical_safety_approved': False,
        'solar_wind_hydrogen_fuel_cell_physical_integration_validated': False,
        'meaning': 'Research provenance, data compatibility and conditional model links only; NOT a validated physical dispatch or safe operating system',
    }
    source_specs = {
        'solar': ('timestamp', 'solar_power_kw'),
        'wind': ('day_index', 'time_of_day'),
        'hydrogen': ('sample_id', 'timestamp', 'renewable_power_kw', 'hydrogen_production_kg_h'),
        'fuel_cell': ('test_id', 'hydrogen_flow_l_min', 'outcome'),
    }
    for component, required in source_specs.items():
        status['components'][component] = _source_diagnostic(root, component, resolved.get(component+'_source'), required)

    solar, status['components']['solar_predictions'] = _predictions_file(
        root, 'Solar', resolved.get('solar_predictions'), ('timestamp', 'predicted_solar_power_kw'))
    wind, status['components']['wind_predictions'] = _predictions_file(
        root, 'Wind', resolved.get('wind_predictions'), ('predicted_wind_power_kw',))
    candidates, status['components']['phase3_candidates'] = _predictions_file(
        root, 'Hydrogen Phase3', resolved.get('phase3_passed'),
        ('candidate_id', 'source_row_number', 'predicted_hydrogen_production_kg_h'))
    fc_history, status['components']['fuel_cell_history_predictions'] = _predictions_file(
        root, 'Fuel Cell history', resolved.get('fuel_cell_predictions'), ('predicted_class_raw',))
    if fc_history is not None:
        status['components']['fuel_cell_history_predictions']['detail'] = (
            'Historical dataset predictions ONLY, not predictions for Hydrogen candidate flows')
    hydro_src = None
    if resolved.get('hydrogen_source') and resolved['hydrogen_source'].is_file():
        try:
            hydro_src = pd.read_csv(resolved['hydrogen_source'], encoding='utf-8-sig')
        except Exception as exc:
            status['components']['hydrogen']['status'] = 'BLOCKED_SOURCE_READ'
            status['components']['hydrogen']['detail'] = f'{type(exc).__name__}: {exc}'

    if candidates is not None and hydro_src is not None:
        try:
            provenance = _hydrogen_row_map(candidates, hydro_src)
            mismatches = int((~provenance['legacy_source_row_matches_original']).sum())
            status['components']['hydrogen_candidate_provenance'] = _stage(
                'EXACT_UNIQUE_FEATURE_MATCH_VERIFIED_LEGACY_ROW_ID_AUDITED',
                f'Seven original features matched exactly and uniquely; {mismatches} legacy source_row_number values differ from actual sample_id',
                len(provenance), mismatched_legacy_row_ids=mismatches,
                matching_method='exact_unique_seven_source_features')
            _write_csv(provenance, out / 'phase4_candidate_provenance.csv')
        except Exception as exc:
            status['components']['hydrogen_candidate_provenance'] = _stage(
                'BLOCKED_ORIGINAL_ROW_PROVENANCE', f'{type(exc).__name__}: {exc}')

    aligned, status['links']['solar_to_wind'] = _renewables(cfg, solar, wind)
    if aligned is not None:
        _write_csv(aligned, out / 'phase4_renewable_aligned.csv')
    context, status['links']['renewable_to_hydrogen'] = _hydrogen_power(
        root, files, evidence.get('renewable_to_hydrogen'), aligned, candidates, hydro_src)
    if context is not None:
        _write_csv(context, out / 'phase4_hydrogen_power_context.csv')
    fc, status['links']['hydrogen_to_fuel_cell'] = _fuel_cell_link(
        root, cfg, files, evidence.get('hydrogen_to_fuel_cell'), candidates, hydro_src,
        power_context=context)
    if fc is not None:
        _write_csv(fc, out / 'phase4_fuel_cell_candidate_diagnostics.csv')

    statuses = {k: v['status'] for k, v in status['links'].items()}
    full_chain_overlap = (len(set(context['candidate_id']) & set(fc['candidate_id']))
                          if context is not None and fc is not None else 0)
    status['shared_four_component_candidate_count'] = full_chain_overlap
    status['overall_status'] = (
        'RESEARCH_CROSS_MODEL_LINKS_COMPUTED_NOT_PHYSICAL_APPROVAL'
        if all(s.startswith('RESEARCH_') for s in statuses.values()) and full_chain_overlap > 0
        else 'BLOCKED_INCOMPLETE_CROSS_MODEL_EVIDENCE'
    )
    status['unresolved_links'] = [k for k, s in statuses.items() if not s.startswith('RESEARCH_')]
    status['outputs_created'] = [p.name for p in out.iterdir() if p.is_file() and p.name in DERIVED]
    rows = []
    for section in ('components', 'links'):
        for name, val in status[section].items():
            rows.append({'section': section, 'stage': name,
                         'status': val['status'], 'rows': val.get('rows'), 'details': val['detail']})
    _write_csv(pd.DataFrame(rows), out / 'phase4_component_diagnostics.csv')
    status['outputs_created'] = [p.name for p in out.iterdir() if p.is_file() and p.name in DERIVED] + ['phase4_status.json']
    _write_json(status, out / 'phase4_status.json')
    print('[PHASE4] ' + status['overall_status'])
    for name, entry in status['links'].items():
        print(f'  {name}: {entry["status"]} - {entry["detail"][:180]}')
    print(f'Report: {out / "phase4_status.json"}')
    print('No unverified calendar coupling, hydrogen-to-FC gas flow, safe operation, or physical dispatch.')
    return status


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Evidence-gated Phase 4 research integration')
    ap.add_argument('--contract', required=True)
    ap.add_argument('--config', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    run(args.contract, config_path=args.config, output_dir=args.out)


if __name__ == '__main__':
    main()
