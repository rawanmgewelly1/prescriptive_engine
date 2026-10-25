"""Phase 4 standalone illustrative research scenario, NOT physical four-device linkage.

No datetime joins, fabricated measurements, model retraining, config changes, or
hardcoded classification. FC inference is on a *separate observed historical row*;
its model output is NOT a prediction on the hydrogen candidate's future FC inlet.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


class ScenarioError(ValueError):
    pass


def _positive(doc, key):
    value = doc.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ScenarioError(f'{key} must be a finite positive number')
    return float(value)


def _nonnegative(doc, key):
    value = doc.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ScenarioError(f'{key} must be a finite nonnegative number')
    return float(value)


def _relative_file(root: Path, raw: str):
    if not isinstance(raw, str) or not raw or '\x00' in raw:
        raise ScenarioError('File paths must be nonempty relative strings')
    path = Path(raw)
    if path.is_absolute():
        raise ScenarioError('Absolute scenario file paths not allowed')
    result = (root / path).resolve()
    if not result.is_relative_to(root):
        raise ScenarioError('Scenario file path escapes project')
    if not result.is_file():
        raise ScenarioError(f'Required scenario file missing: {result}')
    return result


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False, suffix='.tmp') as f:
        temporary = Path(f.name)
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def _write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False, suffix='.tmp', newline='') as f:
        temporary = Path(f.name)
        frame.to_csv(f, index=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def validate(doc):
    if not isinstance(doc, dict) or doc.get('schema_version') != 1 or doc.get('mode') != 'explicit_hypothetical_independent_research_scenario':
        raise ScenarioError('Require research scenario schema 1 and explicit hypothetical mode')
    if doc.get('assumptions_acknowledged') is not True:
        raise ScenarioError('Scenario must explicitly acknowledge hypothetical assumptions')
    solar = _nonnegative(doc, 'solar_assumed_kw')
    wind = _nonnegative(doc, 'wind_assumed_kw')
    supply = _positive(doc, 'available_power_kw')
    if not math.isclose(solar + wind, supply, rel_tol=1e-10, abs_tol=1e-8):
        raise ScenarioError('Assumed solar + wind must equal assumed total supply')
    if not isinstance(doc.get('renewable_source'), str) or 'ILLUSTRATIVE' not in doc['renewable_source']:
        raise ScenarioError('Renewable power assumptions must be visibly labeled ILLUSTRATIVE')
    sec = _positive(doc, 'specific_energy_kwh_per_kg_h2')
    if not isinstance(doc.get('specific_energy_reference'), str) or not doc['specific_energy_reference'].startswith('https://'):
        raise ScenarioError('Documented literature benchmark URL is required')
    requested = _positive(doc, 'fuel_cell_reference_flow_l_min')
    gap = _positive(doc, 'fuel_cell_reference_max_gap_l_min')
    temp = _positive(doc, 'reference_gas_temperature_k')
    pressure = _positive(doc, 'reference_gas_pressure_pa')
    molar_mass = _positive(doc, 'hydrogen_molar_mass_kg_per_mol')
    if not all(isinstance(doc.get(k), str) and doc[k].strip() for k in
               ('hydrogen_candidates_file','fuel_cell_history_file','reference_gas_note','specific_energy_context')):
        raise ScenarioError('Missing required file or assumption documentation')
    return supply, sec, requested, gap, temp, pressure, molar_mass


def scenario_candidates(candidates, *, supply, sec, temperature, pressure, molar_mass, fc_demand):
    required = ('candidate_id', 'renewable_power_kw', 'predicted_hydrogen_production_kg_h', 'passes_research_screening')
    if any(col not in candidates.columns for col in required):
        raise ScenarioError(f'Missing Phase3 candidate columns: {[c for c in required if c not in candidates.columns]}')
    if candidates.empty or candidates.candidate_id.isna().any() or candidates.candidate_id.duplicated().any():
        raise ScenarioError('Phase3 candidate IDs must exist and be unique')
    if not candidates['passes_research_screening'].map(lambda x: x is True or (isinstance(x, str) and x == 'True')).all():
        # Pandas numpy.bool_ is handled by explicit equality without coalescing NaN.
        if not candidates['passes_research_screening'].eq(True).all():
            raise ScenarioError('Input is not fully Phase3-research-screened')
    frame = candidates[['candidate_id', 'renewable_power_kw', 'predicted_hydrogen_production_kg_h']].copy()
    for col in ('renewable_power_kw', 'predicted_hydrogen_production_kg_h'):
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
        values = frame[col].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ScenarioError(f'Invalid/nonfinite/negative input in {col}')
    # Exact ideal-gas hypothetical reference condition: kg/h -> mol/h -> m3/h -> L/min.
    liters_per_kg_h = (8.31446261815324 * temperature / pressure) * 1000 / (60 * molar_mass)
    frame['assumed_total_supply_kw'] = supply
    frame['power_proxy_within_assumed_supply'] = frame['renewable_power_kw'].le(supply)
    frame['literature_benchmark_required_kw'] = frame['predicted_hydrogen_production_kg_h'] * sec
    frame['literature_benchmark_within_assumed_supply'] = frame['literature_benchmark_required_kw'].le(supply)
    frame['research_scenario_candidate_supported'] = (frame['power_proxy_within_assumed_supply'] &
                                                      frame['literature_benchmark_within_assumed_supply'])
    frame['ideal_gas_h2_capacity_l_min_reference_conditions'] = frame['predicted_hydrogen_production_kg_h'] * liters_per_kg_h
    frame['illustrative_fc_reference_flow_l_min'] = fc_demand
    frame['illustrative_fc_flow_capacity_sufficient'] = frame['ideal_gas_h2_capacity_l_min_reference_conditions'].ge(fc_demand)
    frame['simulated_combined_screen_pass'] = (frame['research_scenario_candidate_supported'] &
                                              frame['illustrative_fc_flow_capacity_sufficient'])
    frame['simulated_fc_usable_flow_not_measured'] = True
    frame['physical_dispatch_approved'] = False
    frame['same_calendar_verified'] = False
    frame['same_device_chain_verified'] = False
    frame['scenario_scope'] = 'ILLUSTRATIVE_INDEPENDENT_RESEARCH_NOT_PHYSICAL_INTEGRATION'
    return frame


def _fuel_cell_reference(fc, requested_flow, gap):
    fields = ['hydrogen_flow_l_min','voltage_v','temperature_c','pressure_bar']
    if any(f not in fc for f in fields) or fc.empty:
        raise ScenarioError('Fuel Cell historical dataset missing input features or empty')
    for f in fields:
        if not np.isfinite(pd.to_numeric(fc[f], errors='coerce').to_numpy(dtype=float)).all():
            raise ScenarioError(f'Fuel Cell historical input not numeric/finite: {f}')
    distances = (fc['hydrogen_flow_l_min'].astype(float) - requested_flow).abs()
    idx = distances.idxmin()
    row = fc.loc[[idx], :].copy()
    actual_flow = float(row['hydrogen_flow_l_min'].iloc[0])
    if abs(actual_flow - requested_flow) > gap:
        raise ScenarioError('No observed Fuel Cell flow within declared matching tolerance')
    return row


def _model_fc_reference(root, row):
    """Real saved FC model on an UNMODIFIED historical reference point; no candidate inlet claim."""
    try:
        from prescriptive_engine import engine
    except ImportError:
        # Direct script is always placed at the project root; retain standalone Python import.
        import sys
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from prescriptive_engine import engine
    cfg = engine.read_config(root / 'config.json')
    engine.require_verified_preprocessing(cfg, 'fuel_cell')
    names = engine.load_feature_file(cfg, 'fuel_cell', cfg['fuel_cell_features'])
    from pathlib import Path as _Path
    model = engine.load_pickle(_Path(cfg['models_dir']) / cfg['fuel_cell_model_file'])
    scaler = (engine.load_pickle(_Path(cfg['models_dir']) / cfg['fuel_cell_scaler_file'])
              if cfg['input_modes']['fuel_cell'] == 'scaled' else None)
    X = engine.prepare_input(
        row, model, names, scaler, cfg['input_modes']['fuel_cell'], 'Phase4 research historical FC reference',
        scaler_order_verified=cfg.get('scaler_feature_order_verified',{}).get('fuel_cell',False),
        model_order_verified=cfg.get('model_feature_order_verified',{}).get('fuel_cell',False),
        scaler_order_reference=cfg.get('scaler_feature_order_reference',{}).get('fuel_cell',''),
        model_order_reference=cfg.get('model_feature_order_reference',{}).get('fuel_cell',''),
        scaler_feature_order=cfg.get('scaler_feature_order',{}).get('fuel_cell'))
    predicted = model.predict(X)
    mapping = cfg['fuel_cell_pass_constraint']
    if mapping.get('class_mapping_verified') is not True or not mapping.get('class_mapping_reference'):
        raise ScenarioError('FC class mapping unverified in config')
    mapping = mapping['class_label_mapping']
    label = mapping.get(str(predicted[0]))
    if label not in ('Pass', 'Fail'):
        raise ScenarioError(f'FC classifier returned unmapped label: {predicted[0]}')
    return label


def run(scenario_file, *, output_dir=None, root_override=None, fc_predictor=None):
    scenario_file = Path(scenario_file).resolve()
    root = Path(root_override).resolve() if root_override else scenario_file.parent
    if not root.is_dir():
        raise ScenarioError('Project root does not exist')
    with scenario_file.open(encoding='utf-8-sig') as file:
        doc = json.load(file)
    supply, sec, requested, gap, temp, pressure, molar_mass = validate(doc)
    candidates = pd.read_csv(_relative_file(root, doc['hydrogen_candidates_file']), encoding='utf-8-sig')
    fc = pd.read_csv(_relative_file(root, doc['fuel_cell_history_file']), encoding='utf-8-sig')
    historical_fc_row = _fuel_cell_reference(fc, requested, gap)
    if fc_predictor is None:
        model_result = _model_fc_reference(root, historical_fc_row)
    else:
        model_result = fc_predictor(historical_fc_row)
    if model_result not in ('Pass', 'Fail'):
        raise ScenarioError('Invalid FC model output')
    flow = float(historical_fc_row['hydrogen_flow_l_min'].iloc[0])
    evaluated = scenario_candidates(candidates, supply=supply, sec=sec, temperature=temp,
                                    pressure=pressure, molar_mass=molar_mass, fc_demand=flow)
    admitted = evaluated[evaluated['simulated_combined_screen_pass']].copy()
    result = 'RESEARCH_SCENARIO_COMPLETE_NOT_PHYSICAL_INTEGRATION' if not admitted.empty else 'RESEARCH_SCENARIO_NO_ADMISSIBLE_CANDIDATES'
    output = Path(output_dir).resolve() if output_dir else root / 'phase4_simulation_results'
    # Generate and write files only after validation and model inference succeeded.
    _write_csv(output / 'phase4_simulated_candidates.csv', evaluated)
    _write_csv(output / 'phase4_simulated_eligible.csv', admitted)
    reference_output = historical_fc_row[['test_id', 'hydrogen_flow_l_min','voltage_v','temperature_c','pressure_bar']].copy() if 'test_id' in historical_fc_row.columns else historical_fc_row[['hydrogen_flow_l_min','voltage_v','temperature_c','pressure_bar']].copy()
    reference_output['predicted_fc_label_at_unchanged_historical_point'] = model_result
    reference_output['fc_candidate_specific_inference'] = False
    _write_csv(output / 'phase4_fc_historical_reference.csv', reference_output)
    counts = {'all_phase3_screened': len(evaluated),
              'proxy_within_assumed_supply': int(evaluated['power_proxy_within_assumed_supply'].sum()),
              'benchmark_within_assumed_supply': int(evaluated['literature_benchmark_within_assumed_supply'].sum()),
              'combined_research_screen_pass': int(evaluated['simulated_combined_screen_pass'].sum())}
    best = None
    if not admitted.empty:
        selected = admitted.sort_values(['predicted_hydrogen_production_kg_h','candidate_id'],ascending=[False,True]).iloc[0]
        best = {'candidate_id': str(selected['candidate_id']),
                'predicted_hydrogen_production_kg_h': float(selected['predicted_hydrogen_production_kg_h']),
                'literature_benchmark_required_kw': float(selected['literature_benchmark_required_kw']),
                'ideal_gas_h2_capacity_l_min': float(selected['ideal_gas_h2_capacity_l_min_reference_conditions'])}
    status = {
       'phase': 4,
       'status': result,
       'research_simulation_scope_closed': result == 'RESEARCH_SCENARIO_COMPLETE_NOT_PHYSICAL_INTEGRATION',
       'created_utc': datetime.now(timezone.utc).isoformat(),
       'scenario': {
          'assumed_solar_kw': doc['solar_assumed_kw'],
          'assumed_wind_kw': doc['wind_assumed_kw'],
          'assumed_total_kw': supply,
          'source': doc['renewable_source'],
          'renewables_are_model_forecasts': False,
          'actual_solar_wind_time_alignment': False,
          'electrolyzer_benchmark_kwh_per_kg': sec,
          'benchmark_reference': doc['specific_energy_reference'],
          'reference_conditions': {'temperature_k':temp,'pressure_pa':pressure,'ideal_dry_h2': True}
       },
       'counts': counts,
       'selected_research_candidate': best,
       'fuel_cell': {
          'historic_reference_test_id': str(historical_fc_row['test_id'].iloc[0]) if 'test_id' in historical_fc_row else None,
          'historic_reference_observed_flow_l_min': flow,
          'saved_model_prediction_at_historic_reference': model_result,
          'candidate_specific_fc_prediction': False,
          'observed_reference_point_not_linked_to_hydrogen_candidates': True,
          'historical_point_within_requested_flow_gap_l_min': gap
       },
       'integration_limits': {
          'physical_safety_approved': False,
          'hydrogen_to_fc_flow_measured': False,
          'same_device_chain_verified': False,
          'solar_wind_hydrogen_calendar_aligned': False,
          'electrolyzer_consumed_power_verified': False,
          'complete_four_model_physical_integration': False,
          'this_is_an_illustrative_benchmark_not_a_physical_dispatch': True
       },
       'output_files': ['phase4_simulated_candidates.csv','phase4_simulated_eligible.csv',
                        'phase4_fc_historical_reference.csv','phase4_simulation_status.json'],
       'important_note': 'Research benchmark scenario only. A separate historical FC model point is NOT an FC operating prediction on produced H2; solar/wind shares are illustrative assumed inputs, not actual forecasts.'
    }
    _write_json(output / 'phase4_simulation_status.json', status)
    print(f'[PHASE4 RESEARCH SIMULATION] {result}; {counts["combined_research_screen_pass"]}/{counts["all_phase3_screened"]} candidates meet hypothetical assumptions')
    print(f'Fuel Cell historical model reference: {model_result}; NOT candidate-specific FC inference')
    print(f'Output: {output / "phase4_simulation_status.json"}')
    print('Research scope only; no verified four-model physical coupling, calendar alignment, or safety approval.')
    return status


def main():
    ap = argparse.ArgumentParser(description='Phase 4 research-only stand-alone hypothetical system scenario')
    ap.add_argument('--scenario', default='phase4_research_scenario.json')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    run(args.scenario, output_dir=args.out)


if __name__ == '__main__':
    main()
