"""Phase4 regression: no invented mapping, no stale outputs, direct Windows-style CLI."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from prescriptive_engine.phase4_integration import (
    Phase4ContractError, _hydrogen_row_map, _hydrogen_power,
    _fuel_cell_link, _renewables, _path, run,
)
from prescriptive_engine import engine

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / 'phase3_results' / 'phase3_research_passed.csv'
SOURCE = ROOT / 'data' / 'green_hydrogen_standardized.csv'


class Phase4Regression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidates = pd.read_csv(CANDIDATES)
        cls.source = pd.read_csv(SOURCE)

    def test_phase3_candidate_provenance_recovers_exact_unique_original_rows(self):
        aligned = _hydrogen_row_map(self.candidates, self.source)
        self.assertEqual(len(aligned), 1622)
        self.assertEqual(int(aligned['legacy_source_row_matches_original'].sum()), 0)
        self.assertEqual(aligned['original_sample_id_matched_by_features'].nunique(), 1622)

    def test_duplicate_original_operating_state_must_not_pick_arbitrary_clock(self):
        duplicated = pd.concat([self.source, self.source.iloc[[0]]], ignore_index=True)
        duplicated.loc[duplicated.index[-1], 'sample_id'] = 987654
        with self.assertRaisesRegex(Phase4ContractError, 'multiple source timestamps'):
            _hydrogen_row_map(self.candidates.iloc[[0]], duplicated)

    def test_default_alignment_requires_documented_evidence(self):
        cfg = engine.read_config(ROOT / 'config.json')
        s = pd.DataFrame({'timestamp': ['2014-10-20 12:00'], 'predicted_solar_power_kw': [12.]})
        w = pd.DataFrame({'timestamp': ['2014-10-20 12:00'], 'predicted_wind_power_kw': [3.], 'wind_output_scope': ['farm_total']})
        actual, status = _renewables(cfg, s, w)
        self.assertIsNone(actual)
        self.assertEqual(status['status'], 'BLOCKED_RENEWABLE_ALIGNMENT')

    def test_documented_two_source_fixture_can_combine_without_unchecked_math(self):
        cfg = engine.read_config(ROOT / 'config.json')
        align = cfg['alignment']
        refs = cfg['alignment_verification_reference']
        for field in ('timestamps_compatible_verified', 'solar_power_semantics_verified',
                      'wind_power_semantics_verified', 'wind_aggregation_semantics_verified'):
            align[field] = True
            refs[field] = 'SYNTHETIC_TEST_FIXTURE_ONLY: controlled clock and unit assumptions'
        align['solar_power_scope'] = 'farm_total'
        align['wind_power_scope'] = 'farm_total'
        refs['solar_power_scope'] = 'SYNTHETIC_TEST_FIXTURE_ONLY: farm-total reference'
        refs['wind_power_scope'] = 'SYNTHETIC_TEST_FIXTURE_ONLY: farm-total reference'
        align['timestamp_timezone_mode'] = 'naive_same_timezone'
        refs['timestamp_timezone_mode'] = 'SYNTHETIC_TEST_FIXTURE_ONLY: common naive clock'
        s = pd.DataFrame({'timestamp': ['2026-02-01 00:00'], 'predicted_solar_power_kw': [150.]})
        w = pd.DataFrame({'timestamp': ['2026-02-01 00:00'], 'predicted_wind_power_kw': [75.],
                          'wind_output_scope': ['farm_total']})
        merged, status = _renewables(cfg, s, w)
        self.assertEqual(status['status'], 'RESEARCH_SOLAR_WIND_ALIGNED_NOT_PHYSICAL_APPROVAL', status)
        self.assertEqual(float(merged['predicted_total_renewable_power_kw'].iloc[0]), 225.)
        self.assertEqual(merged['physical_safety_status'].iloc[0], 'NOT_ESTABLISHED')

    def test_no_cross_dataset_hydrogen_calendar_shift(self):
        c = self.candidates.iloc[[0]].copy()
        stage = _hydrogen_row_map(c, self.source)
        self.assertTrue(stage['timestamp'].notna().all())
        aligned = pd.DataFrame({'timestamp': pd.to_datetime(['2014-10-20 12:00']),
                                'predicted_total_renewable_power_kw': [1000.]})
        evidence = {'common_clock_reference': 'Documented common UTC timestamp source',
                    'hydrogen_timestamp_reference': 'Hydrogen source notebook dates',
                    'hydrogen_timestamp_format': '%m/%d/%Y %H:%M'}
        matched, status = _hydrogen_power(ROOT, {}, evidence, aligned, c, self.source)
        self.assertIsNone(matched)
        self.assertEqual(status['status'], 'BLOCKED_ZERO_TIMESTAMP_OVERLAP')

    def test_exact_timestamp_hydrogen_joins_only_available_supply_proxy(self):
        c = self.candidates.iloc[[0]].copy()
        candidate_source = _hydrogen_row_map(c, self.source).iloc[0]
        aligned = pd.DataFrame({'timestamp': [pd.to_datetime(candidate_source['timestamp'], format='%m/%d/%Y %H:%M')],
                                'predicted_total_renewable_power_kw': [1000.]})
        evidence = {'common_clock_reference': 'Experiment common clock system identity',
                    'hydrogen_timestamp_reference': 'Hydrogen source documented timestamp format',
                    'hydrogen_timestamp_format': '%m/%d/%Y %H:%M'}
        result, status = _hydrogen_power(ROOT, {}, evidence, aligned, c, self.source)
        self.assertEqual(status['status'], 'RESEARCH_CLOCK_MATCHED_POWER_PROXY_ONLY')
        self.assertEqual(len(result), 1)
        self.assertTrue(bool(result['supply_proxy_supports_scenario'].iloc[0]))
        self.assertFalse(bool(result['consumed_electrolyzer_power_verified'].iloc[0]))
        self.assertFalse(bool(result['physical_dispatch_approved'].iloc[0]))

    def test_missing_fc_bridge_never_uses_fc_historical_flows(self):
        cfg = engine.read_config(ROOT / 'config.json')
        result, status = _fuel_cell_link(ROOT, cfg, {}, {}, self.candidates, self.source)
        self.assertIsNone(result)
        self.assertEqual(status['status'], 'BLOCKED_NO_VERIFIED_HYDROGEN_TO_FC_MAPPING')

    def test_fc_inference_only_with_explicit_measured_bridge_and_fc_operating_point(self):
        import joblib
        import numpy as np
        from unittest.mock import patch
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier

        cfg = engine.read_config(ROOT / 'config.json')
        features = cfg['fuel_cell_features']
        c = self.candidates.iloc[[0]].copy()
        source_row = _hydrogen_row_map(c, self.source).iloc[0]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            models = tmp/'models'; models.mkdir()
            selection = tmp/'selection'; selection.mkdir()
            pd.DataFrame({'feature': features}).to_csv(selection/cfg['selected_feature_files']['fuel_cell'], index=False)
            training = pd.DataFrame([
                {'hydrogen_flow_l_min': 15., 'voltage_v': 48., 'temperature_c': 70., 'pressure_bar': 3.},
                {'hydrogen_flow_l_min': 25., 'voltage_v': 45., 'temperature_c': 75., 'pressure_bar': 4.},
            ], columns=features)
            scaler_order = cfg['scaler_feature_order']['fuel_cell']
            scaler = StandardScaler().fit(training[scaler_order])
            scaled = pd.DataFrame(scaler.transform(training[scaler_order]), columns=scaler_order)[features]
            model = DecisionTreeClassifier(random_state=1).fit(scaled, [1, 0])
            joblib.dump(model, models/cfg['fuel_cell_model_file'])
            joblib.dump(scaler, models/cfg['fuel_cell_scaler_file'])
            cfg['models_dir'] = str(models)
            cfg['selected_features_dir'] = str(selection)
            bridge = pd.DataFrame([{
                'candidate_id': c['candidate_id'].iloc[0],
                'timestamp': source_row['timestamp'],
                'measurement_reference': 'independent flow meter document 123',
                'hydrogen_flow_l_min': 15., 'voltage_v': 48.,
                'temperature_c': 70., 'pressure_bar': 3.,
            }])
            bridge_path = tmp/'bridge.csv'
            bridge.to_csv(bridge_path, index=False)
            evidence = {
                'same_device_chain_reference': 'device H2 out -> FC verified design 123',
                'measured_flow_reference': 'independent flow meter document 123',
                'fuel_cell_operating_point_reference': 'FC operating point source 123',
                'shared_clock_reference': 'common lab log UTC clock doc 123',
                'fuel_cell_timestamp_format': '%m/%d/%Y %H:%M',
                'hydrogen_timestamp_format': '%m/%d/%Y %H:%M',
            }
            # Only the temp fixture resolver is overridden; production prevents paths leaving project root.
            with patch('prescriptive_engine.phase4_integration._path', return_value=bridge_path):
                result, state = _fuel_cell_link(ROOT, cfg, {'fuel_cell_bridge': 'bridge.csv'},
                                                evidence, c, self.source)
            self.assertEqual(state['status'], 'RESEARCH_FC_CLASSIFIED_WITH_EXPLICIT_BRIDGE', state)
            self.assertEqual(result['predicted_fc_class'].iloc[0], 'Pass')
            self.assertFalse(bool(result['physical_safety_approved'].iloc[0]))

    def test_invalid_external_file_path_rejected(self):
        with self.assertRaises(Phase4ContractError):
            _path(ROOT, '../outside.csv')
        with self.assertRaises(Phase4ContractError):
            _path(ROOT, '/tmp/nonlocal.csv')

    def test_direct_script_cli_no_circular_import(self):
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run([
                sys.executable, str(ROOT / 'prescriptive_engine' / 'engine.py'),
                '--phase4-contract', str(ROOT / 'phase4_contract.json'),
                '--research-out', str(Path(temp)/'out')], cwd=ROOT,
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            status = json.loads((Path(temp)/'out'/'phase4_status.json').read_text())
            self.assertEqual(status['overall_status'], 'BLOCKED_INCOMPLETE_CROSS_MODEL_EVIDENCE')
            self.assertFalse(status['physical_safety_approved'])
            self.assertEqual(status['components']['hydrogen_candidate_provenance']['rows'], 1622)

    def test_previous_success_outputs_archived_not_reused_when_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out/'phase4_renewable_aligned.csv').write_text('timestamp,power\n2014-01-01,123\n')
            (out/'phase4_status.json').write_text('{"overall_status":"STALE_FAKE_SUCCESS"}')
            status = run(ROOT/'phase4_contract.json', output_dir=out)
            self.assertEqual(status['overall_status'], 'BLOCKED_INCOMPLETE_CROSS_MODEL_EVIDENCE')
            self.assertFalse((out/'phase4_renewable_aligned.csv').exists())
            archive = list((out/'_previous_runs').rglob('phase4_renewable_aligned.csv'))
            self.assertEqual(len(archive),1)

    def test_contract_rejects_uncontrolled_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            obj = json.loads((ROOT/'phase4_contract.json').read_text())
            obj['input_files']['fuel_cell_bridge'] = '../../outside.csv'
            (root/'contract.json').write_text(json.dumps(obj))
            # A minimal config copy allows entering run without hitting real model assets.
            (root/'config.json').write_text((ROOT/'config.json').read_text())
            with self.assertRaises(Phase4ContractError):
                run(root/'contract.json')


if __name__ == '__main__':
    unittest.main()
