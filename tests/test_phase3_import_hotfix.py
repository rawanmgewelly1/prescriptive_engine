"""Regression: a compatibility wrapper must never replace the actual package module."""
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class Phase3ImportFixTests(unittest.TestCase):
    def test_both_import_paths_point_to_one_real_implementation(self):
        import phase3_constraints as cli_wrapper
        from prescriptive_engine import phase3_constraints as core
        self.assertIs(cli_wrapper.run, core.run)
        self.assertIs(cli_wrapper.ContractError, core.ContractError)
        self.assertTrue(callable(core.non_dominated))
        self.assertTrue(callable(core.validated_rules))

    def test_engine_direct_script_is_not_circular(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            pd.DataFrame([{
                'candidate_id': 1,
                'predicted_hydrogen_production_kg_h': 1.1,
                'predicted_kg_per_available_kwh_proxy': 0.012,
            }]).to_csv(temp / 'candidates.csv', index=False)
            (temp / 'rules.json').write_text('{"research_constraints": []}', encoding='utf-8')
            cmd = [sys.executable, str(ROOT / 'prescriptive_engine' / 'engine.py'),
                   '--phase3-rules', str(temp / 'rules.json'),
                   '--candidate-file', str(temp / 'candidates.csv'),
                   '--research-out', str(temp / 'results')]
            result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('BLOCKED_NO_REFERENCED_CONSTRAINTS', result.stdout)
            self.assertFalse(json.loads((temp/'results'/'phase3_status.json').read_text(encoding='utf-8'))['physical_operating_approval'])

    def test_empirical_pareto_never_claims_safety(self):
        from prescriptive_engine.phase3_constraints import run
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            (temp / 'data').mkdir()
            pd.DataFrame({'voltage_v': list(range(100))}).to_csv(temp/'data'/'reference.csv', index=False)
            pd.DataFrame([
                {'candidate_id': 1, 'voltage_v': 30, 'predicted_hydrogen_production_kg_h': 2.0, 'predicted_kg_per_available_kwh_proxy': 0.03},
                {'candidate_id': 2, 'voltage_v': 70, 'predicted_hydrogen_production_kg_h': 3.0, 'predicted_kg_per_available_kwh_proxy': 0.02},
                {'candidate_id': 3, 'voltage_v': 98, 'predicted_hydrogen_production_kg_h': 4.0, 'predicted_kg_per_available_kwh_proxy': 0.04},
            ]).to_csv(temp/'candidates.csv', index=False)
            rules = {'research_constraints': [], 'observed_data_screening': {
                'enabled': True, 'mode': 'empirical_quantiles', 'dataset_file': 'data/reference.csv',
                'columns': ['voltage_v'], 'lower_quantile': 0.05, 'upper_quantile': 0.95,
                'scientific_claim': 'OBSERVED_DATA_SUPPORT_ONLY_NOT_SAFETY'}}
            (temp/'rules.json').write_text(json.dumps(rules), encoding='utf-8')
            summary = run(temp/'candidates.csv', temp/'rules.json', temp/'out')
            self.assertEqual(summary['passed_rows'], 2)
            self.assertEqual(summary['pareto_after_screening'], 2)
            self.assertFalse(summary['engineering_limits_verified'])
            self.assertFalse(summary['physical_operating_approval'])

if __name__ == '__main__':
    unittest.main()
