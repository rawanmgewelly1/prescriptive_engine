import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from prescriptive_engine.phase3_constraints import run, non_dominated, ContractError


class ObservedSupportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.dir = Path(self.temporary.name)
        (self.dir / 'data').mkdir()
        df = pd.DataFrame({
            'voltage_v': [float(v) for v in range(1, 101)],
            'temperature_c': [60.0] * 100,
        })
        df.to_csv(self.dir / 'data' / 'hydrogen.csv', index=False)
        self.rules = self.dir / 'rules.json'
        self.rules.write_text(json.dumps({
            'research_constraints': [],
            'observed_data_screening': {
                'enabled': True, 'mode': 'empirical_quantiles',
                'dataset_file': 'data/hydrogen.csv',
                'columns': ['voltage_v'],
                'lower_quantile': 0.05, 'upper_quantile': 0.95,
                'scientific_claim': 'OBSERVED_DATA_SUPPORT_ONLY_NOT_SAFETY'
            }
        }), encoding='utf-8')

    def test_screen_real_observations_and_pareto(self):
        candidates = pd.DataFrame({
            'candidate_id': [1, 2, 3, 4],
            'voltage_v': [1, 10, 70, 100],
            'predicted_hydrogen_production_kg_h': [5, 4, 3, 2],
            'predicted_kg_per_available_kwh_proxy': [.01, .02, .03, .04],
        })
        path = self.dir / 'candidates.csv'
        candidates.to_csv(path, index=False)
        result = run(path, self.rules, self.dir / 'results')
        self.assertEqual(result['passed_rows'], 2)
        self.assertEqual(result['rejected_rows'], 2)
        self.assertEqual(result['pareto_after_screening'], 2)
        self.assertEqual(result['research_status'], 'SCREENED_OBSERVED_SUPPORT_ONLY')
        self.assertFalse(result['physical_operating_approval'])
        self.assertFalse(result['engineering_limits_verified'])
        self.assertEqual(result['enabled_referenced_rules'], 0)
        self.assertEqual(result['empirical_support_rules'], 1)

    def test_bad_quantiles_block(self):
        cfg = json.loads(self.rules.read_text())
        cfg['observed_data_screening']['lower_quantile'] = 1
        self.rules.write_text(json.dumps(cfg))
        with self.assertRaises(ContractError):
            run(self.dir/'none.csv', self.rules, self.dir/'out')

    def test_absent_reference_blocks(self):
        cfg = json.loads(self.rules.read_text())
        cfg['observed_data_screening']['dataset_file'] = 'missing.csv'
        self.rules.write_text(json.dumps(cfg))
        with self.assertRaises(ContractError):
            run(self.dir/'none.csv', self.rules, self.dir/'out')

    def test_no_safety_label_blocks(self):
        cfg = json.loads(self.rules.read_text())
        cfg['observed_data_screening']['scientific_claim'] = 'APPROVED'
        self.rules.write_text(json.dumps(cfg))
        with self.assertRaises(ContractError):
            run(self.dir/'none.csv', self.rules, self.dir/'out')

    def test_dominance_ties(self):
        samples = pd.DataFrame({
            'predicted_hydrogen_production_kg_h': [5, 4, 3, 3, 6],
            'predicted_kg_per_available_kwh_proxy': [.02, .02, .03, .03, .01],
        })
        self.assertEqual(non_dominated(samples).tolist(), [True, False, True, True, True])


if __name__ == '__main__':
    unittest.main()
