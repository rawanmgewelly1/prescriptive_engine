import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('research_engine', ROOT / 'prescriptive_engine' / 'engine.py')
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)
CONFIG = json.loads((ROOT / 'reference_config.json').read_text(encoding='utf-8'))

def fixture():
    rows = []
    for cid, p, h in ((1, 100, 3), (2, 150, 4), (3, 200, 5), (4, 100, 2)):
        row = {"candidate_id": cid, "renewable_power_kw": p, "predicted_hydrogen_production_kg_h": h}
        row.update({v: 1.0 for v in CONFIG['decision_variables']})
        rows.append(row)
    return pd.DataFrame(rows)

class ResearchDispatchTests(unittest.TestCase):
    def test_filter_and_pareto_tradeoffs(self):
        selected, summary = MOD.research_power_dispatch(CONFIG, fixture(), 180.)
        self.assertEqual(set(selected['candidate_id']), {1,2,4})
        self.assertEqual(set(selected.loc[selected['pareto_frontier'], 'candidate_id']), {1,2})
        self.assertEqual(summary.loc[0,'status'], 'RESEARCH_SCENARIO_NOT_AN_OPERATING_COMMAND')
        self.assertEqual(set(selected['operating_safety']), {'NOT_VERIFIED'})

    def test_invalid_power_is_rejected(self):
        for power in (0, -1, float('nan'), float('inf'), True):
            with self.subTest(power=power), self.assertRaises(ValueError):
                MOD.research_power_dispatch(CONFIG, fixture(), power)

    def test_no_matching_rows_is_not_approved(self):
        selected, summary = MOD.research_power_dispatch(CONFIG, fixture(), 1)
        self.assertTrue(selected.empty)
        self.assertEqual(summary.loc[0, 'matching_candidate_count'], 0)
        self.assertEqual(summary.loc[0, 'physical_feasibility'], 'NOT_VERIFIED')

    def test_missing_features_and_duplicate_ids_block(self):
        with self.assertRaises(ValueError):
            MOD.research_power_dispatch(CONFIG, fixture().drop(columns=['pressure_bar']), 180)
        rows = fixture()
        rows.loc[1,'candidate_id'] = rows.loc[0, 'candidate_id']
        with self.assertRaises(ValueError):
            MOD.research_power_dispatch(CONFIG, rows, 180)

if __name__ == '__main__':
    unittest.main()
