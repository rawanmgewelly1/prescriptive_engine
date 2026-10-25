"""Final acceptance tests for the research-only path. No PKL loads or production changes."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("research_acceptance_engine", ROOT / "prescriptive_engine" / "engine.py")
engine = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = engine
spec.loader.exec_module(engine)
config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def candidate(cid, supply, predicted):
    row = {"candidate_id": cid, "renewable_power_kw": supply,
           "predicted_hydrogen_production_kg_h": predicted}
    row.update({name: 1.0 for name in config["decision_variables"]})
    return row


class FinalAcceptance(unittest.TestCase):
    def test_200kw_reference_dataset(self):
        path = ROOT / "fixture_candidates.csv"
        if not path.exists():
            self.skipTest("The 3000-row fixture is not included in this test environment")
        result, summary = engine.research_power_dispatch(config, pd.read_csv(path), 200)
        self.assertEqual(len(result), 988)
        self.assertEqual(int(result["pareto_frontier"].sum()), 7)
        self.assertAlmostEqual(float(summary.at[0, "best_predicted_hydrogen_kg_h"]), 3.297114, places=5)
        self.assertEqual(summary.at[0, "status"], "RESEARCH_SCENARIO_NOT_AN_OPERATING_COMMAND")
        self.assertFalse(result["research_status"].eq("APPROVED").any())
        self.assertTrue(result["renewable_power_kw"].le(200).all())

    def test_invalid_values_fail_closed(self):
        frame = pd.DataFrame([candidate(1, 100, 2)])
        for value in (-1, 0, float("inf"), float("nan"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                engine.research_power_dispatch(config, frame, value)

    def test_no_matching_candidate_remains_unapproved(self):
        result, summary = engine.research_power_dispatch(config, pd.DataFrame([candidate(1, 100, 2)]), 1)
        self.assertTrue(result.empty)
        self.assertEqual(summary.at[0, "matching_candidate_count"], 0)
        self.assertEqual(summary.at[0, "physical_feasibility"], "NOT_VERIFIED")

    def test_equal_objectives_keep_nondominated_ties(self):
        rows = [candidate(1,100,2),candidate(2,100,2),candidate(3,100,1)]
        result, _ = engine.research_power_dispatch(config, pd.DataFrame(rows), 200)
        self.assertEqual(set(result.loc[result["pareto_frontier"], "candidate_id"]), {1,2})

    def test_missing_column_and_duplicates_fail(self):
        frame = pd.DataFrame([candidate(1,100,2),candidate(2,150,3)])
        with self.assertRaises(ValueError):
            engine.research_power_dispatch(config, frame.drop(columns=[config["decision_variables"][0]]), 200)
        frame.at[1,"candidate_id"] = 1
        with self.assertRaises(ValueError):
            engine.research_power_dispatch(config, frame, 200)

if __name__ == "__main__":
    unittest.main()
