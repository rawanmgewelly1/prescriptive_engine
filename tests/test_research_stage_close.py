"""Research-stage acceptance gates. Never certify physical device feasibility."""
import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from prescriptive_engine.pipeline import run_scenario
from prescriptive_engine.pipeline.optimization import empirical_joint_support

ROOT=Path(__file__).resolve().parents[1]

class ResearchStageCloseTests(unittest.TestCase):
    @staticmethod
    def scenario(name, out):
        return run_scenario(ROOT/'configs'/'scenarios'/f'{name}.json',
                            config_path=ROOT/'config.json',output_dir=out)

    def test_four_model_prediction_runs_but_remains_unverified(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.scenario('RESEARCH_FOUR_MODELS_HYPOTHETICAL', d)
            self.assertEqual(r['status'],'RESEARCH_CONDITIONAL_RESULT_NOT_PHYSICAL_VALIDATION',r.get('reason'))
            self.assertTrue(r['full_four_model_pipeline_executed'])
            self.assertFalse(r['full_four_model_pipeline_verified'])
            self.assertIsNotNone(r['renewable_stage']['solar_prediction_kw'])
            self.assertIsNotNone(r['renewable_stage']['wind_prediction_kw'])
            self.assertAlmostEqual(r['fixed_available_power_kw'],r['renewable_stage']['solar_prediction_kw']+r['renewable_stage']['wind_prediction_kw'])
            c=pd.read_csv(Path(d)/'all_candidates.csv')
            self.assertEqual(c['renewable_power_kw'].nunique(),1)
            self.assertEqual(len(c),r['candidate_count'])
            self.assertGreater(r['empirical_joint_support_pass_count'],0)
            self.assertTrue((c['conditional_research_feasible']<=c['empirical_joint_support_pass']).all())

    def test_regulated_setpoint_classifier_is_declared_non_discriminating(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.scenario('RESEARCH_DEMO_200KW',d)
            self.assertEqual(r['fuel_cell_stage']['distinct_fc_model_input_flow_values'],1)
            self.assertTrue(r['fc_fixed_setpoint_cannot_discriminate_candidates'])
            self.assertFalse(r['fuel_cell_stage']['fc_model_class_is_candidate_discriminating'])
            self.assertFalse(r['power_constraint_verified'])
            self.assertFalse(r['physical_safety_approved'])
            self.assertGreaterEqual(len(r['physical_integration_blockers']),3)

    def test_direct_flow_without_training_support_gives_no_false_pass(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.scenario('RESEARCH_FC_DIRECT_FLOW_SUPPORT_CHECK',d)
            self.assertEqual(r['status'],'NO_RESEARCH_FEASIBLE_CANDIDATE',r.get('reason'))
            self.assertEqual(r['fc_predicted_pass_count'],0)
            self.assertEqual(r['conditional_research_feasible_count'],0)
            c=pd.read_csv(Path(d)/'all_candidates.csv')
            self.assertTrue(c['fuel_cell_status'].str.contains('OUT_OF_OBSERVED_SUPPORT').all())

    def test_strict_missing_bridge_stays_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.scenario('STRICT_NO_FC_BRIDGE',d)
            self.assertEqual(r['status'],'BLOCKED_FC_BRIDGE_NO_FEASIBLE_RESULT')
            self.assertEqual(r['conditional_research_feasible_count'],0)

    def test_joint_support_rejects_extreme_combination(self):
        h=pd.read_csv(ROOT/'data/green_hydrogen_standardized.csv')
        row=h.iloc[[0]].copy()
        mask,dist,cut=empirical_joint_support(row,h)
        self.assertTrue(bool(mask[0]))
        row.loc[:,'renewable_power_kw']=1000000
        mask,dist,cut=empirical_joint_support(row,h)
        self.assertFalse(bool(mask[0]))
