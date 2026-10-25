"""Acceptance criteria for the user's original seven-step pipeline (research scope)."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from prescriptive_engine.pipeline.contracts import validate, PipelineBlocked
from prescriptive_engine.pipeline.optimization import make_candidates
from prescriptive_engine.pipeline.renewable import fixed_available_power
from prescriptive_engine.pipeline.fuel_cell import kg_h_to_reference_l_min
from prescriptive_engine.pipeline import run_scenario

ROOT=Path(__file__).resolve().parents[1]

class OriginalPipelineAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc=json.loads((ROOT/'configs/scenarios/RESEARCH_DEMO_200KW.json').read_text())

    def test_scenario_valid(self):
        validate(self.doc)

    def test_unacknowledged_bridge_blocked(self):
        d=copy.deepcopy(self.doc)
        d['fuel_cell']['acknowledge_unverified_device_chain']=False
        with self.assertRaises(PipelineBlocked):validate(d)

    def test_fixed_power_for_all_candidate_rows(self):
        df=pd.read_csv(ROOT/'data/green_hydrogen_standardized.csv')
        candidates=make_candidates(self.doc,df,200.0)
        self.assertEqual(candidates.renewable_power_kw.nunique(),1)
        self.assertEqual(candidates.renewable_power_kw.iloc[0],200.0)
        self.assertEqual(len(candidates),2501)
        self.assertTrue((candidates.loc[0,'origin']=='BASELINE'))
        self.assertEqual({k:float(candidates.loc[0,k]) for k in self.doc['baseline']},self.doc['baseline'])

    def test_model_source_rejects_missing_solar_model(self):
        from prescriptive_engine.pipeline.inference import Models
        from prescriptive_engine import engine
        cfg=engine.read_config(ROOT/'config.json')
        d=copy.deepcopy(self.doc['renewable_power']);d.update(mode='model_scenario',
            solar_inputs={k:0.0 for k in cfg['solar_features']},
            wind_inputs={k:0.0 for k in cfg['wind_features']},
            same_hypothetical_scenario_acknowledged=True,
            aggregation_scope='ONE_SOLAR_SOURCE_PLUS_ONE_WIND_TURBINE')
        if not (ROOT/'models/best_Solar_Power_model.pkl').exists():
            with self.assertRaisesRegex(PipelineBlocked,'MISSING_MODEL_ARTIFACT'):
                fixed_available_power(d,Models(cfg))

    def test_ideal_gas_reference_conversion(self):
        a=kg_h_to_reference_l_min([1.0],273.15,101325)
        self.assertTrue(170<float(a[0])<200)
        self.assertAlmostEqual(float(kg_h_to_reference_l_min([2.0],273.15,101325)[0]),2*float(a[0]))

    def test_nan_and_wrong_units_rejected(self):
        d=copy.deepcopy(self.doc)
        d['renewable_power']['available_kw']=float('nan')
        with self.assertRaises(PipelineBlocked):validate(d)

    def test_no_fc_bridge_never_reports_pass(self):
        with tempfile.TemporaryDirectory() as out:
            r=run_scenario(ROOT/'configs/scenarios/STRICT_NO_FC_BRIDGE.json',config_path=ROOT/'config.json',output_dir=out)
            self.assertEqual(r['status'],'BLOCKED_FC_BRIDGE_NO_FEASIBLE_RESULT')
            self.assertEqual(r['conditional_research_feasible_count'],0)
            self.assertIsNone(r['best_candidate_id'])
            self.assertFalse(r['physical_safety_approved'])

    def test_research_demo_end_to_end(self):
        with tempfile.TemporaryDirectory() as out:
            r=run_scenario(ROOT/'configs/scenarios/RESEARCH_DEMO_200KW.json',config_path=ROOT/'config.json',output_dir=out)
            self.assertEqual(r['fixed_available_power_kw'],200.0)
            self.assertEqual(r['candidate_count'],2501)
            self.assertFalse(r['physical_safety_approved'])
            self.assertFalse(r['candidate_specific_fc_physical_link_verified'])
            all_rows=pd.read_csv(Path(out)/'all_candidates.csv')
            self.assertEqual(all_rows['renewable_power_kw'].nunique(),1)
            self.assertEqual(all_rows['origin'].iloc[0],'BASELINE')
            self.assertEqual(all_rows['conditional_research_feasible'].sum(),r['conditional_research_feasible_count'])
            self.assertIn(r['status'],('RESEARCH_CONDITIONAL_RESULT_NOT_PHYSICAL_VALIDATION','NO_RESEARCH_FEASIBLE_CANDIDATE'))
