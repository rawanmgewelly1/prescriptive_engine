import json,sys,tempfile,unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'prescriptive_engine'))
from electrolyzer_simulation import *

class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.valid={'mode':'research_simulation','model_type':'fixed_fraction_available_to_load','source_reference':'SYNTHETIC TEST REFERENCE','equation_reference':'SYNTHETIC TEST EQUATION','input_power_boundary':'synthetic available bus','load_power_boundary':'synthetic load','model_applicability_reference':'SYNTHETIC TEST ONLY','conversion_fraction':0.8}
        self.df=pd.DataFrame({'renewable_power_kw':[100,200], 'predicted_hydrogen_production_kg_h':[2,4]})
    def test_missing_model_does_not_compute(self):
        df,meta=evaluate_simulated_consumption(pd.DataFrame(),{'mode':'missing_model','reason':'no model'})
        self.assertTrue(df.empty);self.assertFalse(meta['sec_approved'])
    def test_synthetic_arithmetic_unapproved(self):
        df,meta=evaluate_simulated_consumption(self.df,self.valid)
        self.assertEqual(df['simulated_electrical_load_kw'].tolist(),[80,160])
        self.assertEqual(df['simulated_sec_kwh_per_predicted_kg'].tolist(),[40,40])
        self.assertFalse(meta['dispatch_approved']);self.assertFalse(meta['sec_approved'])
    def test_no_defaults_and_invalid_fraction(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'model.json'
            for value in [0,1.1,True]:
                obj=dict(self.valid,conversion_fraction=value);f.write_text(json.dumps(obj))
                with self.assertRaises(SimulationContractError):load_simulation_contract(f)
            f.write_text(json.dumps({'mode':'research_simulation'}))
            with self.assertRaises(SimulationContractError):load_simulation_contract(f)
    def test_invalid_candidates_blocked(self):
        self.df.loc[0,'renewable_power_kw']=-1
        with self.assertRaises(SimulationContractError):evaluate_simulated_consumption(self.df,self.valid)
    def test_missing_contract_clears_stale_results(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);contract=d/'contract.json';contract.write_text(json.dumps({'mode':'missing_model','reason':'no reference'}))
            (d/'electrolyzer_simulated_candidates.csv').write_text('stale')
            meta=run_simulation(contract,d/'missing.csv',d)
            self.assertFalse((d/'electrolyzer_simulated_candidates.csv').exists())
            self.assertEqual(meta['simulation_status'],'BLOCKED_MISSING_DOCUMENTED_MODEL')
if __name__=='__main__':unittest.main()
