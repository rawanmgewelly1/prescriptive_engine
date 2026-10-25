import unittest, tempfile, json
from pathlib import Path
import pandas as pd
from prescriptive_engine.electrolyzer_simulation import load_simulation_contract, evaluate_simulated_consumption, SimulationContractError

class Phase2BenchmarkTests(unittest.TestCase):
    def get_contract(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'contract.json'
            p.write_text(Path(__file__).resolve().parents[1].joinpath('electrolyzer_simulation_model.json').read_text())
            return load_simulation_contract(p)
    def test_benchmark_equations_and_shortfall(self):
        data=pd.DataFrame({'renewable_power_kw':[200,100], 'predicted_hydrogen_production_kg_h':[3,3]})
        rows,meta=evaluate_simulated_consumption(data,self.get_contract())
        self.assertEqual(rows['benchmark_electrical_demand_kw'].tolist(),[165,165])
        self.assertEqual(rows['benchmark_demand_within_available_power'].tolist(),[True,False])
        self.assertEqual(rows['benchmark_power_shortfall_kw'].tolist(),[0,65])
        self.assertEqual(meta['benchmark_insufficient_supply_rows'],1)
        self.assertFalse(meta['dispatch_approved'])
    def test_reference_not_editable_into_unverified_numbers(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.get_contract(); c['specific_energy_kwh_per_kg']=45
            p=Path(d)/'x.json';p.write_text(json.dumps(c))
            with self.assertRaises(SimulationContractError): load_simulation_contract(p)
    def test_reject_empty_or_nonfinite_data(self):
        for data in [pd.DataFrame(columns=['renewable_power_kw','predicted_hydrogen_production_kg_h']),
                     pd.DataFrame({'renewable_power_kw':[200], 'predicted_hydrogen_production_kg_h':[float('nan')]})]:
            with self.assertRaises(SimulationContractError): evaluate_simulated_consumption(data,self.get_contract())
    def test_missing_model_still_blocks(self):
        rows,meta=evaluate_simulated_consumption(pd.DataFrame(),{'mode':'missing_model','reason':'none'})
        self.assertTrue(rows.empty)
        self.assertEqual(meta['simulation_status'],'BLOCKED_MISSING_DOCUMENTED_MODEL')
