import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from phase4_research_simulation import ScenarioError, validate, scenario_candidates, run

ROOT = Path(__file__).resolve().parents[1]


class Phase4ResearchTests(unittest.TestCase):
    def doc(self):
        return json.loads((ROOT/'phase4_research_scenario.json').read_text(encoding='utf-8'))

    def test_refuse_unacknowledged_assumptions(self):
        d = self.doc(); d['assumptions_acknowledged'] = False
        with self.assertRaises(ScenarioError): validate(d)

    def test_refuse_unbalanced_renewables(self):
        d = self.doc(); d['wind_assumed_kw'] = 150
        with self.assertRaisesRegex(ScenarioError,'must equal'): validate(d)

    def test_refuse_nan_benchmark(self):
        d = self.doc(); d['specific_energy_kwh_per_kg_h2'] = float('nan')
        with self.assertRaises(ScenarioError): validate(d)

    def test_supply_and_benchmark_are_both_required(self):
        points = pd.DataFrame({'candidate_id':[1,2,3], 'renewable_power_kw':[100.,210.,100.],
                               'predicted_hydrogen_production_kg_h':[3.,1.,5.],
                               'passes_research_screening':[True]*3})
        result = scenario_candidates(points, supply=200.,sec=55.,temperature=273.15,
                    pressure=101325.,molar_mass=0.00201588,fc_demand=15.)
        self.assertEqual(result['simulated_combined_screen_pass'].tolist(),[True,False,False])
        self.assertFalse(result['same_calendar_verified'].any())
        self.assertFalse(result['physical_dispatch_approved'].any())

    def test_flow_conversion_zero_and_reference_calculation(self):
        points = pd.DataFrame({'candidate_id':[1], 'renewable_power_kw':[100.],
                               'predicted_hydrogen_production_kg_h':[1.],
                               'passes_research_screening':[True]})
        result = scenario_candidates(points, supply=200.,sec=55.,temperature=273.15,
                    pressure=101325.,molar_mass=0.00201588,fc_demand=15.)
        self.assertTrue(180 < result['ideal_gas_h2_capacity_l_min_reference_conditions'].iloc[0] < 190)
        self.assertTrue(bool(result['simulated_combined_screen_pass'].iloc[0]))

    def test_refuse_unverified_phase3_candidates(self):
        points = pd.DataFrame({'candidate_id':[1], 'renewable_power_kw':[100.],
                               'predicted_hydrogen_production_kg_h':[1.],
                               'passes_research_screening':[False]})
        with self.assertRaises(ScenarioError):
            scenario_candidates(points, supply=200.,sec=55.,temperature=273.15,
                    pressure=101325.,molar_mass=0.00201588,fc_demand=15.)

    def test_end_to_end_real_phase3_data_mock_only_fc_model(self):
        if not (ROOT/'phase3_results/phase3_research_passed.csv').is_file():
            self.skipTest('Original 1622 candidate fixture unavailable')
        with tempfile.TemporaryDirectory() as temp:
            status = run(ROOT/'phase4_research_scenario.json', output_dir=temp,
                         fc_predictor=Mock(return_value='Pass'))
            self.assertEqual(status['status'], 'RESEARCH_SCENARIO_COMPLETE_NOT_PHYSICAL_INTEGRATION')
            self.assertEqual(status['counts']['all_phase3_screened'],1622)
            self.assertEqual(status['counts']['combined_research_screen_pass'],535)
            self.assertTrue(status['research_simulation_scope_closed'])
            self.assertFalse(status['integration_limits']['complete_four_model_physical_integration'])
            self.assertFalse(status['fuel_cell']['candidate_specific_fc_prediction'])
            file = Path(temp)/'phase4_simulation_status.json'
            self.assertTrue(file.exists())

    def test_fc_simulation_no_false_class_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ScenarioError, 'Invalid FC model output'):
                run(ROOT/'phase4_research_scenario.json',output_dir=temp,
                    fc_predictor=Mock(return_value='Definitely Safe'))
            self.assertFalse((Path(temp)/'phase4_simulation_status.json').exists())


if __name__ == '__main__':
    unittest.main()
