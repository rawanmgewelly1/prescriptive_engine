import unittest
from pathlib import Path
import importlib.util

MODULE = Path(__file__).resolve().parents[1] / 'prescriptive_engine' / 'power_contract.py'
spec = importlib.util.spec_from_file_location('power_contract', MODULE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class PowerContractTests(unittest.TestCase):
    def test_research_input(self):
        result=mod.validate_power_contract({'mode':'research_scenario','source':'user_entered_available_power','available_power_kw':200})
        self.assertEqual(result['scientific_status'],'RESEARCH_SCENARIO_ONLY')
        self.assertEqual(result['consumed_power_status'],'NOT_AVAILABLE')
    def test_bad_power(self):
        for value in [-1,0,float('nan'),True]:
            with self.assertRaises(mod.PowerContractError):
                mod.validate_power_contract({'mode':'research_scenario','source':'user_entered_available_power','available_power_kw':value})
    def test_no_invented_consumption(self):
        with self.assertRaises(mod.PowerContractError):
            mod.validate_power_contract({'mode':'research_scenario','source':'user_entered_available_power','available_power_kw':200,'consumed_power_kw':175})
    def test_no_unsupported_alignment_claim(self):
        with self.assertRaises(mod.PowerContractError):
            mod.validate_power_contract({'mode':'research_scenario','source':'user_entered_available_power','available_power_kw':200,'solar_wind_historical_alignment_verified':True})
    def test_external_measurement_evidence(self):
        x={'mode':'research_scenario','source':'external_measurement','available_power_kw':200,'source_reference':'measurement-id-1','measurement_timestamp':'2026-10-23T20:00:00+03:00','measurement_point':'AC bus','measurement_basis':'instantaneous available power'}
        self.assertEqual(mod.validate_power_contract(x)['source'],'external_measurement')
        x['measurement_timestamp']='2026-10-23T20:00:00'
        with self.assertRaises(mod.PowerContractError):mod.validate_power_contract(x)
    def test_no_physical_dispatch(self):
        with self.assertRaises(mod.PowerContractError):mod.validate_power_contract({'mode':'operational','source':'user_entered_available_power','available_power_kw':200})
