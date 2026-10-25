import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'prescriptive_engine'))
from electrolyzer_contract import assess_electrolyzer_contract, ElectrolyzerContractError
from engine import run_electrolyzer_contract

class ElectrolyzerTests(unittest.TestCase):
    def valid(self):
        return {'mode':'observed_interval','measurement_start':'2026-10-01T09:00:00+03:00',
                'measurement_end':'2026-10-01T10:00:00+03:00', 'electrical_energy_kwh':100.0,
                'hydrogen_mass_kg':2.0,'electric_meter_reference':'meter-A','hydrogen_meter_reference':'meter-B',
                'data_source_reference':'record-1','electrical_boundary':'AC inlet','measurement_basis':'paired interval'}
    def test_missing_data_blocked(self):
        r=assess_electrolyzer_contract({'mode':'missing_measurements','reason':'No physical meters'})
        self.assertIsNone(r['sec_kwh_per_kg']); self.assertEqual(r['physical_dispatch'],'NOT_APPROVED')
    def test_observed_sec_only(self):
        r=assess_electrolyzer_contract(self.valid())
        self.assertEqual(r['sec_kwh_per_kg'],50)
        self.assertEqual(r['operating_recommendation'],'NOT_APPROVED')
    def test_reject_unpaired_and_unzoned(self):
        for change in ({'measurement_end':'2026-10-01T09:00:00+03:00'},
                       {'measurement_start':'2026-10-01T09:00:00'}, {'hydrogen_mass_kg':0},
                       {'electric_meter_reference':''}, {'electrical_energy_kwh':True}):
            d=self.valid();d.update(change)
            with self.subTest(change=change),self.assertRaises(ElectrolyzerContractError): assess_electrolyzer_contract(d)
    def test_no_estimated_numbers_in_missing_mode(self):
        with self.assertRaises(ElectrolyzerContractError):
            assess_electrolyzer_contract({'mode':'missing_measurements','reason':'missing','electrical_energy_kwh':100})
    def test_output_file(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp); file=p/'contract.json';file.write_text(json.dumps({'mode':'missing_measurements','reason':'missing'}))
            run_electrolyzer_contract(file,output_dir=p/'out')
            self.assertIsNone(json.loads((p/'out'/'electrolyzer_consumption_status.json').read_text())['sec_kwh_per_kg'])

if __name__=='__main__': unittest.main()
