"""Phase 5 tests use miniature independent fixtures; no model deserialization."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from phase5_final_audit import audit, AuditError, _safe, _sklearn_embedded_version


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding='utf-8')


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Phase5AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        contract = {
            'schema_version': 1, 'purpose': 'FINAL_RESEARCH_AUDIT_NOT_HARDWARE_CERTIFICATION',
            'phase3_passed': 'phase3_results/phase3_research_passed.csv',
            'phase4_scenario': 'phase4_research_scenario.json',
            'phase4_simulation_status': 'phase4_simulation_results/phase4_simulation_status.json',
            'phase4_simulated_candidates': 'phase4_simulation_results/phase4_simulated_candidates.csv',
            'phase4_simulated_eligible': 'phase4_simulation_results/phase4_simulated_eligible.csv',
            'phase4_fc_historical_reference': 'phase4_simulation_results/phase4_fc_historical_reference.csv',
            'engine_config':'config.json', 'engine_source':'prescriptive_engine/engine.py',
            'output_dir': 'phase5_results', 'audit_tolerance': 1e-7, 'run_project_tests': True,
        }
        self.contract = self.root / 'phase5_audit_contract.json'
        save_json(self.contract, contract)
        self.scenario = {
            'schema_version': 1, 'mode': 'explicit_hypothetical_independent_research_scenario',
            'assumptions_acknowledged': True, 'available_power_kw': 200., 'solar_assumed_kw': 100.,
            'wind_assumed_kw': 100., 'specific_energy_kwh_per_kg_h2': 55.,
            'specific_energy_reference': 'https://www.energy.gov/cmei/fuels/technical-targets-liquid-alkaline-electrolysis',
            'reference_gas_temperature_k': 273.15, 'reference_gas_pressure_pa': 101325.,
            'hydrogen_molar_mass_kg_per_mol': 0.00201588,
        }
        save_json(self.root / 'phase4_research_scenario.json', self.scenario)
        save_json(self.root / 'config.json', {'models_dir': 'models'})
        self.p3 = [
            {'candidate_id': '10', 'renewable_power_kw':'100', 'predicted_hydrogen_production_kg_h': '2', 'passes_research_screening':'True', 'predicted_kg_per_available_kwh_proxy':'0.02', 'pareto_nondominated_after_screening':'True'},
            {'candidate_id': '20', 'renewable_power_kw':'210', 'predicted_hydrogen_production_kg_h': '1', 'passes_research_screening':'True', 'predicted_kg_per_available_kwh_proxy':'0.00476', 'pareto_nondominated_after_screening':'False'},
        ]
        save_csv(self.root / 'phase3_results/phase3_research_passed.csv', self.p3)
        fac = 8.31446261815324 * 273.15/101325 *1000/(60*0.00201588)
        def row(i, p, h, pass_flag):
            return {
                'candidate_id': i, 'renewable_power_kw': p, 'predicted_hydrogen_production_kg_h': h,
                'assumed_total_supply_kw': 200., 'power_proxy_within_assumed_supply': pass_flag,
                'literature_benchmark_required_kw': h*55., 'literature_benchmark_within_assumed_supply': True,
                'research_scenario_candidate_supported': pass_flag,
                'ideal_gas_h2_capacity_l_min_reference_conditions': h*fac,
                'illustrative_fc_reference_flow_l_min': 15.,
                'illustrative_fc_flow_capacity_sufficient':True,
                'simulated_combined_screen_pass':pass_flag,
                'simulated_fc_usable_flow_not_measured':True,
                'physical_dispatch_approved':False, 'same_calendar_verified':False, 'same_device_chain_verified':False,
            }
        self.rows = [row('10',100.,2.,True),row('20',210.,1.,False)]
        self.all_csv = self.root / 'phase4_simulation_results/phase4_simulated_candidates.csv'
        save_csv(self.all_csv, self.rows)
        save_csv(self.root / 'phase4_simulation_results/phase4_simulated_eligible.csv', self.rows[:1])
        save_csv(self.root / 'phase4_simulation_results/phase4_fc_historical_reference.csv', [
            {'hydrogen_flow_l_min':15., 'predicted_fc_label_at_unchanged_historical_point':'Pass',
             'fc_candidate_specific_inference':False}])
        save_json(self.root / 'phase4_simulation_results/phase4_simulation_status.json', {
            'status': 'RESEARCH_SCENARIO_COMPLETE_NOT_PHYSICAL_INTEGRATION',
            'research_simulation_scope_closed': True,
            'counts': {'all_phase3_screened':2,'proxy_within_assumed_supply':1,
                       'benchmark_within_assumed_supply':2, 'combined_research_screen_pass':1},
            'scenario': {'assumed_total_kw':200.,'assumed_solar_kw':100.,'assumed_wind_kw':100.,
                         'electrolyzer_benchmark_kwh_per_kg':55.,
                         'renewables_are_model_forecasts':False,'actual_solar_wind_time_alignment':False},
            'selected_research_candidate': {'candidate_id':'10','predicted_hydrogen_production_kg_h':2., 'literature_benchmark_required_kw':110., 'ideal_gas_h2_capacity_l_min':2*fac},
            'fuel_cell': {'candidate_specific_fc_prediction':False, 'saved_model_prediction_at_historic_reference':'Pass', 'historic_reference_observed_flow_l_min':15.},
            'integration_limits': {k:False for k in (
                'physical_safety_approved','hydrogen_to_fc_flow_measured','same_device_chain_verified',
                'solar_wind_hydrogen_calendar_aligned','electrolyzer_consumed_power_verified',
                'complete_four_model_physical_integration')}
        })

    def call_audit(self):
        with patch('phase5_final_audit._run_tests', return_value={'status':'PASS','count':126,'returncode':0}):
            return audit(self.contract)

    def test_valid_complete_independent_audit(self):
        res = self.call_audit()
        self.assertEqual(res['status'],'RESEARCH_SCENARIO_AUDIT_PASSED_WITH_DISCLOSED_LIMITATIONS')
        self.assertEqual(res['eligible_rows'],1)
        self.assertFalse(res['physical_safety_approved'])
        self.assertTrue((self.root/'phase5_results/phase5_scientific_report_EN.md').exists())
        self.assertTrue((self.root/'phase5_results/phase5_scientific_report_EN.html').exists())
        self.assertTrue((self.root/'phase5_results/phase5_sha256_manifest.csv').exists())

    def test_report_language_is_english_only(self):
        self.call_audit()
        for ext in ("md", "html"):
            report = self.root / "phase5_results" / f"phase5_scientific_report_EN.{ext}"
            content = report.read_text(encoding="utf-8")
            self.assertFalse(any("\u0600" <= char <= "\u06ff" for char in content))
            self.assertIn("Phase 5", content)
        html_content = (self.root / "phase5_results/phase5_scientific_report_EN.html").read_text(encoding="utf-8")
        self.assertIn('lang="en" dir="ltr"', html_content)

    def test_failure_reports_are_english_only(self):
        self.rows[0]['literature_benchmark_required_kw'] = 42.0
        save_csv(self.all_csv, self.rows)
        result = self.call_audit()
        self.assertEqual(result['status'], 'RESEARCH_SCENARIO_AUDIT_FAILED')
        for ext in ('html', 'md'):
            text = (self.root / 'phase5_results' / f'phase5_scientific_report_EN.{ext}').read_text(encoding='utf-8')
            self.assertFalse(any('\u0600' <= char <= '\u06ff' for char in text))
            self.assertIn('Phase 5', text)

    def test_tampered_electric_benchmark_is_detected(self):
        self.rows[0]['literature_benchmark_required_kw'] = 42.0
        save_csv(self.all_csv, self.rows)
        res = self.call_audit()
        self.assertEqual(res['status'], 'RESEARCH_SCENARIO_AUDIT_FAILED')
        self.assertFalse(res['scope_accepted'])

    def test_fake_candidate_fc_flow_is_rejected(self):
        self.rows[0]['same_device_chain_verified'] = True
        save_csv(self.all_csv, self.rows)
        res = self.call_audit()
        self.assertEqual(res['status'],'RESEARCH_SCENARIO_AUDIT_FAILED')

    def test_scenario_change_is_detected(self):
        self.scenario['specific_energy_kwh_per_kg_h2']=30.
        save_json(self.root / 'phase4_research_scenario.json', self.scenario)
        res=self.call_audit()
        self.assertEqual(res['status'],'RESEARCH_SCENARIO_AUDIT_FAILED')

    def test_tampered_eligible_details_are_detected(self):
        changed = [dict(self.rows[0])]
        changed[0]['renewable_power_kw'] = 99.0
        save_csv(self.root/'phase4_simulation_results/phase4_simulated_eligible.csv', changed)
        self.assertEqual(self.call_audit()['status'],'RESEARCH_SCENARIO_AUDIT_FAILED')

    def test_tampered_pareto_frontier_is_detected(self):
        self.p3[0]['pareto_nondominated_after_screening'] = 'False'
        save_csv(self.root/'phase3_results/phase3_research_passed.csv',self.p3)
        self.assertEqual(self.call_audit()['status'],'RESEARCH_SCENARIO_AUDIT_FAILED')

    def test_fake_fc_summary_is_detected(self):
        path = self.root / 'phase4_simulation_results/phase4_simulation_status.json'
        doc = json.loads(path.read_text(encoding='utf-8'))
        doc['fuel_cell']['candidate_specific_fc_prediction'] = True
        save_json(path, doc)
        self.assertEqual(self.call_audit()['status'],'RESEARCH_SCENARIO_AUDIT_FAILED')

    def test_source_mismatch_is_detected(self):
        self.p3[0]['renewable_power_kw']='130'
        save_csv(self.root/'phase3_results/phase3_research_passed.csv',self.p3)
        self.assertEqual(self.call_audit()['status'],'RESEARCH_SCENARIO_AUDIT_FAILED')

    def test_no_tests_cannot_qualify_as_final_acceptance(self):
        res = audit(self.contract, run_tests_override=False)
        self.assertEqual(res['status'],'RESEARCH_SCENARIO_AUDIT_FAILED')

    def test_path_escape_is_rejected(self):
        with self.assertRaises(AuditError):
            _safe(self.root, '../private.key')

    def test_pickle_metadata_is_scanned_without_deserialization(self):
        artifact = self.root/'dummy.pkl'
        artifact.write_bytes(b'NOT_A_PICKLE _sklearn_version xyz 1.8.0 blah')
        self.assertEqual(_sklearn_embedded_version(artifact),'1.8.0')


if __name__ == '__main__':
    unittest.main()
