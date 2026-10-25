"""Regression tests for Phase 6 fail-closed research evidence release."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from phase6_release import create_release, RELEASE_ACCEPT


class Phase6ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self._put('config.json', 'initial source config')
        self._put('tests/test_smoke.py', 'import unittest\nclass Smoke(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n')
        self._put('phase5_results/phase5_scientific_report_EN.md', '# Research only')
        self._put('phase5_results/phase5_scientific_report_EN.html', '<html lang="en">Research only</html>')
        self._put('phase3_results/phase3_research_passed.csv', 'candidate_id\n123\n')
        self._put('phase5_final_audit.py', '# source audit script')
        self._put('phase4_simulation_results/phase4_simulation_status.json', json.dumps({
            'phase': 4, 'status': 'RESEARCH_SCENARIO_COMPLETE_NOT_PHYSICAL_INTEGRATION'}))
        self.phase5 = {
            'phase': 5,
            'status': 'RESEARCH_SCENARIO_AUDIT_PASSED_WITH_DISCLOSED_LIMITATIONS',
            'scope_accepted': True, 'research_scope_only': True,
            'physical_safety_approved': False, 'all_four_models_physically_integrated': False,
            'candidate_rows': 1622, 'eligible_rows': 535, 'phase3_pareto_points': 7,
            'unit_tests': {'count': 139, 'status': 'PASS', 'returncode': 0},
            'checks': [{'check': 'test', 'result': 'PASS', 'detail': 'Unit tests pass'},
                       {'check': 'version', 'result': 'WARN', 'detail': 'Model versions not proven'}],
            'hypothetical_solar_kw': 100, 'hypothetical_wind_kw': 100,
            'hypothetical_supply_kw': 200, 'benchmark_kwh_per_kg': 55,
            'historical_fc_prediction': 'Pass', 'runtime': {'python': '3.14'},
        }
        self._put('phase5_results/phase5_audit_status.json', json.dumps(self.phase5))
        rows = [dict(relative_path='config.json', exists='True', bytes=len(b'initial source config'),
                     sha256=hashlib.sha256(b'initial source config').hexdigest()),
                dict(relative_path='models/not_present.pkl', exists='False', bytes='', sha256='')]
        self._write_manifest(rows)
        self.contract = {
            'schema_version': 1,
            'release_scope': 'RESEARCH_EVIDENCE_ONLY',
            'phase5_audit_status': 'phase5_results/phase5_audit_status.json',
            'phase5_sha256_manifest': 'phase5_results/phase5_sha256_manifest.csv',
            'phase4_simulation_status': 'phase4_simulation_results/phase4_simulation_status.json',
            'output_dir': 'phase6_results',
            'required_files': [
                'phase5_results/phase5_audit_status.json',
                'phase5_results/phase5_scientific_report_EN.md',
                'phase5_results/phase5_scientific_report_EN.html',
                'phase5_results/phase5_sha256_manifest.csv',
                'phase4_simulation_results/phase4_simulation_status.json',
                'phase3_results/phase3_research_passed.csv',
                'phase5_final_audit.py',
            ],
            'optional_files': ['models/not_present.pkl'],
        }
        self._write_contract()

    def _put(self, rel, value):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding='utf-8')

    def _write_manifest(self, rows):
        p = self.root / 'phase5_results/phase5_sha256_manifest.csv'
        p.parent.mkdir(exist_ok=True)
        with p.open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['relative_path','exists','bytes','sha256'])
            w.writeheader()
            w.writerows(rows)

    def _write_contract(self):
        self._put('phase6_release_contract.json', json.dumps(self.contract))

    def _run(self):
        return create_release(self.root / 'phase6_release_contract.json')

    def test_valid_package_and_embedded_verifier(self):
        original = (self.root / 'config.json').read_bytes()
        result = self._run()
        self.assertEqual(result['status'], RELEASE_ACCEPT)
        self.assertEqual(result['phase5_unit_tests_passed'], 139)
        self.assertEqual(result['eligible_rows'], 535)
        self.assertFalse(result['physical_safety_approved'])
        self.assertEqual((self.root/'config.json').read_bytes(), original)
        target = Path(result['archive'])
        with zipfile.ZipFile(target) as z:
            self.assertIn('RELEASE_README_EN.md', z.namelist())
            self.assertIn('VERIFY_RELEASE.py', z.namelist())
            self.assertNotIn('models/not_present.pkl', z.namelist())
            self.assertEqual(z.testzip(), None)
            with tempfile.TemporaryDirectory() as temp:
                z.extractall(temp)
                p = subprocess.run([sys.executable, str(Path(temp)/'VERIFY_RELEASE.py')],
                                   capture_output=True, text=True)
                self.assertEqual(p.returncode, 0, p.stdout+p.stderr)
                self.assertIn('RELEASE_INTEGRITY_VERIFIED', p.stdout)
                (Path(temp)/'phase5_results/phase5_scientific_report_EN.md').write_text('tampered', encoding='utf-8')
                p = subprocess.run([sys.executable, str(Path(temp)/'VERIFY_RELEASE.py')],
                                   capture_output=True, text=True)
                self.assertNotEqual(p.returncode, 0)

    def test_modified_source_blocks_release(self):
        self._put('config.json', 'change not audited')
        result = self._run()
        self.assertEqual(result['status'], 'RESEARCH_EVIDENCE_RELEASE_FAILED')
        self.assertIn('changed since Phase 5', result['error'])

    def test_rejected_phase5_status_blocks_release(self):
        self.phase5['status'] = 'RESEARCH_SCENARIO_AUDIT_FAILED'
        self._put('phase5_results/phase5_audit_status.json', json.dumps(self.phase5))
        result = self._run()
        self.assertEqual(result['status'], 'RESEARCH_EVIDENCE_RELEASE_FAILED')

    def test_failed_phase5_unit_tests_block_release(self):
        self.phase5['unit_tests']['status'] = 'FAIL'
        self._put('phase5_results/phase5_audit_status.json', json.dumps(self.phase5))
        self.assertIn('tests', self._run()['error'])

    def test_unsupported_physical_approval_blocks_release(self):
        self.phase5['physical_safety_approved'] = True
        self._put('phase5_results/phase5_audit_status.json', json.dumps(self.phase5))
        self.assertIn('physical-system', self._run()['error'])

    def test_path_traversal_blocks_release(self):
        self.contract['required_files'][0] = '../secrets.txt'
        self._write_contract()
        self.assertIn('Unsafe', self._run()['error'])

    def test_duplicate_member_names_block_release(self):
        self.contract['required_files'].append('phase5_final_audit.py')
        self._write_contract()
        self.assertIn('duplicate', self._run()['error'].lower())

    def test_physical_scenario_status_is_not_research_acceptance(self):
        self._put('phase4_simulation_results/phase4_simulation_status.json',
                  json.dumps({'status': 'BLOCKED_INCOMPLETE_CROSS_MODEL_EVIDENCE'}))
        self.assertIn('Phase 4 simulation', self._run()['error'])

    def test_no_arabic_documentation_members(self):
        result = self._run()
        with zipfile.ZipFile(result['archive']) as z:
            for name in z.namelist():
                self.assertNotIn('_AR.', name)
            html = z.read('phase5_results/phase5_scientific_report_EN.html').decode('utf-8')
            self.assertIn('lang="en"', html)


if __name__ == '__main__':
    unittest.main()
