"""Test the local dashboard bridge without approving any engineering claims."""
from __future__ import annotations
import json
import socket
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from pathlib import Path

from prescriptive_engine import dashboard_server as ds


class DashboardContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.demo = ds.load_template(cls.root, 'RESEARCH_DEMO_200KW.json')
        cls.four = ds.load_template(cls.root, 'RESEARCH_FOUR_MODELS_HYPOTHETICAL.json')

    def request(self, template='RESEARCH_DEMO_200KW.json', *, mode='fixed_user', fc_mode='ASSUMED_REGULATED_DRAW'):
        spec = ds.load_template(self.root, template)
        return {
            'template': template, 'acknowledge_research': True,
            'power_mode': mode, 'fixed_power_kw': 200.0,
            'solar_inputs': self.four['renewable_power']['solar_inputs'],
            'wind_inputs': self.four['renewable_power']['wind_inputs'],
            'acknowledge_hypothetical_power': mode == 'model_scenario',
            'baseline': spec['baseline'], 'bounds': spec['bounds'],
            'candidate_count': 40, 'seed': 42,
            'fc_mode': fc_mode, 'acknowledge_assumed_fc': fc_mode != 'BLOCKED_NO_MAPPING',
            'fc_flow_demand_l_min': self.demo['fuel_cell']['flow_demand_l_min'],
            'fc_operating_point': self.demo['fuel_cell']['operating_point']
        }

    def test_fixed_power_and_six_variables_are_preserved(self):
        spec = ds.make_scenario(self.root, self.request())
        self.assertEqual(spec['renewable_power'], {'mode':'fixed_user','available_kw':200.0})
        self.assertEqual(set(spec['baseline']), set(ds.DECISION_VARS))
        self.assertEqual(spec['optimization']['candidate_count'], 40)
        self.assertIn('not manufacturer-approved', spec['bounds_basis'])

    def test_model_power_mode_requires_explicit_assumption(self):
        r = self.request(mode='model_scenario')
        r['acknowledge_hypothetical_power'] = False
        with self.assertRaisesRegex(ValueError, 'acknowledged'):
            ds.make_scenario(self.root, r)
        r['acknowledge_hypothetical_power'] = True
        spec = ds.make_scenario(self.root, r)
        self.assertEqual(spec['renewable_power']['mode'],'model_scenario')
        self.assertEqual(len(spec['renewable_power']['solar_inputs']),8)
        self.assertEqual(len(spec['renewable_power']['wind_inputs']),12)

    def test_strict_bridge_is_really_blocked(self):
        spec = ds.make_scenario(self.root, self.request(fc_mode='BLOCKED_NO_MAPPING'))
        self.assertEqual(spec['fuel_cell'],{'bridge_mode':'BLOCKED_NO_MAPPING'})

    def test_research_acknowledgement_required(self):
        r = self.request();r['acknowledge_research'] = False
        with self.assertRaises(ValueError):ds.make_scenario(self.root,r)

    def test_fc_bridge_acknowledgement_required(self):
        r = self.request();r['acknowledge_assumed_fc'] = False
        with self.assertRaises(ValueError):ds.make_scenario(self.root,r)

    def test_boolean_numbers_and_invalid_ranges_rejected(self):
        r = self.request();r['baseline']['voltage_v']=True
        with self.assertRaises(ValueError):ds.make_scenario(self.root,r)
        r=self.request();r['bounds']['voltage_v']=[2,1]
        with self.assertRaises(ValueError):ds.make_scenario(self.root,r)

    def test_candidate_limit_rejected(self):
        r = self.request();r['candidate_count']=100001
        with self.assertRaises(ValueError):ds.make_scenario(self.root,r)

    def test_template_path_traversal_rejected(self):
        with self.assertRaises(ValueError):ds.load_template(self.root,'../../secrets.json')

    def test_dashboard_preserves_existing_graph(self):
        content = ds.dashboard_html(self.root)
        self.assertIn('id="scenario-workspace"',content)
        self.assertIn('id="chart"',content)
        self.assertIn('id="embedded"',content)
        self.assertEqual(content.count('id="scenario-workspace"'),1)

    def test_saved_run_identifier_rejected(self):
        with self.assertRaises(ValueError):ds.read_run(self.root, '../../test')


class DashboardHttpTests(DashboardContractTests):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
        cls.server = ds.create_server(root=cls.root, port=port)
        cls.port=port
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join(timeout=5)

    def get(self, path, host=None):
        req=Request(f'http://127.0.0.1:{self.port}{path}', headers={'Host': host} if host else {})
        with urlopen(req,timeout=30) as r:return r.status,r.read()

    def post(self, path, doc, extra_headers=None):
        headers={'Content-Type':'application/json', 'Origin':f'http://127.0.0.1:{self.port}'}
        headers.update(extra_headers or {})
        req=Request(f'http://127.0.0.1:{self.port}{path}',data=json.dumps(doc).encode(),headers=headers)
        with urlopen(req,timeout=120) as r:return r.status,json.load(r)

    def test_dashboard_serves_pane_with_assets(self):
        status, page = self.get('/')
        self.assertEqual(status,200)
        self.assertIn(b'id="scenario-workspace"',page)
        self.assertIn(b'/dashboard/scenario_workspace.js',page)
        _,js = self.get('/dashboard/scenario_workspace.js')
        self.assertIn(b'RESEARCH ONLY',page)
        self.assertIn(b'buildRequest',js)

    def test_template_api_and_forbidden_file(self):
        _, data = self.get('/api/templates')
        self.assertEqual(len(json.loads(data)['templates']),4)
        with self.assertRaises(HTTPError) as ctx:self.get('/models/best_Solar_Power_model.pkl')
        self.assertEqual(ctx.exception.code,404)

    def test_origin_check(self):
        with self.assertRaises(HTTPError) as ctx:
            self.post('/api/run',self.request(),{'Origin':'https://untrusted.example'})
        self.assertEqual(ctx.exception.code,403)

    def test_no_safety_ack_rejected_over_http(self):
        r=self.request();r['acknowledge_research']=False
        with self.assertRaises(HTTPError) as ctx:self.post('/api/run',r)
        self.assertEqual(ctx.exception.code,400)

    def test_pipeline_run_and_retrieve_saved_results(self):
        # Runs the real four artifacts and the original optimization pipeline.
        status, doc = self.post('/api/run',self.request())
        self.assertEqual(status,200)
        self.assertEqual(doc['status']['scientific_status'],'RESEARCH_ONLY_NOT_VALIDATED_ENGINEERING')
        self.assertFalse(doc['status']['physical_safety_approved'])
        self.assertEqual(doc['status']['candidate_count'],41)
        self.assertEqual(doc['status']['fixed_available_power_kw'],200)
        self.assertTrue(doc['baseline'])
        self.assertIn('scenario.json',doc['downloadable'])
        _, data=self.get('/api/run/'+doc['run_id'])
        self.assertEqual(json.loads(data)['run_id'],doc['run_id'])
        _, data=self.get('/api/download/'+doc['run_id']+'/pipeline_status.json')
        self.assertEqual(json.loads(data)['scientific_status'],'RESEARCH_ONLY_NOT_VALIDATED_ENGINEERING')


if __name__=='__main__':unittest.main()
