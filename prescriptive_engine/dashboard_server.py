"""Local-only, research-only dashboard bridge for the existing seven-stage pipeline.

The historical Pareto dashboard stays unchanged; its HTML is augmented in memory.
No external dependencies are added by the server. The Python inference process
continues to use the project's pinned environment and the original pipeline.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import subprocess
import sys
import threading
import uuid
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from prescriptive_engine.pipeline.contracts import DECISION_VARS, validate as validate_scenario

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_NAMES = (
    'RESEARCH_DEMO_200KW.json',
    'RESEARCH_FOUR_MODELS_HYPOTHETICAL.json',
    'RESEARCH_FC_DIRECT_FLOW_SUPPORT_CHECK.json',
    'STRICT_NO_FC_BRIDGE.json',
)
POWER_MODES = ('fixed_user', 'model_scenario')
FC_MODES = ('BLOCKED_NO_MAPPING', 'ASSUMED_REGULATED_DRAW', 'ASSUMED_CANDIDATE_FLOW')
ARTIFACTS = frozenset(('pipeline_status.json', 'scenario.json', 'baseline.csv', 'best_conditional_candidate.csv',
                      'conditionally_feasible.csv', 'all_candidates.csv'))
MAX_CANDIDATES = 10_000
MAX_REQUEST_BYTES = 48_000
RUN_GUARD = threading.Lock()
RUN_ID_PATTERN = re.compile(r'^dashboard_[0-9]{8}T[0-9]{6}_[a-f0-9]{12}$')


def finite_number(value, label, *, min_value=None, max_value=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite JSON number.')
    value = float(value)
    if min_value is not None and value < min_value:
        raise ValueError(f'{label} must be >= {min_value}.')
    if max_value is not None and value > max_value:
        raise ValueError(f'{label} must be <= {max_value}.')
    return value


def load_template(root, name):
    if name not in TEMPLATE_NAMES:
        raise ValueError('Unknown scenario template.')
    path = root / 'configs' / 'scenarios' / name
    return json.loads(path.read_text(encoding='utf-8'))


def _numerical_record(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f'{label} requires exactly: {", ".join(expected)}')
    return {key: finite_number(value[key], f'{label}.{key}') for key in expected}


def make_scenario(root, request):
    """Assemble a reviewed scenario from an allowlisted template and editable fields."""
    if not isinstance(request, dict):
        raise ValueError('Expected a JSON object.')
    allowed = {'template', 'acknowledge_research', 'power_mode', 'fixed_power_kw', 'solar_inputs', 'wind_inputs',
               'acknowledge_hypothetical_power', 'baseline', 'bounds', 'candidate_count', 'seed',
               'fc_mode', 'acknowledge_assumed_fc', 'fc_flow_demand_l_min', 'fc_operating_point'}
    if set(request) - allowed:
        raise ValueError('Unexpected request fields.')
    if request.get('acknowledge_research') is not True:
        raise ValueError('Explicit research-only acknowledgment required.')
    original = load_template(root, request.get('template'))
    scenario = copy.deepcopy(original)
    scenario['description'] = 'Dashboard-run research simulation; inputs may be modified by the user. Not engineering validation.'
    scenario['assumptions_acknowledged'] = True

    mode = request.get('power_mode')
    if mode not in POWER_MODES:
        raise ValueError('Invalid renewable power mode.')
    if mode == 'fixed_user':
        fixed = finite_number(request.get('fixed_power_kw'), 'fixed_power_kw', min_value=0.001, max_value=1_000_000)
        scenario['renewable_power'] = {'mode': 'fixed_user', 'available_kw': fixed}
    else:
        if request.get('acknowledge_hypothetical_power') is not True:
            raise ValueError('Hypothetical Solar+Wind combination must be explicitly acknowledged.')
        reference = load_template(root, 'RESEARCH_FOUR_MODELS_HYPOTHETICAL.json')['renewable_power']
        features = json.loads((root/'config.json').read_text(encoding='utf-8'))
        solar = _numerical_record(request.get('solar_inputs'), features['solar_features'], 'solar_inputs')
        wind = _numerical_record(request.get('wind_inputs'), features['wind_features'], 'wind_inputs')
        scenario['renewable_power'] = {
            'mode': 'model_scenario', 'same_hypothetical_scenario_acknowledged': True,
            'aggregation_scope': reference['aggregation_scope'],
            'solar_inputs': solar, 'wind_inputs': wind,
            'inputs_provenance': 'User-entered hypothetical research inputs; no verified historical calendar alignment.'
        }

    scenario['baseline'] = _numerical_record(request.get('baseline'), DECISION_VARS, 'baseline')
    scenario['baseline_provenance'] = 'User-submitted dashboard scenario values; not automatically verified measurements.'
    raw_bounds = request.get('bounds')
    if not isinstance(raw_bounds, dict) or set(raw_bounds) != set(DECISION_VARS):
        raise ValueError('bounds must contain exactly the six decision variables.')
    scenario['bounds'] = {}
    for key in DECISION_VARS:
        pair = raw_bounds[key]
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f'bounds.{key} must be [minimum, maximum].')
        scenario['bounds'][key] = [finite_number(pair[0], f'bounds.{key}[0]'),
                                   finite_number(pair[1], f'bounds.{key}[1]')]
    scenario['bounds_basis'] = 'User-submitted empirical research ranges, not manufacturer-approved safety limits.'
    count = request.get('candidate_count')
    if type(count) is not int or not 20 <= count <= MAX_CANDIDATES:
        raise ValueError(f'candidate_count must be an integer from 20 to {MAX_CANDIDATES}.')
    seed = request.get('seed')
    if type(seed) is not int or abs(seed) > 2**31 - 1:
        raise ValueError('seed must be a 32-bit signed integer.')
    scenario['optimization'] = {'candidate_count': count, 'seed': seed}

    fc_mode = request.get('fc_mode')
    if fc_mode not in FC_MODES:
        raise ValueError('Invalid Fuel Cell bridge mode.')
    if fc_mode == 'BLOCKED_NO_MAPPING':
        scenario['fuel_cell'] = {'bridge_mode': fc_mode}
    else:
        if request.get('acknowledge_assumed_fc') is not True:
            raise ValueError('Fuel Cell assumed bridge must be explicitly acknowledged.')
        reference_fc = load_template(root, 'RESEARCH_DEMO_200KW.json')['fuel_cell']
        fc = copy.deepcopy(reference_fc)
        fc['bridge_mode'] = fc_mode
        fc['acknowledge_unverified_device_chain'] = True
        fc['operating_point'] = _numerical_record(request.get('fc_operating_point'),
                               ('voltage_v', 'temperature_c', 'pressure_bar'), 'fc_operating_point')
        fc['operating_point_provenance'] = 'Dashboard user input; research hypothetical device setpoint.'
        if fc_mode == 'ASSUMED_REGULATED_DRAW':
            fc['flow_demand_l_min'] = finite_number(request.get('fc_flow_demand_l_min'),
                                         'fc_flow_demand_l_min', min_value=0.001, max_value=1_000_000)
            fc['flow_demand_provenance'] = 'Dashboard user assumption, not measured Fuel Cell draw.'
        else:
            fc.pop('flow_demand_l_min', None)
            fc['flow_demand_provenance'] = 'Candidate output mapped under research-only ideal gas reference assumptions.'
        scenario['fuel_cell'] = fc
    validate_scenario(scenario)
    return scenario


def _read_csv_first(path, limit=1):
    if not path.is_file():
        return []
    with path.open(encoding='utf-8-sig', newline='') as f:
        return [row for _, row in zip(range(limit), csv.DictReader(f))]


def read_run(root, run_id):
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError('Invalid run identifier.')
    folder = root / 'outputs' / 'dashboard_runs' / run_id
    status_path = folder / 'pipeline_status.json'
    if not status_path.is_file():
        raise FileNotFoundError('Run was not found.')
    status = json.loads(status_path.read_text(encoding='utf-8'))
    baseline = _read_csv_first(folder/'baseline.csv')
    best = _read_csv_first(folder/'best_conditional_candidate.csv')
    top = _read_csv_first(folder/'conditionally_feasible.csv', limit=12)
    return {'run_id': run_id, 'status': status, 'baseline': baseline[0] if baseline else None,
            'best': best[0] if best else None, 'top_candidates': top,
            'downloadable': sorted(name for name in ARTIFACTS if (folder/name).is_file())}


def dashboard_html(root):
    """Preserve all original offline history while injecting the live workspace."""
    path = root / 'dashboard' / 'research_dashboard.html'
    original = path.read_text(encoding='utf-8')
    panel = (root / 'dashboard' / 'scenario_workspace.html').read_text(encoding='utf-8')
    before = '<div class="cards">'
    if before not in original:
        raise RuntimeError('Existing dashboard layout marker was not found.')
    if 'id="scenario-workspace"' in original:
        raise RuntimeError('Scenario workspace was already embedded in the source dashboard.')
    original = original.replace(before, panel + '\n' + before, 1)
    original = original.replace('</head>', '<link rel="stylesheet" href="/dashboard/scenario_workspace.css"></head>', 1)
    return original.replace('</body>', '<script src="/dashboard/scenario_workspace.js" defer></script></body>', 1)


def _make_handler(root, bind_host, bind_port):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'GreenHydrogenLocalDashboard/1.0'

        def _headers(self, code, content_type, length):
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(length))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'DENY')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.end_headers()

        def send_bytes(self, data, content_type='application/json; charset=utf-8', code=200):
            self._headers(code, content_type, len(data))
            self.wfile.write(data)

        def send_json(self, value, code=200):
            self.send_bytes(json.dumps(value, ensure_ascii=True, allow_nan=False).encode('utf-8'), code=code)

        def send_error_json(self, message, code=400):
            self.send_json({'error': str(message)}, code=code)

        def _check_host(self):
            # Reject Host-header DNS rebinding: the server must remain loopback-only.
            valid = {f'127.0.0.1:{bind_port}', f'localhost:{bind_port}'}
            return self.headers.get('Host', '') in valid

        def do_GET(self):
            if not self._check_host():
                return self.send_error_json('Invalid local host header.', HTTPStatus.FORBIDDEN)
            url = urlsplit(self.path)
            try:
                if url.path in ('/', '/dashboard/research_dashboard.html'):
                    return self.send_bytes(dashboard_html(root).encode('utf-8'), 'text/html; charset=utf-8')
                if url.path in ('/dashboard/scenario_workspace.css', '/dashboard/scenario_workspace.js'):
                    name = url.path.rsplit('/', 1)[-1]
                    data = (root/'dashboard'/name).read_bytes()
                    mime = 'text/css; charset=utf-8' if name.endswith('.css') else 'text/javascript; charset=utf-8'
                    return self.send_bytes(data, mime)
                if url.path == '/api/templates':
                    return self.send_json({'templates': list(TEMPLATE_NAMES), 'research_only': True})
                if url.path == '/api/template':
                    name = parse_qs(url.query).get('name', [''])[0]
                    return self.send_json({'name': name, 'scenario': load_template(root, name)})
                if url.path == '/api/runs':
                    folder = root/'outputs'/'dashboard_runs'
                    ids = sorted((p.name for p in folder.iterdir() if p.is_dir() and RUN_ID_PATTERN.fullmatch(p.name)
                                  and (p/'pipeline_status.json').is_file()), reverse=True)[:30] if folder.exists() else []
                    return self.send_json({'runs': [dict(run_id=i, status=json.loads((folder/i/'pipeline_status.json').read_text(encoding='utf-8')).get('status')) for i in ids]})
                if url.path.startswith('/api/run/'):
                    return self.send_json(read_run(root, url.path.removeprefix('/api/run/')))
                if url.path.startswith('/api/download/'):
                    parts = url.path.split('/')
                    if len(parts) != 5 or parts[3] not in ('',) and not RUN_ID_PATTERN.fullmatch(parts[3]):
                        return self.send_error_json('Invalid download path.', 404)
                    run_id, name = parts[3], parts[4]
                    if not RUN_ID_PATTERN.fullmatch(run_id) or name not in ARTIFACTS:
                        return self.send_error_json('Download is not allowlisted.', 404)
                    # Require recorded run; do not allow arbitrary project file browsing.
                    read_run(root, run_id)
                    data = (root/'outputs'/'dashboard_runs'/run_id/name).read_bytes()
                    mime = 'application/json' if name.endswith('.json') else 'text/csv'
                    self.send_response(200)
                    self.send_header('Content-Type', mime)
                    self.send_header('Content-Length', str(len(data)))
                    self.send_header('Content-Disposition', f'attachment; filename="{name}"')
                    self.send_header('X-Content-Type-Options', 'nosniff')
                    self.end_headers()
                    return self.wfile.write(data)
                return self.send_error_json('Not found.', 404)
            except (ValueError, FileNotFoundError) as exc:
                return self.send_error_json(str(exc), 404)
            except Exception:
                return self.send_error_json('Dashboard content unavailable. Check the local project files.', 500)

        def do_POST(self):
            if not self._check_host():
                return self.send_error_json('Invalid local host header.', HTTPStatus.FORBIDDEN)
            if urlsplit(self.path).path != '/api/run':
                return self.send_error_json('Not found.', 404)
            origin = self.headers.get('Origin')
            if origin and origin not in {f'http://127.0.0.1:{bind_port}', f'http://localhost:{bind_port}'}:
                return self.send_error_json('Cross-origin requests are not allowed.', 403)
            if self.headers.get('Content-Type', '').split(';')[0].strip().lower() != 'application/json':
                return self.send_error_json('JSON content type required.', 415)
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    return self.send_error_json('Invalid or oversized request.', 413)
                request = json.loads(self.rfile.read(length).decode('utf-8'))
                spec = make_scenario(root, request)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                return self.send_error_json(str(exc), 400)
            if not RUN_GUARD.acquire(blocking=False):
                return self.send_error_json('Another scenario is already running.', 409)
            try:
                run_id = 'dashboard_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:12]
                folder = root/'outputs'/'dashboard_runs'/run_id
                folder.mkdir(parents=True, exist_ok=False)
                scenario_path = folder/'scenario.json'
                scenario_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
                cmd = [sys.executable, str(root/'run_pipeline.py'), '--scenario', str(scenario_path),
                       '--config', str(root/'config.json'), '--out', str(folder)]
                try:
                    result = subprocess.run(cmd, cwd=root, capture_output=True, text=True,
                                            timeout=180, check=False)
                except subprocess.TimeoutExpired:
                    return self.send_error_json(f'Run {run_id} exceeded 180 seconds. The scenario input is saved; inspect its output.', 504)
                if not (folder/'pipeline_status.json').is_file():
                    return self.send_error_json('Pipeline did not produce status output. ' + result.stderr[-1200:], 500)
                summary = read_run(root, run_id)
                summary['runner_exit_code'] = result.returncode
                return self.send_json(summary)
            finally:
                RUN_GUARD.release()

        def log_message(self, format, *args):
            sys.stderr.write('[DASHBOARD] ' + format % args + '\n')
    return Handler


def create_server(root=ROOT, port=8765):
    root = Path(root).resolve()
    if not (root/'run_pipeline.py').is_file() or not (root/'dashboard/research_dashboard.html').is_file():
        raise FileNotFoundError('Existing project root and historical dashboard are required.')
    return ThreadingHTTPServer(('127.0.0.1', port), _make_handler(root, '127.0.0.1', port))


def main(argv=None):
    ap = argparse.ArgumentParser(description='Local-only research scenario dashboard; never use for equipment control.')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--no-browser', action='store_true')
    args = ap.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        ap.error('Port must be from 1024 to 65535.')
    server = create_server(port=args.port)
    url = f'http://127.0.0.1:{args.port}/'
    print(f'[DASHBOARD] Local research workspace: {url}', flush=True)
    print('[DASHBOARD] Local computer only. Not a validated physical control application.', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print('\n[DASHBOARD] Stopped.')
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
