"""Phase 6: fail-closed research evidence release (not device certification).

Uses the existing, independently audited Phase 5 evidence. Does not unpickle
models, execute engines, invent four-model coupling, or modify project inputs.
All release names, statuses, generated reports and documentation are English.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import subprocess
import tempfile
from datetime import datetime, timezone
import uuid
import zipfile

FIX_VERSION = '2026-10-24-PHASE6-V3'

AUDIT_ACCEPT = 'RESEARCH_SCENARIO_AUDIT_PASSED_WITH_DISCLOSED_LIMITATIONS'
SIM_ACCEPT = 'RESEARCH_SCENARIO_COMPLETE_NOT_PHYSICAL_INTEGRATION'
RELEASE_ACCEPT = 'RESEARCH_EVIDENCE_RELEASE_COMPLETE_WITH_DISCLOSED_LIMITATIONS'
MANIFEST_NAME = 'RELEASE_MANIFEST_SHA256.csv'
MAX_SOURCE_BYTES = 150 * 1024 * 1024

class ReleaseError(RuntimeError):
    pass


def _json(path: Path):
    try:
        result = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as exc:
        raise ReleaseError(f'Cannot read {path}: {exc}') from exc
    if not isinstance(result, dict):
        raise ReleaseError(f'Expected JSON object: {path}')
    return result


def _canonical_relative_path(raw: str, origin: str = 'project path') -> str:
    """Accept Windows or POSIX project-relative paths and emit safe POSIX paths.

    Phase 5 generated Windows backslashes in its inventory. Some user-edited
    release contracts also contain them. Reject drive/UNC roots, traversal,
    invalid segments and ambiguous separators, then use one canonical form.
    """
    if not isinstance(raw, str) or not raw or '\x00' in raw:
        raise ReleaseError(f'Invalid {origin}: {raw!r}')
    if raw.startswith(('\\', '/')) or ':' in raw:
        raise ReleaseError(f'Unsafe {origin} (absolute, drive, or URI): {raw!r}')
    normalized = raw.replace('\\', '/')
    if any(segment in ('', '.', '..') for segment in normalized.split('/')):
        raise ReleaseError(f'Unsafe {origin} (empty/dot/traversal segment): {raw!r}')
    # A valid release path is a project-relative string with path separators,
    # never a platform-dependent absolute path or a drive-specific URI.
    return normalized


def _safe(root: Path, raw: str, *, required: bool = True, origin: str = 'project file') -> Path:
    relative = _canonical_relative_path(raw, origin)
    resolved = (root / Path(relative)).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ReleaseError(f'Unsafe {origin} (resolves outside project): {raw!r}')
    if required and not resolved.is_file():
        raise ReleaseError(f'Required file missing for {origin}: {relative}')
    if resolved.is_symlink() or (root / Path(relative)).is_symlink():
        raise ReleaseError(f'Symlinks are not permitted in evidence files: {relative}')
    return resolved


def _phase5_manifest_path(raw: str) -> str:
    return _canonical_relative_path(raw, 'Phase 5 SHA-256 inventory path')


def _hash(data: bytes):
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _csv_bytes(rows: list[dict], names: list[str]):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=names)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode('utf-8')


def _atomic(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.phase6.', suffix='.tmp', delete=False) as f:
        tmp = Path(f.name)
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


VERIFIER = r'''"""Verify an extracted Phase 6 research evidence archive (standard library only)."""
import csv
import hashlib
from pathlib import Path
import sys

def main():
    root = Path(__file__).resolve().parent
    rows = list(csv.DictReader((root / 'RELEASE_MANIFEST_SHA256.csv').open(encoding='utf-8', newline='')))
    if not rows:
        raise SystemExit('FAIL: Empty release manifest')
    seen = set()
    for row in rows:
        rel = row['relative_path']
        p = Path(rel)
        if rel in seen or p.is_absolute() or '..' in p.parts or '\\' in rel or not (root / p).resolve().is_relative_to(root):
            raise SystemExit('FAIL: Unsafe/duplicate release manifest path')
        seen.add(rel)
        path = root / p
        if not path.is_file() or path.is_symlink() or path.stat().st_size != int(row['bytes']):
            raise SystemExit('FAIL: Missing or changed ' + rel)
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1048576), b''):
                h.update(chunk)
        if h.hexdigest() != row['sha256']:
            raise SystemExit('FAIL: SHA-256 mismatch ' + rel)
    print('RELEASE_INTEGRITY_VERIFIED; files=' + str(len(rows)))
    print('Integrity is not model validity or physical device approval.')

if __name__ == '__main__':
    main()
'''


def _check_audit(root: Path, contract: dict):
    audit_path = _safe(root, contract['phase5_audit_status'])
    status = _json(audit_path)
    if status.get('status') != AUDIT_ACCEPT or status.get('scope_accepted') is not True:
        raise ReleaseError('Phase 5 final audit is not accepted: rerun Phase 5')
    tests = status.get('unit_tests', {})
    if tests.get('status') != 'PASS' or not isinstance(tests.get('count'), int) or tests['count'] < 1:
        raise ReleaseError('Phase 5 project unit tests are not confirmed PASS')
    checks = status.get('checks') or []
    if not checks or any(item.get('result') == 'FAIL' for item in checks):
        raise ReleaseError('Phase 5 has failed or missing acceptance checks')
    if status.get('physical_safety_approved') is not False or status.get('all_four_models_physically_integrated') is not False:
        raise ReleaseError('Unsafe or ambiguous physical-system approval flag in Phase 5')
    if status.get('research_scope_only') is not True:
        raise ReleaseError('Phase 5 research-only scope must remain explicit')
    sim = _json(_safe(root, contract['phase4_simulation_status']))
    if sim.get('overall_status', sim.get('research_status')) != SIM_ACCEPT and sim.get('status') != SIM_ACCEPT:
        raise ReleaseError('Phase 4 simulation is not in expected complete research-only status')
    total = status.get('candidate_rows')
    eligible = status.get('eligible_rows')
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in [total, eligible]) or not 0 <= eligible <= total:
        raise ReleaseError('Invalid Phase 5 accepted candidate counts')
    return status, sim


def _validate_sha_inventory(root: Path, contract: dict):
    path = _safe(root, contract['phase5_sha256_manifest'])
    try:
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error) as exc:
        raise ReleaseError(f'Cannot read Phase 5 source inventory: {exc}') from exc
    if not rows or any(not {'relative_path','exists','bytes','sha256'}.issubset(row) for row in rows):
        raise ReleaseError('Invalid Phase 5 inventory columns')
    missing = []
    verified = []
    seen = set()
    for row in rows:
        rel = _phase5_manifest_path(row['relative_path'])
        if rel in seen:
            raise ReleaseError('Duplicate Phase 5 inventory path after separator normalization: ' + rel)
        seen.add(rel)
        path = _safe(root, rel, required=False)
        if row['exists'] == 'True':
            if not path.is_file() or not row['sha256'] or not re.fullmatch(r'[0-9a-f]{64}', row['sha256']):
                raise ReleaseError(f'Source missing/invalid since Phase 5: {rel}')
            if int(row['bytes']) != path.stat().st_size or _file_sha256(path) != row['sha256']:
                raise ReleaseError(f'Source changed since Phase 5 audit: {rel}; rerun Phase 5')
            verified.append(rel)
        elif row['exists'] == 'False':
            missing.append(rel)
        else:
            raise ReleaseError('Unknown exists value in Phase 5 manifest')
    return verified, missing


def _fresh_tests(root: Path):
    if not (root / 'tests').is_dir():
        raise ReleaseError('Project tests directory missing; current acceptance cannot be verified')
    try:
        completed = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-q'],
                                   cwd=root, capture_output=True, text=True, encoding='utf-8',
                                   errors='replace', timeout=240, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReleaseError(f'Current project unit tests could not run: {exc}') from exc
    log = (completed.stdout + '\n' + completed.stderr).encode('utf-8')
    text = log.decode('utf-8')
    found = re.search(r'Ran\s+(\d+)\s+tests?', text)
    if completed.returncode != 0 or found is None or int(found.group(1)) < 1 or re.search(r'(?:^|\n)OK\s*(?:\n|$)', text) is None:
        raise ReleaseError('Current project unit tests failed; release blocked. Run python -m unittest discover -s tests -q')
    return int(found.group(1)), log


def _verify_zip(path: Path, rows: list[dict]):
    expected = {r['relative_path']: r for r in rows}
    try:
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None:
                raise ReleaseError('ZIP CRC verification failed')
            names = z.namelist()
            if len(names) != len(set(names)) or set(names) != set(expected) | {MANIFEST_NAME}:
                raise ReleaseError('ZIP members do not match release manifest')
            for rel, r in expected.items():
                if _hash(z.read(rel)) != r['sha256'] or len(z.read(rel)) != int(r['bytes']):
                    raise ReleaseError(f'ZIP SHA-256 mismatch: {rel}')
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise ReleaseError(f'ZIP verification failed: {exc}') from exc


def create_release(contract_file: str | Path, *, output_override: str | Path | None = None):
    contract_path = Path(contract_file).resolve()
    root = contract_path.parent
    config = _json(contract_path)
    if config.get('schema_version') != 1 or config.get('release_scope') != 'RESEARCH_EVIDENCE_ONLY':
        raise ReleaseError('Expected Phase 6 research-evidence contract schema_version=1')
    output = (Path(output_override).resolve() if output_override else (root / _canonical_relative_path(config.get('output_dir','phase6_results'), 'release output directory')).resolve())
    if not output.is_relative_to(root):
        raise ReleaseError('Release output directory must be inside project')
    output.mkdir(parents=True, exist_ok=True)
    report_file = output / 'phase6_release_status.json'
    result = {'phase': 6, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'RESEARCH_EVIDENCE_RELEASE_FAILED', 'fix_version': FIX_VERSION, 'research_scope_only': True,
              'physical_safety_approved': False, 'all_four_models_physically_integrated': False,
              'standalone_full_project_reproducibility_certified': False}
    try:
        phase5, phase4 = _check_audit(root, config)
        current_test_count, current_test_log = _fresh_tests(root)
        verified, missing = _validate_sha_inventory(root, config)
        required_raw = config.get('required_files')
        if not isinstance(required_raw, list) or len(required_raw) < 7:
            raise ReleaseError('Missing/invalid release required_files')
        required = [_canonical_relative_path(rel, 'required_files entry') for rel in required_raw]
        if len(set(required)) != len(required):
            raise ReleaseError('Duplicate required files after Windows path normalization')
        optional_raw = config.get('optional_files', [])
        if not isinstance(optional_raw, list):
            raise ReleaseError('Invalid optional_files')
        optional = [_canonical_relative_path(rel, 'optional_files entry') for rel in optional_raw]
        if len(set(optional)) != len(optional) or set(required) & set(optional):
            raise ReleaseError('Duplicate or overlapping optional_files after normalization')
        file_map = {}
        for rel in required:
            file_map[rel] = _safe(root, rel, origin='required_files entry').read_bytes()
        optional_missing = []
        for rel in optional:
            p = _safe(root, rel, required=False, origin='optional_files entry')
            if p.is_file():
                file_map[rel] = p.read_bytes()
            else:
                optional_missing.append(rel)
        for name, content in file_map.items():
            if len(content) > MAX_SOURCE_BYTES:
                raise ReleaseError(f'Single evidence file exceeds size limit (150 MiB): {name}')
        if any('_AR.' in rel for rel in file_map):
            raise ReleaseError('Arabic report selected for English-only delivery')
        warnings = [item.get('detail', '') for item in phase5['checks'] if item.get('result') == 'WARN']
        warnings.append('Phase 4 is a hypothetical scenario, not verified contemporaneous Solar/Wind/Hydrogen/Fuel Cell operation.')
        if missing:
            warnings.append('Some original Phase 5 inventory items were not present at audit time; see release metadata.')
        info = {'phase': 6, 'release_scope': 'RESEARCH_EVIDENCE_ONLY',
                'phase5_audit_status': phase5['status'],
                'phase5_verified_unit_test_count': phase5['unit_tests']['count'],
                'candidates_audited': phase5['candidate_rows'],
                'scenario_eligible_candidates': phase5['eligible_rows'],
                'phase3_pareto_points': phase5.get('phase3_pareto_points'),
                'phase4_scenario_status': SIM_ACCEPT,
                'assumed_solar_kw': phase5.get('hypothetical_solar_kw'),
                'assumed_wind_kw': phase5.get('hypothetical_wind_kw'),
                'assumed_total_supply_kw': phase5.get('hypothetical_supply_kw'),
                'benchmark_kwh_per_kg': phase5.get('benchmark_kwh_per_kg'),
                'historical_fuel_cell_reference_not_candidate_specific': phase5.get('historical_fc_prediction'),
                'runtime_at_phase5_audit': phase5.get('runtime', {}),
                'source_files_sha256_verified_against_phase5': len(verified),
                'phase5_inventory_items_missing_at_audit': missing,
                'optional_files_unavailable': optional_missing,
                'phase5_disclosed_warnings': warnings,
                'physical_safety_approved': False,
                'actual_renewable_calendar_alignment_verified': False,
                'hydrogen_to_fuel_cell_device_coupling_verified': False,
                'standalone_full_project_reproducibility_certified': False}
        readme = f'''# Green Hydrogen Prescriptive Engine — Research Evidence Release

## Release scope
This archive preserves audited **research scenario evidence only**. It does not establish physical Solar/Wind/Hydrogen/Fuel Cell linkage, electrolysis consumption measurements, equipment safety, or independently validated model persistence.

## Verified Phase 5 outcomes
- Phase 5: `{phase5['status']}`.
- Project tests when Phase 5 ran: **{phase5['unit_tests']['count']} passed**.
- Fresh tests at Phase 6 release: **{current_test_count} passed**.
- Audited Hydrogen candidates: **{phase5['candidate_rows']}**.
- Candidates meeting the hypothetical scenario: **{phase5['eligible_rows']}**.
- Recomputed Phase 3 Pareto points: **{phase5.get('phase3_pareto_points', 'not available')}**.
- Renewable assumptions: **{phase5.get('hypothetical_solar_kw')} kW** Solar + **{phase5.get('hypothetical_wind_kw')} kW** Wind, not timestamp-aligned forecasts.
- Benchmark: **{phase5.get('benchmark_kwh_per_kg')} kWh/kg H2**, literature reference, not measured electrolyzer SEC.
- Fuel Cell: historical **{phase5.get('historical_fc_prediction')}** reference, not candidate-specific integration.

## Verify the archive after extraction
Run `python VERIFY_RELEASE.py` at the extracted archive root. SHA-256 verification confirms file integrity, **not scientific validity**.

## Reproduce in the complete original project
1. Use the original project with its full source datasets, model artifacts, approved environment, and tests.
2. Run `python .\\phase4_research_simulation.py --scenario .\\phase4_research_scenario.json`.
3. Run `python .\\phase5_final_audit.py --contract .\\phase5_audit_contract.json`.
4. Run this Phase 6 release command again to check source manifest immutability.

This archive is an **evidence snapshot**, not a guaranteed complete standalone install. Some model/version warnings may remain; refer to `RELEASE_METADATA.json` and `phase5_results/phase5_model_compatibility.csv`.

## Distribution caution
Datasets and serialized model artifacts (if included) may have ownership, licensing, privacy, and pickle trust considerations. Use trusted artifacts and assess sharing permissions before distribution.
'''
        file_map['phase6_current_unit_tests.log'] = current_test_log
        info['current_project_unit_tests_passed'] = current_test_count
        file_map['RELEASE_README_EN.md'] = readme.encode('utf-8')
        file_map['RELEASE_METADATA.json'] = (json.dumps(info, indent=2, ensure_ascii=True, allow_nan=False)+'\n').encode('utf-8')
        file_map['VERIFY_RELEASE.py'] = VERIFIER.encode('utf-8')
        rows = [{'relative_path': k, 'bytes': len(v), 'sha256': _hash(v)} for k, v in sorted(file_map.items())]
        manifest = _csv_bytes(rows, ['relative_path', 'bytes', 'sha256'])
        archive_name = 'Green_Hydrogen_Research_Evidence_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8] + '.zip'
        archive = output / archive_name
        with tempfile.NamedTemporaryFile(dir=output, prefix='.phase6.', suffix='.zip.tmp', delete=False) as f:
            temp = Path(f.name)
        try:
            with zipfile.ZipFile(temp, mode='w', compression=zipfile.ZIP_DEFLATED, compresslevel=7, allowZip64=True) as z:
                for name, data in sorted(file_map.items()):
                    z.writestr(name, data)
                z.writestr(MANIFEST_NAME, manifest)
            _verify_zip(temp, rows)
            os.replace(temp, archive)
        finally:
            temp.unlink(missing_ok=True)
        _atomic(output / 'phase6_release_manifest.csv', manifest)
        _atomic(output / 'RELEASE_README_EN.md', readme.encode('utf-8'))
        result.update({'status': RELEASE_ACCEPT, 'archive': str(archive), 'archive_sha256': _file_sha256(archive),
                       'archive_size_bytes': archive.stat().st_size,
                       'verified_release_members': len(rows),
                       'phase5_unit_tests_passed': phase5['unit_tests']['count'],
                       'current_unit_tests_passed': current_test_count,
                       'candidate_rows': phase5['candidate_rows'], 'eligible_rows': phase5['eligible_rows'],
                       'phase3_pareto_points': phase5.get('phase3_pareto_points'),
                       'phase5_inventory_verified_count': len(verified),
                       'phase5_inventory_missing_at_audit': missing,
                       'optional_files_missing': optional_missing,
                       'warnings': warnings,
                       'integrity_verified': True})
        _atomic(report_file, (json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False)+'\n').encode('utf-8'))
        print(f"[PHASE6] {RELEASE_ACCEPT}; audited={phase5['candidate_rows']}; eligible={phase5['eligible_rows']}; fresh tests={current_test_count}")
        print(f'Release: {archive}')
        print(f'SHA256: {result["archive_sha256"]}')
        print('Research evidence archive only; no physical integration or safe operation certification.')
        return result
    except (ReleaseError, OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        result['error'] = str(exc)
        _atomic(report_file, (json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False)+'\n').encode('utf-8'))
        print(f'[PHASE6] RESEARCH_EVIDENCE_RELEASE_FAILED: {exc}')
        print(f'Report: {report_file}')
        return result


def main():
    ap = argparse.ArgumentParser(description='Phase 6 research evidence release; no device certification')
    ap.add_argument('--contract', default='phase6_release_contract.json')
    args = ap.parse_args()
    result = create_release(args.contract)
    return 0 if result['status'] == RELEASE_ACCEPT else 1

if __name__ == '__main__':
    raise SystemExit(main())
