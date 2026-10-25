"""Phase 2 observational electrolyzer measurement contract. Never grants operational approval."""
from __future__ import annotations
import json
import math
from datetime import datetime, timezone
from pathlib import Path

class ElectrolyzerContractError(ValueError):
    pass

def _instant(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ElectrolyzerContractError(f'{name}: ISO 8601 timestamp with UTC offset required')
    try:
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ElectrolyzerContractError(f'{name}: invalid ISO 8601 timestamp') from exc
    if dt.utcoffset() is None:
        raise ElectrolyzerContractError(f'{name}: timezone offset required')
    return dt.astimezone(timezone.utc)

def _positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ElectrolyzerContractError(f'{name}: positive finite number required')
    return float(value)

def _required_text(obj, field):
    value = obj.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ElectrolyzerContractError(f'{field}: nonempty evidence required')
    return value.strip()

def assess_electrolyzer_contract(document: dict) -> dict:
    if not isinstance(document, dict):
        raise ElectrolyzerContractError('Contract must be a JSON object')
    allowed = {'mode','reason','measurement_start','measurement_end','electrical_energy_kwh','hydrogen_mass_kg',
               'electric_meter_reference','hydrogen_meter_reference','data_source_reference','electrical_boundary','measurement_basis'}
    if set(document)-allowed:
        raise ElectrolyzerContractError(f'Unknown keys: {sorted(set(document)-allowed)}')
    mode = document.get('mode')
    report = {'contract_version':'phase2_v1', 'mode':mode, 'consumption_status':'NOT_AVAILABLE',
              'sec_kwh_per_kg':None, 'sec_status':'NOT_CALCULABLE',
              'physical_dispatch':'NOT_APPROVED','operating_recommendation':'NOT_APPROVED',
              'fuel_cell_integration':'NOT_VALIDATED','scientific_status':'EVIDENCE_GATE_ONLY'}
    if mode == 'missing_measurements':
        if set(document)-{'mode','reason'}:
            raise ElectrolyzerContractError('missing_measurements cannot contain measurement or estimated values')
        report['reason'] = _required_text(document,'reason')
        return report
    if mode != 'observed_interval':
        raise ElectrolyzerContractError('mode must be missing_measurements or observed_interval')
    if 'reason' in document:
        raise ElectrolyzerContractError('reason is reserved for missing_measurements')
    start = _instant(document.get('measurement_start'),'measurement_start')
    end = _instant(document.get('measurement_end'),'measurement_end')
    if end <= start:
        raise ElectrolyzerContractError('measurement_end must be later than measurement_start')
    energy = _positive_number(document.get('electrical_energy_kwh'),'electrical_energy_kwh')
    mass = _positive_number(document.get('hydrogen_mass_kg'),'hydrogen_mass_kg')
    for name in ('electric_meter_reference','hydrogen_meter_reference','data_source_reference','electrical_boundary','measurement_basis'):
        _required_text(document,name)
    report.update({'consumption_status':'OBSERVED_INTERVAL_REPORTED_NOT_INDEPENDENTLY_VERIFIED',
                   'sec_kwh_per_kg':energy / mass,
                   'sec_status':'OBSERVATIONAL_RATIO_NOT_ENGINEERING_APPROVAL',
                   'measurement_start_utc':start.isoformat(), 'measurement_end_utc':end.isoformat(),
                   'electrical_energy_kwh':energy, 'hydrogen_mass_kg':mass,
                   'electrical_boundary':document['electrical_boundary'],
                   'measurement_basis':document['measurement_basis'],
                   'electric_meter_reference':document['electric_meter_reference'],
                   'hydrogen_meter_reference':document['hydrogen_meter_reference'],
                   'data_source_reference':document['data_source_reference'],
                   'reason':'SEC is from user-supplied paired measured totals; meter calibration, gas purity, boundary and physical feasibility remain unverified.'})
    return report

def read_electrolyzer_contract(path):
    return assess_electrolyzer_contract(json.loads(Path(path).read_text(encoding='utf-8-sig')))
