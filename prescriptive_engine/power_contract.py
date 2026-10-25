"""Evidence-aware renewable supply intake. Not load dispatch or engineering approval."""
from __future__ import annotations
import json
import math
from datetime import datetime
from pathlib import Path

class PowerContractError(ValueError):
    pass


def validate_power_contract(document: dict) -> dict:
    if not isinstance(document, dict):
        raise PowerContractError('Power contract must be a JSON object.')
    allowed = {'mode','source','available_power_kw','measurement_timestamp','timestamp_timezone','source_reference','measurement_point','measurement_basis','consumed_power_kw','loss_fraction','solar_wind_historical_alignment_verified'}
    extras = set(document)-allowed
    if extras:
        raise PowerContractError(f'Unexpected power contract keys: {sorted(extras)}')
    mode = document.get('mode')
    if mode != 'research_scenario':
        raise PowerContractError('Only research_scenario is supported; real dispatch is not approved.')
    source = document.get('source')
    if source not in ('user_entered_available_power','external_measurement'):
        raise PowerContractError('Source must be user_entered_available_power or external_measurement.')
    power = document.get('available_power_kw')
    if isinstance(power, bool) or not isinstance(power,(int,float)) or not math.isfinite(power) or power<=0:
        raise PowerContractError('available_power_kw must be a positive, finite number.')
    if 'consumed_power_kw' in document or 'loss_fraction' in document:
        raise PowerContractError('Consumption and loss calculations are not supported without a validated power flow model.')
    if document.get('solar_wind_historical_alignment_verified') is not None:
        raise PowerContractError('This research contract cannot certify historical Solar/Wind alignment.')
    ref = document.get('source_reference')
    timestamp = document.get('measurement_timestamp')
    tz = document.get('timestamp_timezone')
    point = document.get('measurement_point')
    basis = document.get('measurement_basis')
    if source == 'external_measurement':
        if any(not isinstance(v,str) or not v.strip() for v in (ref,timestamp,point,basis)):
            raise PowerContractError('External measurement needs source_reference, measurement_timestamp, measurement_point and measurement_basis.')
        try:
            parsed = datetime.fromisoformat(timestamp.replace('Z','+00:00'))
        except ValueError as ex:
            raise PowerContractError('measurement_timestamp must be ISO 8601.') from ex
        if parsed.utcoffset() is None:
            raise PowerContractError('External measurement timestamp needs an explicit UTC offset.')
    elif timestamp is not None:
        raise PowerContractError('Do not attach a measurement timestamp to user-entered hypothetical power.')
    if tz is not None and (not isinstance(tz,str) or not tz.strip()):
        raise PowerContractError('timestamp_timezone must be a non-empty string if provided.')
    return {
        'mode':mode,'source':source,'available_power_kw':float(power),
        'source_reference':ref if isinstance(ref,str) else None,
        'measurement_timestamp':timestamp,'measurement_point':point,
        'measurement_basis':basis,
        'available_power_semantics':'SUPPLY_AVAILABLE_NOT_CONSUMED',
        'historical_alignment':'NOT_VERIFIED',
        'consumed_power_status':'NOT_AVAILABLE',
        'physical_dispatch':'NOT_APPROVED',
        'scientific_status':'RESEARCH_SCENARIO_ONLY',
    }


def read_power_contract(path: str | Path) -> dict:
    raw = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    return validate_power_contract(raw)
