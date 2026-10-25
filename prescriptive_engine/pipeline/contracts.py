"""Parse and validate scenario contracts. Do not invent missing evidence."""
from __future__ import annotations
import json
import math
from pathlib import Path

DECISION_VARS = (
    "voltage_v", "temperature_c", "current_density_a_cm2",
    "electrolyte_concentration_wt_pct", "water_flow_l_min", "pressure_bar")

class PipelineBlocked(ValueError):
    """Requested calculation lacks evidence or a valid scenario contract."""

def real_number(x, name, minimum=None):
    if isinstance(x, bool) or not isinstance(x,(float,int)) or not math.isfinite(x):
        raise PipelineBlocked(f"{name} must be a finite numeric value")
    if minimum is not None and x < minimum:
        raise PipelineBlocked(f"{name} must be >= {minimum}")
    return float(x)

def validate(doc):
    if not isinstance(doc,dict) or doc.get('schema_version')!=1:
        raise PipelineBlocked('schema_version must be 1')
    if doc.get('mode')!='RESEARCH_SIMULATION':
        raise PipelineBlocked('Only explicitly labeled RESEARCH_SIMULATION is supported')
    if doc.get('assumptions_acknowledged') is not True:
        raise PipelineBlocked('Research assumptions must be explicitly acknowledged')
    power=doc.get('renewable_power',{})
    if not isinstance(power,dict) or power.get('mode') not in ('fixed_user','model_scenario'):
        raise PipelineBlocked('renewable_power.mode must be fixed_user or model_scenario')
    if power['mode']=='fixed_user':
        real_number(power.get('available_kw'),'renewable_power.available_kw',0.001)
    else:
        if power.get('same_hypothetical_scenario_acknowledged') is not True:
            raise PipelineBlocked('Model combination needs explicit same-scenario acknowledgement')
        if not all(isinstance(power.get(k),dict) for k in ('solar_inputs','wind_inputs')):
            raise PipelineBlocked('Model scenario requires explicit solar_inputs and wind_inputs')
        if power.get('aggregation_scope') not in ('ONE_SOLAR_SOURCE_PLUS_ONE_WIND_TURBINE',):
            raise PipelineBlocked('Explicit component scope required; no farm aggregation inferred')
    initial=doc.get('baseline',{})
    if not isinstance(initial,dict) or set(initial)!=set(DECISION_VARS):
        raise PipelineBlocked('baseline must contain exactly six decision variables')
    for k in DECISION_VARS: real_number(initial[k],f'baseline.{k}')
    bounds=doc.get('bounds',{})
    if not isinstance(bounds,dict) or set(bounds)!=set(DECISION_VARS):
        raise PipelineBlocked('bounds must contain exactly six decision variables')
    for k in DECISION_VARS:
        b=bounds[k]
        if not isinstance(b,list) or len(b)!=2: raise PipelineBlocked(f'bounds.{k} must be [min,max]')
        lo=real_number(b[0],f'bounds.{k}[0]')
        hi=real_number(b[1],f'bounds.{k}[1]')
        if lo>=hi: raise PipelineBlocked(f'bounds.{k} must increase')
        if not lo <= initial[k] <= hi: raise PipelineBlocked(f'baseline.{k} outside scenario bounds')
    n=doc.get('optimization',{}).get('candidate_count')
    if type(n)!=int or not (20<=n<=100000):
        raise PipelineBlocked('optimization.candidate_count must be integer from 20 to 100000')
    seed=doc.get('optimization',{}).get('seed')
    if type(seed)!=int: raise PipelineBlocked('optimization.seed must be integer')
    energy=doc.get('energy_benchmark',{})
    if not isinstance(energy,dict) or energy.get('use_as')!='RESEARCH_COMPARISON_ONLY':
        raise PipelineBlocked('energy_benchmark must be research-only')
    real_number(energy.get('kwh_per_kg'),'energy_benchmark.kwh_per_kg',0.001)
    if not isinstance(energy.get('source'),str) or not energy['source'].strip():
        raise PipelineBlocked('Document the research benchmark source')
    fc=doc.get('fuel_cell',{})
    if not isinstance(fc,dict) or fc.get('bridge_mode') not in ('BLOCKED_NO_MAPPING','ASSUMED_REGULATED_DRAW','ASSUMED_CANDIDATE_FLOW'):
        raise PipelineBlocked('fuel_cell.bridge_mode invalid')
    if fc['bridge_mode'] in ('ASSUMED_REGULATED_DRAW','ASSUMED_CANDIDATE_FLOW'):
        if fc.get('acknowledge_unverified_device_chain') is not True:
            raise PipelineBlocked('Assumed FC bridge requires explicit research acknowledgement')
        if fc['bridge_mode']=='ASSUMED_REGULATED_DRAW':
            real_number(fc.get('flow_demand_l_min'),'fuel_cell.flow_demand_l_min',0.001)
        real_number(fc.get('gas_reference_temperature_k'),'fuel_cell.gas_reference_temperature_k',0.001)
        real_number(fc.get('gas_reference_pressure_pa'),'fuel_cell.gas_reference_pressure_pa',0.001)
        point=fc.get('operating_point')
        if not isinstance(point,dict) or set(point)!={'voltage_v','temperature_c','pressure_bar'}:
            raise PipelineBlocked('FC operating point requires voltage_v, temperature_c, pressure_bar')
        for k in point: real_number(point[k],f'fuel_cell.operating_point.{k}',0.001 if k=='pressure_bar' else None)
    return doc

def read_scenario(path):
    try: data=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError,json.JSONDecodeError) as exc: raise PipelineBlocked(f'Invalid scenario JSON: {exc}') from exc
    return validate(data)
