"""Research electrical consumption simulation; NEVER authorizes operation."""
from __future__ import annotations
import json
import math
from pathlib import Path
import pandas as pd

class SimulationContractError(ValueError):
    pass

REQUIRED = {'mode','model_type','source_reference','equation_reference','input_power_boundary','load_power_boundary', 'model_applicability_reference','conversion_fraction'}

BENCHMARK_URL = "https://www.energy.gov/cmei/fuels/technical-targets-liquid-alkaline-electrolysis"
BENCHMARK_FIELDS = {'mode', 'model_type', 'technology', 'specific_energy_kwh_per_kg',
                    'reference_url', 'reference_metric', 'electricity_boundary', 'scope'}

def load_simulation_contract(path):
    doc=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(doc,dict):
        raise SimulationContractError('Contract must be a JSON object')
    if doc.get('mode')=='missing_model':
        if set(doc)-{'mode','reason'} or not isinstance(doc.get('reason'),str) or not doc['reason'].strip():
            raise SimulationContractError('missing_model requires a reason and no model values')
        return doc
    if doc.get('mode') == 'research_simulation' and doc.get('model_type') == 'literature_specific_energy_benchmark':
        if set(doc) != BENCHMARK_FIELDS:
            raise SimulationContractError('Benchmark contract fields must match documented schema exactly')
        expected = {
            'technology': 'liquid_alkaline_electrolysis',
            'specific_energy_kwh_per_kg': 55,
            'reference_url': BENCHMARK_URL,
            'reference_metric': 'DOE 2022 status; system energy efficiency, kWh/kg hydrogen',
            'electricity_boundary': 'electrolyzer_system',
            'scope': 'research_comparison_only_not_device_validated',
        }
        for key, value in expected.items():
            if type(doc.get(key)) is not type(value) or doc.get(key) != value:
                raise SimulationContractError(f'Invalid/unverified benchmark field: {key}')
        return doc
    if set(doc)!=REQUIRED or doc.get('mode')!='research_simulation'  or doc.get('model_type')!='fixed_fraction_available_to_load':
        raise SimulationContractError('Model requires exact documented research fields; no defaults')
    for name in REQUIRED - {'conversion_fraction'}:
        if not isinstance(doc[name],str) or not doc[name].strip():
            raise SimulationContractError(f'{name} requires a nonempty string')
    frac=doc['conversion_fraction']
    if isinstance(frac,bool) or not isinstance(frac,(int,float)) or not math.isfinite(frac) or not (0<frac<=1):
        raise SimulationContractError('conversion_fraction must lie in (0,1]')
    return doc

def evaluate_simulated_consumption(candidates, contract):
    meta={'simulation_status':'BLOCKED_MISSING_DOCUMENTED_MODEL', 'scientific_status':'RESEARCH_ONLY_NOT_VALIDATED',
          'dispatch_approved':False, 'sec_approved':False, 'physical_safety_approved':False,
          'solar_wind_historical_alignment_verified':False,'fuel_cell_integration_verified':False}
    if contract['mode']=='missing_model':
        meta['reason']=contract['reason']
        return pd.DataFrame(),meta
    power='renewable_power_kw'
    prediction='predicted_hydrogen_production_kg_h'
    for c in (power,prediction):
        if c not in candidates:
            raise SimulationContractError(f'Missing candidate column: {c}')
    if candidates.columns.duplicated().any():
        raise SimulationContractError('Duplicate candidate column names')
    df=candidates.copy()
    for c in (power,prediction):
        df[c]=pd.to_numeric(df[c],errors='coerce')
    import numpy as np
    clean=np.isfinite(df[power]) & np.isfinite(df[prediction]) & (df[power]>0)&(df[prediction]>0)
    if not clean.all():
        raise SimulationContractError(f'Cannot simulate: {(~clean).sum()} invalid or nonpositive candidate power/production values')
    if len(df)==0:
        raise SimulationContractError('Candidate dataset is empty')
    if contract.get('model_type') == 'literature_specific_energy_benchmark':
        # Demand is the electricity a *reference* liquid-alkaline system would need
        # to produce the model-predicted hydrogen flow at a steady hourly rate.
        # This is not observed consumption, device SEC, or time-resolved dispatch.
        benchmark = float(contract['specific_energy_kwh_per_kg'])
        df['benchmark_electrical_demand_kw'] = df[prediction] * benchmark
        df['benchmark_specific_energy_kwh_per_kg'] = benchmark
        df['benchmark_demand_within_available_power'] = df['benchmark_electrical_demand_kw'] <= df[power]
        df['benchmark_power_shortfall_kw'] = (df['benchmark_electrical_demand_kw'] - df[power]).clip(lower=0)
        df['simulation_claim'] = 'DOE_2022_ALKALINE_SYSTEM_REFERENCE_SCENARIO_ONLY'
        df['physical_operating_approval'] = False
        meta.update({
            'simulation_status':'LITERATURE_BENCHMARK_COMPARISON_COMPLETE',
            'scientific_status':'RESEARCH_BENCHMARK_NOT_DEVICE_VALIDATED',
            'reason':'Reference 2022 alkaline-system status, not a validated device-specific energy model or real dispatch',
            'rows':int(len(df)),
            'benchmark_kwh_per_kg':benchmark,
            'benchmark_sufficient_supply_rows':int(df['benchmark_demand_within_available_power'].sum()),
            'benchmark_insufficient_supply_rows':int((~df['benchmark_demand_within_available_power']).sum()),
            'source_reference':contract['reference_url'],
            'benchmark_technology':contract['technology'],
            'electricity_boundary':contract['electricity_boundary'],
        })
        return df, meta
    fraction=float(contract['conversion_fraction'])
    df['simulated_electrical_load_kw']=df[power]*fraction
    df['simulated_sec_kwh_per_predicted_kg']=df['simulated_electrical_load_kw']/df[prediction]
    df['simulation_claim']='RESEARCH_MODEL_ASSUMPTION_NOT_MEASURED_OR_VALIDATED'
    df['physical_operating_approval']=False
    meta.update({'simulation_status':'SIMULATED_ASSUMPTION_NOT_VALIDATED', 'reason':'Research-only fixed fraction from user-supplied cited model; not independently verified, SEC is simulated ratio, not measured electrical SEC',
       'rows':int(len(df)), 'conversion_fraction':fraction,'model_type':contract['model_type'],
       'input_power_boundary':contract['input_power_boundary'],'load_power_boundary':contract['load_power_boundary'],
       'source_reference':contract['source_reference'],'equation_reference':contract['equation_reference'],
       'model_applicability_reference':contract['model_applicability_reference']})
    return df,meta

def run_simulation(contract_file, candidate_file, output_dir):
    contract=load_simulation_contract(contract_file)
    if contract['mode']=='missing_model':
        candidates=pd.DataFrame()
    else:
        candidates=pd.read_csv(candidate_file)
    output,meta=evaluate_simulated_consumption(candidates,contract)
    dest=Path(output_dir);dest.mkdir(parents=True,exist_ok=True)
    from tempfile import NamedTemporaryFile
    import os
    def atomic_bytes(path,data):
        if path.is_symlink():
            raise SimulationContractError('Refusing to overwrite symbolic link')
        with NamedTemporaryFile('wb',dir=dest,prefix='.'+path.name+'.',suffix='.tmp',delete=False) as f:
            tmp=Path(f.name)
            try:
                f.write(data);f.flush();os.fsync(f.fileno())
            except BaseException:
                tmp.unlink(missing_ok=True);raise
        try: os.replace(tmp,path)
        finally: tmp.unlink(missing_ok=True)
    report=dest/'electrolyzer_simulation_status.json'
    atomic_bytes(report,json.dumps(meta,indent=2,ensure_ascii=False).encode('utf-8'))
    # Prevent stale success-like results persisting when model is absent.
    csv=dest/'electrolyzer_simulated_candidates.csv'
    if output.empty:
        if csv.exists():csv.unlink()
    else:
        atomic_bytes(csv,output.to_csv(index=False).encode('utf-8'))
    print(f"[ELECTROLYZER SIMULATION] {meta['simulation_status']}. Output: {report}")
    print('Research assumption only; NOT measured SEC, safe operation, dispatch, or independently validated engineering integration.')
    return meta
