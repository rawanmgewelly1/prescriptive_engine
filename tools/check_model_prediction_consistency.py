"""Compare four frozen model spot predictions across local runtime versions.

Passing is NOT evidence of scientific calibration, device safety or full-model
compatibility; it checks only four archived reference operating points.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prescriptive_engine.pipeline.inference import Models
from prescriptive_engine import engine


def predictions():
    doc=json.loads((ROOT/'configs/scenarios/RESEARCH_FOUR_MODELS_HYPOTHETICAL.json').read_text())
    fixed=json.loads((ROOT/'configs/scenarios/RESEARCH_DEMO_200KW.json').read_text())
    models=Models(engine.read_config(ROOT/'config.json'))
    power_inputs=doc['renewable_power']
    h2={'renewable_power_kw':fixed['renewable_power']['available_kw'],**fixed['baseline']}
    fc=fixed['fuel_cell']
    fc_in={'hydrogen_flow_l_min':fc['flow_demand_l_min'],**fc['operating_point']}
    return {
        'solar_kw':float(models.regression('solar',pd.DataFrame([power_inputs['solar_inputs']]))[0]),
        'wind_kw':float(models.regression('wind',pd.DataFrame([power_inputs['wind_inputs']]))[0]),
        'hydrogen_baseline_kg_h':float(models.regression('hydrogen',pd.DataFrame([h2]))[0]),
        'fuel_cell_class':str(models.fuel_cell(pd.DataFrame([fc_in]))[0])
    }


def main():
    ref=json.loads((ROOT/'configs/reference_model_predictions.json').read_text())
    actual=predictions()
    diffs={}
    for key,value in ref['reference_predictions'].items():
        observed=actual[key]
        if isinstance(value,str):
            same=value==observed
        else:
            same=abs(observed-value)<=ref['absolute_tolerance']+ref['relative_tolerance']*abs(value)
        if not same:diffs[key]={'reference':value,'current':observed}
    result={'status':'FOUR_MODEL_SPOT_CHECK_PASSED_NOT_FULL_VALIDATION' if not diffs else 'MODEL_PREDICTION_VERSION_DRIFT_DETECTED',
            'reference_versions':ref['reference_versions'], 'current_predictions':actual,
            'mismatches':diffs,'scientific_validation':False}
    path=ROOT/'outputs/model_compatibility_status.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('[MODEL SPOT CHECK]',result['status'])
    print('Report:',path)
    return 0 if not diffs else 2

if __name__=='__main__':raise SystemExit(main())
