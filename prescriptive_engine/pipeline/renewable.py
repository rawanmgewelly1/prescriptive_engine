"""One scenario, one fixed power value; historical time-series are not silently joined."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .contracts import PipelineBlocked

def fixed_available_power(spec,models):
    if spec['mode']=='fixed_user':
        return float(spec['available_kw']),{'mode':'USER_ENTERED_FIXED_POWER','solar_prediction_kw':None,
             'wind_prediction_kw':None,'historical_calendar_alignment_verified':False,
             'supply_is_measured_consumption':False}
    solar=pd.DataFrame([spec['solar_inputs']]);wind=pd.DataFrame([spec['wind_inputs']])
    for kind,frame in [('solar',solar),('wind',wind)]:
        required=models.cfg[kind+'_features']
        if not set(required).issubset(frame):
            raise PipelineBlocked(f'{kind} scenario missing model input fields: {sorted(set(required)-set(frame))}')
    solar_kw=float(models.regression('solar',solar)[0]);wind_kw=float(models.regression('wind',wind)[0])
    if not np.isfinite([solar_kw,wind_kw]).all() or solar_kw < 0 or wind_kw < 0:
        raise PipelineBlocked('Predicted component power is negative or invalid; not clipped')
    if spec['aggregation_scope']!='ONE_SOLAR_SOURCE_PLUS_ONE_WIND_TURBINE':
        raise PipelineBlocked('Unsupported component aggregation scope')
    return solar_kw+wind_kw,{'mode':'HYPOTHETICAL_SAME_SCENARIO_MODEL_PREDICTIONS',
        'solar_prediction_kw':solar_kw,'wind_prediction_kw':wind_kw,
        'historical_calendar_alignment_verified':False,'supply_is_measured_consumption':False}
