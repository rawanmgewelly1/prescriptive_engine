"""Conditional candidate FC inference; no device-coupling or safety certification."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .contracts import PipelineBlocked
R_GAS=8.31446261815324
M_H2_G_PER_MOL=2.01588

def kg_h_to_reference_l_min(h2_kg_h,temp_k,pressure_pa):
    # Ideal gas conversion at *specified reference gas conditions* only.
    h2=np.asarray(h2_kg_h,dtype=float)
    return (h2*1000/M_H2_G_PER_MOL)*(R_GAS*temp_k/pressure_pa)*1000/60

def annotate(frame,spec,models,support):
    f=frame.copy()
    f['model_fc_class']='NOT_EVALUATED'
    f['fc_support_check']=False
    f['fc_flow_demand_met']=False
    f['fc_available_hydrogen_l_min']=np.nan
    f['fc_assumed_draw_l_min']=np.nan
    f['fuel_cell_research_eligible']=False
    if spec['bridge_mode']=='BLOCKED_NO_MAPPING':
        f['fuel_cell_status']='NOT_VERIFIED_NO_BRIDGE'
        return f,{'link':'BLOCKED_NO_VERIFIED_CANDIDATE_MAPPING'}
    t=float(spec['gas_reference_temperature_k']);pr=float(spec['gas_reference_pressure_pa'])
    direct_flow=spec['bridge_mode']=='ASSUMED_CANDIDATE_FLOW'
    demand=float(spec.get('flow_demand_l_min', 0)) if not direct_flow else None
    f['fc_available_hydrogen_l_min']=kg_h_to_reference_l_min(
        f['predicted_hydrogen_kg_h'].clip(lower=0).to_numpy(),t,pr)
    if direct_flow:
        # Deliberately test the modeled flow per candidate; out-of-training-support
        # values MUST NOT receive a fabricated Pass class.
        f['fc_assumed_draw_l_min']=f['fc_available_hydrogen_l_min']
        f['fc_flow_demand_met']=f['hydrogen_nonnegative'] & f['fc_available_hydrogen_l_min'].gt(0)
    else:
        f['fc_assumed_draw_l_min']=np.minimum(f['fc_available_hydrogen_l_min'],demand)
        f['fc_flow_demand_met']=(f['fc_available_hydrogen_l_min']>=demand)
    point=spec['operating_point']
    # A valid historical model response is required only for candidates whose H2 supply could meet draw.
    capable=f['fc_flow_demand_met'] & f['hydrogen_nonnegative']
    # FC voltage, temperature, and pressure are DIFFERENT device variables from
    # the electrolyzer controls; they come only from an explicit FC scenario.
    point_support={k:float(point[k]) for k in ['voltage_v','temperature_c','pressure_bar']}
    for k,v in point_support.items():
        lo,hi=support[k]
        if not (lo<=v<=hi):
            f['fuel_cell_status']='NOT_VERIFIED_FC_INPUT_OUT_OF_SUPPORT'
            return f,{'link':'BLOCKED_ASSUMED_OPERATING_POINT_OUTSIDE_OBSERVED_SUPPORT',
                'field':k,'value':v,'support_range':[lo,hi]}
    lo,hi=support['hydrogen_flow_l_min']
    eligible_flow=f['fc_assumed_draw_l_min'].between(lo,hi,inclusive='both')
    supported=capable & eligible_flow
    f['fc_support_check']=supported
    f['fuel_cell_status']='NOT_EVALUATED_OR_NOT_PASS'
    f.loc[capable & ~eligible_flow,'fuel_cell_status']='NOT_VERIFIED_FC_FLOW_OUT_OF_OBSERVED_SUPPORT'
    if supported.any():
        count=int(supported.sum())
        fc_input=pd.DataFrame({
            'hydrogen_flow_l_min':f.loc[supported,'fc_assumed_draw_l_min'].to_numpy(dtype=float),
            'voltage_v':np.full(count,point_support['voltage_v']),
            'temperature_c':np.full(count,point_support['temperature_c']),
            'pressure_bar':np.full(count,point_support['pressure_bar'])})
        f.loc[supported,'model_fc_class']=models.fuel_cell(fc_input)
    f['fuel_cell_research_eligible']=(f['model_fc_class']=='Pass') & supported
    f.loc[f['fuel_cell_research_eligible'],'fuel_cell_status']='PREDICTED_PASS_UNDER_ASSUMED_BRIDGE_NOT_PHYSICAL'
    return f,{'link':'RESEARCH_ASSUMED_FLOW_CONVERSION_AND_FC_INFERENCE',
        'physical_link_verified':False,'specified_reference_temperature_k':t,
        'specified_reference_pressure_pa':pr,
        'assumed_fc_flow_demand_l_min':demand,'fc_operating_point':point,
        'method':('DIRECT_CANDIDATE_REFERENCE_FLOW_RESEARCH_ONLY' if direct_flow
                  else 'IDEAL_GAS_REFERENCE_VOLUME_WITH_ASSUMED_REGULATED_FC_DRAW'),
        'fc_input_policy':('CANDIDATE_DEPENDENT_FLOW_WITH_EMPIRICAL_SUPPORT_GATE' if direct_flow
                           else 'CONSTANT_REGULATED_SETPOINT_NON_DISCRIMINATING_CLASSIFIER'),
        'fc_model_evaluated_candidate_count':int(supported.sum()),
        'fc_flow_out_of_support_candidate_count':int((capable & ~eligible_flow).sum()),
        'distinct_fc_model_input_flow_values':int(f.loc[supported,'fc_assumed_draw_l_min'].nunique()),
        'fc_model_class_is_candidate_discriminating':bool(direct_flow and f.loc[supported,'fc_assumed_draw_l_min'].nunique()>1),
        'fuel_cell_observed_flow_support_l_min':[lo,hi],
        'candidate_classification_is_simulation_not_measured':True}
