"""User scenario -> fixed renewable power -> baseline -> six-variable search -> FC -> results."""
from __future__ import annotations
import json
import os
import tempfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from prescriptive_engine import engine
from .contracts import PipelineBlocked, read_scenario, DECISION_VARS
from .inference import Models
from .renewable import fixed_available_power
from .optimization import make_candidates, empirical_joint_support
from .fuel_cell import annotate

def _json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,delete=False,suffix='.tmp') as h:
        json.dump(obj,h,indent=2,allow_nan=False)
        tmp=h.name
    os.replace(tmp,path)

def run_scenario(scenario_path, *, config_path=None, output_dir=None):
    scenario_path=Path(scenario_path).resolve()
    root=Path(config_path).resolve().parent if config_path else Path(__file__).resolve().parents[2]
    output=Path(output_dir).resolve() if output_dir else root/'outputs'/'scenarios'/scenario_path.stem
    output.mkdir(parents=True,exist_ok=True)
    status={'schema_version':1,'scenario':scenario_path.name,'created_utc':datetime.now(timezone.utc).isoformat(),
        'pipeline':'ORIGINAL_SEVEN_STAGE_RESEARCH_IMPLEMENTATION','physical_safety_approved':False,
        'candidate_specific_fc_physical_link_verified':False,'full_four_model_pipeline_verified':False,
        'scientific_status':'RESEARCH_ONLY_NOT_VALIDATED_ENGINEERING'}
    try:
        spec=read_scenario(scenario_path)
        cfg=engine.read_config(config_path or (root/'config.json'))
        mod=Models(cfg)
        hdf=pd.read_csv(Path(cfg['dataset_dir'])/cfg['hydrogen_dataset_file'])
        fcdf=pd.read_csv(Path(cfg['dataset_dir'])/cfg['fuel_cell_dataset_file'])
        fixed_power,power_status=fixed_available_power(spec['renewable_power'],mod)
        status['renewable_stage']=power_status
        status['fixed_available_power_kw']=fixed_power
        candidates=make_candidates(spec,hdf,fixed_power)
        support_ok,support_distance,support_cutoff=empirical_joint_support(candidates,hdf)
        candidates['empirical_joint_support_pass']=support_ok
        candidates['empirical_joint_support_distance']=support_distance
        # All candidates, including baseline, use exactly the same available power.
        assert candidates['renewable_power_kw'].nunique()==1
        candidates['predicted_hydrogen_kg_h']=mod.regression('hydrogen',candidates)
        candidates['hydrogen_nonnegative']=candidates['predicted_hydrogen_kg_h'].ge(0)
        candidates['benchmark_required_kw']=candidates['predicted_hydrogen_kg_h']*float(spec['energy_benchmark']['kwh_per_kg'])
        candidates['research_power_comparison_pass']=(candidates['hydrogen_nonnegative'] &
            candidates['benchmark_required_kw'].le(fixed_power))
        fc_support={}
        for k in ['hydrogen_flow_l_min','voltage_v','temperature_c','pressure_bar']:
            s=pd.to_numeric(fcdf[k],errors='coerce').dropna()
            if s.empty:raise PipelineBlocked(f'No FC historical support for {k}')
            fc_support[k]=(float(s.min()),float(s.max()))
        candidates,fc_status=annotate(candidates,spec['fuel_cell'],mod,fc_support)
        candidates['conditional_research_feasible']=(candidates['research_power_comparison_pass'] &
            candidates['empirical_joint_support_pass'] & candidates['fuel_cell_research_eligible'])
        baseline=candidates.iloc[0]
        accepted=candidates[candidates['conditional_research_feasible']].copy()
        ordered=accepted.sort_values('predicted_hydrogen_kg_h',ascending=False)
        best=ordered.head(1)
        # Missing bridge => results are predictions and *cannot* become "feasible".
        if spec['fuel_cell']['bridge_mode']=='BLOCKED_NO_MAPPING':
            assert accepted.empty
        # Never assert that the optimizer has proved a global optimum.
        for name,data in [('all_candidates.csv',candidates),('conditionally_feasible.csv',ordered),
                          ('best_conditional_candidate.csv',best),('baseline.csv',candidates.head(1))]:
            engine.atomic_to_csv(data,output/name,index=False)
        status.update({
            'status':'RESEARCH_CONDITIONAL_RESULT_NOT_PHYSICAL_VALIDATION' if len(best) else
                     ('BLOCKED_FC_BRIDGE_NO_FEASIBLE_RESULT' if spec['fuel_cell']['bridge_mode']=='BLOCKED_NO_MAPPING' else 'NO_RESEARCH_FEASIBLE_CANDIDATE'),
            'candidate_count':int(len(candidates)), 'baseline_predicted_kg_h':float(baseline['predicted_hydrogen_kg_h']),
            'research_power_comparison_pass_count':int(candidates['research_power_comparison_pass'].sum()),
            'empirical_joint_support_pass_count':int(candidates['empirical_joint_support_pass'].sum()),
            'empirical_joint_support_cutoff':float(support_cutoff),
            'empirical_joint_support_is_safety_approval':False,
            'full_four_model_pipeline_executed':power_status['mode']=='HYPOTHETICAL_SAME_SCENARIO_MODEL_PREDICTIONS',
            'power_constraint_verified':False,
            'power_constraint_status':'RESEARCH_SEC_BENCHMARK_ONLY_NOT_MEASURED',
            'fc_fixed_setpoint_cannot_discriminate_candidates':bool(fc_status.get('fc_input_policy')=='CONSTANT_REGULATED_SETPOINT_NON_DISCRIMINATING_CLASSIFIER'),
            'fc_predicted_pass_count':int((candidates['model_fc_class']=='Pass').sum()),
            'conditional_research_feasible_count':int(len(accepted)),
            'best_conditional_predicted_kg_h':float(best.iloc[0]['predicted_hydrogen_kg_h']) if len(best) else None,
            'best_candidate_id':int(best.iloc[0]['candidate_id']) if len(best) else None,
            'baseline_fc_model_class':str(baseline['model_fc_class']),
            'fuel_cell_stage':fc_status,
            'power_comparison':'55-kWh/kg or other documented benchmark, NOT measured electrolyzer consumption',
            'optimization_method':'seeded Latin Hypercube research search; no global optimum certified',
            'ranges':'user-supplied marginal research bounds within observed univariate support; NOT safety limits',
            'fuel_cell_result':'predicted class under separately assumed operating point and regulated draw, not physical proof',
            'research_stage_software_status':'END_TO_END_RESEARCH_IMPLEMENTED_WITH_EXPLICIT_UNVERIFIED_GATES',
            'physical_integration_blockers':[
                'Unverified Solar/Wind scope and shared real-world timestamps',
                'Missing measured electrolyzer energy-consumption function',
                'Missing validated hydrogen-production-to-FC device bridge and independently specified FC operating conditions',
                'Model versions and device-domain generalization not independently verified'],
            'files':['all_candidates.csv','conditionally_feasible.csv','best_conditional_candidate.csv','baseline.csv']})
    except Exception as exc:
        status['status']='BLOCKED_INVALID_OR_MISSING_EVIDENCE'
        status['reason']=f'{type(exc).__name__}: {exc}'
    _json(output/'pipeline_status.json',status)
    print('[PIPELINE] '+status['status'])
    print(f'Report: {output/"pipeline_status.json"}')
    return status
