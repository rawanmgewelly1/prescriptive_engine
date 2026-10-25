"""One command research-only closure gate: four models + three FC modes + tests.

Never reports approval for physical devices or verified engineering feasibility.
"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prescriptive_engine.pipeline import run_scenario
from tools.check_model_prediction_consistency import main as check_models


def main():
    reports={}
    ok=check_models()==0
    scenarios={
        'four_models':'RESEARCH_FOUR_MODELS_HYPOTHETICAL',
        'regulated_draw':'RESEARCH_DEMO_200KW',
        'direct_flow_support_gate':'RESEARCH_FC_DIRECT_FLOW_SUPPORT_CHECK',
        'strict_no_bridge':'STRICT_NO_FC_BRIDGE',
    }
    for name,scenario in scenarios.items():
        result=run_scenario(ROOT/'configs'/'scenarios'/f'{scenario}.json',config_path=ROOT/'config.json')
        reports[name]=result
        if result.get('status')=='BLOCKED_INVALID_OR_MISSING_EVIDENCE':ok=False
    four=reports['four_models'];regulated=reports['regulated_draw']
    direct=reports['direct_flow_support_gate'];strict=reports['strict_no_bridge']
    gates={
        'four_models_executed':four.get('full_four_model_pipeline_executed') is True,
        'nonzero_fixed_model_power':four.get('fixed_available_power_kw',0)>0,
        'research_solution_generated':four.get('conditional_research_feasible_count',0)>0,
        'four_model_joint_support_gate_used':four.get('empirical_joint_support_pass_count',0)>0,
        'fc_regulated_setpoint_non_discriminating_disclosed':regulated.get('fc_fixed_setpoint_cannot_discriminate_candidates') is True,
        'direct_fc_out_of_support_rejected':direct.get('fc_predicted_pass_count')==0 and direct.get('conditional_research_feasible_count')==0,
        'no_bridge_no_false_pass':strict.get('conditional_research_feasible_count')==0,
        'physical_approval_explicitly_false':all(v.get('physical_safety_approved') is False for v in reports.values()),
        'power_physically_verified_false':all(v.get('power_constraint_verified') is False for v in reports.values()),
    }
    ok=ok and all(gates.values())
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-q'],cwd=ROOT,
                         capture_output=True,text=True)
    gates['full_unit_test_suite_passed']=tests.returncode==0
    ok=ok and tests.returncode==0
    out={'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
         'status':'RESEARCH_SOFTWARE_STAGE_ACCEPTED_WITH_DISCLOSED_LIMITATIONS' if ok else 'RESEARCH_SOFTWARE_STAGE_NOT_ACCEPTED',
         'physical_engineering_approved':False,'gates':gates,
         'model_spot_check':str(ROOT/'outputs/model_compatibility_status.json'),
         'four_model_status':four.get('status'),'four_model_power_kw':four.get('fixed_available_power_kw'),
         'four_model_best_predicted_kg_h':four.get('best_conditional_predicted_kg_h'),
         'full_four_model_pipeline_verified':False,
         'actual_device_integration_blockers':four.get('physical_integration_blockers',[]),
         'unit_test_tail':'\n'.join((tests.stdout+'\n'+tests.stderr).splitlines()[-15:])}
    target=ROOT/'outputs/RESEARCH_STAGE_ACCEPTANCE.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(out,indent=2)+'\n')
    print('[STAGE ACCEPTANCE]',out['status'])
    for gate,value in gates.items():print('PASS' if value else 'FAIL',gate)
    print('Report:',target)
    return 0 if ok else 2

if __name__=='__main__':raise SystemExit(main())
