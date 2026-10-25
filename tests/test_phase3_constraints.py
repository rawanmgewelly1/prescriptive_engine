import tempfile, unittest, json
from pathlib import Path
import pandas as pd
from phase3_constraints import run, non_dominated, validated_rules, ContractError
class Tests(unittest.TestCase):
 def test_empty_gates(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t); pd.DataFrame([{'candidate_id':1,'predicted_hydrogen_production_kg_h':2,'predicted_kg_per_available_kwh_proxy':.01}]).to_csv(p/'c.csv',index=False)
   (p/'r.json').write_text('{"research_constraints":[]}')
   s=run(p/'c.csv',p/'r.json',p/'out');self.assertEqual(s['research_status'],'BLOCKED_NO_REFERENCED_CONSTRAINTS');self.assertEqual(s['passed_rows'],0)
 def test_constraints_and_pareto(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)
   pd.DataFrame([{'candidate_id':1,'temperature_c':50,'predicted_hydrogen_production_kg_h':2,'predicted_kg_per_available_kwh_proxy':.01},
   {'candidate_id':2,'temperature_c':55,'predicted_hydrogen_production_kg_h':1,'predicted_kg_per_available_kwh_proxy':.02},
   {'candidate_id':3,'temperature_c':90,'predicted_hydrogen_production_kg_h':3,'predicted_kg_per_available_kwh_proxy':.03}]).to_csv(p/'c.csv',index=False)
   (p/'r.json').write_text(json.dumps({'research_constraints':[{'name':'t','enabled':True,'column':'temperature_c','min':40,'max':60,'units':'degC','source_reference':'test only','scope':'test','evidence_type':'published_research_range'}]}))
   s=run(p/'c.csv',p/'r.json',p/'out');self.assertEqual((s['passed_rows'],s['rejected_rows'],s['pareto_after_screening']),(2,1,2))
 def test_missing_evidence(self):
  with self.assertRaises(ContractError): validated_rules({'research_constraints':[{'name':'t','enabled':True,'column':'temperature_c','min':40,'max':60,'units':'degC'}]})
 def test_dominated(self):
  d=pd.DataFrame([{'predicted_hydrogen_production_kg_h':2,'predicted_kg_per_available_kwh_proxy':.03},{'predicted_hydrogen_production_kg_h':1,'predicted_kg_per_available_kwh_proxy':.02}]);self.assertEqual(non_dominated(d).tolist(),[True,False])
if __name__=='__main__':unittest.main()
