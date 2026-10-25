import unittest, pathlib, tempfile, csv
from build_dashboard import build
class DashboardTests(unittest.TestCase):
 def test_real_data(self):
  root=pathlib.Path('/mnt/data')
  if not (root/'available_power_proxy_candidates.csv').exists(): self.skipTest('Real data not available')
  with tempfile.TemporaryDirectory() as t:
   o=pathlib.Path(t)/'dashboard.html'
   build(root/'available_power_proxy_candidates.csv',root/'pareto_hydrogen_results/pareto_frontier.csv',root/'research_scenario_summary.csv',o)
   text=o.read_text()
   self.assertIn('Green Hydrogen',text)
   self.assertIn('"pareto":[2021,1053',text)
   self.assertIn('"candidate_id":2021',text)
 def test_rejects_bad_frontier(self):
  with tempfile.TemporaryDirectory() as t:
   t=pathlib.Path(t); a=t/'a.csv';p=t/'p.csv';s=t/'s.csv'
   fields=['candidate_id','renewable_power_kw','predicted_hydrogen_production_kg_h','predicted_kg_per_available_kwh_proxy','voltage_v','temperature_c','current_density_a_cm2','electrolyte_concentration_wt_pct','water_flow_l_min','pressure_bar','available_power_scenario']
   with a.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(dict.fromkeys(fields,1))
   with p.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(dict.fromkeys(fields,2))
   with s.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['scenario_id','candidate_count','power_min_kw','power_max_kw','max_predicted_hydrogen_kg_h']);w.writeheader();w.writerow(dict(scenario_id=1,candidate_count=1,power_min_kw=1,power_max_kw=1,max_predicted_hydrogen_kg_h=1))
   with self.assertRaises(ValueError): build(a,p,s,t/'out.html')
if __name__=='__main__':unittest.main()
