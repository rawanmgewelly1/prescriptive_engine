import tempfile, unittest, json, subprocess, sys
from pathlib import Path
import pandas as pd

ENGINE=Path(__file__).resolve().parents[1]/'prescriptive_engine'/'engine.py'

class Phase3CLI(unittest.TestCase):
    def test_no_rules_blocks_and_keeps_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'data').mkdir(); (p/'models').mkdir(); (p/'selected_features').mkdir(); (p/'outputs').mkdir()
            config=json.loads((Path(__file__).resolve().parents[1]/'reference_config.json').read_text(encoding='utf-8'))
            (p/'config.json').write_text(json.dumps(config),encoding='utf-8')
            pd.DataFrame([{'candidate_id':1,'predicted_hydrogen_production_kg_h':1.1,'predicted_kg_per_available_kwh_proxy':0.01}]).to_csv(p/'sample.csv',index=False)
            (p/'rules.json').write_text('{"research_constraints": []}',encoding='utf-8')
            result=subprocess.run([sys.executable,str(ENGINE),'--config',str(p/'config.json'),'--phase3-rules',str(p/'rules.json'),'--candidate-file',str(p/'sample.csv'),'--research-out',str(p/'results')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('BLOCKED_NO_REFERENCED_CONSTRAINTS',result.stdout)
            self.assertEqual(json.loads((p/'results'/'phase3_status.json').read_text())['physical_operating_approval'],False)
    def test_refuses_competing_modes(self):
        result=subprocess.run([sys.executable,str(ENGINE),'--phase3-rules','x','--research-power-kw','200'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('cannot be combined',result.stderr)
