"""Run the original seven-stage research scenario pipeline; Python models require trusted local artifacts."""
import argparse
from pathlib import Path
from prescriptive_engine.pipeline import run_scenario

def main():
    ap=argparse.ArgumentParser(description='Fixed-power research prescriptive optimization')
    ap.add_argument('--scenario',required=True)
    ap.add_argument('--config',default='config.json')
    ap.add_argument('--out')
    args=ap.parse_args()
    result=run_scenario(Path(args.scenario),config_path=Path(args.config),output_dir=args.out)
    if result['status']=='BLOCKED_INVALID_OR_MISSING_EVIDENCE': raise SystemExit(2)

if __name__=='__main__':main()
