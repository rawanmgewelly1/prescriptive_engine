"""Compatibility entrypoint: one implementation in prescriptive_engine.phase3_constraints."""
from prescriptive_engine.phase3_constraints import (ContractError, FEATURE_UNITS,
    OUTPUT_PRODUCTION, OUTPUT_PROXY, validated_rules, non_dominated, run)

if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--rules', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.candidates, args.rules, args.out), indent=2))
