"""Build an offline, research-only HTML dashboard from reviewed CSV outputs."""
import argparse, json, csv, html, pathlib, sys

def load_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def number(row, field):
    try:
        value=float(row[field]);
        if not __import__('math').isfinite(value): raise ValueError('nonfinite')
        return value
    except (KeyError, ValueError, TypeError) as ex: raise ValueError(f'Invalid {field} in CSV: {ex}')

def build(candidates, frontier, scenarios, output):
    a=load_csv(candidates); p=load_csv(frontier); s=load_csv(scenarios)
    if not a or not p or not s: raise ValueError('Input CSV empty')
    fields=['candidate_id','renewable_power_kw','predicted_hydrogen_production_kg_h','predicted_kg_per_available_kwh_proxy','voltage_v','temperature_c','current_density_a_cm2','electrolyte_concentration_wt_pct','water_flow_l_min','pressure_bar','available_power_scenario']
    def take(row):
        return {k: (int(number(row,k)) if k in ('candidate_id','available_power_scenario') else number(row,k)) for k in fields}
    data=[take(r) for r in a]; ps=[take(r) for r in p]
    if len({x['candidate_id'] for x in data})!=len(data): raise ValueError('Duplicate candidate identifiers')
    ids={r['candidate_id'] for r in data}
    if not all(r['candidate_id'] in ids for r in ps): raise ValueError('Pareto points not found in candidates')
    # independently verify dominance in both maximization objectives
    objectives=lambda x: (x['predicted_hydrogen_production_kg_h'],x['predicted_kg_per_available_kwh_proxy'])
    true_p=[]
    for x in data:
        ax,bx=objectives(x)
        if not any((y['predicted_hydrogen_production_kg_h']>=ax-1e-12 and y['predicted_kg_per_available_kwh_proxy']>=bx-1e-12 and (y['predicted_hydrogen_production_kg_h']>ax+1e-12 or y['predicted_kg_per_available_kwh_proxy']>bx+1e-12)) for y in data):true_p.append(x['candidate_id'])
    if set(true_p)!={x['candidate_id'] for x in ps}:raise ValueError('Pareto CSV differs from independently calculated frontier')
    scen=[]
    for row in s:scen.append({'scenario':int(number(row,'scenario_id')),'count':int(number(row,'candidate_count')),'minimum':number(row,'power_min_kw'),'maximum':number(row,'power_max_kw'),'maxproduction':number(row,'max_predicted_hydrogen_kg_h')})
    if sum(x['count'] for x in scen)!=len(data):raise ValueError('Scenario counts do not match candidates')
    payload=json.dumps({'candidates':data,'pareto':list(true_p),'scenarios':scen},ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    template=(pathlib.Path(__file__).parent / 'dashboard' / 'dashboard_template.html').read_text(encoding='utf-8')
    result=template.replace('__EMBED_DATA__',payload)
    output=pathlib.Path(output);output.parent.mkdir(parents=True, exist_ok=True);output.write_text(result,encoding='utf-8')
    print(f'Dashboard created: {output.resolve()} | candidates={len(data)} | pareto={len(ps)} | scenarios={len(scen)}')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Build offline research dashboard without changing engine')
    parser.add_argument('--candidates',default='available_power_analysis/available_power_proxy_candidates.csv')
    parser.add_argument('--pareto',default='pareto_hydrogen_results/pareto_frontier.csv')
    parser.add_argument('--scenarios',default='research_scenarios/research_scenario_summary.csv')
    parser.add_argument('--out',default='dashboard/research_dashboard.html')
    v=parser.parse_args()
    try: build(v.candidates,v.pareto,v.scenarios,v.out)
    except Exception as e: print('ERROR:',e,file=sys.stderr);sys.exit(1)
