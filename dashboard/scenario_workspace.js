/* Existing historical research visualizations are intentionally left intact.
 * All new scenario data is handled as text, never interpreted as HTML.
 */
(() => {
  'use strict';
  const el = id => document.getElementById(id);
  const VARS = ['voltage_v','temperature_c','current_density_a_cm2',
    'electrolyte_concentration_wt_pct','water_flow_l_min','pressure_bar'];
  const DISPLAY = {'voltage_v':'Voltage (V)','temperature_c':'Temperature (°C)',
    'current_density_a_cm2':'Current density (A/cm²)',
    'electrolyte_concentration_wt_pct':'Electrolyte concentration (%)',
    'water_flow_l_min':'Water flow (L/min)','pressure_bar':'Pressure (bar)'};
  const FC_KEYS = ['voltage_v','temperature_c','pressure_bar'];
  let templates = [], solarDefaults = {}, windDefaults = {}, fcDefaults = {};

  async function request(url, options) {
    const result = await fetch(url, options);
    let data;
    try { data = await result.json(); } catch (_) { throw new Error('The local Python service returned an invalid response.'); }
    if (!result.ok) throw new Error(data.error || `HTTP ${result.status}`);
    return data;
  }
  function textNode(tag, text, className) {
    const node = document.createElement(tag);
    node.textContent = text;
    if (className) node.className = className;
    return node;
  }
  function clear(node) { node.replaceChildren(); }
  function makeNumber(id, value, labelText) {
    const label = document.createElement('label');
    if (labelText) label.append(textNode('span', labelText));
    const input = document.createElement('input');
    input.id = id; input.type = 'number'; input.step = 'any';
    input.value = String(value);
    input.required = true;
    label.append(input);
    return label;
  }
  function buildFeatureFields(containerId, prefix, fields) {
    const holder = el(containerId); clear(holder);
    Object.entries(fields).forEach(([name, value]) => holder.append(makeNumber(`${prefix}-${name}`, value, name)));
  }
  function buildDecisionFields(spec) {
    const holder = el('ws-decision-fields'); clear(holder);
    VARS.forEach(name => {
      const row = textNode('div', '', 'ws-bound-row');
      row.append(textNode('span', DISPLAY[name], 'ws-bound-label'));
      row.append(makeNumber(`ws-base-${name}`, spec.baseline[name]));
      row.append(makeNumber(`ws-min-${name}`, spec.bounds[name][0]));
      row.append(makeNumber(`ws-max-${name}`, spec.bounds[name][1]));
      holder.append(row);
    });
  }
  function updateModes() {
    const isModel = el('ws-power-mode').value === 'model_scenario';
    el('ws-model-wrap').classList.toggle('ws-hidden', !isModel);
    el('ws-fixed-wrap').classList.toggle('ws-hidden', isModel);
    const mode = el('ws-fc-mode').value;
    el('ws-fc-fields').classList.toggle('ws-hidden', mode === 'BLOCKED_NO_MAPPING');
    el('ws-fc-draw-wrap').classList.toggle('ws-hidden', mode !== 'ASSUMED_REGULATED_DRAW');
  }
  async function loadTemplate(name) {
    const response = await request('/api/template?name=' + encodeURIComponent(name));
    const spec = response.scenario;
    const power = spec.renewable_power;
    el('ws-power-mode').value = power.mode;
    el('ws-fixed-power').value = power.available_kw ?? 200;
    buildFeatureFields('ws-solar-fields','ws-solar',power.solar_inputs || solarDefaults);
    buildFeatureFields('ws-wind-fields','ws-wind',power.wind_inputs || windDefaults);
    el('ws-candidates').value = String(spec.optimization.candidate_count);
    el('ws-seed').value = String(spec.optimization.seed);
    buildDecisionFields(spec);
    const fc = spec.fuel_cell;
    el('ws-fc-mode').value = fc.bridge_mode;
    el('ws-fc-draw').value = fc.flow_demand_l_min ?? fcDefaults.flow_demand_l_min;
    buildFeatureFields('ws-fc-point-fields','ws-fc',fc.operating_point || fcDefaults.operating_point);
    el('ws-power-ack').checked = false;
    el('ws-fc-ack').checked = false;
    el('ws-research-ack').checked = false;
    updateModes();
    el('ws-connection').textContent = 'Connected to local Python service. Template loaded; adjust inputs and acknowledge the research limitations before running.';
  }
  function numeric(id) {
    const input = el(id);
    if (!input || input.value.trim() === '' || !Number.isFinite(Number(input.value))) {
      throw new Error('Enter a finite number for ' + id);
    }
    return Number(input.value);
  }
  function getFeatureRecord(prefix, spec) {
    const out = {};
    Object.keys(spec).forEach(name => { out[name] = numeric(`${prefix}-${name}`); });
    return out;
  }
  function buildRequest() {
    const baseline = {}, bounds = {};
    VARS.forEach(name => {
      baseline[name] = numeric('ws-base-' + name);
      bounds[name] = [numeric('ws-min-' + name), numeric('ws-max-' + name)];
    });
    const candidateCount = numeric('ws-candidates');
    const seed = numeric('ws-seed');
    if (!Number.isInteger(candidateCount) || !Number.isInteger(seed)) {
      throw new Error('Candidate count and random seed must be integers.');
    }
    return {
      template:el('ws-template').value,
      acknowledge_research:el('ws-research-ack').checked,
      power_mode:el('ws-power-mode').value,
      fixed_power_kw:numeric('ws-fixed-power'),
      acknowledge_hypothetical_power:el('ws-power-ack').checked,
      solar_inputs:getFeatureRecord('ws-solar', solarDefaults),
      wind_inputs:getFeatureRecord('ws-wind', windDefaults),
      baseline, bounds, candidate_count:candidateCount, seed,
      fc_mode:el('ws-fc-mode').value,
      acknowledge_assumed_fc:el('ws-fc-ack').checked,
      fc_flow_demand_l_min:numeric('ws-fc-draw'),
      fc_operating_point:getFeatureRecord('ws-fc', fcDefaults.operating_point)
    };
  }
  function format(value, decimals = 3) {
    const v = Number(value);
    return value === null || value === undefined || value === '' || !Number.isFinite(v) ? '—' : v.toFixed(decimals);
  }
  function metric(name, value) {
    const box = textNode('div','','ws-metric');
    box.append(textNode('span', name, 'muted'));
    box.append(textNode('b', value));
    return box;
  }
  function table(columns, rows) {
    const wrap = textNode('div','','ws-table');
    const table = document.createElement('table'), thead = document.createElement('thead'),
      hrow = document.createElement('tr'), tbody = document.createElement('tbody');
    columns.forEach(c => hrow.append(textNode('th',c)));
    thead.append(hrow);
    rows.forEach(row => {
      const tr = document.createElement('tr');
      row.forEach(cell => tr.append(textNode('td',String(cell))));
      tbody.append(tr);
    });
    table.append(thead,tbody); wrap.append(table); return wrap;
  }
  function showResult(data) {
    const status = data.status, baseline = data.baseline, best = data.best;
    el('ws-results').classList.remove('ws-hidden');
    el('ws-result-heading').textContent = 'Saved scenario — ' + data.run_id;
    el('ws-result-status').textContent = 'Status: ' + status.status + '\nScientific status: ' + status.scientific_status +
       '\nPower source: ' + (status.renewable_stage?.mode || 'Unavailable') +
       '\nPhysical safety approved: NO. Candidate-level physical Fuel Cell bridge verified: NO.' +
       (status.reason ? '\nBlocker: ' + status.reason : '');
    const metrics = el('ws-metrics'); clear(metrics);
    const delta = Number(status.best_conditional_predicted_kg_h) - Number(status.baseline_predicted_kg_h);
    const gainPct = status.best_conditional_predicted_kg_h == null || !Number.isFinite(delta) || Number(status.baseline_predicted_kg_h) <= 0
      ? '—' : format(100*delta/Number(status.baseline_predicted_kg_h),2) + '%';
    [['Fixed power',format(status.fixed_available_power_kw)+' kW'],
      ['Baseline H₂',format(status.baseline_predicted_kg_h)+' kg/h'],
      ['Best conditional H₂',format(status.best_conditional_predicted_kg_h)+' kg/h'],
      ['Predicted change',gainPct],
      ['Candidates',String(status.candidate_count ?? '—')],
      ['Empirical support passes',String(status.empirical_joint_support_pass_count ?? '—')],
      ['Conditional research eligible',String(status.conditional_research_feasible_count ?? '—')],
      ['Model FC Pass predictions',String(status.fc_predicted_pass_count ?? '—')]].forEach(([k,v]) => metrics.append(metric(k,v)));
    const plot=el('ws-production-chart');clear(plot);
    const plotMax=Math.max(0,Number(status.baseline_predicted_kg_h)||0,
      Number(status.best_conditional_predicted_kg_h)||0);
    [['Baseline',status.baseline_predicted_kg_h,false],
      ['Best conditional',status.best_conditional_predicted_kg_h,true]].forEach(([name, val, optimized])=>{
      const line=textNode('div','','ws-bar-line'), track=textNode('div','','ws-track');
      const fill=textNode('div','',optimized?'ws-fill ws-optimized':'ws-fill');
      const amount=Number(val);
      fill.style.width=plotMax>0 && Number.isFinite(amount)? Math.max(0,Math.min(100,amount/plotMax*100))+'%' : '0%';
      track.append(fill);line.append(textNode('span',name),track,textNode('span',format(val)+' kg/h'));plot.append(line);
    });
    const comparison = el('ws-comparison');clear(comparison);
    const cols = ['Input / metric','Baseline','Best conditional'];
    const rows = [['Predicted hydrogen (kg/h)',format(baseline?.predicted_hydrogen_kg_h),format(best?.predicted_hydrogen_kg_h)],
      ['Available renewable power (kW)',format(baseline?.renewable_power_kw),format(best?.renewable_power_kw)]];
    VARS.forEach(k=>rows.push([DISPLAY[k],format(baseline?.[k]),format(best?.[k])]));
    rows.push(['Fuel Cell status',baseline?.fuel_cell_status || '—',best?.fuel_cell_status || '—']);
    rows.push(['Empirical joint support',baseline?.empirical_joint_support_pass || '—',best?.empirical_joint_support_pass || '—']);
    comparison.append(table(cols,rows));
    const top = el('ws-top');clear(top);
    const candidates = data.top_candidates || [];
    top.append(candidates.length ? table(['Candidate','Predicted H₂ (kg/h)','Power comparison','FC class','Conditional'],
      candidates.map(x=>[x.candidate_id,format(x.predicted_hydrogen_kg_h),x.research_power_comparison_pass,x.model_fc_class,x.conditional_research_feasible]))
      : textNode('p','No candidate satisfied all research screening gates. This is not an engineering failure/success verdict.','warning'));
    const downloads = el('ws-downloads');clear(downloads);
    (data.downloadable||[]).forEach(name=>{
      const link = textNode('a','Download '+name);
      link.href = '/api/download/'+encodeURIComponent(data.run_id)+'/'+encodeURIComponent(name);
      link.download = name;
      downloads.append(link);
    });
    el('ws-disclosure').textContent = 'MODEL PREDICTIONS ONLY. Power comparison is an illustrative kWh/kg benchmark, not measured electrolyzer consumption. ' +
      'Empirical feature support does not certify safe operating bounds. Fuel Cell decisions depend on research-only flow assumptions unless blocked. ' +
      'Solar/Wind hypothetical combinations do not prove temporal synchronization. Model version compatibility is not independently certified.';
  }
  async function refreshHistory() {
    const result = await request('/api/runs');
    const select = el('ws-run-history'), current = select.value;clear(select);
    const def = document.createElement('option'); def.value='';def.textContent='Select a saved run';select.append(def);
    result.runs.forEach(run=>{
      const option = document.createElement('option');option.value=run.run_id;
      option.textContent=run.run_id+' • '+run.status;select.append(option);
    });
    if ([...select.options].some(o=>o.value===current)) select.value=current;
  }
  async function runScenario() {
    const button=el('ws-run');
    try {
      const payload=buildRequest();
      button.disabled=true;
      el('ws-connection').textContent='Running the local models and constrained research search…';
      const result=await request('/api/run',{
        method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)
      });
      showResult(result);
      await refreshHistory();
      el('ws-run-history').value=result.run_id;
      el('ws-connection').textContent='Run saved successfully. The status below distinguishes research eligibility from physical verification.';
      el('ws-results').scrollIntoView({behavior:'smooth',block:'start'});
    } catch(error) {
      el('ws-connection').textContent='Run rejected or failed: '+error.message;
    } finally {button.disabled=false;}
  }
  async function initialize() {
    try {
      const response=await request('/api/templates');
      templates=response.templates;
      const [modelRef,fcRef]=await Promise.all([
        request('/api/template?name=RESEARCH_FOUR_MODELS_HYPOTHETICAL.json'),
        request('/api/template?name=RESEARCH_DEMO_200KW.json')
      ]);
      solarDefaults=modelRef.scenario.renewable_power.solar_inputs;
      windDefaults=modelRef.scenario.renewable_power.wind_inputs;
      fcDefaults=fcRef.scenario.fuel_cell;
      const selector=el('ws-template');clear(selector);
      templates.forEach(name=>{
        const option=document.createElement('option');option.value=name;
        option.textContent=name.replace(/\.json$/,'').replaceAll('_',' ');
        selector.append(option);
      });
      await loadTemplate(templates[0]);
      await refreshHistory();
      selector.addEventListener('change',async()=>{
        try {await loadTemplate(selector.value);} catch(error){el('ws-connection').textContent=error.message;}
      });
      el('ws-power-mode').addEventListener('change',updateModes);
      el('ws-fc-mode').addEventListener('change',updateModes);
      el('ws-run').addEventListener('click',runScenario);
      el('ws-run-history').addEventListener('change',async()=>{
        if (!el('ws-run-history').value) return;
        try {showResult(await request('/api/run/'+encodeURIComponent(el('ws-run-history').value)));}
        catch(error){el('ws-connection').textContent=error.message;}
      });
    } catch(error) {
      el('ws-connection').textContent='Dashboard service unavailable: '+error.message+' Use python run_dashboard.py, not file://.';
      el('ws-run').disabled=true;
    }
  }
  initialize();
})();
