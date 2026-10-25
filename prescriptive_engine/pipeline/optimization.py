"""Fixed-power single-objective search; research candidate support ≠ safe limits."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import qmc
from .contracts import DECISION_VARS, PipelineBlocked

def make_candidates(spec,observations,fixed_power):
    n=spec['optimization']['candidate_count']
    bounds=np.array([spec['bounds'][k] for k in DECISION_VARS],dtype=float)
    baseline=np.array([spec['baseline'][k] for k in DECISION_VARS],dtype=float)
    for j,k in enumerate(DECISION_VARS):
        obs=pd.to_numeric(observations[k],errors='coerce').to_numpy(dtype=float)
        obs=obs[np.isfinite(obs)]
        if len(obs)<10: raise PipelineBlocked(f'Not enough observed support for {k}')
        if bounds[j,0] < obs.min() or bounds[j,1] > obs.max():
            raise PipelineBlocked(f'{k} scenario research range outside observed min/max')
    lhs=qmc.LatinHypercube(d=6,seed=spec['optimization']['seed'])
    generated=qmc.scale(lhs.random(n),bounds[:,0],bounds[:,1])
    combined=np.vstack([baseline,generated])
    out=pd.DataFrame(combined,columns=DECISION_VARS)
    out.insert(0,'candidate_id',np.arange(len(out)))
    out['renewable_power_kw']=float(fixed_power)
    out['origin']=np.where(out['candidate_id']==0,'BASELINE','LHS_RESEARCH_SEARCH')
    # The six marginal intervals do not prove multivariate interpolation or safe operation.
    return out


def empirical_joint_support(candidates, observations, *, quantile=0.95):
    """Research-only 7-D nearest-neighbor gate, NOT a physical safe envelope.

    Derive the cutoff from each observed row's nearest OTHER observed neighbor.
    The renewable-power dimension is included to avoid accepting extrapolated
    fixed-power conditions merely because six individual variables are in range.
    """
    from scipy.spatial import cKDTree
    cols=['renewable_power_kw',*DECISION_VARS]
    observed=observations.loc[:,cols].apply(pd.to_numeric,errors='coerce')
    observed=observed.replace([np.inf,-np.inf],np.nan).dropna().to_numpy(float)
    if len(observed)<20:raise PipelineBlocked('Insufficient joint observed support')
    lows=np.quantile(observed,0.25,axis=0)
    highs=np.quantile(observed,0.75,axis=0)
    scale=np.where(highs>lows,highs-lows,1.0)
    tree=cKDTree(observed/scale)
    nearest_nonself=tree.query(observed/scale,k=2)[0][:,1]
    cutoff=float(np.quantile(nearest_nonself,quantile))
    candidate_values=candidates.loc[:,cols].to_numpy(float)
    if not np.isfinite(candidate_values).all():raise PipelineBlocked('Invalid joint candidate values')
    distance=tree.query(candidate_values/scale,k=1)[0]
    return distance<=cutoff, distance, cutoff
