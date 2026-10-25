"""Reuses verified feature-order/scaler contracts from the legacy engine."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .contracts import PipelineBlocked
from prescriptive_engine import engine

class Models:
    def __init__(self, cfg):
        self.cfg=cfg
        self._models={}
        self._scalers={}
    def _get(self,kind):
        if kind not in self._models:
            model_path=Path(self.cfg['models_dir'])/self.cfg[f'{kind}_model_file']
            if not model_path.is_file(): raise PipelineBlocked(f'MISSING_MODEL_ARTIFACT: {model_path.name}')
            self._models[kind]=engine.load_pickle(model_path)
            mode=self.cfg['input_modes'][kind]
            if mode=='scaled':
                scaler_path=Path(self.cfg['models_dir'])/self.cfg[f'{kind}_scaler_file']
                if not scaler_path.is_file():raise PipelineBlocked(f'Missing scaler: {scaler_path.name}')
                self._scalers[kind]=engine.load_pickle(scaler_path)
            else: self._scalers[kind]=None
        return self._models[kind],self._scalers[kind]
    def features(self,kind):
        return engine.load_feature_file(self.cfg,kind,self.cfg[kind+'_features'])
    def prepared(self,kind,frame):
        model,scaler=self._get(kind)
        cfg=self.cfg
        return model,engine.prepare_input(
            frame, model, self.features(kind),scaler,cfg['input_modes'][kind],kind,
            scaler_order_verified=cfg.get('scaler_feature_order_verified',{}).get(kind,False),
            model_order_verified=cfg.get('model_feature_order_verified',{}).get(kind,False),
            scaler_order_reference=cfg.get('scaler_feature_order_reference',{}).get(kind,''),
            model_order_reference=cfg.get('model_feature_order_reference',{}).get(kind,''),
            scaler_feature_order=cfg.get('scaler_feature_order',{}).get(kind))
    def regression(self,kind,frame):
        model,X=self.prepared(kind,frame)
        a=np.asarray(model.predict(X),dtype=float).reshape(-1)
        if len(a)!=len(frame) or not np.isfinite(a).all():
            raise PipelineBlocked(f'{kind} returned invalid predictions')
        return a
    def fuel_cell(self,frame):
        model,X=self.prepared('fuel_cell',frame)
        a=np.asarray(model.predict(X)).reshape(-1)
        if len(a)!=len(frame): raise PipelineBlocked('Fuel-cell prediction shape mismatch')
        m=self.cfg['fuel_cell_pass_constraint']
        if m.get('class_mapping_verified') is not True or not m.get('class_mapping_reference'):
            raise PipelineBlocked('Fuel-cell class label mapping is not verified')
        labelmap=m['class_label_mapping']
        if any(str(v) not in labelmap for v in a):raise PipelineBlocked('Unmapped fuel-cell class output')
        return [labelmap[str(v)] for v in a]
