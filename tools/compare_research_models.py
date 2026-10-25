"""Research-only model comparison. No production models are loaded or written.
Existing holdout was already viewed; this is NOT a fresh blind test.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, balanced_accuracy_score, recall_score, precision_score, confusion_matrix
from catboost import CatBoostRegressor
from xgboost import XGBClassifier

SEED=42

def regression_metrics(y,p):
    return {'MAE':float(mean_absolute_error(y,p)),'RMSE':float(np.sqrt(mean_squared_error(y,p))),'R2':float(r2_score(y,p))}

def classification_metrics(y,p):
    return {'balanced_accuracy':float(balanced_accuracy_score(y,p)), 'F1_Pass':float(f1_score(y,p,pos_label=1,zero_division=0)), 'recall_Pass':float(recall_score(y,p,pos_label=1,zero_division=0)), 'precision_Pass':float(precision_score(y,p,pos_label=1,zero_division=0)), 'recall_Fail':float(recall_score(y,p,pos_label=0,zero_division=0)), 'precision_Fail':float(precision_score(y,p,pos_label=0,zero_division=0)), 'false_negative_Fail_as_Pass':int(((np.asarray(y)==0)&(np.asarray(p)==1)).sum()),'n':int(len(y))}

class ResearchPredictor:
    def __init__(self, model, k, classification=False):
        self.scaler=StandardScaler()
        self.selector=SelectKBest(f_classif if classification else f_regression,k=k)
        self.model=model
    def fit(self,x,y):
        scaled=self.scaler.fit_transform(x)
        selected=self.selector.fit_transform(scaled,y)
        self.model.fit(selected,y)
        return self
    def transform(self,x):
        return self.selector.transform(self.scaler.transform(x))
    def predict(self,x):
        return self.model.predict(self.transform(x))
    def predict_proba(self,x):
        return self.model.predict_proba(self.transform(x))

def pipeline(model,k,classification=False):
    return ResearchPredictor(model,k,classification)

def run_hydrogen(path,out):
    df=pd.read_csv(path)
    required={'timestamp','hydrogen_production_kg_h','sample_id','efficiency_pct'}
    if not required.issubset(df):raise ValueError('Missing Hydrogen fields '+str(required-set(df)))
    df['timestamp']=pd.to_datetime(df['timestamp'],format='%m/%d/%Y %H:%M',errors='raise')
    if df['timestamp'].duplicated().any():raise ValueError('Duplicate timestamp')
    df=df.sort_values('timestamp').reset_index(drop=True)
    cols=[c for c in df.select_dtypes(include='number') if c not in ['hydrogen_production_kg_h','sample_id','efficiency_pct']]
    x=df[cols];y=df['hydrogen_production_kg_h']
    if not np.isfinite(x.to_numpy()).all() or not np.isfinite(y.to_numpy()).all(): raise ValueError('Hydrogen nonfinite')
    a=int(len(df)*.6);b=int(len(df)*.8)
    models={'RandomForest':RandomForestRegressor(n_estimators=120,min_samples_leaf=2,random_state=SEED,n_jobs=-1),'CatBoost':CatBoostRegressor(iterations=250,depth=6,learning_rate=.05,loss_function='RMSE',verbose=False,thread_count=2,random_seed=SEED,allow_writing_files=False)}
    selection=[]
    for name,m in models.items():
        p=pipeline(m,min(6,len(cols)));p.fit(x.iloc[:a],y.iloc[:a]);pred=p.predict(x.iloc[a:b]);selection.append({'model':name,**{'validation_'+k:v for k,v in regression_metrics(y.iloc[a:b],pred).items()}})
    pd.DataFrame(selection).to_csv(out/'hydrogen_validation_comparison.csv',index=False)
    winner=min(selection,key=lambda d:d['validation_MAE'])['model']
    p=pipeline(models[winner],min(6,len(cols)));p.fit(x.iloc[:b],y.iloc[:b]); pred=p.predict(x.iloc[b:])
    result={'selected_on_validation':winner,**{'test_'+k:v for k,v in regression_metrics(y.iloc[b:],pred).items()},'train_rows':a,'validation_rows':b-a,'test_rows':len(df)-b,'test_start':str(df['timestamp'].iloc[b]),'test_end':str(df['timestamp'].iloc[-1]),'research_status':'REUSED_HOLDOUT_ALREADY_VIEWED_NOT_PRODUCTION_VALIDATION'}
    pd.DataFrame([result]).to_csv(out/'hydrogen_selected_holdout.csv',index=False)
    return result

def run_fuel(path,out):
    df=pd.read_csv(path)
    cols=['voltage_v','temperature_c','pressure_bar','hydrogen_flow_l_min']
    if not {'test_id','outcome',*cols}.issubset(df):raise ValueError('Fuel Cell fields missing')
    groups=df.groupby('test_id',sort=True)['outcome']
    if (groups.nunique(dropna=False)!=1).any():raise ValueError('Inconsistent label within test')
    gs=groups.first()
    if not gs.isin(['Fail','Pass']).all():raise ValueError('Invalid labels')
    dev_ids,test_ids=train_test_split(gs.index.to_numpy(),test_size=.2,stratify=gs.to_numpy(),random_state=42)
    train_ids,val_ids=train_test_split(dev_ids,test_size=.25,stratify=gs.loc[dev_ids].to_numpy(),random_state=43)
    tr=df.test_id.isin(train_ids);va=df.test_id.isin(val_ids);te=df.test_id.isin(test_ids);dev=tr|va
    x=df[cols];y=df.outcome.map({'Fail':0,'Pass':1}).astype(int)
    if not np.isfinite(x.to_numpy()).all():raise ValueError('Fuel nonfinite')
    models={'RandomForest':RandomForestClassifier(n_estimators=120,min_samples_leaf=2,random_state=SEED,n_jobs=-1),'XGBoost':XGBClassifier(n_estimators=170,max_depth=4,learning_rate=.06,subsample=.9,colsample_bytree=.9,random_state=SEED,n_jobs=2,eval_metric='logloss',tree_method='hist')}
    selection=[]
    for name,m in models.items():
        p=pipeline(m,'all',True);p.fit(x.loc[tr],y.loc[tr]);pred=p.predict(x.loc[va]);selection.append({'model':name,**{'validation_'+k:v for k,v in classification_metrics(y.loc[va],pred).items()}})
    pd.DataFrame(selection).to_csv(out/'fuel_cell_validation_comparison.csv',index=False)
    winner=max(selection,key=lambda d:(d['validation_balanced_accuracy'],d['validation_recall_Fail'],d['validation_F1_Pass']))['model']
    p=pipeline(models[winner],'all',True);p.fit(x.loc[dev],y.loc[dev]);pred=p.predict(x.loc[te]); probs=p.predict_proba(x.loc[te])[:,1]
    row={'selected_on_validation':winner,**{'test_'+k:v for k,v in classification_metrics(y.loc[te],pred).items()},'train_tests':len(train_ids),'validation_tests':len(val_ids),'test_tests':len(test_ids),'research_status':'REUSED_HOLDOUT_ALREADY_VIEWED_NOT_PRODUCTION_VALIDATION'}
    pd.DataFrame([row]).to_csv(out/'fuel_cell_selected_holdout_rows.csv',index=False)
    # A priori research rule: average row Pass probability >= .5 => predicted Pass for the entire test.
    group=pd.DataFrame({'test_id':df.loc[te,'test_id'].to_numpy(),'actual':y.loc[te].to_numpy(),'predicted_pass_probability':probs}).groupby('test_id').agg(actual=('actual','first'),mean_pass_probability=('predicted_pass_probability','mean'),rows=('actual','size'))
    group['predicted']=(group.mean_pass_probability>=.5).astype(int)
    group.to_csv(out/'fuel_cell_selected_holdout_by_test.csv')
    gm=classification_metrics(group.actual,group.predicted)
    gresult={'selected_on_validation':winner,'aggregation_rule':'mean_row_pass_probability_gte_0.5',**{'group_'+k:v for k,v in gm.items()},'research_status':'GROUP_EVALUATION_REUSED_HOLDOUT_NOT_SAFETY_APPROVAL'}
    pd.DataFrame([gresult]).to_csv(out/'fuel_cell_selected_holdout_group_metrics.csv',index=False)
    return row,gresult

def main():
    a=argparse.ArgumentParser();a.add_argument('--hydrogen',type=Path,required=True);a.add_argument('--fuel-cell',type=Path,required=True);a.add_argument('--out',type=Path,default=Path('research_comparison_results'))
    args=a.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    h=run_hydrogen(args.hydrogen,args.out)
    f,g=run_fuel(args.fuel_cell,args.out)
    print('HYDROGEN',json.dumps(h,indent=2));print('FUEL CELL ROWS',json.dumps(f,indent=2));print('FUEL CELL TEST GROUPS',json.dumps(g,indent=2))
    print('Research comparison ONLY. Existing .pkl models are not loaded or changed. Holdout previously viewed: NOT blind.')
if __name__=='__main__':main()
