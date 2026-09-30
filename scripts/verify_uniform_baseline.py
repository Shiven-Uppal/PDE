"""Exact uniform-boundary one-episode baseline for equivalence checking."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from pyproj import Transformer
from scipy.optimize import minimize_scalar
from src.adr_solver import ADRConfig
from src.daily_adr import interpolate_to_station_coordinates,load_inventory_template
from src.transient_adr import advance_adr_implicit_many
O,DT=5.,3600.
def pq(b,r,y):
 def f(q): return float(np.mean((np.log(b+q*r+O)-np.log(y+O))**2))
 hi,old,new=1e-8,f(0),f(1e-8)
 while new<old and hi<1: old,hi,new=new,hi*10,f(hi)
 z=minimize_scalar(f,bounds=(0,hi),method='bounded',options={'xatol':1e-10});return float(z.x),float(z.fun)
def main():
 p=argparse.ArgumentParser();p.add_argument('--processed-dir',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--episode-id',default='calibration_013');p.add_argument('--parameters',nargs=3,type=float,required=True);a=p.parse_args()
 d,l,k=a.parameters;root=Path(a.processed_dir);out=Path(a.output_dir)
 f=pd.read_csv(root/'transient_hourly_episode_panel.csv',parse_dates=['time']);b=f[(f.split=='calibration')&(f.episode_id==a.episode_id)].sort_values('time')
 o=pd.read_csv(root/'pm25_hourly_clean.csv',parse_dates=['hour']);s=pd.read_csv(root/'station_metadata.csv');x,y=Transformer.from_crs(4326,32643,always_xy=True).transform(s.longitude,s.latitude);m=s.assign(easting_m=x,northing_m=y).set_index('station_id');t=load_inventory_template(root/'edgar_pm25_inventory_template.csv')
 fields=np.stack((np.full((50,50),float(b.iloc[0].pm25_inflow_ug_m3)),np.zeros((50,50))));src=np.stack((np.zeros((50,50)),t));rows=[];res=0.
 for n,(_,r) in enumerate(b.iterrows(),1):
  c=ADRConfig(50000,50000,50,50,d,k*float(r.u10_m_s),k*float(r.v10_m_s),l);fields,z=advance_adr_implicit_many(c,fields,src,np.array([float(r.pm25_inflow_ug_m3),0.]),DT);res=max(res,z)
  if bool(r.score_hour):
   q=o[o.hour==r.time].merge(m[['easting_m','northing_m']],left_on='station_id',right_index=True);q['background_pm25']=interpolate_to_station_coordinates(fields[0],c,q.easting_m.to_numpy(),q.northing_m.to_numpy());q['unit_source_response']=interpolate_to_station_coordinates(fields[1],c,q.easting_m.to_numpy(),q.northing_m.to_numpy());rows.append(q[['hour','station_id','pm25_hourly','background_pm25','unit_source_response']])
  if n%24==0: print(f'hour {n}/{len(b)}',flush=True)
 q=pd.concat(rows,ignore_index=True);scale,loss=pq(q.background_pm25.to_numpy(),q.unit_source_response.to_numpy(),q.pm25_hourly.to_numpy());q['model_pm25']=q.background_pm25+scale*q.unit_source_response;record={'episode_id':a.episode_id,'n_station_hours':len(q),'q':scale,'log_mse':loss,'rmse_ug_m3':float(np.sqrt(np.mean((q.model_pm25-q.pm25_hourly)**2))),'max_relative_linear_residual':res};out.mkdir(parents=True,exist_ok=True);q.to_csv(out/'uniform_baseline_predictions.csv',index=False);(out/'uniform_baseline.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
if __name__=='__main__':main()
