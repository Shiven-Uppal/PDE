"""One-episode ADR diagnostic using physically specified inflow-only NCR boundaries."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from pyproj import Transformer
from scipy.optimize import minimize_scalar
from src.adr_solver import ADRConfig
from src.daily_adr import interpolate_to_station_coordinates,load_inventory_template
from src.transient_adr import advance_adr_implicit
O,DT=5.,3600.
def fitq(b,r,y):
 def f(q):return float(np.mean((np.log(b+q*r+O)-np.log(y+O))**2))
 hi,old,new=1e-8,f(0),f(1e-8)
 while new<old and hi<1:old,hi,new=new,hi*10,f(hi)
 z=minimize_scalar(f,bounds=(0,hi),method='bounded',options={'xatol':1e-10});return float(z.x),float(z.fun)
def main():
 p=argparse.ArgumentParser();p.add_argument('--processed-dir',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--episode-id',default='calibration_013');p.add_argument('--parameters',nargs=3,type=float,required=True);a=p.parse_args();d,l,k=a.parameters;root=Path(a.processed_dir)
 f=pd.read_csv(root/'transient_hourly_episode_panel.csv',parse_dates=['time']);b=f[(f.split=='calibration')&(f.episode_id==a.episode_id)].sort_values('time');bd=pd.read_csv(root/'boundary_pm25_hourly_panel.csv',parse_dates=['time']);o=pd.read_csv(root/'pm25_hourly_clean.csv',parse_dates=['hour']);s=pd.read_csv(root/'station_metadata.csv');e,n=Transformer.from_crs(4326,32643,always_xy=True).transform(s.longitude,s.latitude);m=s.assign(easting_m=e,northing_m=n).set_index('station_id');t=load_inventory_template(root/'edgar_pm25_inventory_template.csv')
 bg=np.full((50,50),float(b.iloc[0].pm25_inflow_ug_m3));r=np.zeros((50,50));rows=[];res=0;fallback=0
 for ix,(_,z) in enumerate(b.iterrows(),1):
  v=bd[bd.time==z.time].set_index('boundary_role').pm25_inflow_ug_m3.to_dict();c=ADRConfig(50000,50000,50,50,d,k*z.u10_m_s,k*z.v10_m_s,l);need={'W'} if c.u_m_s>=0 else {'E'};need|={'S','SE'} if c.v_m_s>=0 else {'N_NW'}
  good=all(role in v and np.isfinite(v[role]) for role in need)
  if not good:v={role:float(z.pm25_inflow_ug_m3) for role in ['W','E','S','SE','N_NW']};fallback+=1
  def inflow(x,y,v=v,L=c.lx_m):
   if np.isclose(x,0):return float(v['W'])
   if np.isclose(x,L):return float(v['E'])
   if np.isclose(y,0):return float((1-x/L)*v['S']+(x/L)*v['SE'])
   return float(v['N_NW'])
  bg,a1=advance_adr_implicit(c,bg,np.zeros((50,50)),inflow,DT);r,a2=advance_adr_implicit(c,r,t,0.,DT);res=max(res,a1,a2)
  if bool(z.score_hour):
   q=o[o.hour==z.time].merge(m[['easting_m','northing_m']],left_on='station_id',right_index=True);it=q[['hour','station_id','pm25_hourly']].copy();it['background']=interpolate_to_station_coordinates(bg,c,q.easting_m,q.northing_m);it['response']=interpolate_to_station_coordinates(r,c,q.easting_m,q.northing_m);rows.append(it)
  if ix%24==0:print(f'hour {ix}/{len(b)}',flush=True)
 q=pd.concat(rows);scale,loss=fitq(q.background.to_numpy(),q.response.to_numpy(),q.pm25_hourly.to_numpy());q['model_pm25']=q.background+scale*q.response;out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);record={'purpose':'one-episode physically specified inflow-only boundary diagnostic; calibration only','episode_id':a.episode_id,'n_station_hours':len(q),'q':scale,'log_mse':loss,'rmse_ug_m3':float(np.sqrt(np.mean((q.model_pm25-q.pm25_hourly)**2))),'inflow_missing_fallback_hours':fallback,'max_relative_linear_residual':res};q.to_csv(out/'inflow_only_predictions.csv',index=False);(out/'inflow_only_boundary.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
if __name__=='__main__':main()
