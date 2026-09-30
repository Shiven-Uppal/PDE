"""Calibrate BLH-constrained ADR transport parameters on locked calibration episodes."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from pyproj import Transformer
from scipy.optimize import differential_evolution,minimize_scalar
from src.adr_solver import ADRConfig
from src.daily_adr import interpolate_to_station_coordinates,load_inventory_template
from src.transient_adr import advance_adr_implicit_many
O,DT=5.,3600.
def pq(b,r,y):
 def f(q):return float(np.mean((np.log(b+q*r+O)-np.log(y+O))**2))
 hi,old,new=1e-8,f(0),f(1e-8)
 while new<old and hi<1:old,hi,new=new,hi*10,f(hi)
 z=minimize_scalar(f,bounds=(0,hi),method='bounded',options={'xatol':1e-10});return z.x,z.fun
def main():
 p=argparse.ArgumentParser();p.add_argument('--processed-dir',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--maxiter',type=int,default=12);p.add_argument('--popsize',type=int,default=6);p.add_argument('--seed',type=int,default=20260902);a=p.parse_args();root=Path(a.processed_dir);f=pd.read_csv(root/'transient_hourly_episode_panel_physical.csv',parse_dates=['time'])
 f=f[f.split=='calibration'].copy();mu,sd=f.boundary_layer_height_m.mean(),f.boundary_layer_height_m.std();f['z_blh']=(f.boundary_layer_height_m-mu)/sd
 o=pd.read_csv(root/'pm25_hourly_clean.csv',parse_dates=['hour']);s=pd.read_csv(root/'station_metadata.csv');x,y=Transformer.from_crs(4326,32643,always_xy=True).transform(s.longitude,s.latitude);m=s.assign(easting_m=x,northing_m=y).set_index('station_id');t=load_inventory_template(root/'edgar_pm25_inventory_template.csv');eps=[]
 for eid,b in f.groupby('episode_id',sort=True):
  obs={time:o[o.hour==time].merge(m[['easting_m','northing_m']],left_on='station_id',right_index=True) for time in b.time};eps.append((eid,b.sort_values('time'),obs))
 trace=[]
 def ev(z):
  D,lam,k,al,be=np.exp(z[0]),np.exp(z[1]),z[2],z[3],z[4];loss=[];weights=[]
  for _,b,obs in eps:
   F=np.stack((np.full((50,50),b.iloc[0].pm25_inflow_ug_m3),np.zeros((50,50))));S=np.stack((np.zeros((50,50)),t));rr=[]
   for _,r in b.iterrows():
    c=ADRConfig(50000,50000,50,50,D*np.exp(al*r.z_blh),k*r.u10_m_s,k*r.v10_m_s,lam*np.exp(-be*r.z_blh));F,_=advance_adr_implicit_many(c,F,S,np.array([r.pm25_inflow_ug_m3,0.]),DT)
    if r.score_hour:
     q=obs[r.time];rr.append((interpolate_to_station_coordinates(F[0],c,q.easting_m.to_numpy(),q.northing_m.to_numpy()),interpolate_to_station_coordinates(F[1],c,q.easting_m.to_numpy(),q.northing_m.to_numpy()),q.pm25_hourly.to_numpy()))
   B=np.concatenate([v[0] for v in rr]);R=np.concatenate([v[1] for v in rr]);Y=np.concatenate([v[2] for v in rr]);_,L=pq(B,R,Y);loss.append(L);weights.append(len(Y))
  v=float(np.average(loss,weights=weights));trace.append({'D0':D,'lambda0':lam,'kappa':k,'alpha_blh':al,'beta_blh':be,'log_mse':v});return v
 bounds=[(np.log(300),np.log(5000)),(np.log(2e-6),np.log(8e-5)),(.05,1),(-1.5,1.5),(-1.5,1.5)]
 r=differential_evolution(ev,bounds,maxiter=a.maxiter,popsize=a.popsize,seed=a.seed,polish=False);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);pd.DataFrame(trace).to_csv(out/'physical_calibration_trace.csv',index=False);record={'split':'calibration','success':bool(r.success),'message':str(r.message),'objective_log_mse':float(r.fun),'D0_m2_s':float(np.exp(r.x[0])),'lambda0_s_inv':float(np.exp(r.x[1])),'kappa':float(r.x[2]),'alpha_blh':float(r.x[3]),'beta_blh':float(r.x[4]),'blh_calibration_mean_m':float(mu),'blh_calibration_sd_m':float(sd)};(out/'physical_calibrated_parameters.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
if __name__=='__main__':main()
