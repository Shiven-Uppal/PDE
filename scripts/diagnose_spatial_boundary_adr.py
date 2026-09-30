"""Compare uniform and spatially varying NCR boundary forcing on one episode."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from pyproj import Transformer
from scipy.optimize import minimize_scalar
from src.adr_solver import ADRConfig
from src.daily_adr import interpolate_to_station_coordinates, load_inventory_template
from src.transient_adr import advance_adr_implicit
LOG_OFFSET, DT_S = 5.0, 3600.0

def profile_q(b, r, y):
    def f(q):
        return float(np.mean((np.log(np.maximum(b + q*r, 0)+LOG_OFFSET)-np.log(y+LOG_OFFSET))**2))
    hi, old, new = 1e-8, f(0), f(1e-8)
    while new < old and hi < 1:
        old, hi, new = new, hi*10, f(hi)
    z=minimize_scalar(f,bounds=(0,hi),method="bounded")
    return float(z.x),float(z.fun)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--processed-dir",required=True);p.add_argument("--output-dir",required=True)
    p.add_argument("--episode-id",default="calibration_013")
    p.add_argument("--parameters",nargs=3,type=float,required=True,metavar=("D","LAMBDA","KAPPA"))
    a=p.parse_args(); d,lam,k=a.parameters; root=Path(a.processed_dir); out=Path(a.output_dir)
    force=pd.read_csv(root/"transient_hourly_episode_panel.csv",parse_dates=["time"])
    block=force[(force.split=="calibration")&(force.episode_id==a.episode_id)].sort_values("time")
    boundary=pd.read_csv(root/"boundary_pm25_hourly_panel.csv",parse_dates=["time"])
    obs=pd.read_csv(root/"pm25_hourly_clean.csv",parse_dates=["hour"])
    st=pd.read_csv(root/"station_metadata.csv"); e,n=Transformer.from_crs(4326,32643,always_xy=True).transform(st.longitude,st.latitude)
    sm=st.assign(easting_m=e,northing_m=n).set_index("station_id")
    template=load_inventory_template(root/"edgar_pm25_inventory_template.csv")
    uniform=np.full((50,50),float(block.iloc[0].pm25_inflow_ug_m3)); spatial=uniform.copy(); response=np.zeros((50,50))
    rows=[]; resid=0.
    for ix,(_,r) in enumerate(block.iterrows(),1):
        values=boundary[boundary.time==r.time].set_index("boundary_role").pm25_inflow_ug_m3.to_dict()
        if set(values)!={"N_NW","W","E","SE","S"}: raise RuntimeError(f"Missing boundary roles at {r.time}")
        cfg=ADRConfig(50000,50000,50,50,d,k*float(r.u10_m_s),k*float(r.v10_m_s),lam)
        L=cfg.lx_m
        def bfun(x,y,v=values):
            if np.isclose(x,0): return float(v["W"])
            if np.isclose(x,L): return float(v["E"])
            if np.isclose(y,0): return float((1-x/L)*v["S"]+(x/L)*v["SE"])
            return float((1-x/L)*v["N_NW"]+(x/L)*v["E"])
        uniform,ru=advance_adr_implicit(cfg,uniform,np.zeros((50,50)),float(r.pm25_inflow_ug_m3),DT_S)
        spatial,rs=advance_adr_implicit(cfg,spatial,np.zeros((50,50)),bfun,DT_S)
        response,rr=advance_adr_implicit(cfg,response,template,0.0,DT_S);resid=max(resid,ru,rs,rr)
        print(f"hour {ix:03d}/{len(block)} {r.time}",flush=True)
        if not bool(r.score_hour): continue
        o=obs[obs.hour==r.time].merge(sm[["easting_m","northing_m"]],left_on="station_id",right_index=True)
        item=o[["hour","station_id","station_name","pm25_hourly"]].copy()
        item["uniform_background"]=interpolate_to_station_coordinates(uniform,cfg,o.easting_m,o.northing_m)
        item["spatial_background"]=interpolate_to_station_coordinates(spatial,cfg,o.easting_m,o.northing_m)
        item["unit_source_response"]=interpolate_to_station_coordinates(response,cfg,o.easting_m,o.northing_m);rows.append(item)
    x=pd.concat(rows,ignore_index=True); result={}
    for name in ["uniform","spatial"]:
        q,l=profile_q(x[f"{name}_background"].to_numpy(),x.unit_source_response.to_numpy(),x.pm25_hourly.to_numpy())
        x[f"{name}_model"]=x[f"{name}_background"]+q*x.unit_source_response
        result[name]={"q":q,"log_mse":l,"rmse_ug_m3":float(np.sqrt(np.mean((x[f"{name}_model"]-x.pm25_hourly)**2)))}
    out.mkdir(parents=True,exist_ok=True);x.to_csv(out/"spatial_boundary_comparison_predictions.csv",index=False)
    result.update({"purpose":"one-episode spatial-boundary code-path diagnostic; not calibration","episode_id":a.episode_id,"n_station_hours":len(x),"max_relative_linear_residual":resid})
    (out/"spatial_boundary_comparison.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=="__main__": main()
