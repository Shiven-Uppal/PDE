"""Create Figure 3: a selected 24-hour conditional ADR model field."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.adr_solver import ADRConfig
from src.daily_adr import load_inventory_template
from src.transient_adr import advance_adr_implicit_many

def main():
    root=Path(__file__).resolve().parents[1]; p=root/'data/processed'
    f=pd.read_csv(p/'transient_hourly_episode_panel.csv',parse_dates=['time'])
    e=f[(f.episode_id=='calibration_013')].sort_values('time').iloc[:24]
    q=float(pd.read_csv(root/'outputs/transient_screen_b/transient_episode_source_scales.csv').source_scale_q.median())
    t=load_inventory_template(p/'edgar_pm25_inventory_template.csv')
    fields=np.stack((np.full((50,50),float(e.iloc[0].pm25_inflow_ug_m3)),np.zeros((50,50))))
    src=np.stack((np.zeros((50,50)),t)); residual=0
    for _,r in e.iterrows():
        c=ADRConfig(50000,50000,50,50,2000,.15*r.u10_m_s,.15*r.v10_m_s,1.6e-5)
        fields,residual=advance_adr_implicit_many(c,fields,src,np.array([r.pm25_inflow_ug_m3,0.]),3600)
    field=fields[0]+q*fields[1]; out=root/'outputs/figures'; out.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(7.4,6)); im=ax.imshow(field,origin='lower',extent=[0,50,0,50],cmap='magma',aspect='equal')
    cb=fig.colorbar(im,ax=ax,pad=.02); cb.set_label('Conditional model concentration (µg/m³)')
    ax.set(xlabel='Easting in ADR domain (km)',ylabel='Northing in ADR domain (km)',title='Figure 3. Selected 24-hour conditional ADR field')
    ax.text(.5,-.16,'Model-generated field for calibration_013; not an observed map or forecast.',transform=ax.transAxes,ha='center',fontsize=8.5)
    fig.tight_layout(); fig.savefig(out/'figure_3_selected_conditional_adr_field.png',dpi=300,bbox_inches='tight'); fig.savefig(out/'figure_3_selected_conditional_adr_field.pdf',bbox_inches='tight')
    np.save(out/'figure_3_selected_conditional_adr_field.npy',field)
if __name__=='__main__': main()
