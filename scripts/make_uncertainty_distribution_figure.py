"""Create Figure 4 from locked 500-draw conditional uncertainty outputs."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def main():
 root=Path(__file__).resolve().parents[1]; d=pd.read_csv(root/'outputs/conditional_uncertainty/conditional_scenario_outcomes.csv'); out=root/'outputs/figures'; out.mkdir(exist_ok=True)
 factors=sorted(d.scenario_source_factor.unique()); groups=[d.loc[d.scenario_source_factor.eq(x),'final_domain_mean_ug_m3'] for x in factors]
 fig,ax=plt.subplots(figsize=(7.2,5.4)); parts=ax.violinplot(groups,showmedians=True,showextrema=False); [b.set_facecolor('#0c6b9d') or b.set_alpha(.55) for b in parts['bodies']]
 ax.boxplot(groups,widths=.14,showfliers=False,medianprops={'color':'#7b2013','lw':1.8},boxprops={'color':'#27364b'},whiskerprops={'color':'#27364b'},capprops={'color':'#27364b'})
 ax.set(xticks=range(1,len(factors)+1),xticklabels=[f'{x:.1f}' for x in factors],xlabel='Nominal local-primary-source input factor',ylabel='Final conditional domain-mean PM₂.₅ (µg/m³)',title='Figure 4. Conditional uncertainty across 500 Latin-hypercube draws')
 ax.grid(axis='y',alpha=.25); ax.text(.5,-.17,'Model-input scaling only; not a real-world policy-effect estimate.',transform=ax.transAxes,ha='center',fontsize=8.5); fig.tight_layout(); fig.savefig(out/'figure_4_conditional_uncertainty_distribution.png',dpi=300,bbox_inches='tight'); fig.savefig(out/'figure_4_conditional_uncertainty_distribution.pdf',bbox_inches='tight')
if __name__=='__main__': main()
