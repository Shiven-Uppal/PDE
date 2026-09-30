from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
def main():
 r=Path(__file__).resolve().parents[1]; d=pd.read_csv(r/'outputs/conditional_uncertainty/rank_sensitivity.csv').sort_values('spearman_rho_with_baseline_final_domain_mean')
 labels=d.parameter.str.replace('_',' '); colors=['#bd3b24' if x<0 else '#0c6b9d' for x in d.spearman_rho_with_baseline_final_domain_mean]
 fig,ax=plt.subplots(figsize=(7.2,4.6)); ax.barh(labels,d.spearman_rho_with_baseline_final_domain_mean,color=colors); ax.axvline(0,color='#333',lw=.8); ax.set(xlabel='Spearman rank correlation with final conditional domain mean',title='Figure 5. Conditional-model sensitivity to declared uncertain inputs')
 ax.grid(axis='x',alpha=.25); ax.text(.5,-.18,'Rank sensitivity within the declared ADR priors; not causal source attribution.',transform=ax.transAxes,ha='center',fontsize=8.5); fig.tight_layout(); o=r/'outputs/figures'; o.mkdir(exist_ok=True); fig.savefig(o/'figure_5_rank_sensitivity.png',dpi=300,bbox_inches='tight'); fig.savefig(o/'figure_5_rank_sensitivity.pdf',bbox_inches='tight')
if __name__=='__main__':main()
