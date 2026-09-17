"""Render the bounded auxiliary-supervision experiment without mixing total and text loss."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/attention-aux'
r=json.loads((OUT/'results.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
for name,arm in r['arms'].items():
    steps=sorted(map(int,arm['evaluations']))
    ev=[arm['evaluations'][str(s)] for s in steps]
    for ax,metric,title in [(axes[0,0],'lm','Primary text CE (lower is better)'),(axes[0,1],'attention_ce','Direct attention text CE (lower is better)'),(axes[1,0],'rank','Attention centered effective rank'),(axes[1,1],'centered_energy','Attention centered energy fraction')]:
        for subset,style in [('fixed','-'),('extra','--')]:
            ax.plot(steps,[e[subset][metric] for e in ev],style,marker='o',label=f'{name}, {subset}')
        ax.set_title(title);ax.set_xlabel('Updates from step 12810');ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8)
fig.suptitle('Same starting weights, data, LR, and Adam reset; auxiliary weight = 0.1')
fig.savefig(OUT/'comparison.png',dpi=170)
fig.savefig(OUT/'comparison.pdf')
