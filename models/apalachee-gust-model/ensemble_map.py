"""Render assumed-perturbation exceedance and peak panels; no official warnings."""
import os
os.environ['APALACHEE_SKIP_CACHE'] = '1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import stormmodel as sm
from plotkit import COLORS,FILL_LEVELS,INK,MUTED,ACCENT

with np.load(os.path.join(sm.HERE,'ensemble.npz')) as data:
    if str(data['model_signature'])!=sm.signature(): raise RuntimeError('Stale ensemble: run ensemble.py again')
    fields = [data[k] for k in ('p58','p74','mean_peak','max_peak')]
    n = int(data['n_members']); step = float(data['step_h'])
fig,axes = plt.subplots(2,2,figsize=(12,12.2),dpi=180,facecolor='white')
fig.subplots_adjust(left=.035,right=.965,bottom=.12,top=.885,wspace=.07,hspace=.32)
fig.text(.035,.975,'Apalachee Bay Sou’easter: Scenario Uncertainty',fontsize=20,weight='bold',color=INK)
fig.text(.035,.950,'SIMULATED SCENARIO · uncalibrated frequencies under assumed perturbations',fontsize=11,weight='bold',color=ACCENT)
fig.text(.035,.929,f'{n} perturbed members · {step*60:.0f}-minute integration · track, pressure, size, mixing, sting jet and convective cells',fontsize=9,color=MUTED)
titles = ['Members with peak gust ≥58 mph','Members with peak gust ≥74 mph',
          'Mean of each member’s peak gust','Highest peak among all members']
cmap = LinearSegmentedColormap.from_list('pink_plum',COLORS)
track = sm.track_xy(np.arange(30,114,1))
for i,(ax,field,title) in enumerate(zip(axes.ravel(),fields,titles)):
    sm.m.drawmapboundary(fill_color='#e6ecf1',linewidth=0,ax=ax)
    sm.m.fillcontinents(color='#f5f3ee',lake_color='#e6ecf1',ax=ax,zorder=0)
    if i<2:
        ax.pcolormesh(sm.X,sm.Y,np.ma.masked_where(field==0,field),cmap=cmap,vmin=0,vmax=100,
                      shading='auto',rasterized=True,zorder=2)
    else:
        ax.contourf(sm.X,sm.Y,field,levels=FILL_LEVELS,colors=COLORS,zorder=2)
    sm.m.drawstates(linewidth=.45,color='#57575d',ax=ax,zorder=3)
    sm.m.drawcoastlines(linewidth=.5,color='#3a3a42',ax=ax,zorder=3)
    ax.plot(track[:,0],track[:,1],color=INK,lw=.7,zorder=4)
    for name,lo,la in [('Atlanta',-84.39,33.75),('Tampa',-82.46,27.95),('Savannah',-81.09,32.08)]:
        x,y = sm.m(lo,la); ax.plot(x,y,'o',ms=2,color=INK,zorder=5)
        ax.annotate(name,(x,y),xytext=(4,2),textcoords='offset points',fontsize=7,zorder=6,
                    bbox=dict(fc='white',alpha=.7,ec='none',pad=.2))
    ax.set_title(title,fontsize=11,weight='bold',loc='left',pad=8,color=INK)
# One shared bar per row, leaving room between rows and at the bottom.
cax = fig.add_axes([.15,.515,.70,.011])
cb = fig.colorbar(ScalarMappable(norm=Normalize(0,100),cmap=cmap),cax=cax,orientation='horizontal',ticks=[0,25,50,75,100])
cb.ax.tick_params(labelsize=8); cb.set_label('Fraction of members (%) · unshaded = 0%',fontsize=8)
from plotkit import draw_cbar
draw_cbar(fig,[.15,.077,.70,.012])
fig.text(.15,.102,'Peak gust (mph) · unshaded below 40',fontsize=8,color=INK)
fig.text(.035,.026,'Fictional event. Frequencies depend on chosen perturbations; not real-world forecast probabilities. Thin line = control low track.\n'
         'Mean of peaks ≠ peak of ensemble-mean winds. Highest-member panel is sensitive to ensemble size.',fontsize=8,color=MUTED,linespacing=1.6)
fig.savefig(os.path.join(sm.HERE,'apalachee_ensemble_v3.png'),facecolor='white')
print('saved apalachee_ensemble_v3.png')
