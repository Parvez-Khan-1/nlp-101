import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, torch
from matplotlib.colors import LinearSegmentedColormap
exec(open('dates_seq2seq.py').read().split('def train_model')[0])
model=Seq2Seq(attention=True); model.load_state_dict(torch.load('dates_attention.pt')); model.eval()
text="7th of march, 2011"
out,w=model.translate(text)
# recompute weights for the full output sequence with teacher forcing on the produced output
src=enc_src([text]); y=torch.tensor([[ti["<s>"]]+[ti[c] for c in out]])
with torch.no_grad(): _,W=model(src,y)
W=W[0].numpy()[:len(out)]   # rows: produce each output char
print(out, W.shape)
SURF,T1,T2='#fcfcfb','#0b0b0b','#52514e'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'figure.facecolor':SURF,'axes.facecolor':SURF,'text.color':T1,'xtick.color':T2,'ytick.color':T2,'axes.labelcolor':T2})
cmap=LinearSegmentedColormap.from_list('b',['#f1f6fd','#9cc1ef','#2a78d6','#0d3a75'])
fig,ax=plt.subplots(figsize=(9,5.4),dpi=160)
im=ax.imshow(W,cmap=cmap,vmin=0,vmax=1,aspect='auto')
ax.set_xticks(range(len(text)),[c if c!=' ' else '␣' for c in text]); ax.set_yticks(range(len(out)),list(out))
ax.xaxis.set_ticks_position('top'); ax.xaxis.set_label_position('top')
ax.set_xlabel('Input characters (messy date)'); ax.set_ylabel('Output characters (ISO date)')
for s in ax.spines.values(): s.set_visible(False)
ax.set_xticks(np.arange(-.5,len(text),1),minor=True); ax.set_yticks(np.arange(-.5,len(out),1),minor=True)
ax.grid(which='minor',color=SURF,lw=1.5); ax.tick_params(which='minor',length=0)
cb=fig.colorbar(im,ax=ax,fraction=0.03); cb.outline.set_visible(False); cb.set_label('Attention weight',color=T2)
fig.suptitle(f'Real attention weights from our trained model: "{text}" → {out}',x=0.03,ha='left',fontsize=12.5)
fig.tight_layout(); fig.savefig('/home/claude/site/static/images/unit5/attention_dates.png'); print('ok')
