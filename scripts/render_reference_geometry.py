"""Optional descriptive figure from retained analysis; no numerical recomputation.

Requires matplotlib outside the canonical backend. Plotting never selects layers,
changes the stored matrices, or enters model execution/verification.
"""
import argparse
import json
from pathlib import Path


def render(directory, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    study = json.loads((directory/'analysis.json').read_bytes())
    request = json.loads((directory/'request.json').read_bytes())
    labels = [f"F{i//8+1}:{r['task'][:3]}:{r['carrier'][0]}" for i,r in enumerate(request['probes'])]
    fig, axes = plt.subplots(1,3,figsize=(15,5.7),layout='constrained')
    for ax,c in zip(axes,('native','stock','modelfile')):
        values = study['diagnostics'][c+':final']['cosine_distance_hex']
        matrix = [[float('nan') if x is None else float.fromhex(x) for x in row] for row in values]
        image = ax.imshow(matrix,vmin=0,vmax=2,cmap='viridis',interpolation='nearest')
        ax.set_title({'native':'Trinite trained arithmetic','stock':'Qwen stock','modelfile':'Qwen SYSTEM-only profile'}[c])
        ax.set_xticks(range(16),labels,rotation=90,fontsize=7)
        ax.set_yticks(range(16),labels,fontsize=7)
        ax.axhline(7.5,color='white',linewidth=.8);ax.axvline(7.5,color='white',linewidth=.8)
    fig.colorbar(image,ax=axes,shrink=.8,label='Within-model cosine distance (shared 0–2 scale)')
    fig.suptitle('Final normalized states • full-prompt last user-content token',fontsize=14)
    fig.supxlabel('16 visible native-training probes; two paired families. Descriptive; no cross-model coordinate cosine.',fontsize=10)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=160,metadata={'Description':'Derived from exact retained reference geometry v1 analysis.'})
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();render(args.directory,args.output)
