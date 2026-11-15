"""Single training-only audit of hard loss and available offset capacity; no tuning."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_objectives import historical, core, CRITICAL, local_loss, decision_loss


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'analysis'/'optimization_audit')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    allrows=[]
    for dataset in ('MNIST','CIFAR10'):
        run=HERE/'runs'/'objectives'/f'{dataset}41'
        meta,_,_,folded,loaded,_,data_root=historical.load_teacher(dataset,41,historical.DEFAULT_ASSETS,
            CRITICAL/'runs'/'allocation'/f'{dataset}41',torch.device('cuda'))
        net=core.CalibratedGCN(folded,loaded.thresholds).cuda().eval()
        train=GNNBenchmarkDataset(root=str(data_root),name=dataset,split='train')
        batches=[]
        for b in DataLoader(train[meta['train_indices']],batch_size=32,shuffle=False):
            b=b.cuda();del b.y
            with torch.no_grad():a,acts=core.teacher_sites(folded,b)
            batches.append((b,a,acts))
        for kind in ('local_t1','local_joint','logits','kl','ranking'):
            params=np.load(run/f'{kind}_parameters.npz')
            for alpha in (0.,.25,.5,1.):
                with torch.no_grad():
                    for i,o in enumerate(net.offsets):o.copy_(torch.as_tensor(params[f'offset_{i}'],device='cuda')*alpha)
                    losses=[];agreement=[];third=[]
                    for b,a,acts in batches:
                        pair=[]
                        for T in (1,2):
                            z,sites=net(b,T)
                            pair.append(local_loss(sites,acts) if kind.startswith('local') else decision_loss(z,a,kind))
                            agreement.append(float((z.argmax(1)==a.argmax(1)).float().mean()))
                            top2=a.topk(2,1).indices
                            third.append(float((~(z.argmax(1)[:,None]==top2).any(1)).float().mean()))
                        losses.append(float(pair[0] if kind=='local_t1' else sum(pair)/2))
                allrows.append(dict(dataset=dataset,method=kind,parameter_interpolation=alpha,
                    full_train_hard_loss=float(np.mean(losses)),prefix_mean_teacher_agreement=float(np.mean(agreement)),
                    prefix_mean_other_than_teacher_top2=float(np.mean(third)),
                    offset_max_abs=max(float(o.abs().max()) for o in net.offsets),
                    offsets_at_bound=sum(int((o.abs()>=.5-1e-6).sum()) for o in net.offsets),parameters=693))
    (out/'results.json').write_text(json.dumps(dict(training_only=True,validation_access=False,
        purpose='one predeclared optimization/capacity audit, no hyperparameter selection',rows=allrows),indent=2)+'\n')
    print(json.dumps([r for r in allrows if r['parameter_interpolation'] in (0.,1.)],indent=2))


if __name__=='__main__':main()
