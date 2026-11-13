"""Single full CIFAR10 test of a completed frozen GatedGCN teacher."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from gated_teacher_train import DATA_ROOT, OUT_ROOT, RUN_NAME, sha256
from gated_teacher_model import GatedTeacher

def main():
    here=Path(__file__).resolve().parent
    run=OUT_ROOT/RUN_NAME
    out, pred=run/'gtest_v1.json',run/'gtest_v1_pred.npz'
    if out.exists() or pred.exists():
        raise SystemExit('Existing output; refusing repeat/overwrite')
    done=json.loads((run/'done.json').read_text(encoding='utf-8'))
    if done.get('status')!='finished':
        raise RuntimeError('Training must be finished')
    checkpoint=run/'best.pt'
    if sha256(checkpoint)!=done['best_checkpoint_sha256']:
        raise RuntimeError('Best checkpoint hash mismatch')
    state=torch.load(checkpoint,map_location='cuda',weights_only=False)
    meta=state['metadata']
    if meta['dataset']!='CIFAR10' or meta['seed']!=41 or meta['official_counts']!={'train':45000,'val':5000}:
        raise RuntimeError('Teacher provenance mismatch')
    for name in ('gated_teacher_model.py','gated_teacher_train.py','gated_teacher_train_protocol_v1.md'):
        if sha256(here/name)!=meta['provenance']['source_sha256'][name]:
            raise RuntimeError(f'Training source changed: {name}')
    for name,digest in meta['provenance']['processed_data_sha256'].items():
        if sha256(DATA_ROOT/'CIFAR10'/'processed'/name)!=digest:
            raise RuntimeError(f'Processed data changed: {name}')
    model=GatedTeacher(**{k:meta['model'][k] for k in ('node_dim','edge_dim','hidden_dim','out_dim','layers','n_classes','readout','residual','batch_norm','dropout')}).cuda()
    model.load_state_dict(state['model']); model.eval()
    dataset=GNNBenchmarkDataset(root=str(DATA_ROOT),name='CIFAR10',split='test')
    if len(dataset)!=10000:
        raise RuntimeError('Unexpected test size')
    loader=DataLoader(dataset,batch_size=128,shuffle=False,num_workers=0)
    ys,zs=[],[]
    start=time.monotonic()
    with torch.no_grad():
        for batch in loader:
            batch=batch.to('cuda')
            zs.append(model(batch).cpu().numpy())
            ys.append(batch.y.view(-1).cpu().numpy())
    y,z=np.concatenate(ys),np.concatenate(zs)
    if len(y)!=10000 or z.shape!=(10000,10):
        raise RuntimeError('Incomplete test')
    np.savez_compressed(pred,index=np.arange(len(y)),labels=y,logits=z,predictions=z.argmax(1))
    result={'dataset':'CIFAR10','seed':41,'split':'full official test, historically exposed','graphs':len(y),
            'checkpoint_epoch':state['epoch'],'checkpoint_sha256':sha256(checkpoint),
            'accuracy':float(np.mean(z.argmax(1)==y)),'prediction_sha256':sha256(pred),
            'script_sha256':sha256(__file__),'protocol_sha256':sha256(here/'gated-test-protocol-v1.md'),
            'device':torch.cuda.get_device_name(),'elapsed_sec':time.monotonic()-start,
            'qualification':'PyG GatedGCN adaptation; prior scientific test exposure; teacher result only'}
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    main()
