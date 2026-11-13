"""Actual author-API compatibility probe on synthetic graphs, never an accuracy result."""
import argparse,copy,hashlib,json,os,sys,traceback
from pathlib import Path
from types import SimpleNamespace
import torch
from torch_geometric.data import Data,Batch
from torch_geometric.loader import DataLoader

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--repo-root',type=Path,required=True)
    p.add_argument('--author-repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.repo_root=a.repo_root.resolve();a.author_repo=a.author_repo.resolve()
    root=Path(__file__).resolve().parents[1];out=a.output.resolve()
    if not out.is_relative_to(root) or out.exists():raise RuntimeError('fresh output under continuation root required')
    source=a.repo_root/'research/visual-graphs/takeover_20261112'
    sys.path.insert(0,str(source))
    from gated_affine_base_v1 import require_author_sources
    from gated_teacher_model import GatedTeacher
    _,_,provenance=require_author_sources(a.author_repo)
    from converter.converter import Converter
    from converter.threshold_getter import Threshold_Getter,ThreHook
    from forwards import forward_replace
    torch.manual_seed(918);torch.set_num_threads(2)
    graphs=[Data(x=torch.randn(4,3),edge_attr=torch.randn(5,1),
                 edge_index=torch.tensor([[0,1,2,3,0],[1,2,3,0,2]])) for _ in range(2)]
    batch=Batch.from_data_list(graphs)
    model=GatedTeacher(node_dim=3,edge_dim=1,hidden_dim=4,out_dim=4,layers=2).eval()
    args=SimpleNamespace(model_name='gated_gcn',num_thresholds=8,time=4,tau=1.0,
        task='classification',mode='test_snn',coding_type='diff_rate',step_mode='s',neuron_name='MTH_diff_line')
    report={'scope':'synthetic architecture/API compatibility only; no official graphs or calibration',
            'official_data_loaded':False,'device':'cpu','pinned_author':provenance,'steps':[]}
    def safe(s):
        for prefix,label in [(str(a.author_repo),'DCGS_REPO'),(str(a.repo_root),'REPO_ROOT'),(str(Path.home()),'LOCAL_HOME')]:
            s=s.replace(prefix,label).replace(prefix.replace('\\','/'),label)
        return s
    def attempt(name,fn):
        row={'step':name}
        try:
            result=fn();row.update(status='executed',result=result)
        except Exception as exc:
            row.update(status='error',error_type=type(exc).__name__,message=safe(str(exc)),
                frames=[{'file':safe(x.filename),'line':x.lineno,'function':x.name} for x in traceback.extract_tb(exc.__traceback__)])
        report['steps'].append(row)
    def count(m):return sum(x.__class__.__name__.startswith('MTH') for x in m.modules())
    def direct():
        converted=Converter('MTH_diff_line',args,T=4,step_mode='s',fuse_flag=False)(copy.deepcopy(model))
        return {'registered_relu_modules':sum(isinstance(x,torch.nn.ReLU) for x in model.modules()),
                'converted_source_modules':count(converted),
                'interpretation':'zero converted modules is not a successful conversion'}
    attempt('author_converter_direct_without_threshold_hooks',direct)
    attempt('author_threshold_getter_native_pyg_loader',lambda: str(type(
        Threshold_Getter(DataLoader(graphs,batch_size=2),mode='99.9%',level='layer',device='cpu')(copy.deepcopy(model))).__name__))
    adapted={}
    def calibrate_adapter():
        # Dummy targets satisfy the author image-loader API but never enter the model.
        wrapped=Threshold_Getter([(batch,torch.zeros(2,dtype=torch.long))],
            mode='99.9%',level='layer',device='cpu')(copy.deepcopy(model))
        adapted['hooked']=wrapped
        return {'hooked_modules':sum(x.__class__.__name__=='ThreHook' for x in wrapped.modules()),
                'calibration_scope':'two synthetic unlabeled graphs only; not usable experimental thresholds'}
    attempt('author_threshold_getter_tuple_adapter',calibrate_adapter)
    if 'hooked' not in adapted:
        # Explicit runtime metadata adapter, not a change to numerical source.
        # The graph model inspects embedding dimensions; the author's wrapper
        # does not forward these attributes. Preserve the first real failure.
        ThreHook.in_features=property(lambda self:self.out.in_features)
        ThreHook.out_features=property(lambda self:self.out.out_features)
        report['runtime_adapter']='ThreHook forwards read-only in_features/out_features metadata; pinned source bytes unchanged'
        attempt('author_threshold_getter_tuple_and_metadata_adapter',calibrate_adapter)
    if 'hooked' in adapted:
        def convert_and_forward():
            converted=Converter('MTH_diff_line',args,T=4,step_mode='s',fuse_flag=False)(copy.deepcopy(adapted['hooked']))
            converted.embedding_h.in_features=model.embedding_h.in_features
            report['generic_converted_source_modules']=count(converted)
            temporal=forward_replace(args,converted)
            output=temporal(batch)
            return {'output_type':type(output).__name__,'scope':'execution only; accuracy/faithfulness unassessed'}
        attempt('author_generic_converter_and_temporal_forward',convert_and_forward)
        def tensor_payload_adapter():
            converted=Converter('MTH_diff_line',args,T=4,step_mode='s',fuse_flag=False)(copy.deepcopy(adapted['hooked']))
            converted.embedding_h.in_features=model.embedding_h.in_features
            class TensorGraph(torch.nn.Module):
                def __init__(self,inner,template):
                    super().__init__();self.inner=inner;self.template=template
                def forward(self,x):
                    own=self.template.clone();own.x=x
                    return self.inner(own)
            wrapped=TensorGraph(converted,batch)
            temporal=forward_replace(args,wrapped)
            output=temporal(batch.x)
            return {'output_shape':list(output.shape),'finite':bool(torch.isfinite(output).all()),
                    'converted_source_modules':count(converted),
                    'scope':'fixed synthetic topology/edge attributes, Tensor node payload; no fidelity or dynamic-graph claim'}
        attempt('author_generic_with_tensor_payload_adapter',tensor_payload_adapter)
        attempt('author_converter_fx_fusion',lambda:str(type(Converter('MTH_diff_line',args,T=4,
            step_mode='s',fuse_flag=True)(copy.deepcopy(adapted['hooked']))).__name__))
        def fx_after_metadata():
            converted=Converter.replace_by_neuron(copy.deepcopy(adapted['hooked']),'MTH_diff_line',args,step_mode='s',T=4)
            converted.embedding_h.in_features=model.embedding_h.in_features
            graph=torch.fx.symbolic_trace(converted)
            return str(type(Converter.fuse(graph,True)).__name__)
        attempt('author_fx_path_after_explicit_metadata_adapter',fx_after_metadata)
    names=('converter/converter.py','converter/threshold_getter.py','forwards/__init__.py','forwards/encoder.py','forwards/decoder.py')
    report['author_files_sha256']={n:sha(a.author_repo/n) for n in names}
    report['project_model_sha256']=sha(source/'gated_teacher_model.py')
    report['limitations']='No full-train threshold iteration, no official validation, no fidelity numbers. API failure is not a universal graph-conversion impossibility.'
    def scrub_tree(value):
        if isinstance(value,dict):return {safe(k):scrub_tree(v) for k,v in value.items()}
        if isinstance(value,list):return [scrub_tree(v) for v in value]
        return safe(value) if isinstance(value,str) else value
    report=scrub_tree(report)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'output':out.name,'steps':[{'step':x['step'],'status':x['status'],'error_type':x.get('error_type'),'result':x.get('result')} for x in report['steps']]},indent=2))
if __name__=='__main__':main()
