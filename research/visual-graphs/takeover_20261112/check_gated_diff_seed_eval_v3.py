"""Exercise generic evaluator reporting/label/seed guards on synthetic data."""
import argparse
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from torch_geometric.data import Data
from gated_teacher_model import GatedTeacher
import gated_diff_seed_eval_v3 as evaluator


def main(author_repo):
    torch.set_num_threads(4)
    torch.manual_seed(816)
    metadata={"seed":42,"model":{"node_dim":3,"edge_dim":1,"hidden_dim":4,"out_dim":4,
        "n_classes":10,"layers":4,"dropout":0.,"batch_norm":True,"residual":True,"readout":"mean"}}
    model=GatedTeacher(node_dim=3,hidden_dim=4,out_dim=4,layers=4).eval()
    graphs=[Data(x=torch.randn(2,3),edge_index=torch.tensor([[0,1],[1,0]]),
                 edge_attr=torch.rand(2,1),y=torch.tensor([i%10])) for i in range(5000)]
    loaded=[]
    def dataset(*args,**kwargs):
        assert kwargs["split"]=="val"
        loaded.append("synthetic_val")
        return graphs
    original_builder=evaluator.build_teacher
    def build(state,device):
        result=original_builder(state,device)
        def check_labels(module,args):
            assert getattr(args[0],"y",None) is None
        result.register_forward_pre_hook(check_labels)
        return result
    with tempfile.TemporaryDirectory(prefix=".ge-",dir=evaluator.OUT_ROOT) as folder:
        run=Path(folder)
        checkpoint=run/"b.pt"
        torch.save({"metadata":metadata,"model":model.state_dict(),"epoch":1},checkpoint)
        cal_path=run/"c.json"
        cal_path.write_text("{}",encoding="utf-8")
        widths=(4,4,4,4,2,1)
        cal={"teacher_run":str(run),"teacher_metadata":metadata,
             "calibration":{"channel_thresholds":{name:[1.]*w for name,w in zip(evaluator.SITE_NAMES,widths)}},
             "data":{"processed_files_sha256":{"fixture":"synthetic"}}}
        args=argparse.Namespace(split="val",output=run/"o.json",author_repo=author_repo,
            teacher_run=run,calibration=cal_path,teacher_seed=42,device="cpu",batch_size=128)
        with patch.object(evaluator,"validate_calibration",return_value=(cal,checkpoint,evaluator.sha256(checkpoint))), \
             patch.object(evaluator,"GNNBenchmarkDataset",side_effect=dataset), \
             patch.object(evaluator,"build_teacher",side_effect=build):
            evaluator.evaluate(args)
            result=json.loads(args.output.read_text(encoding="utf-8"))
            assert result["graphs"]==5000 and result["teacher_seed"]==42 and not result["test_loaded"]
            assert len(result["rows"])==10
            with np.load(args.output.with_suffix(".npz"),allow_pickle=False) as archive:
                assert np.array_equal(archive["indices"],np.arange(5000))
                assert archive["dc_T8"].shape==(5000,10)
                assert "report_json" in archive.files
            args.teacher_seed=43
            args.output=run/"bad.json"
            try:
                evaluator.evaluate(args)
            except RuntimeError as error:
                assert "metadata mismatch" in str(error)
            else:
                raise AssertionError("mismatched teacher seed accepted")
            assert loaded==["synthetic_val"]
    print(json.dumps({"synthetic_rows":5000,"report_and_archive":True,
                     "label_free_forward":True,"seed_mismatch_rejected_before_data":True,
                     "official_data_loaded":False,"gpu_used":False}))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--author-repo",required=True,type=Path)
    main(parser.parse_args().author_repo)
