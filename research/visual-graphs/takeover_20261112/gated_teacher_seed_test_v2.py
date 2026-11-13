"""Frozen generic GatedGCN seed teacher on the complete exposed test split."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from gated_frozen_gate_calibrate import DATA_ROOT,OUT_ROOT,validate_teacher_run,build_teacher,processed_file_hashes,sha256

HERE=Path(__file__).resolve().parent


@torch.no_grad()
def evaluate(args):
    run=args.teacher_run.resolve()
    run.relative_to(OUT_ROOT.resolve())
    output,archive=run/"gtest_v2.json",run/"gtest_v2_pred.npz"
    if output.exists() or archive.exists():
        raise FileExistsError("refuse test repeat/overwrite")
    _,metadata,done,checkpoint,checkpoint_hash=validate_teacher_run(run)
    if metadata["seed"]!=args.teacher_seed:
        raise RuntimeError("teacher seed mismatch")
    processed=DATA_ROOT/"CIFAR10"/"processed"
    if processed_file_hashes(processed)!=metadata["provenance"]["processed_data_sha256"]:
        raise RuntimeError("processed data hash mismatch")
    device=torch.device(args.device)
    if device.type=="cpu":
        torch.set_num_threads(4)
    state=torch.load(checkpoint,map_location=device,weights_only=False)
    if state["metadata"]!=metadata:
        raise RuntimeError("checkpoint metadata mismatch")
    model=build_teacher(state,device)
    dataset=GNNBenchmarkDataset(root=str(DATA_ROOT),name="CIFAR10",split="test")
    if len(dataset)!=10000:
        raise RuntimeError("full official test size mismatch")
    labels,parts=[],[]
    started=time.monotonic()
    for batch in DataLoader(dataset,batch_size=128,shuffle=False,num_workers=0):
        labels.append(batch.y.view(-1).long().numpy())
        del batch.y
        parts.append(model(batch.to(device)).cpu().numpy())
    y,z=np.concatenate(labels),np.concatenate(parts)
    if len(y)!=10000 or z.shape!=(10000,10) or not np.isfinite(z).all():
        raise RuntimeError("invalid/incomplete test outputs")
    result={"status":"complete","dataset":"CIFAR10","seed":args.teacher_seed,
            "split":"all official test; historically exposed","graphs":10000,
            "checkpoint_sha256":checkpoint_hash,"checkpoint_epoch":int(state["epoch"]),
            "teacher_metadata_sha256":sha256(run/"metadata.json"),"teacher_done_sha256":sha256(run/"done.json"),
            "accuracy":float(np.mean(z.argmax(1)==y)),"script_sha256":sha256(__file__),
            "protocol_sha256":sha256(HERE/"gated-teacher-seed-protocol-v2.md"),
            "processed_data_sha256":metadata["provenance"]["processed_data_sha256"],
            "device":str(device),"elapsed_sec":time.monotonic()-started,
            "qualification":"teacher only; PyG public-architecture adaptation; prior project test exposure"}
    with archive.open("xb") as stream:
        np.savez_compressed(stream,index=np.arange(10000),labels=y,logits=z,predictions=z.argmax(1),
                            report_json=np.array(json.dumps(result)))
    result["prediction_sha256"]=sha256(archive)
    with output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--teacher-run",required=True,type=Path)
    parser.add_argument("--teacher-seed",required=True,type=int,choices=(42,43))
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    evaluate(parser.parse_args())
