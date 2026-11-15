"""Select offline capacity controls from a fixed bank; no deployable method claim."""
import csv
import json
from pathlib import Path
import numpy as np
from analyze_cached import accuracy_bootstrap,sha256

HERE=Path(__file__).resolve().parent


def class_margins(z,teacher_class):
    # candidate, graph, class; same frozen teacher class for every candidate.
    other=z.copy()
    other[:,np.arange(z.shape[1]),teacher_class]=-np.inf
    return z[:,np.arange(z.shape[1]),teacher_class]-other.max(2)


def budget_mask(a,margins,budget):
    valid=(a['pairs'][:,:,0]>=0);valid[0]=True
    mask=valid.copy()
    if budget!='first_site_only':
        for T in (1,2):
            events=a[f'events_T{T}'];delivery=a[f'deliveries_T{T}'];amp=a[f'amplitude_T{T}']
            mask&=(events==events[:1]) if budget=='exact_events' else (events<=events[:1])
            mask&=delivery<=delivery[:1]
            mask&=amp<=amp[:1]+1e-9*np.maximum(1,amp[:1])
    mask&=margins[2]>=margins[2][:1]-1e-7
    assert mask[0].all()
    return mask


def main():
    out=HERE/'analysis/swap_oracle';out.mkdir(parents=True,exist_ok=True)
    rows=[];graphrows=[];gate={};hashes={};rng=np.random.default_rng(20261117)
    for ds in ('MNIST','CIFAR10'):
        root=HERE/'runs/swap_oracle'/f'{ds}41'
        metadata=json.loads((root/'metadata.json').read_text())
        path=root/'candidates.npz'
        assert sha256(path)==metadata['candidate_npz_sha256']
        assert metadata['training_only'] and not metadata['validation_loaded'] and not metadata['test_loaded']
        for name,digest in metadata['source_sha256'].items():assert sha256(root/'source'/name)==digest
        hashes[str(path.relative_to(HERE))]=sha256(path)
        a=dict(np.load(path));cls=a['teacher'].argmax(1);labels=a['labels'];n=len(labels);g=np.arange(n)
        assert n==128 and len(a['logits_T1'])==9
        margins={T:class_margins(a[f'logits_T{T}'],cls) for T in (1,2)}
        choices_by_budget={}
        for budget in ('exact_events','nonincrease','first_site_only'):
            feasible=budget_mask(a,margins,budget)
            scores={'local':a['local_mse'],'message':a['message_mse'],'shuffled':a['shuffled_mse'],'oracle':-margins[1]}
            selected={key:np.where(feasible,value,np.inf).argmin(0) for key,value in scores.items()}
            choices_by_budget[budget]=selected
            feasible_graphs=int(feasible[1:].any(0).sum())
            for method,chosen in selected.items():
                for T in (1,2):
                    z=a[f'logits_T{T}'][chosen,g];pred=z.argmax(1);base=a[f'logits_T{T}'][0].argmax(1)
                    delta=(pred==labels).astype(float)-(base==labels).astype(float)
                    mean,lo,hi=accuracy_bootstrap(delta,rng)
                    event=a[f'events_T{T}'][chosen,g];delivery=a[f'deliveries_T{T}'][chosen,g]
                    amplitude=a[f'amplitude_T{T}'][chosen,g]
                    if budget!='first_site_only':
                        assert np.all(event<=a[f'events_T{T}'][0])
                        assert np.all(delivery<=a[f'deliveries_T{T}'][0])
                        assert np.all(amplitude<=a[f'amplitude_T{T}'][0]+1e-9*np.maximum(1,a[f'amplitude_T{T}'][0]))
                    if budget=='exact_events':assert np.array_equal(event,a[f'events_T{T}'][0])
                    row=dict(dataset=ds,budget=budget,policy=method,T=T,graphs=n,
                        feasible_nontrivial_graphs=feasible_graphs,selected_swap_graphs=int((chosen!=0).sum()),
                        accuracy_percent=float(100*np.mean(pred==labels)),baseline_accuracy_percent=float(100*np.mean(base==labels)),
                        teacher_agreement_percent=float(100*np.mean(pred==cls)),baseline_teacher_agreement_percent=float(100*np.mean(base==cls)),
                        mean_teacher_class_margin_change=float((margins[T][chosen,g]-margins[T][0]).mean()),
                        accuracy_delta_pp=mean,paired_graph_bootstrap95_low_pp=lo,paired_graph_bootstrap95_high_pp=hi,
                        events=int(event.sum(dtype=np.int64)),baseline_events=int(a[f'events_T{T}'][0].sum(dtype=np.int64)),
                        branch_delivery_proxy=int(delivery.sum(dtype=np.int64)),amplitude_sum=float(amplitude.sum()))
                    rows.append(row)
                    for j,idx in enumerate(a['indices']):
                        graphrows.append(dict(dataset=ds,budget=budget,policy=method,T=T,index=int(idx),choice=int(chosen[j]),
                            feasible_nontrivial_candidates=int(feasible[1:,j].sum()),
                            teacher_margin_change=float(margins[T][chosen[j],j]-margins[T][0,j]),
                            baseline_events=int(a[f'events_T{T}'][0,j]),events=int(event[j]),
                            baseline_amplitude=float(a[f'amplitude_T{T}'][0,j]),amplitude=float(amplitude[j]),
                            teacher_class=int(cls[j]),label=int(labels[j]),prediction=int(pred[j]),baseline_prediction=int(base[j])))
        sel=choices_by_budget['nonincrease']['oracle']
        improve=(margins[1][sel,g]-margins[1][0])>=.1
        gain1=100*float(np.mean(a['logits_T1'][sel,g].argmax(1)==cls)-np.mean(a['logits_T1'][0].argmax(1)==cls))
        gain2=100*float(np.mean(a['logits_T2'][sel,g].argmax(1)==cls)-np.mean(a['logits_T2'][0].argmax(1)==cls))
        gate[ds]=dict(graphs_with_margin_gain_at_least_0_1=int(improve.sum()),fraction=float(improve.mean()),
            T1_teacher_agreement_gain_pp=gain1,T2_teacher_agreement_gain_pp=gain2,
            capacity_admission_pass=bool(improve.mean()>.1 and gain1>=.5 and gain2>=0))
    for name,data in [('endpoints.csv',rows),('per_graph.csv',graphrows)]:
        with (out/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    verdict=dict(capacity_gate_pass=all(v['capacity_admission_pass'] for v in gate.values()),per_dataset=gate,
        proposal_rng_seed=20261117,teacher_seed=41,proposal_seed_evidence='propose_swaps default in saved source/event_swap.py; graph-ID-derived independent streams',
        deployable_method_established=False,independent_graph_contribution_established=False,
        qualification='training-only optimistic oracle; selection and evaluation use same graphs; descriptive bootstrap is not generalization',source_hashes=hashes)
    (out/'gate.json').write_text(json.dumps(verdict,indent=2)+'\n')
    lines=['# Two-event reassignment capacity diagnostic','',verdict['qualification'], '',
        '|Dataset|Budget|Policy|T|Accuracy %|Teacher agreement %|Feasible graphs|Selected swaps|Total events / baseline|',
        '|---|---|---|---:|---:|---:|---:|---:|---|']
    for r in rows:lines.append(f"|{r['dataset']}|{r['budget']}|{r['policy']}|{r['T']}|{r['accuracy_percent']:.4f}|{r['teacher_agreement_percent']:.4f}|{r['feasible_nontrivial_graphs']}|{r['selected_swap_graphs']}|{r['events']}/{r['baseline_events']}|")
    lines+=['','Capacity continuation gate: **'+('PASS' if verdict['capacity_gate_pass'] else 'FAIL')+'**. This is not a deployment/method gate or independent structure proof. All candidate logits, labels, graph IDs, coordinate pairs and costs are retained.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(verdict,indent=2))


if __name__=='__main__':main()
