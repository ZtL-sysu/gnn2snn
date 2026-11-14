"""Build manuscript-facing facts from accepted summaries; no experiment or model call."""
from pathlib import Path
import csv,hashlib,json,shutil
ROOT=Path(__file__).resolve().parent
def rows(p):
    with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def main():
    output=ROOT/'writing';output.mkdir(exist_ok=True)
    source=ROOT/'inputs/historical';source.mkdir(exist_ok=True)
    old=ROOT.parent/'20261114_story'
    for name,origin in [('frozen_test.csv','analysis/frozen_test/aggregate.csv'),('ideal.csv','analysis/ideal/aggregate.csv'),('confirmation.csv','analysis/confirmation/aggregate.csv')]:
        shutil.copyfile(old/origin,source/name)
    evidence=lambda p:[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())]
    facts=[]
    def fact(id,kind,statement,path,**kwargs):
        facts.append(dict(id=id,kind=kind,statement=statement,status='observed_verified',evidence=evidence(path),**kwargs))
    fact('design','setup','Frozen four-layer residual GCNs classify MNIST and CIFAR10 superpixel graphs. Four node branches and two readout activations use signed eight-level differential events. Subsequent studies preserve teacher weights.',ROOT/'allocation_core.py')
    fact('selection','setup','New comparisons use 512 training calibration graphs and 512 validation graphs from fixed indices. Both official validation and test sets have historical exposure. New methods are selected without loading test; these are exploratory comparisons, not independent external confirmation.',ROOT/'RESCUE_PROTOCOL.md')
    fact('budget','method','The allocation candidate supplements only the first site at its first tick with signed quantized residual events for ceil(n/4) nodes per graph. Local, feature-only, permuted and actual-message policies share candidates and opportunity budgets. Actual nonzero event counts can differ.',ROOT/'allocation.py')
    fact('scope','method','For a fixed folded graph convolution, first-site error E changes the next preactivation by A E W transpose. Residual identity and later nonlinearities remain separate. This linear identity motivates a proxy, not a novel theorem or an exact final-classification model.',ROOT/'objective.py')
    for r in rows(source/'frozen_test.csv'):
        if r['method'] not in ['baseline','node','graph']:continue
        id=f"historical_{r['dataset']}_{r['method']}_T{r['T']}"
        mean=float(r['accuracy_percent_mean']);sd=float(r['accuracy_percent_sample_sd'])
        fact(id,'result',f"{r['dataset']} historical {r['method']} T{r['T']} accuracy is {mean:.4f}% ± {sd:.4f}% sample SD across three teacher seeds.",source/'frozen_test.csv',
             model='four-layer GCN teachers 41/42/43',condition=f"{r['dataset']}, {r['method']}, T{r['T']}",metric='accuracy',value=mean,unit='percent',aggregation='three-teacher mean',uncertainty=dict(sample_sd=sd,n_seeds=3),measurement_scope='10000 historically exposed official test graphs, descriptive; not new method selection')
        if r['method']=='baseline' and r['T']=='1':
            teacher=float(r['teacher_accuracy_percent_mean']);teacher_sd=float(r['teacher_accuracy_percent_sample_sd'])
            fact(f"teacher_{r['dataset']}",'result',f"{r['dataset']} ANN teacher accuracy averages {teacher:.4f}% with sample SD {teacher_sd:.4f}%.",source/'frozen_test.csv',model='GCN teachers41/42/43',condition=r['dataset']+' ANN teacher',metric='accuracy',value=teacher,unit='percent',aggregation='three-teacher mean',uncertainty=dict(sample_sd=teacher_sd,n_seeds=3),measurement_scope='10000 historically exposed official test graphs; descriptive teacher quality')
    for r in rows(source/'ideal.csv'):
        if r['scope']!='full_validation' or r['T']!='1' or r['method'] not in ['site0','nodes0_3','head4_5','all6']:continue
        mean=float(r['agreement_gap_closed_mean']);id=f"ideal_{r['dataset']}_{r['method']}"
        fact(id,'result',f"Ideal continuous recovery at {r['method']} closes {100*mean:.2f}% of the teacher-prediction disagreement gap on {r['dataset']} at T1.",source/'ideal.csv',
             model='GCN teachers41/42/43',condition=f"{r['dataset']} T1 ideal {r['method']}",metric='teacher_disagreement_gap_closed',value=mean,unit='fraction',aggregation='three-teacher mean',uncertainty=dict(sample_sd=float(r['agreement_gap_closed_sample_sd']),n_seeds=3),measurement_scope='5000 historically exposed validation graphs; conditional continuous diagnostic, unequal group sizes, non-additive, not deployable SNN')
    for filename,phase in [('endpoints.csv','initial'),('rescue_endpoints.csv','rescue')]:
        path=ROOT/'analysis'/filename
        for i,r in enumerate(rows(path)):
            if r['T']!='1':continue
            mean=float(r['accuracy_percent']);id=f'{phase}_{i}'
            fact(id,'result',f"{r['dataset']} {r['phase']} {r['method']} T1 accuracy is {mean:.4f}%.",path,
                 model='GCN teacher41',condition=f"{r['dataset']}, {r['phase']}, {r['method']}, T1",metric='accuracy',value=mean,unit='percent',aggregation='one frozen teacher over 512 validation graphs',uncertainty='no teacher-seed confidence interval; exploratory',measurement_scope='historically exposed validation512; different parameter/event/optimizer budgets must be distinguished')
    for r in rows(ROOT/'analysis/rescue_paired.csv'):
        if r['control'] not in ['local','baseline','identity','shuffled','strong_local_lr01']:continue
        d=float(r['delta_pp']);id=f"allocation_delta_{r['dataset']}_{r['T']}_{r['control']}"
        fact(id,'result',f"Actual-message allocation minus {r['control']} on {r['dataset']} T{r['T']} is {d:+.4f} accuracy percentage points.",ROOT/'analysis/rescue_paired.csv',
             model='GCN teacher41',condition=f"{r['dataset']}, T{r['T']}, message vs {r['control']}",metric='paired_accuracy_difference',value=d,unit='percentage points',aggregation='paired 512-graph mean difference',uncertainty=dict(descriptive_normal_low=float(r['paired_normal_low']),descriptive_normal_high=float(r['paired_normal_high'])),measurement_scope='one teacher, exploratory; interval concerns graph sampling only, not seed variability, multiplicity or external generalization')
    costpath=ROOT/'analysis/allocation_mechanism_cost.csv'
    for r in rows(costpath):
        if r['policy'] not in ['local','message']:continue
        value=float(r['median_ms_per_graph'])
        fact(f"runtime_{r['dataset']}_{r['policy']}",'result',f"{r['dataset']} T1 {r['policy']} allocation has median synchronized diagnostic runtime {value:.4f} ms per graph.",costpath,
             model='GCN teacher41 converted implementation',condition=f"{r['dataset']} T1 {r['policy']}, cached batch32",metric='synchronized_host_runtime',value=value,unit='ms/graph',aggregation='median of20 randomized-interleaved repeats after5 warmups',uncertainty='repeat variation retained in raw records; no population CI',measurement_scope='RTX3060 diagnostic implementation, includes policy instrumentation, excludes teacher forward and data transfer; not energy or optimized deployment bound')
        mse=float(r['actual_normalized_message_mse_after'])
        fact(f"messageMSE_{r['dataset']}_{r['policy']}",'result',f"{r['dataset']} {r['policy']} allocation has mean normalized actual next-message MSE {mse:.6f} after correction.",costpath,
             model='GCN teacher41',condition=f"{r['dataset']} T1 {r['policy']}",metric='normalized_next_message_MSE',value=mse,unit='dimensionless',aggregation='mean over512 validation graphs',uncertainty='one teacher, no seed-level interval',measurement_scope='actual normalized A E W_next transpose, all policies measured under true operator')
    brief=dict(question='Does critical-site sensitivity translate into realizable, graph-specific improvements in low-timestep GCN-to-SNN conversion?',
        answer='The tested strong-teacher setting separates ideal sensitivity from practical calibration. Message objectives and fixed-budget residual selection do not establish robust structural advantage over matched local and counterfactual controls.',
        audience='Researchers designing conversion and graph quantization methods who need to distinguish realizability, target mismatch and event-budget effects.',
        domain_bridge='Graph propagation makes quantization errors directional. Actual next-message perturbations and within-graph operator controls test whether exploiting that direction yields classification gains beyond local reconstruction.',
        priorities={'main':['sensitivity versus realizability','matched structure contribution tests'],'supporting':['strong-teacher fidelity','local calibration gains'],'diagnostic':['surrogate and hard optimization','alphabet capacity','event budget and dense runtime']},
        terminology={'T':'active ticks','event':'signed graded scalar amplitude','ideal recovery':'continuous causal diagnostic','opportunity budget':'upper bound on extra per-channel events, not equal actual energy'},
        paragraph_plan=[],qualification_placement=[],section_jobs={'Introduction':'Present low-prefix graph conversion and sensitivity-realizability question.','Related Work':'Separate known calibration, coding, graph quantization and adaptive budgets from this empirical test.','Method':'Define teachers, exact event dynamics, loss comparisons, ideal diagnostic and bounded allocation.','Experiments and Results':'Organize comparisons by scientific question with their own matched scopes and uncertainty.','Discussion':'Explain practical capacity and proxy limitations and hardware implications.','Conclusion':'State evidence-scaled empirical answer and remaining method gap.'},
        author_constraints={'edit_scope':['structural','scientific prose'],'protected_sections':[],'protected_artifacts':[],'declaration_mode':'author-managed','allow_new_experiments':False,'editing_round_budget':3},
        journal_requirements={'source':'no target venue specified','length':'content-led; no invented submission page limit','explicit_author_requirements':['do not claim successful method without evidence','preserve negative results','distinguish old and new work']})
    packet=dict(schema_version=1,brief=brief,facts=facts,internal={'state':'prepared scientific packet, not accepted P6/ARC execution; diagnostic cost facts audited; bibliography and final scientific content to be reviewed separately'})
    target=output/'scientific_packet_v1.json';target.write_text(json.dumps(packet,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(facts=len(facts),output=str(target))))
if __name__=='__main__':main()
