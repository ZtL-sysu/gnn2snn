"""Generate paper tables directly from scientific CSVs with source hashes."""
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent/'tables'
def read(name):
    with (ROOT/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write(name,text):
    (OUT/name).write_text(text+'\n',encoding='utf-8')
def main():
    OUT.mkdir(exist_ok=True)
    files=['inputs/historical/frozen_test.csv','analysis/endpoints.csv','analysis/rescue_endpoints.csv','analysis/allocation_mechanism_cost.csv','analysis/rescue_paired.csv']
    historical=read(files[0]);initial=read(files[1]);rescue=read(files[2]);cost=read(files[3])
    text=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Conversion & $T=1$ & $T=2$ & $T=4$ & $T=8$ \\',r'\midrule']
    for dataset in ['MNIST','CIFAR10']:
        ref=next(r for r in historical if r['dataset']==dataset)
        text.append(r'\multicolumn{5}{l}{'+dataset+': teacher '+f"{float(ref['teacher_accuracy_percent_mean']):.2f}"+r'$\pm$'+f"{float(ref['teacher_accuracy_percent_sample_sd']):.2f}"+r'} \\')
        for method,label in [('baseline','Uncalibrated'),('node','Local'),('graph','Graph logits')]:
            cells=[]
            for t in ['1','2','4','8']:
                r=next(r for r in historical if r['dataset']==dataset and r['method']==method and r['T']==t)
                cells.append(f"{float(r['accuracy_percent_mean']):.2f}"+r'$\pm$'+f"{float(r['accuracy_percent_sample_sd']):.2f}")
            text.append(label+' & '+' & '.join(cells)+r' \\')
        text.append(r'\midrule')
    text[-1]=r'\bottomrule';text.append(r'\end{tabular}');write('historical.tex','\n'.join(text))
    conditions=[('screen','baseline','Uncalibrated','0'),('screen','all_local',r'All-local ($10^{-3}$)','693'),('optimizer_check','all_local',r'All-local ($10^{-2}$)','693'),('screen','first_local',r'First-local ($10^{-3}$)','146'),('screen','first_message',r'First-message ($10^{-3}$)','146'),('optimizer_check','first_local',r'First-local ($10^{-2}$)','146'),('optimizer_check','first_message',r'First-message ($10^{-2}$)','146'),('hard_probe','first_local','Hard-grid local','146'),('hard_probe','first_message','Hard-grid message','146'),('book_screen','book_local','Local alphabet','1168'),('book_screen','book_message','Diagonal message alphabet','1168'),('full_book_screen','book_local','Refined local alphabet','1168'),('full_book_screen','book_message','Refined message alphabet','1168')]
    text=[r'\begin{tabular}{lrrr}',r'\toprule',r'Condition & Scalars & MNIST & CIFAR10 \\',r'\midrule']
    for phase,method,label,n in conditions:
        cells=[]
        for d in ['MNIST','CIFAR10']:
            r=next(r for r in initial if r['phase']==phase and r['dataset']==d and r['method']==method and r['T']=='1')
            cells.append(f"{float(r['accuracy_percent']):.2f}")
        text.append(label+' & '+n+' & '+' & '.join(cells)+r' \\')
    text.extend([r'\bottomrule',r'\end{tabular}']);write('realizability.tex','\n'.join(text))
    text=[r'\begin{tabular}{lrrrr}',r'\toprule',r' & \multicolumn{2}{c}{MNIST} & \multicolumn{2}{c}{CIFAR10} \\',r'Policy & $T=1$ & $T=2$ & $T=1$ & $T=2$ \\',r'\midrule']
    labels={'baseline':'Base (no extra events)','random':'Random','local':'Local residual','identity':'Feature-only','shuffled':'Permuted graph','message':'Actual graph'}
    for method,label in labels.items():
        values=[]
        for d in ['MNIST','CIFAR10']:
            for t in ['1','2']:
                r=next(r for r in rescue if r['phase']=='allocation' and r['dataset']==d and r['method']==method and r['T']==t)
                values.append(f"{float(r['accuracy_percent']):.2f}")
        text.append(label+' & '+' & '.join(values)+r' \\')
    text.extend([r'\bottomrule',r'\end{tabular}']);write('allocation.tex','\n'.join(text))
    text=[r'\begin{tabular}{lrrrr}',r'\toprule',r' & \multicolumn{2}{c}{Message MSE ($\times10^{-3}$)} & \multicolumn{2}{c}{Time (ms/graph)} \\',r'Policy & MNIST & CIFAR10 & MNIST & CIFAR10 \\',r'\midrule']
    for method,label in [('baseline','Base'),('local','Local residual'),('message','Actual graph')]:
        mse=[];runtime=[]
        for d in ['MNIST','CIFAR10']:
            r=next(x for x in cost if x['dataset']==d and x['policy']==('message' if method=='baseline' else method))
            mse.append(f"{1000*float(r['actual_normalized_message_mse_before' if method=='baseline' else 'actual_normalized_message_mse_after']):.3f}")
            runtime.append(f"{float(r['baseline_median_ms_per_graph' if method=='baseline' else 'median_ms_per_graph']):.3f}")
        text.append(label+' & '+' & '.join(mse+runtime)+r' \\')
    text.extend([r'\bottomrule',r'\end{tabular}']);write('mechanism_cost.tex','\n'.join(text))
    provenance={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
    (OUT/'sources.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print('Generated four source-bound paper tables')
if __name__=='__main__':main()
