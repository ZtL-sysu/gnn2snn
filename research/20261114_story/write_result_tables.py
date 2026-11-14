"""Render paper tables directly from validated aggregate CSVs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

METHODS=('baseline','node','graph','mix','output','shuffled')
LABELS={'baseline':'Uncalibrated','node':'Local','graph':'Graph','mix':'Mixed','output':'Output bias','shuffled':'Shuffled targets'}


def read(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def fmt(row,metric='accuracy_percent'):
    return f"{float(row[metric+'_mean']):.2f} $\\pm$ {float(row[metric+'_sample_sd']):.2f}"


def table(rows,scope,caption,label):
    index={(r['dataset'],r['method'],int(r['T'])):r for r in rows if r['scope']==scope}
    lines=[r'\begin{table*}[t]',r'\centering\scriptsize',r'\begin{tabular}{llrrrrrr}',r'\toprule',
           'Task & $T$ & '+' & '.join(LABELS[m] for m in METHODS)+r' \\',r'\midrule']
    for dataset in ['MNIST','CIFAR10']:
        for t in [1,2,4,8]:
            lines.append(f'{dataset} & {t} & '+' & '.join(fmt(index[(dataset,m,t)]) for m in METHODS)+r' \\')
        if dataset=='MNIST':lines.append(r'\midrule')
    lines.extend([r'\bottomrule',r'\end{tabular}',f'\\caption{{{caption}}}',f'\\label{{{label}}}',r'\end{table*}'])
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--test-analysis',type=Path)
    p.add_argument('--paper',type=Path,required=True)
    args=p.parse_args();args.paper.mkdir(parents=True,exist_ok=True)
    rows=read(args.analysis/'aggregate.csv');per=read(args.analysis/'per_seed.csv')
    text=[table(rows,'full_validation',
                'Full official validation accuracy (percent), mean $\\pm$ sample SD across three teacher seeds. Each seed uses the same 5,000 graphs; calibration uses 512 training graphs and 64 updates. All six frozen conditions are retained.',
                'tab:validation')]
    lines=[r'\begin{table*}[t]',r'\centering\small',r'\begin{tabular}{llrrrrr}',r'\toprule',
           r'Task & $T$ & Seed 41 & Seed 42 & Seed 43 & Mean $\pm$ SD & $t$-95\% interval \\',r'\midrule']
    for ds in ['MNIST','CIFAR10']:
        for t in [1,2]:
            agg=next(r for r in rows if r['dataset']==ds and r['scope']=='screen_excluded' and r['method']=='paired_graph-minus-node' and int(r['T'])==t)
            vals=[next(r for r in per if r['dataset']==ds and r['scope']=='screen_excluded' and r['method']=='paired_graph-minus-node' and int(r['T'])==t and int(r['seed'])==seed) for seed in [41,42,43]]
            ci=f"[{float(agg['accuracy_percent_t95_low']):+.2f}, {float(agg['accuracy_percent_t95_high']):+.2f}]"
            lines.append(f'{ds} & {t} & '+' & '.join(f"{float(r['accuracy_percent']):+.2f}" for r in vals)+f' & {fmt(agg)} & {ci}'+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\caption{Graph-minus-local accuracy differences (percentage points) on the 4,488 validation graphs outside the current screen. Intervals use three seed-level differences and two degrees of freedom; they remain descriptive on historically exposed validation data.}',r'\label{tab:paired}',r'\end{table*}'])
    text.append('\n'.join(lines))
    if args.test_analysis:
        tests=read(args.test_analysis/'aggregate.csv')
        text.append(table(tests,'historically_exposed_test',
                    'Once-frozen full official test endpoints (10,000 historically exposed graphs per task), mean $\\pm$ sample SD over three teacher seeds. No updates or model selection occur in this evaluation.',
                    'tab:test'))
    (args.paper/'result_tables.tex').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    sources=[args.analysis/'aggregate.csv',args.analysis/'per_seed.csv']
    if args.test_analysis:sources.append(args.test_analysis/'aggregate.csv')
    (args.paper/'table_sources.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n',encoding='utf-8')
    print('Wrote tables from',len(sources),'validated source CSVs')


if __name__=='__main__': main()
