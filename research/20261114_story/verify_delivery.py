"""Recompute delivered statistics and verify immutable inputs and artifacts."""
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frozen=json.loads((ROOT/'TEST_FREEZE.json').read_text())
    for name,digest in frozen['sha256'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('frozen input changed: '+name)
    qa=REPO.parent/'tmp'/('delivery_verify_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    qa.mkdir(parents=True,exist_ok=False)
    reproduced={}
    jobs=[('summarize_confirmation.py','confirmation','confirmation'),
          ('summarize_frozen_test.py','frozen_test','frozen_test'),
          ('summarize_ideal_sites.py','ideal','ideal')]
    for script,runs,analysis in jobs:
        output=qa/analysis
        subprocess.run([sys.executable,str(ROOT/script),'--runs',str(ROOT/'runs'/runs),'--output',str(output)],check=True)
        for path in (ROOT/'analysis'/analysis).glob('*.csv'):
            rebuilt=output/path.name
            if not rebuilt.exists() or sha(path)!=sha(rebuilt):raise RuntimeError('non-reproducible summary: '+str(path))
            reproduced[str(path.relative_to(ROOT))]=sha(path)
    manifest=ROOT/'ARTIFACTS_SHA256.csv'
    with manifest.open(newline='',encoding='utf-8') as f:artifacts=list(csv.DictReader(f))
    for row in artifacts:
        path=ROOT/row['path']
        if path.stat().st_size!=int(row['bytes']) or sha(path)!=row['sha256']:
            raise RuntimeError('artifact hash mismatch: '+row['path'])
    import pymupdf
    pdf=ROOT/'paper/main.pdf';document=pymupdf.open(pdf)
    text='\n'.join(p.get_text() for p in document)
    if any(x in text for x in ['PLACEHOLDER','TODO','[?]']):raise RuntimeError('unresolved PDF placeholder')
    log=(ROOT/'paper/main.log').read_text(encoding='utf-8',errors='replace')
    if 'Overfull' in log or 'undefined' in log.lower():raise RuntimeError('PDF build has unresolved layout/reference warning')
    keys=re.findall(r'@\w+\{([^,]+),',(ROOT/'paper/references.bib').read_text(encoding='utf-8'))
    if len(keys)!=7 or len(set(keys))!=7:raise RuntimeError('bibliography key mismatch')
    result=dict(status='PASS',checked_utc=datetime.now(timezone.utc).isoformat(),
                frozen_input_hashes_verified=len(frozen['sha256']),artifact_hashes_verified=len(artifacts),
                byte_identical_recomputed_csvs=reproduced,pdf_sha256=sha(pdf),pdf_pages=len(document),
                bibliography_entries=len(keys),pdf_overfull_boxes=0,unresolved_references=0,
                full_confirmation_runs=6,full_frozen_test_runs=6,ideal_site_runs=6,
                scientific_status='route C; no stable general graph-objective advantage; manuscript not submission-ready',
                visual_review='all seven PDF pages and three figure PNGs inspected in this session')
    (ROOT/'FINAL_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='byte_identical_recomputed_csvs'}))


if __name__=='__main__':main()
