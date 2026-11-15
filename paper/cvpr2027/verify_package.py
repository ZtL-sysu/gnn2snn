from pathlib import Path
import re,json,hashlib,zipfile,subprocess,shutil
import pandas as pd
import pymupdf as fitz
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks={}
m=json.loads((HERE/'source_sha_manifest.json').read_text())
checks['input_hashes']=all(sha(ROOT/x['path'])==x['sha256'] for x in m['source_files'])
for ds in ['MNIST','CIFAR10']:
 d=pd.read_csv(ROOT/f'research/20261115_decision/runs/intervention_v2/{ds}41/graphs.csv')
 for t in [1,2]:
  q=d[d['T']==t];improved=q.changed_message_mse<q.baseline_message_mse;bad=q.changed_margin<q.baseline_margin
  expect={('MNIST',1):(811,332),('MNIST',2):(811,395),('CIFAR10',1):(928,296),('CIFAR10',2):(928,409)}[(ds,t)]
  checks[f'raw_interventions_{ds}_{t}']=len(q)==2048 and (int(improved.sum()),int((improved&bad).sum()))==expect
for name in ['main','supplement']:
 doc=fitz.open(HERE/(name+'.pdf'));text='\n'.join(p.get_text() for p in doc)
 checks[name+'_anonymous']=not any(x in text for x in ['ZtL','xhf','github.com','C:\\','Institution1','First Author'])
 checks[name+'_resolved']=not any(x in text for x in ['??','[?]','TBD','[VERIFY]'])
 log=(HERE/(name+'.log')).read_text(errors='replace')
 checks[name+'_no_overfull']=not re.search(r'Overfull|undefined|LaTeX Error',log)
 out=HERE/'audit/rendered';out.mkdir(exist_ok=True)
 thumbs=[]
 for i,p in enumerate(doc):
  p.get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(out/f'{name}{i+1}.png')
  im=Image.open(out/f'{name}{i+1}.png').convert('RGB');im.thumbnail((430,580));thumbs.append(im)
 sheet=Image.new('RGB',(1290,600*((len(thumbs)+2)//3)),'#ddd')
 for i,im in enumerate(thumbs):sheet.paste(im,((i%3)*430,(i//3)*600))
 sheet.save(out/f'{name}_contact.png')
 if name=='main':
  llm=next(i+1 for i,p in enumerate(doc) if 'LLM Usage Statement' in p.get_text())
  checks['main_body_pages']=llm-1;checks['main_page_limit']=llm-1<=8
  checks['disclosure_before_references']=text.index('LLM Usage Statement')<text.index('References\n')
 checks[name+'_pages']=len(doc)
 fonts=subprocess.run(['pdffonts',str(HERE/(name+'.pdf'))],capture_output=True,text=True,check=True).stdout
 (HERE/'audit'/(name+'-fonts.txt')).write_text(fonts)
 checks[name+'_fonts_embedded']=all(' yes ' in line for line in fonts.splitlines()[2:] if line.strip())
src=(HERE/'main.tex').read_text();keys=set(re.findall(r'@\w+\{([^,]+),',(HERE/'references.bib').read_text()))
cited={k for a in re.findall(r'\\cite\{([^}]+)\}',src) for k in a.split(',')};checks['citation_keys']=cited<=keys
checks['official_style']=sha(HERE/'cvpr.sty')==sha(HERE/'audit/official-template/cvpr.sty')
shutil.copyfile(ROOT/'research/20261114_story/analysis/frozen_test/aggregate.csv',HERE/'data/historical_full_test.csv')
m['outputs']=[{'path':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted([*(HERE/'data').glob('*.csv'),*(HERE/'figures').glob('*.pdf'),*(HERE/'figures').glob('*.png')])]
(HERE/'source_sha_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
with zipfile.ZipFile(HERE/'paper_source.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in [*(HERE.glob('*.tex')),HERE/'cvpr.sty',HERE/'ieeenat_fullname.bst',HERE/'references.bib',*(HERE/'figures').glob('*.pdf')]:z.write(p,p.relative_to(HERE))
with zipfile.ZipFile(HERE/'evidence_inputs.zip','w',zipfile.ZIP_DEFLATED) as z:
 for x in m['source_files']:z.write(ROOT/x['path'],x['path'])
 for p in (HERE/'early/raw').glob('*'):z.write(p,p.relative_to(ROOT))
 for p in (HERE/'data/centered_validation').glob('*'):z.write(p,p.relative_to(ROOT))
checks['all_checks_pass']=all(v for k,v in checks.items() if isinstance(v,bool))
(HERE/'audit/verification.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
if not checks['all_checks_pass']:raise SystemExit(1)
