from pathlib import Path
import json,re,hashlib,collections,zipfile,subprocess
import pymupdf as fitz
from PIL import Image
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks={}
manifest=json.loads((R/'input-manifest.json').read_text())
checks['originals_unchanged']=all(sha(Path(x['path']))==x['sha256'] and sha(R/x['snapshot'])==x['sha256'] for x in manifest)
for name in ['main.tex','supplement.tex']:
 before=(R/'sources'/name).read_text(encoding='utf-8');after=(R/'revised'/name).read_text(encoding='utf-8')
 checks[name+'_numbers']=collections.Counter(re.findall(r'(?<![A-Za-z])\d+(?:\.\d+)?',before))==collections.Counter(re.findall(r'(?<![A-Za-z])\d+(?:\.\d+)?',after))
 checks[name+'_math']=collections.Counter(re.findall(r'\$[^$]*\$',before))==collections.Counter(re.findall(r'\$[^$]*\$',after))
 for env in ['equation','table','table*','figure']:
  pat=r'\\begin\{'+re.escape(env)+r'\}.*?\\end\{'+re.escape(env)+r'\}'
  checks[name+'_'+env]=re.findall(pat,before,re.S)==re.findall(pat,after,re.S)
 checks[name+'_citations']=re.findall(r'\\cite\{[^}]+\}',before)==re.findall(r'\\cite\{[^}]+\}',after)
checks['disclosure_verbatim']=(R/'sources/main.tex').read_text().split('\\section*{LLM Usage Statement}')[1]==(R/'revised/main.tex').read_text().split('\\section*{LLM Usage Statement}')[1]
for f in ['references.bib','cvpr.sty','ieeenat_fullname.bst']:checks[f+'_unchanged']=(R/'sources'/f).read_bytes()==(R/'revised'/f).read_bytes()
checks['graphics_unchanged']=all(p.read_bytes()==(R.parent/'figures'/p.name).read_bytes() for p in (R/'revised/figures').glob('*') if p.is_file())
out=R/'audit/rendered';out.mkdir(exist_ok=True)
for name in ['main','supplement']:
 d=fitz.open(R/'revised'/(name+'.pdf'));text='\n'.join(p.get_text() for p in d)
 checks[name+'_resolved']=not any(x in text for x in ['??','[?]','[VERIFY]','TBD'])
 checks[name+'_anonymous']=not any(x in text for x in ['xhf','ZtL','github.com','Institution1'])
 checks[name+'_no_overflow']=not re.search('Overfull|undefined|LaTeX Error',(R/'revised'/(name+'.log')).read_text(errors='replace'))
 checks[name+'_pages']=len(d)
 if name=='main':
  n=next(i for i,p in enumerate(d) if 'LLM Usage Statement' in p.get_text());checks['main_body_pages']=n;checks['within_8_pages']=n<=8
 thumbs=[]
 for i,p in enumerate(d):
  p.get_pixmap(matrix=fitz.Matrix(1.2,1.2)).save(out/f'{name}{i+1}.png')
  im=Image.open(out/f'{name}{i+1}.png').convert('RGB');im.thumbnail((430,580));thumbs.append(im)
 sheet=Image.new('RGB',(1290,600*((len(thumbs)+2)//3)),'#ddd')
 for i,im in enumerate(thumbs):sheet.paste(im,((i%3)*430,(i//3)*600))
 sheet.save(out/f'{name}-contact.png')
 fonts=subprocess.run(['pdffonts',str(R/'revised'/(name+'.pdf'))],check=True,capture_output=True,text=True).stdout
 checks[name+'_fonts_embedded']=all(' yes ' in row for row in fonts.splitlines()[2:] if row.strip())
 (R/'audit'/(name+'-fonts.txt')).write_text(fonts)
with zipfile.ZipFile(R/'polished-source.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in ['main.tex','supplement.tex','cvpr.sty','ieeenat_fullname.bst','references.bib']:z.write(R/'revised'/name,name)
 text=(R/'revised/main.tex').read_text()
 for f in re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}',text):z.write(R/'revised'/f,f)
checks['all_pass']=all(v for k,v in checks.items() if isinstance(v,bool))
checks['final_hashes']={str(p.relative_to(R)):sha(p) for p in [R/'revised/main.tex',R/'revised/supplement.tex',R/'revised/main.pdf',R/'revised/supplement.pdf',R/'polished-source.zip']}
(R/'audit/verification.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
if not checks['all_pass']:raise SystemExit(1)
