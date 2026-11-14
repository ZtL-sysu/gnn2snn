"""Package the accepted paper and prove clean-extraction compilation consistency."""
import hashlib,json,os,re,subprocess,unicodedata,zipfile
from pathlib import Path
import pymupdf
ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def normalized(text):return re.sub(r'\s+','',unicodedata.normalize('NFKC',text))
def main():
    out=ROOT/'delivery';out.mkdir(exist_ok=True)
    archive=out/'paper_source.zip';assert not archive.exists(),'Version rather than overwrite an accepted archive'
    files=[ROOT/'paper'/n for n in ['main.tex','main.pdf','references.bib','manuscript.md','manuscript-metadata.json','captions.json','render_tex.py','build_tables.py']]
    files+=sorted((ROOT/'paper/tables').glob('*.tex'))
    files+=[ROOT/'figures'/n for n in ['framework.png','ideal_sensitivity.pdf','allocation_contrasts.pdf','mechanism_runtime.pdf','ideal_sensitivity.svg','allocation_contrasts.svg','mechanism_runtime.svg','plot_study.py','source_hashes.json','framework_generation.md']]
    readme=out/'SOURCE_README.md'
    assert readme.is_file()
    file_manifest={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in files}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,str(p.relative_to(ROOT)).replace('\\','/'))
        z.write(readme,'SOURCE_README.md')
        z.writestr('source_manifest.json',json.dumps(file_manifest,indent=2)+'\n')
    clean=out/'clean_compile';assert not clean.exists()
    clean.mkdir()
    with zipfile.ZipFile(archive) as z:
        for n in z.namelist():assert not Path(n).is_absolute() and '..' not in Path(n).parts
        z.extractall(clean)
    for name,h in file_manifest.items():assert sha(clean/name)==h,name
    env=os.environ.copy();env['PATH']='C:/texlive/2026/bin/windows;'+env.get('PATH','')
    # Remove only the extracted PDF file to force a fresh source build. No recursive deletion.
    (clean/'paper/main.pdf').unlink()
    res=subprocess.run(['C:/texlive/2026/bin/windows/latexmk.exe','-pdf','-interaction=nonstopmode','-halt-on-error','main.tex'],cwd=clean/'paper',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    (out/'clean_compile_log.txt').write_text(res.stdout,encoding='utf-8');assert res.returncode==0
    log=(clean/'paper/main.log').read_text(encoding='utf-8',errors='replace')
    warnings=[s for s in log.splitlines() if any(q in s for q in ['Overfull','undefined','LaTeX Warning','Package natbib Warning'])];assert not warnings,warnings
    accepted=pymupdf.open(ROOT/'paper/main.pdf');rebuilt=pymupdf.open(clean/'paper/main.pdf')
    assert len(accepted)==len(rebuilt)==12
    text_equal=normalized(''.join(p.get_text() for p in accepted))==normalized(''.join(p.get_text() for p in rebuilt));assert text_equal
    pixel_equal=[]
    for a,b in zip(accepted,rebuilt):
        equal=hashlib.sha256(a.get_pixmap().samples).digest()==hashlib.sha256(b.get_pixmap().samples).digest()
        pixel_equal.append(equal);assert equal
    result=dict(archive_sha256=sha(archive),accepted_pdf_sha256=sha(ROOT/'paper/main.pdf'),clean_pdf_sha256=sha(clean/'paper/main.pdf'),source_files_verified=len(file_manifest),pages=12,normalized_pdf_text_identical=text_equal,all_page_render_pixels_identical=all(pixel_equal),warnings=warnings,note='PDF file hashes may differ through build metadata; source entries, text and all rendered pages are compared. Clean compilation is not an experiment rerun.')
    (out/'package_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
if __name__=='__main__':main()
