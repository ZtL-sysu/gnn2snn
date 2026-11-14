"""Hash contained immutable artifacts while excluding reproducible/ignored caches."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    candidates=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(ROOT);name=rel.as_posix()
        if any(x in rel.parts for x in ['__pycache__','.pytest_cache','clean_compile','pdf_preview']):continue
        if name.startswith('audits/pdf_final/page-') or p.suffix in ['.aux','.bbl','.blg','.fdb_latexmk','.fls','.out','.pyc','.tmp']:continue
        if name in ['artifact-manifest.json','pipeline-state.json','pipeline-state.md']:continue
        assert p.resolve().is_relative_to(ROOT)
        candidates.append(p)
    # Binary NUL framing prevents Windows text-mode CRLF from becoming part of
    # filenames sent to Git's stdin protocol.
    names=b'\0'.join(p.relative_to(ROOT).as_posix().encode('utf-8') for p in candidates)+b'\0'
    ignored=subprocess.run(['git','check-ignore','-z','--stdin'],input=names,stdout=subprocess.PIPE,cwd=ROOT)
    assert ignored.returncode in [0,1]
    ignore_set={(ROOT/p.decode('utf-8')).resolve() for p in ignored.stdout.split(b'\0') if p}
    entries=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in candidates if p.resolve() not in ignore_set]
    data=dict(root=str(ROOT),all_listed_paths_within_root=True,entries=entries,excluded_mutable=['artifact-manifest.json','pipeline-state.json','pipeline-state.md'],excluded_reproducible_cache=['git-ignored compiler/Python/cache files','clean source compilation directory','per-page PDF previews'],external_inputs='Preexisting historical evidence, datasets/checkpoints and installed ARC/skills/runtime are authorized input/cache locations; claim-facing snapshots and final artifacts are in this run root.')
    (ROOT/'artifact-manifest.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    print('Contained portable artifact entries:',len(entries),'ignored candidates:',len(ignore_set))
if __name__=='__main__':main()
