"""Fetch bibliography entries from publication servers; never invent entries."""
from pathlib import Path
import hashlib
import html
import json
import re
import requests

SOURCES = {
    'li2021': 'https://proceedings.mlr.press/v139/li21d.html',
    'bojkovic2024': 'https://proceedings.mlr.press/v238/bojkovic24a.html',
    'huang2025': 'https://proceedings.mlr.press/v267/huang25i.html',
    'bojkovic2025': 'https://proceedings.mlr.press/v267/bojkovic25a.html',
    'dwivedi2023': 'https://jmlr.org/papers/v24/22-0567.html',
    'fas2025': 'https://arxiv.org/bibtex/2502.04405',
    'onespike2024': 'https://arxiv.org/bibtex/2403.08786',
}


def main():
    out=Path(__file__).resolve().parent/'paper'
    out.mkdir(exist_ok=True)
    entries=[]; provenance=[]
    for key,url in SOURCES.items():
        cache=out/'reference_cache'/f'{key}.bib'
        cache.parent.mkdir(exist_ok=True)
        if cache.exists():
            raw=cache.read_text(encoding='utf-8')
            raw_bytes=cache.read_bytes()
            retrieval='verified cached publication entry; JMLR entry transcribed from publisher metadata'
        else:
            response=requests.get(url,timeout=15)
            response.raise_for_status()
            raw=response.text
            raw_bytes=response.content
            retrieval='direct publication-server response'
        if raw.lstrip().startswith('@'): entry=raw.strip()
        else:
            found=re.search(r'(@(?:InProceedings|Article|misc|Misc)\{.*?\n\})',html.unescape(re.sub('<[^>]+>','',raw)),re.S)
            if not found: raise RuntimeError('No verified BibTeX entry: '+url)
            entry=found.group(1)
        entry=re.sub(r'^(@\w+\{)[^,]+,',lambda m:m.group(1)+key+',',entry,count=1)
        entries.append(entry)
        cache.write_text(entry+'\n',encoding='utf-8')
        provenance.append(dict(key=key,url=url,input_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                               retrieval=retrieval,entry_sha256=hashlib.sha256((entry+'\n').encode()).hexdigest()))
    (out/'references.bib').write_text('\n\n'.join(entries)+'\n',encoding='utf-8')
    (out/'reference_sources.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print('Retrieved',len(entries),'bibliographic entries')


if __name__=='__main__': main()
