"""Verify and unpack the compact local evidence bundle. No training or pickle loading."""
from pathlib import Path
import hashlib,json,os,shutil,zipfile


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    root=Path(__file__).resolve().parents[1]
    if os.name=='nt' and not str(root).startswith('\\\\?\\'):root=Path('\\\\?\\'+str(root))
    folder=root/'artifacts';manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    archive=folder/manifest['zip']
    if archive.stat().st_size!=manifest['zip_bytes'] or sha(archive)!=manifest['zip_sha256']:
        raise RuntimeError('evidence ZIP hash mismatch')
    restored=0
    with zipfile.ZipFile(archive) as z:
        for item in manifest['files']:
            target=(root/item['path']).resolve()
            if not target.is_relative_to(root.resolve()):raise RuntimeError('unsafe path')
            if target.exists():
                if target.stat().st_size==item['bytes'] and sha(target)==item['sha256']:continue
                raise FileExistsError('refusing different existing file: '+str(target))
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+'.restoring')
            with z.open(item['path']) as source,temporary.open('xb') as out:shutil.copyfileobj(source,out,4*1024*1024)
            if temporary.stat().st_size!=item['bytes'] or sha(temporary)!=item['sha256']:
                raise RuntimeError('restored file hash mismatch: '+item['path'])
            temporary.rename(target);restored+=1
    print(json.dumps({'status':'complete','files_verified':len(manifest['files']),'newly_restored':restored}))


if __name__=='__main__':main()
