"""Restore the complete original Wind dataset from individually downloaded ZIP parts."""
from pathlib import Path
import hashlib
import gzip
import json
import os
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parent
PARTS=ROOT/'data'/'_wind_parts'
DEST=ROOT/'data'/'wind_standardized.csv'
MANIFEST=ROOT/'WIND_DELIVERY_MANIFEST.json'

def sha256_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def run():
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    if DEST.exists() and DEST.stat().st_size==manifest['original_bytes'] and sha256_file(DEST)==manifest['original_sha256']:
        print('WIND DATA ALREADY VERIFIED:',DEST)
        return 0
    required=[PARTS/x['filename'] for x in manifest['parts']]
    missing=[str(p) for p in required if not p.is_file()]
    if missing:
        print('MISSING WIND PARTS:',*missing,sep='\n  ',file=sys.stderr)
        return 2
    for p, info in zip(required,manifest['parts']):
        if p.stat().st_size!=info['size_bytes'] or sha256_file(p)!=info['sha256']:
            print('DAMAGED WIND PART:',p,file=sys.stderr)
            return 3
    DEST.parent.mkdir(parents=True,exist_ok=True)
    compressed=ROOT/'data'/'_wind_data_temp.gz'
    uncompressed=ROOT/'data'/'_wind_data_temp.csv'
    try:
        with compressed.open('wb') as dst:
            for part in required:
                with part.open('rb') as src:shutil.copyfileobj(src,dst,2*1024*1024)
        if compressed.stat().st_size!=manifest['gzip_bytes'] or sha256_file(compressed)!=manifest['gzip_sha256']:
            raise ValueError('Compressed Wind data checksum mismatch')
        with gzip.open(compressed,'rb') as src,uncompressed.open('wb') as dst:
            shutil.copyfileobj(src,dst,2*1024*1024)
        if uncompressed.stat().st_size!=manifest['original_bytes'] or sha256_file(uncompressed)!=manifest['original_sha256']:
            raise ValueError('Wind CSV checksum mismatch')
        os.replace(uncompressed,DEST)
        print('WIND DATA RESTORED AND VERIFIED:',DEST)
        return 0
    except Exception as exc:
        print('RESTORE FAILED:',exc,file=sys.stderr)
        return 1
    finally:
        compressed.unlink(missing_ok=True)
        uncompressed.unlink(missing_ok=True)
if __name__=='__main__':sys.exit(run())
