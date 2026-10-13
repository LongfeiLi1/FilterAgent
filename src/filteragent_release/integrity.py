"""Verify the public code and independent benchmark manifests."""
import hashlib,json
from .paths import CODE_ROOT,BENCHMARK_ROOT

def verify():
    for root in (CODE_ROOT,BENCHMARK_ROOT):
        entries=json.loads((root/'MANIFEST.json').read_text())['files']
        for name,wanted in entries.items():
            path=(root/name).resolve()
            if not path.is_relative_to(root) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=wanted:
                raise ValueError('Release integrity mismatch: '+name)
    return hashlib.sha256((CODE_ROOT/'MANIFEST.json').read_bytes()).hexdigest()

def manifest():
    digest=verify()
    return {'files':json.loads((CODE_ROOT/'MANIFEST.json').read_text())['files'],
            'source_manifest_sha256':digest,'public_release':True}
