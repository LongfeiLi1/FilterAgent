"""Frozen Full policy identifiers and canonical hashes."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE_SHA='cbb3ba4347b76606e0fa65f93c0344a8f6ee3ecf24b0b023e4174957bd976d16'
VERSION='filteragent-r7a-full'
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
