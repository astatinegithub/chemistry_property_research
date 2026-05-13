import os, re
from pathlib import Path

def str2path(path: str) -> str:
    """input example : 
    code\\utils\\path.py

    result example :"""
    rlt = re.sub(r"\\", "/", path)
    print(rlt)
    rlt = Path.resolve(Path(rlt))
    return rlt