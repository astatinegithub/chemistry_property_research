from collections import Counter
from io import StringIO
import logging
import re


from tqdm import tqdm
from rdkit import Chem, rdBase
from isonet.config import ROOT

rdBase.LogToPythonLogger()

logger = logging.getLogger("rdkit")
stream = StringIO()
handler = logging.StreamHandler(stream)
logger.addHandler(handler)

warnings = Counter()

mols = Chem.SDMolSupplier(
    ROOT + "dataset/raw/Compound_000000001_000500000.sdf"
)

for mol in mols:
    logs = stream.getvalue()
    stream.seek(0)
    stream.truncate(0)

    for line in logs.splitlines():
        line = re.sub(r"^\[[^\]]+\]\s*", "", line.strip())
        line = re.sub(r"atom \d+", "atom N", line)

        if line:
            warnings[line] += 1

logger.removeHandler(handler)

for warning, count in warnings.most_common():
    print(count, "|", warning)