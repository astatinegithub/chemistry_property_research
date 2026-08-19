import re
import logging

from io import StringIO
from collections import Counter
from rdkit import rdBase


__all__ = [
    "RDKitLogCapture",
    "MolValidator"
]



class RDKitLogCapture:
    def __init__(self):
        rdBase.LogToPythonLogger()

        self.logger = logging.getLogger("rdkit")
        self.stream = StringIO()
        self.handler = logging.StreamHandler(self.stream)


    def __enter__(self):
        self.logger.addHandler(self.handler)
        return self


    def get(self):
        logs = self.stream.getvalue()

        self.stream.seek(0)
        self.stream.truncate(0)

        return logs


    def __exit__(self, *args):
        self.logger.removeHandler(self.handler)



class MolValidator:
    def __init__(self, allowed_atoms):
        self.allowed_atoms = allowed_atoms
        self.removed = Counter()
        self.total_removed = 0


    def _warning_types(self, logs):
        warning_types = set()

        for line in logs.splitlines():
            line = re.sub(r"^\[[^\]]+\]\s*", "", line.strip())
            line = re.sub(r"atom \d+", "atom N", line)

            if not line:
                continue

            if "ambiguous stereochemistry" in line:
                warning_types.add("ambiguous_stereo")

            elif "not removing hydrogen atom without neighbors" in line:
                warning_types.add("isolated_h")

            elif "Explicit valence" in line:
                warning_types.add("valence")

            elif "Could not sanitize molecule" in line:
                warning_types.add("sanitize")

            else:
                warning_types.add("other")

        return warning_types


    def validate(self, mol, logs=""):
        reasons = set()

        warning_types = self._warning_types(logs)
        reasons.update(warning_types)

        if mol is None:
            reasons.add("invalid")
        else:
            if any(
                atom.GetAtomicNum() not in self.allowed_atoms
                for atom in mol.GetAtoms()
            ):
                reasons.add("unsupported_atom")

            if mol.GetNumBonds() == 0:
                reasons.add("no_bond")

        if reasons:
            self.total_removed += 1

            for reason in reasons:
                self.removed[reason] += 1

            return False

        return True

    def report(self):
        print("====================")
        print("removed molecules:", self.total_removed)

        for reason, count in self.removed.most_common():
            print(f"{reason}: {count}")