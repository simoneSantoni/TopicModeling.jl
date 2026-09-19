"""Shared I/O for the Python reference runs: LDA-C corpora in, raw float64 topic matrices out."""
import json, sys, time
import numpy as np

def read_ldac(path):
    docs = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            pairs = [p.split(":") for p in parts[1:]]
            docs.append([(int(t), int(c)) for t, c in pairs])
    return docs

def read_vocab(path):
    with open(path) as f:
        return [l.rstrip("\n") for l in f]

def expand(doc):
    out = []
    for t, c in doc:
        out.extend([t] * c)
    return out

def save_result(prefix, phi, meta, alpha=None):
    """phi: K x V (row-major float64). meta: dict with timings and settings."""
    phi = np.ascontiguousarray(phi, dtype="<f8")
    phi.tofile(prefix + ".phi.bin")
    meta = dict(meta, K=int(phi.shape[0]), V=int(phi.shape[1]))
    if alpha is not None:
        meta["alpha"] = [float(a) for a in np.atleast_1d(alpha)]
    with open(prefix + ".json", "w") as f:
        json.dump(meta, f, indent=1)
