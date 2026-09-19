"""CTM reference run: tomotopy.CTModel (C++; Gibbs sampling with a logistic-normal prior, Mimno et al. 2008)."""
import argparse, time
import numpy as np, tomotopy as tp
from common import read_ldac, read_vocab, expand, save_result
ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True); ap.add_argument("--vocab", required=True); ap.add_argument("--out", required=True)
ap.add_argument("-K", type=int, default=20); ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--threads", type=int, default=1); ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
docs = read_ldac(a.train); vocab = read_vocab(a.vocab); V = len(vocab)
m = tp.CTModel(k=a.K, seed=a.seed)
for d in docs:
    m.add_doc([vocab[t] for t in expand(d)])
t0 = time.perf_counter(); m.train(a.iters, workers=a.threads); secs = time.perf_counter() - t0
idx = {w: i for i, w in enumerate(vocab)}; cols = [idx[w] for w in m.used_vocabs]
phi = np.zeros((a.K, V))
for k in range(a.K):
    phi[k, cols] = m.get_topic_word_dist(k)
save_result(a.out, phi / phi.sum(1, keepdims=True), dict(impl="tomotopy CTModel", seconds=secs, iters=a.iters, threads=a.threads, ll_per_word=float(m.ll_per_word), version=tp.__version__))
print("tomotopy CTM done in %.2fs" % secs)
