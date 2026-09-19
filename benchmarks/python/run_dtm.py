"""DTM reference runs: gensim LdaSeqModel (port of Blei & Lafferty's variational Kalman code) and tomotopy DTModel (C++, SGLD-based Gibbs)."""
import argparse, time
import numpy as np
from common import read_ldac, read_vocab, expand
ap = argparse.ArgumentParser()
ap.add_argument("--impl", required=True, choices=["gensim", "tomotopy"])
ap.add_argument("--train", required=True); ap.add_argument("--times", required=True); ap.add_argument("--vocab", required=True); ap.add_argument("--out", required=True)
ap.add_argument("-K", type=int, default=5); ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--chain_variance", type=float, default=0.005); ap.add_argument("--alpha", type=float, default=0.01)
ap.add_argument("--threads", type=int, default=1); ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
docs = read_ldac(a.train); vocab = read_vocab(a.vocab); V = len(vocab)
times = [int(l) for l in open(a.times)]; T = max(times)            # 1-based, documents sorted by time
if a.impl == "gensim":
    import gensim
    from gensim.models import LdaSeqModel
    slices = [times.count(t) for t in range(1, T + 1)]
    t0 = time.perf_counter()
    m = LdaSeqModel(corpus=docs, id2word=dict(enumerate(vocab)), time_slice=slices, num_topics=a.K, chain_variance=a.chain_variance,
                    alphas=a.alpha, random_state=a.seed, em_max_iter=a.iters)
    secs = time.perf_counter() - t0
    phi = np.stack([np.array([np.exp(m.topic_chains[k].e_log_prob[:, t]) for k in range(a.K)]) for t in range(T)])   # T x K x V
    version = gensim.__version__
else:
    import tomotopy as tp
    m = tp.DTModel(k=a.K, t=T, seed=a.seed)
    for d, t in zip(docs, times):
        m.add_doc([vocab[w] for w in expand(d)], timepoint=t - 1)
    t0 = time.perf_counter(); m.train(a.iters, workers=a.threads); secs = time.perf_counter() - t0
    idx = {w: i for i, w in enumerate(vocab)}; cols = [idx[w] for w in m.used_vocabs]
    phi = np.zeros((T, a.K, V))
    for t in range(T):
        for k in range(a.K):
            phi[t, k, cols] = m.get_topic_word_dist(k, t)
    version = tp.__version__
phi = phi / phi.sum(-1, keepdims=True)
np.ascontiguousarray(phi, dtype="<f8").tofile(a.out + ".phi.bin")
open(a.out + ".json", "w").write('{"impl": "%s", "seconds": %.4f, "T": %d, "K": %d, "V": %d, "version": "%s"}' % (a.impl, secs, T, a.K, V, version))
print(a.impl, "DTM done in %.2fs" % secs)
