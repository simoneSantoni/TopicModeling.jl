"""LDA reference runs: tomotopy (C++ collapsed Gibbs), gensim (online VB), scikit-learn (batch VB)."""
import argparse, os, time
import numpy as np
from common import read_ldac, read_vocab, expand, save_result

ap = argparse.ArgumentParser()
ap.add_argument("--impl", required=True, choices=["tomotopy", "gensim", "sklearn"])
ap.add_argument("--train", required=True); ap.add_argument("--vocab", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("-K", type=int, default=50); ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--alpha", type=float, default=0.1); ap.add_argument("--eta", type=float, default=0.01)
ap.add_argument("--threads", type=int, default=1); ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()

docs = read_ldac(a.train); vocab = read_vocab(a.vocab); V = len(vocab)
meta = dict(impl=a.impl, iters=a.iters, threads=a.threads, seed=a.seed)

if a.impl == "tomotopy":
    import tomotopy as tp
    m = tp.LDAModel(k=a.K, alpha=a.alpha, eta=a.eta, seed=a.seed)
    for d in docs:
        m.add_doc([vocab[t] for t in expand(d)])
    t0 = time.perf_counter()
    m.train(a.iters, workers=a.threads)
    meta["seconds"] = time.perf_counter() - t0
    idx = {w: i for i, w in enumerate(vocab)}
    cols = [idx[w] for w in m.used_vocabs]
    phi = np.full((a.K, V), 0.0)
    for k in range(a.K):
        phi[k, cols] = m.get_topic_word_dist(k)
    phi /= phi.sum(1, keepdims=True)
    meta["ll_per_word"] = float(m.ll_per_word); meta["version"] = tp.__version__
    save_result(a.out, phi, meta, alpha=m.alpha)
elif a.impl == "gensim":
    import gensim
    from gensim.models import LdaModel, LdaMulticore
    id2word = dict(enumerate(vocab))
    t0 = time.perf_counter()
    kw = dict(corpus=docs, id2word=id2word, num_topics=a.K, eta=a.eta, passes=a.iters, random_state=a.seed)
    m = LdaModel(alpha=a.alpha, **kw) if a.threads == 1 else LdaMulticore(alpha=a.alpha, workers=a.threads, **kw)
    meta["seconds"] = time.perf_counter() - t0; meta["version"] = gensim.__version__
    save_result(a.out, m.get_topics(), meta, alpha=m.alpha)
else:
    import sklearn
    from scipy.sparse import csr_matrix
    from sklearn.decomposition import LatentDirichletAllocation
    rows = [i for i, d in enumerate(docs) for _ in d]; cols = [t for d in docs for t, _ in d]; vals = [c for d in docs for _, c in d]
    X = csr_matrix((vals, (rows, cols)), shape=(len(docs), V))
    m = LatentDirichletAllocation(n_components=a.K, doc_topic_prior=a.alpha, topic_word_prior=a.eta,
                                  learning_method="batch", max_iter=a.iters, n_jobs=a.threads, random_state=a.seed)
    t0 = time.perf_counter(); m.fit(X)
    meta["seconds"] = time.perf_counter() - t0; meta["version"] = sklearn.__version__; meta["n_iter"] = int(m.n_iter_)
    phi = m.components_ / m.components_.sum(1, keepdims=True)
    save_result(a.out, phi, meta, alpha=[a.alpha] * a.K)
print(a.impl, "done in %.2fs" % meta["seconds"])
