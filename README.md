# TopicModeling.jl

[![Text Analysis](https://img.shields.io/badge/Text-Analysis-orange.svg)](https://github.com/simoneSantoni/TopicModeling.jl)
[![Build Status](https://github.com/simoneSantoni/TopicModeling.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/simoneSantoni/TopicModeling.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/simoneSantoni/TopicModeling.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/simoneSantoni/TopicModeling.jl)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://simoneSantoni.github.io/TopicModeling.jl/stable/)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://simoneSantoni.github.io/TopicModeling.jl/dev/)
[![Julia](https://img.shields.io/badge/Julia-1.10+-purple.svg)](https://julialang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Probabilistic topic models in pure Julia: fast, multi-threaded, with one API for all of them and
no dependencies beyond `SpecialFunctions` and `StatsAPI`.

| Model | Inference | Reference |
|---|---|---|
| `LDA` — Latent Dirichlet Allocation | collapsed Gibbs sampling (threaded, with asymmetric-α optimisation); batch and online variational Bayes | Blei, Ng & Jordan (2003); Griffiths & Steyvers (2004); Hoffman, Blei & Bach (2010) |
| `STM` — Structural Topic Model | Laplace-variational EM with prevalence covariates, spectral (anchor-word) initialisation, `estimate_effect` | Roberts, Stewart & Airoldi (2016) |
| `DTM` — Dynamic Topic Model | variational Kalman smoothing | Blei & Lafferty (2006) |
| `CTM` — Correlated Topic Model | logistic-normal variational EM | Blei & Lafferty (2007) |

Every model is benchmarked against the reference implementations in C++, Python and R
(tomotopy, gensim, scikit-learn, `stm`, `topicmodels`, `lda`) and against results from the
literature: see [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/simoneSantoni/TopicModeling.jl")
```

Start Julia with threads (`julia -t auto`) — all models use them.

## Usage

```julia
using TopicModeling

# From raw text (a minimal tokenizer is built in; bring your own pipeline for anything serious) ...
corpus = Corpus(texts; min_df=5, max_df=0.5, stopwords=my_stopwords)
# ... or from tokens, a documents × terms count matrix, or Blei's LDA-C / the UCI bag-of-words format.
corpus = Corpus(tokenized_docs)
corpus = Corpus(counts, vocab)
corpus = read_ldac("ap.dat"; vocab="vocab.txt")

# LDA
lda = fit(LDA, corpus, 50)                                    # collapsed Gibbs, 1000 sweeps
lda = fit(LDA, corpus, 50; method=:vb)                        # batch variational Bayes
lda = fit(LDA, corpus, 50; method=:vb, batchsize=1024)        # online VB for large corpora

# STM: topic prevalence as a function of document covariates (D × P matrix, no intercept column)
stm = fit(STM, corpus, 20; prevalence=X, keep_nu=true)
coef, se = estimate_effect(stm)                               # P × K effects on topic proportions

# DTM: one time label per document
dtm = fit(DTM, corpus, years, 10)
topwords(dtm, 3)                                              # topics as of the third period

# CTM
ctm = fit(CTM, corpus, 20)
topic_correlations(ctm)

# Shared by all models
topwords(lda; n=10)               # most probable words per topic
topicword(lda)                    # K × V topic-word probabilities
doctopic(lda)                     # D × K document-topic proportions
transform(lda, new_corpus)        # topic proportions of unseen documents
coherence(lda, corpus)            # NPMI (or measure=:umass) per topic
heldout_perplexity(lda, test)     # document-completion perplexity
```

`?LDA`, `?STM`, `?DTM`, `?CTM` and `?fit` document every keyword, and the
[documentation](https://simoneSantoni.github.io/TopicModeling.jl/dev) has a tutorial,
a Models page (generative model, inference algorithm, parameters, when to use) and the full API
reference. Build it
locally with

```
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs -t 4 docs/make.jl        # output in docs/build/index.html
```

## Things worth knowing

**Local optima are real, and the objective tells you about them.** All of these models have
multimodal objectives. On a synthetic corpus with 8 known topics, batch VB found all of them
from about half of the random starts — 14/24 here, 11/24 for scikit-learn on the same data
(`benchmarks/RESULTS.md`). The good and bad solutions differ clearly in ELBO (Gibbs:
log-likelihood), which is reported in `model.trace`: fit a few seeds and keep the best.

**STM and CTM default to spectral initialisation**, which is deterministic and in our
experiments reached the same optimum as starting EM *at the true topics*, where `init=:lda` and
`init=:random` ended in worse optima — the finding of Roberts, Stewart & Tingley (2016).
Anchor words must be reasonably frequent to be reliable: candidates need a document frequency
of `max(10, D/4K)`, a heuristic (see `?spectral_init`) you may want to lower for very rare topics.

**The DTM is the model of Blei & Lafferty, optimised differently.** In their variational Kalman
family the smoothed covariance is the same for every word and the smoothed mean is an invertible
linear function of the variational pseudo-observations. Optimising over the means directly gives
a concave problem per topic with a tridiagonal prior, which L-BFGS solves in a fraction of the
time of per-word conjugate gradients. Same family, same optimum (`src/dtm.jl` has the argument).

**What is not implemented**: STM content covariates (topics whose word distributions vary with a
covariate); hyper-parameter learning for the DTM variances; sparse/alias Gibbs samplers for very
large K. The threaded Gibbs sampler partitions documents and vocabulary into blocks (Yan, Xu & Qi
2009) so that threads never work on stale counts; a different thread count is a different chain.

## Tests

```
julia --project -t 4 -e 'using Pkg; Pkg.test()'
```

The suite checks analytic gradients and Hessians against finite differences, that every
variational objective is non-decreasing, and parameter recovery on data simulated from each
model: the "bars" of Griffiths & Steyvers (2004) for LDA, known covariate effects for the STM,
drifting topics for the DTM (which must beat a static LDA), and a planted topic correlation for
the CTM.

## References

- Arora, Ge, Halpern, Mimno, Moitra, Sontag, Wu & Zhu (2013). A practical algorithm for topic modeling with provable guarantees. *ICML*.
- Blei & Lafferty (2006). Dynamic topic models. *ICML*.
- Blei & Lafferty (2007). A correlated topic model of Science. *Annals of Applied Statistics* 1(1).
- Blei, Ng & Jordan (2003). Latent Dirichlet allocation. *JMLR* 3.
- Griffiths & Steyvers (2004). Finding scientific topics. *PNAS* 101.
- Hoffman, Blei & Bach (2010). Online learning for latent Dirichlet allocation. *NeurIPS*.
- Newman, Asuncion, Smyth & Welling (2009). Distributed algorithms for topic models. *JMLR* 10.
- Roberts, Stewart & Airoldi (2016). A model of text for experimentation in the social sciences. *JASA* 111(515).
- Roberts, Stewart & Tingley (2016). Navigating the local modes of big data: the case of topic models. In *Computational Social Science*, CUP.
- Wallach, Murray, Salakhutdinov & Mimno (2009). Evaluation methods for topic models. *ICML*.
- Yan, Xu & Qi (2009). Parallel inference for latent Dirichlet allocation on graphics processing units. *NeurIPS*.
