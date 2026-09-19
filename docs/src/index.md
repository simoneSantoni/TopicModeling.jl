# TopicModeling.jl

A Julia package for probabilistic topic models: fast, multi-threaded, with one API for all of them.

## Overview

Topic models describe a collection of documents as mixtures of a small number of latent
*topics*, each a probability distribution over the vocabulary:

- **Topic-word distributions** (``K \times V``): which words a topic is made of
- **Document-topic proportions** (``D \times K``): which topics a document is made of

The family has grown to let topics correlate, depend on document covariates, or drift over
time. TopicModeling.jl implements four of these models in pure Julia behind a single `fit`
interface, together with the tools needed to evaluate them and to compare them with other
software on equal terms.

## Features

- **Four models**: [`LDA`](@ref), [`STM`](@ref), [`DTM`](@ref) and [`CTM`](@ref)
- **Several inference algorithms**: collapsed Gibbs sampling, batch and online variational
  Bayes, Laplace and mean-field variational EM, variational Kalman smoothing
- **Multi-threaded** E-steps and Gibbs sweeps in every model, reproducible from a seed
- **Spectral (anchor-word) initialisation**, deterministic, for the logistic-normal models
- **Covariate effects with uncertainty** for the STM via [`estimate_effect`](@ref)
- **Implementation-agnostic evaluation**: held-out perplexity, NPMI and UMass coherence, topic
  diversity, and optimal topic matching all work on a bare topic-word matrix
- **Simulators** for every model, for parameter-recovery checks and power analysis
- **Consistent API** with shared accessors across all models, and no dependencies beyond
  `SpecialFunctions` and `StatsAPI`

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/simoneSantoni/TopicModeling.jl")
```

Or for development:

```julia
using Pkg
Pkg.develop(path="/path/to/TopicModeling.jl")
```

Start Julia with threads (`julia -t auto`): every model uses them.

## Quick Start

```julia
using TopicModeling

# Build a corpus from raw text (or from tokens, a count matrix, LDA-C or UCI files)
corpus = Corpus(texts; min_df=5, max_df=0.5, stopwords=my_stopwords)

# Fit a model
model = fit(LDA, corpus, 20)

# Access results
println("Top words: ", topwords(model; n=10))
println("Topic-word matrix: ", size(topicword(model)))      # K × V
println("Document-topic matrix: ", size(doctopic(model)))   # D × K
println("Coherence: ", coherence(model, corpus))

# Apply to unseen documents
theta = transform(model, new_corpus)
```

## Choosing a Model

| Use Case | Recommended Model |
|----------|-------------------|
| General-purpose topics, best held-out likelihood | [`LDA`](@ref) with `method=:gibbs` |
| Smooth objective for comparing runs and models | [`LDA`](@ref) with `method=:vb` |
| Very large corpora | [`LDA`](@ref) with `method=:vb, batchsize=1024` |
| Topic prevalence depends on document covariates | [`STM`](@ref) with [`estimate_effect`](@ref) |
| Topics that co-occur or exclude each other | [`CTM`](@ref) with [`topic_correlations`](@ref) |
| Topics whose vocabulary drifts over time | [`DTM`](@ref) |
| Comparing implementations or software packages | [`heldout_perplexity`](@ref) and [`coherence`](@ref) on the topic matrix |
| Checking recovery on known ground truth | [`simulate_lda`](@ref) and [`match_topics`](@ref) |

## Performance snapshots

### TopicModeling.jl compared with C++, Python and R

LDA with ``K = 50`` on the Associated Press corpus (2021 training and 225 test documents).
Every implementation hands over its topic-word matrix, and one function scores all of them:
held-out perplexity by document completion (mean ± sd over 5 token splits; lower is better)
and NPMI coherence of the top-10 words.

| Implementation | Algorithm | Threads | Time (s) | Held-out perplexity | NPMI |
|:---------------|:----------|--------:|---------:|--------------------:|-----:|
| **TopicModeling.jl** | collapsed Gibbs, 1000 sweeps | 1 | 14.0 | 2367 ± 20 | 0.232 |
| **TopicModeling.jl** | collapsed Gibbs, 1000 sweeps | 8 | 7.6 | 2375 ± 30 | 0.243 |
| tomotopy (C++) | collapsed Gibbs, 1000 sweeps | 1 | 16.8 | 2462 ± 16 | 0.255 |
| tomotopy (C++) | collapsed Gibbs, 1000 sweeps | 8 | 7.8 | 2472 ± 22 | 0.214 |
| lda (R/C) | collapsed Gibbs, 1000 sweeps | 1 | 32.5 | 2556 ± 16 | 0.229 |
| topicmodels (R/C++) | collapsed Gibbs, 1000 sweeps | 1 | 78.2 | 2527 ± 24 | 0.239 |
| **TopicModeling.jl** | batch VB, 100 passes | 1 | 14.9 | 2844 ± 30 | 0.167 |
| **TopicModeling.jl** | batch VB, 100 passes | 8 | 5.4 | 2844 ± 30 | 0.167 |
| scikit-learn (Cython) | batch VB, 100 passes | 1 | 68.2 | 2859 ± 40 | 0.144 |
| gensim (NumPy) | online VB, 100 passes | 1 | 111.8 | 3675 ± 45 | 0.107 |
| topicmodels (lda-c) | VEM, α estimated | 1 | 339.7 | 2972 ± 33 | 0.177 |

Absolute timings are host-specific, and run-to-run noise on the benchmark machine is around
±20%, so small differences mean nothing. See [Performance](@ref) for threading, reproducibility
and practical advice, and
[`benchmarks/RESULTS.md`](https://github.com/simoneSantoni/TopicModeling.jl/blob/main/benchmarks/RESULTS.md)
for the methodology and the checks against the literature.

## Documentation

```@contents
Pages = ["tutorial.md", "models.md", "performance.md", "api.md"]
Depth = 2
```

## References

1. Arora, S., Ge, R., Halpern, Y., Mimno, D., Moitra, A., Sontag, D., Wu, Y., Zhu, M. (2013). A practical algorithm for topic modeling with provable guarantees. *ICML*.
2. Blei, D.M., Lafferty, J.D. (2006). Dynamic topic models. *ICML*.
3. Blei, D.M., Lafferty, J.D. (2007). A correlated topic model of Science. *The Annals of Applied Statistics*, 1(1), 17-35.
4. Blei, D.M., Ng, A.Y., Jordan, M.I. (2003). Latent Dirichlet allocation. *Journal of Machine Learning Research*, 3, 993-1022.
5. Bouma, G. (2009). Normalized (pointwise) mutual information in collocation extraction. *Proceedings of GSCL*.
6. Dieng, A.B., Ruiz, F.J.R., Blei, D.M. (2020). Topic modeling in embedding spaces. *Transactions of the ACL*, 8, 439-453.
7. Drugowitsch, J. (2013). Variational Bayesian inference for linear and logistic regression. *arXiv:1310.5438*.
8. Griffiths, T.L., Steyvers, M. (2004). Finding scientific topics. *PNAS*, 101(suppl. 1), 5228-5235.
9. Hoffman, M.D., Blei, D.M., Bach, F. (2010). Online learning for latent Dirichlet allocation. *NeurIPS*.
10. Lau, J.H., Newman, D., Baldwin, T. (2014). Machine reading tea leaves: automatically evaluating topic coherence and topic model quality. *EACL*.
11. Mimno, D., Wallach, H.M., Talley, E., Leenders, M., McCallum, A. (2011). Optimizing semantic coherence in topic models. *EMNLP*.
12. Minka, T.P. (2000). Estimating a Dirichlet distribution. *Technical report, MIT*.
13. Newman, D., Asuncion, A., Smyth, P., Welling, M. (2009). Distributed algorithms for topic models. *Journal of Machine Learning Research*, 10, 1801-1828.
14. Roberts, M.E., Stewart, B.M., Airoldi, E.M. (2016). A model of text for experimentation in the social sciences. *Journal of the American Statistical Association*, 111(515), 988-1003.
15. Roberts, M.E., Stewart, B.M., Tingley, D. (2016). Navigating the local modes of big data: the case of topic models. In *Computational Social Science: Discovery and Prediction*. Cambridge University Press.
16. Wallach, H.M., Murray, I., Salakhutdinov, R., Mimno, D. (2009). Evaluation methods for topic models. *ICML*.
17. Yan, F., Xu, N., Qi, Y. (2009). Parallel inference for latent Dirichlet allocation on graphics processing units. *NeurIPS*.

## Citation

```biblatex
@misc{SantoniTopicModelingJL,
  author = {Santoni, Simone},
  title = {TopicModeling.jl: Probabilistic Topic Models in Julia},
  year = {2026},
  url = {https://github.com/simoneSantoni/TopicModeling.jl},
  note = {Homepage: https://www.bayes.citystgeorges.ac.uk/faculties-and-research/experts/simone-santoni; GitHub: https://github.com/simoneSantoni}
}
```
