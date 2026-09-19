# Tutorial

This tutorial walks through common use cases for TopicModeling.jl. The examples with output
are executed when the documentation is built, on small simulated corpora so that results can
be checked against known ground truth.

## Creating Corpora

A [`Corpus`](@ref) is a vector of sparse bag-of-words [`Document`](@ref)s over a shared
vocabulary.

### From Raw Text

```@example tutorial
using TopicModeling, Random

texts = ["The cat sat on the mat.", "The dog chased the cat.", "Dogs and cats are pets."]
corpus = Corpus(texts; stopwords=["the", "on", "and", "are"])
```

```@example tutorial
corpus.vocab
```

The built-in [`tokenize`](@ref) is deliberately minimal. For anything language-specific
(stemming, n-grams, lemmatisation) run your own pipeline and pass the tokens.

### From Tokens

```julia
tokenized = [["topic", "models", "find", "topics"], ["models", "of", "text"]]

# Drop terms in fewer than 5 documents or in more than half of them
corpus = Corpus(tokenized; min_df=5, max_df=0.5, stopwords=my_stopwords)
```

Documents left empty by the filters are kept by default (`keep_empty=true`) so that document
indices stay aligned with your metadata.

### From a Count Matrix

```@example tutorial
counts = [2 0 1;
          0 3 0]                       # documents × terms
Corpus(counts, ["x", "y", "z"])
```

Sparse matrices are accepted as well, and [`dtm`](@ref) goes the other way.

### From Files

```julia
corpus = read_ldac("ap.dat"; vocab="vocab.txt")        # Blei's LDA-C format
corpus = read_uci("docword.nips.txt"; vocab="vocab.nips.txt")   # UCI bag-of-words format
```

### Inspecting and Pruning

```julia
ndocs(corpus), nterms(corpus), ntokens(corpus)
docfreq(corpus)                        # number of documents each term occurs in
pruned, kept = prune(corpus; min_df=5, max_df=0.5, max_terms=10_000)
```

## Basic Topic Modeling

For the rest of this tutorial we simulate a corpus with known topics:

```@example tutorial
corpus, phi_true, theta_true = simulate_lda(D=400, K=6, V=300, doclen=80, rng=Xoshiro(42))
train, test, train_idx, test_idx = train_test_split(corpus; test=0.1, rng=Xoshiro(1))
train
```

### Fitting a Model

```@example tutorial
model = fit(LDA, train, 6; iters=300, rng=Xoshiro(1))
```

Pass an `rng` for a reproducible fit.

### Understanding Results

All models share the same accessors:

```@example tutorial
# Most probable words of each topic
topwords(model; n=6)
```

```@example tutorial
# Topic-word probabilities (K × V) and document-topic proportions (D × K)
size(topicword(model)), size(doctopic(model))
```

```julia
# Objective per iteration, iterations run, seconds spent
model.trace, model.iterations, model.elapsed
```

### New Documents

```@example tutorial
theta_new = transform(model, test)     # topics held fixed
size(theta_new)
```

## Evaluating Models

### Held-Out Perplexity

Document completion: the tokens of every test document are split in two; topic proportions are
inferred from one half and the other half is scored.

```@example tutorial
perplexity, loglik = heldout_perplexity(model, test; rng=Xoshiro(1))
perplexity
```

### Coherence and Diversity

```@example tutorial
npmi = coherence(model, train)                      # one value per topic, in [-1, 1]
round.(npmi; digits=2)
```

```@example tutorial
topic_diversity(topicword(model); n=25)             # share of unique top words
```

Read the two together: a model can buy coherence by repeating one good topic.

### Recovering Known Topics

[`match_topics`](@ref) aligns estimated with true topics (Hungarian algorithm) and returns the
total-variation distance of each pair: 0 is identical, 1 is disjoint.

```@example tutorial
perm, dist = match_topics(phi_true, topicword(model))
round.(dist; digits=3)
```

## Comparing Algorithms and Seeds

Topic-model objectives are multimodal, so a single run can land in a poor optimum. The
objective in `model.trace` tells good from bad runs: fit a few seeds and keep the best.

```@example tutorial
fits = [fit(LDA, train, 6; method=:vb, rng=Xoshiro(s)) for s in 1:4]
[round(f.trace[end]; digits=4) for f in fits]       # final ELBO per token
```

```@example tutorial
best = fits[argmax([f.trace[end] for f in fits])]
round(sum(last(match_topics(phi_true, topicword(best)))) / 6; digits=3)
```

### Scoring Other Software

The evaluation functions accept a bare ``K \times V`` topic-word matrix, so a model fitted
elsewhere (gensim, tomotopy, the R packages) can be scored with exactly the same code:

```julia
phi_other = ...                                     # K × V, rows sum to one
heldout_perplexity(phi_other, 0.1, test)            # 0.1 = Dirichlet α used to infer θ
coherence(phi_other, train)
```

## Working with Specific Models

### Covariates with the STM

The prevalence design matrix is ``D \times P`` without an intercept column (one is added).
Here a binary covariate raises the first simulated topic and lowers the second:

```@example tutorial
rng = Xoshiro(7)
D, K = 600, 4
x = Float64.(rand(rng, Bool, D))
Gamma = [0.0 0.0 0.0; 1.5 -1.0 0.0]
mu = hcat(ones(D), x) * Gamma
stm_corpus, _, _ = simulate_logistic_normal(; D, K, V=300, mu, Sigma=[0.5 0 0; 0 0.5 0; 0 0 0.5], rng)

stm = fit(STM, stm_corpus, K; prevalence=reshape(x, :, 1), keep_nu=true)
coef, se = estimate_effect(stm; rng=Xoshiro(1))
round.(coef[2, :]; digits=3)                        # effect of x on each topic's proportion
```

Topics are identified only up to a permutation: the positive and negative effects appear on
whichever fitted topics correspond to the simulated ones. `keep_nu=true` stores the
per-document covariances that [`estimate_effect`](@ref) needs to propagate uncertainty.

### Topic Correlations with the CTM

```@example tutorial
Sigma = [1.0 0.8 0.0; 0.8 1.0 0.0; 0.0 0.0 1.0]    # topics 1 and 2 co-occur
ctm_corpus, _, _ = simulate_logistic_normal(D=600, K=4, V=300, Sigma=Sigma, rng=Xoshiro(11))
ctm = fit(CTM, ctm_corpus, 4)
round.(topic_correlations(ctm); digits=2)
```

Correlations are of log-odds against the reference (last) topic, in the fitted topic order.

### Topics over Time with the DTM

Each document carries a time label (any sortable values); the sorted unique labels become
slices `1:T`.

```@example tutorial
dtm_corpus, times, _, _ = simulate_dtm(T=6, docs_per_slice=60, K=3, V=150, rng=Xoshiro(5))
dyn = fit(DTM, dtm_corpus, times, 3; alpha=0.1, chain_variance=0.01, iters=15, rng=Xoshiro(1))
```

```@example tutorial
# Topic 1 in the first and in the last slice
topwords(dyn, 1; n=5)[1], topwords(dyn, 6; n=5)[1]
```

```julia
topicword(dyn, 3)                      # K × V matrix at slice 3
transform(dyn, new_corpus, new_times)  # new documents use the topics of their own slice
```

## Simulating Corpora

Every model has a generator for data from its own generative process; they are used by the
test suite for parameter-recovery checks and are handy for power analysis.

```@example tutorial
# The "bars" of Griffiths & Steyvers (2004): each topic is a row or a column of a pixel grid
phi = bars_topics(4)
reshape(phi[1, :], 4, 4)', reshape(phi[5, :], 4, 4)'
```

```julia
corpus, phi, theta = simulate_lda(D=1000, K=10, V=25, topics=bars_topics(5))
corpus, phi, theta = simulate_logistic_normal(D=500, K=5, V=500, Sigma=S)
corpus, times, phi, theta = simulate_dtm(T=10, docs_per_slice=100, K=5, V=500)
```
