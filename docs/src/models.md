# Models

TopicModeling.jl implements 4 topic models and 6 inference algorithms. This page describes
each model, its theoretical foundations, how it is fitted, and when to use it.

## Model Categories

The models can be grouped by the prior they put on the document-topic proportions ``\theta_d``:

1. **Dirichlet models**: [`LDA`](@ref), and the [`DTM`](@ref), which adds topics that drift
   over time
2. **Logistic-normal models**: the [`CTM`](@ref), which lets topics correlate, and the
   [`STM`](@ref), which also lets document covariates shift topic prevalence

## Common Interface

```julia
model = fit(Model, corpus, K; kwargs...)          # Model ∈ (LDA, STM, CTM)
model = fit(DTM, corpus, times, K; kwargs...)
```

`fit` is the `StatsAPI` generic. Every fitted model is an immutable
[`AbstractTopicModel`](@ref TopicModeling.AbstractTopicModel) that supports [`topicword`](@ref), [`doctopic`](@ref),
[`vocabulary`](@ref), [`topwords`](@ref) and [`transform`](@ref), and carries `trace` (the
objective per iteration), `iterations` and `elapsed` (seconds).

**Shared parameters**:

- `iters`: Maximum iterations (sweeps, passes or EM iterations; default depends on the model)
- `tol`: Relative change of the objective at which to stop (variational methods)
- `rng`: Random-number generator; pass e.g. `Xoshiro(1)` for a reproducible fit
- `nthreads`: Number of worker tasks (default: `Threads.nthreads()`)
- `verbose`: Print the objective while fitting (default: `false`)

## Latent Dirichlet Allocation

```julia
model = fit(LDA, corpus, K; method=:gibbs, alpha=0.1, eta=0.01, iters=1000)
```

The model of Blei, Ng and Jordan (2003). Each topic is a distribution over the vocabulary and
each document a mixture of topics:

```math
\varphi_k \sim \operatorname{Dirichlet}(\eta), \qquad
\theta_d \sim \operatorname{Dirichlet}(\alpha), \qquad
z_{dn} \sim \operatorname{Categorical}(\theta_d), \qquad
w_{dn} \sim \operatorname{Categorical}(\varphi_{z_{dn}})
```

### Collapsed Gibbs Sampling

```julia
model = fit(LDA, corpus, K; method=:gibbs, iters=1000, burnin=100, optimize_alpha=true)
```

**Algorithm**: The sampler of Griffiths and Steyvers (2004). ``\theta`` and ``\varphi`` are
integrated out and each token's topic is resampled from

```math
p(z_{dn} = k \mid z_{\neg dn}, w) \;\propto\;
(n_{dk} + \alpha_k)\,\frac{n_{kw} + \eta}{n_{k} + V\eta}
```

with the counts excluding the current token. With more than one thread the sweep is
parallelised by partitioning (Yan, Xu and Qi 2009): documents and vocabulary are each cut into
``T`` blocks of equal token mass, and in each of ``T`` sub-rounds task ``t`` samples the tokens
of document block ``t`` that fall into word block ``t + r`` (mod ``T``), so concurrent tasks
touch disjoint rows and columns of the count tables and no stale copies are needed. An
asymmetric ``\alpha`` is learnt after burn-in with Minka's (2000) fixed-point iteration. The
returned ``\varphi`` and ``\theta`` are averages over several states of the chain, which
lowers held-out perplexity at no cost in sweeps.

**Parameters**:

- `alpha`: Document-topic Dirichlet parameter, scalar or length-`K` vector (default: 0.1)
- `eta`: Topic-word Dirichlet parameter (default: 0.01)
- `iters`: Number of sweeps (default: 1000)
- `burnin`: Sweeps before ``\alpha`` optimisation starts (default: 100)
- `optimize_alpha`, `optimize_interval`: Learn an asymmetric ``\alpha`` every so many sweeps
  (default: `true`, 10)
- `nsamples`, `sample_lag`: Average ``\varphi`` and ``\theta`` over this many post-burn-in
  states, taken every `sample_lag` sweeps and ending with the last (default: 10, 10;
  `nsamples=1` gives the final state alone)
- `eval_every`: Record ``\log p(w, z)`` per token every so many sweeps (default: 0, only at the end)

**When to use**: The default. Best held-out likelihood on small and medium corpora.

### Variational Bayes

```julia
model = fit(LDA, corpus, K; method=:vb, iters=200, tol=1e-6)
model = fit(LDA, corpus, K; method=:vb, batchsize=1024, tau0=64, kappa=0.7)
```

**Algorithm**: Variational Bayes with smoothed topics: batch coordinate ascent (Blei, Ng and
Jordan 2003) or, with `batchsize > 0`, the stochastic natural-gradient algorithm of Hoffman,
Blei and Bach (2010) with step size ``\rho_t = (\tau_0 + t)^{-\kappa}``. Token-level
responsibilities are never stored, only their normaliser per distinct term, which keeps the
E-step allocation-free.

**Parameters**:

- `iters`: Maximum passes over the corpus (default: 200)
- `tol`: Relative ELBO change at which to stop (default: 1e-6)
- `batchsize`: 0 for batch VB, otherwise the minibatch size of online VB (default: 0)
- `tau0`, `kappa`: Online learning-rate schedule (default: 64, 0.7)
- `optimize_alpha`: Newton updates of ``\alpha`` (default: `false`)
- `doc_maxiter`, `doc_tol`: Per-document coordinate-ascent limits (default: 100, 1e-3)
- `init`: Optional ``K \times V`` matrix of initial topics
- `warm_start`: Batch only; `n > 0` resumes every document from its previous variational
  posterior after pass `n`. Several times faster per pass, but documents can no longer pick up
  topics they have dropped, which costs ELBO and held-out likelihood (default: 0, off)

**When to use**: When you want a smooth objective (the ELBO) for comparing runs and models, or
— in online mode — when a full pass over the corpus is expensive.

## Structural Topic Model

```julia
model = fit(STM, corpus, K; prevalence=X, init=:spectral, iters=500, tol=1e-6, keep_nu=false)
```

The model of Roberts, Stewart and Airoldi (2016). Document covariates ``x_d`` shift the
*prevalence* of topics through a logistic-normal prior:

```math
\eta_d \sim \mathcal N(\Gamma^\top x_d,\; \Sigma), \qquad
\theta_d = \operatorname{softmax}([\eta_d;\, 0]), \qquad
w_{dn} \sim \operatorname{Categorical}(\beta_{z_{dn}})
```

**Algorithm**: Partially collapsed variational EM, as in the paper and the R package `stm`.
For each document, ``q(\eta_d) = \mathcal N(\lambda_d, \nu_d)`` is a Laplace approximation:
``\lambda_d`` minimises

```math
f(\eta) = -\sum_w c_{dw} \log \sum_k e^{\eta_k}\beta_{kw} + N_d \log \sum_k e^{\eta_k}
          + \tfrac12 (\eta-\mu_d)^\top \Sigma^{-1} (\eta-\mu_d)
```

by L-BFGS with an analytic gradient, warm-started across EM iterations, and ``\nu_d`` is the
inverse of the analytic Hessian there. The M-step updates ``\Gamma`` by variational Bayesian
linear regression with a learnt amount of shrinkage (Drugowitsch 2013),
``\Sigma = \tfrac1D \sum_d [\nu_d + (\lambda_d-\mu_d)(\lambda_d-\mu_d)^\top]``, and ``\beta``
from the expected topic-term counts.

**Parameters**:

- `prevalence`: ``D \times P`` covariate matrix without intercept (default: `nothing`, which
  gives a correlated topic model fitted by the STM algorithm)
- `init`: `:spectral`, `:lda`, `:random`, or a ``K \times V`` matrix (default: `:spectral`)
- `iters`, `tol`: EM iteration cap and relative bound change, which must stay below `tol` for
  two consecutive iterations (default: 500, 1e-6; R's `stm` uses `emtol = 1e-5`)
- `gamma_prior`: `:pooled` (shrinkage) or `:ols` (default: `:pooled`)
- `sigma_prior`: Weight of the diagonal in the ``\Sigma`` update (default: 0.0)
- `keep_nu`: Store the per-document covariances needed by [`estimate_effect`](@ref)
  (default: `false`)
- `doc_maxiter`: L-BFGS iteration cap per document (default: 500)

**Covariate effects**: [`estimate_effect`](@ref) regresses topic proportions on the design
matrix with the "method of composition": it repeatedly draws
``\eta_d \sim \mathcal N(\lambda_d, \nu_d)``, maps to ``\theta``, runs OLS and pools the
estimates, so that uncertainty about topic proportions reaches the standard errors.

**When to use**: When the research question is how topic prevalence varies with document
metadata (author, party, treatment, date).

## Dynamic Topic Model

```julia
model = fit(DTM, corpus, times, K; alpha=0.01, chain_variance=0.005, obs_variance=0.5, iters=50)
```

The model of Blei and Lafferty (2006). Documents carry a time label and topics drift between
time slices as a Gaussian random walk on their natural parameters:

```math
\beta_{t,k} \mid \beta_{t-1,k} \sim \mathcal N(\beta_{t-1,k},\; \sigma^2 I), \qquad
\varphi_{t,k} = \operatorname{softmax}(\beta_{t,k}), \qquad
\theta_d \sim \operatorname{Dirichlet}(\alpha)
```

**Algorithm**: The variational Kalman family of the paper, optimised differently. The smoothed
covariance ``S`` is the same for every word and topic, and the smoothed mean is an invertible
linear map of the variational pseudo-observations, so the optimisation can run over the means
``m`` directly. In terms of ``m`` the bound of a topic is concave,

```math
\sum_t \Big[ \sum_w n_{tw} m_{tw} - n_t \log \sum_w e^{m_{tw} + S_{tt}/2} \Big]
- \tfrac12 \sum_w m_w^\top P\, m_w + \text{const}
```

with ``P`` the tridiagonal random-walk precision. Each M-step is therefore one smooth concave
problem per topic, solved by L-BFGS (topics in parallel), instead of per-word conjugate
gradients with Kalman passes in the inner loop: same family, same optimum, a fraction of the
time. The E-step is the LDA document update, slice by slice.

**Parameters**:

- `times`: One sortable label per document; the sorted unique labels become slices `1:T`
  (treated as equally spaced)
- `alpha`: Document-topic Dirichlet parameter (default: 0.01)
- `chain_variance`: Random-walk variance ``\sigma^2`` (default: 0.005)
- `obs_variance`, `init_variance`: Variational observation variance and variance of the first
  slice (default: 0.5, `1000 * chain_variance`)
- `iters`, `tol`: EM iteration cap and relative ELBO change (default: 50, 1e-5)
- `init`: `:lda` or a ``K \times V`` matrix (default: `:lda`). With a matrix the fit is
  identical for any `nthreads`; the `:lda` Gibbs initialisation depends on it
- `mstep_maxiter`, `doc_maxiter`, `doc_tol`: Inner-solver limits (default: 100, 100, 1e-4)

The variances are fixed, not learnt; the defaults are those of Blei and Lafferty's code.

**When to use**: When the vocabulary of a topic changes over time and you want to follow it,
rather than fit independent models per period.

## Correlated Topic Model

```julia
model = fit(CTM, corpus, K; init=:spectral, iters=500, tol=1e-5, shrinkage=0.0)
```

The model of Blei and Lafferty (2007). A Dirichlet cannot express that some topics tend to
occur together; a logistic normal with a full covariance can:

```math
\eta_d \sim \mathcal N(\mu, \Sigma), \qquad \theta_d = \operatorname{softmax}([\eta_d;\,0])
```

**Algorithm**: Variational EM with the mean-field family of the paper,
``q(\eta_d) = \mathcal N(\lambda_d, \operatorname{diag}(\nu_d^2))``, and the same first-order
bound on ``\mathbb E[\log \sum_k e^{\eta_k}]``. The token responsibilities and the bound
parameter ``\zeta`` are substituted out analytically, leaving one smooth problem per document
in ``(\lambda, \log \nu^2)``,

```math
\mathcal L_d = \sum_w c_{dw} \log \sum_k e^{\lambda_k}\beta_{kw}
 - N_d \log \sum_k e^{\lambda_k + \nu_k^2/2}
 - \tfrac12 (\lambda-\mu)^\top \Sigma^{-1} (\lambda-\mu)
 - \tfrac12 \sum_k \nu_k^2\, (\Sigma^{-1})_{kk}
 + \tfrac12 \sum_k \log \nu_k^2 + \text{const}
```

solved jointly by L-BFGS with an analytic gradient. The M-step is closed form.

**Parameters**:

- `init`: `:spectral`, `:lda`, `:random`, or a ``K \times V`` matrix (default: `:spectral`)
- `iters`, `tol`: EM iteration cap and relative ELBO change, which must stay below `tol` for
  two consecutive iterations (default: 500, 1e-5)
- `shrinkage`: Weight of the diagonal in the ``\Sigma`` update (default: 0.0)
- `doc_maxiter`: L-BFGS iteration cap per document (default: 500)

**When to use**: When the relationships *between* topics are of interest
([`topic_correlations`](@ref)), or when few words of a document are observed: the learnt
covariance lets one topic predict another.

## Spectral Initialisation

```julia
phi, anchors = spectral_init(corpus, K; max_terms=10_000)
```

The STM and the CTM default to the deterministic anchor-word algorithm of Arora et al. (2013),
as `stm` does. In our experiments it reached the same optimum as starting EM *at the true
topics*, where LDA and random initialisation ended lower — the finding of Roberts, Stewart and
Tingley (2016). Anchor words must be reasonably frequent to be reliable: candidates need a
document frequency of `max(10, D/4K)` (`anchor_min_df`), which you may want to lower for very
rare topics.

## Comparison Summary

| Model | Prior on ``\theta`` | Inference | Deterministic | Key Feature |
|-------|---------------------|-----------|---------------|-------------|
| `LDA` (`:gibbs`) | Dirichlet | Collapsed Gibbs | Given `rng` and `nthreads` | Best held-out likelihood |
| `LDA` (`:vb`) | Dirichlet | Batch / online VB | Given `rng` | ELBO; scales to large corpora |
| `STM` | Logistic normal with covariates | Laplace variational EM | **Yes** (spectral init) | Covariate effects on prevalence |
| `CTM` | Logistic normal | Mean-field variational EM | **Yes** (spectral init) | Topic correlations |
| `DTM` | Dirichlet, topics drift | Variational Kalman | Given `rng` | Topics over time |

**What is not implemented**: STM content covariates (topics whose word distributions vary with
a covariate); hyper-parameter learning for the DTM variances; sparse or alias Gibbs samplers
for very large ``K``.
