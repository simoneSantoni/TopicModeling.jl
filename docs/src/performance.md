# Performance

TopicModeling.jl runs the expensive part of every model — Gibbs sweeps and per-document
E-steps — on multiple threads, with per-task scratch buffers so that the inner loops do not
allocate.

## Start Julia with threads

```
julia -t auto          # or: julia -t 8, or JULIA_NUM_THREADS=8
```

Every `fit`, [`transform`](@ref) and [`heldout_perplexity`](@ref) call takes `nthreads`
(default `Threads.nthreads()`). Julia cannot add threads after start-up, so a session started
without `-t` runs everything serially.

## Parallel sections

| Model | Parallel | Serial |
|:---|:---|:---|
| [`LDA`](@ref) Gibbs | sweeps, over document × word blocks (Yan, Xu and Qi 2009); posterior averaging; ``\alpha`` optimisation | — |
| [`LDA`](@ref) VB | E-step over documents; ``\mathbb E[\log\beta]``, ``\lambda`` update and topic part of the ELBO over vocabulary blocks | ``\alpha`` optimisation |
| [`STM`](@ref) | spectral initialisation; E-step over documents (L-BFGS + Hessian + Cholesky per document) | ``\Gamma``, ``\Sigma`` updates |
| [`CTM`](@ref) | spectral initialisation; E-step over documents | ``\mu``, ``\Sigma`` updates |
| [`DTM`](@ref) | E-step over fixed blocks of documents within slices; M-step over topics | bound |

Documents are split into contiguous blocks of near-equal *work* (number of distinct terms), not
equal count, so that a few long documents do not leave threads idle. Each task owns its
scratch buffers and its sufficient-statistics accumulator; nothing is indexed by
`Threads.threadid()`, so results do not depend on task migration.

## Reproducibility

- Pass `rng` (e.g. `Xoshiro(1)`). Per-task generators are seeded from it.
- Variational fits (`LDA` with `method=:vb`, `STM`, `CTM`) give the same result for any
  `nthreads` up to floating-point summation order. [`DTM`](@ref) with a matrix `init` is
  bitwise identical for any `nthreads`; its default `init=:lda` inherits the dependence of the
  Gibbs sampler.
- The threaded Gibbs sampler is a different Markov chain for each `nthreads`: fix both `rng`
  and `nthreads` to reproduce a run exactly. On the AP corpus the 8-thread chain reaches the
  same log-likelihood per token as the serial one after the same number of sweeps.
- [`heldout_perplexity`](@ref) returns the same value for any `nthreads`.

## BLAS threads

[`STM`](@ref) solves one small ``(K-1)\times(K-1)`` Cholesky problem per document inside its
threaded E-step. `fit` sets the number of BLAS threads to 1 for its duration and restores it
afterwards, to avoid oversubscription. If you call `fit` from your own threaded code, set
`nthreads=1` in the inner call.

## Practical advice

- **Small corpora do not benefit from many threads.** The Gibbs sampler only uses one task per
  20 000 tokens; below that the synchronisation costs more than it saves.
- **Prune the vocabulary** ([`prune`](@ref), or `min_df`/`max_df` in the [`Corpus`](@ref)
  constructor). Memory and the serial parts scale with ``K \times V``.
- **Online VB** (`batchsize=1024`, say) for corpora where a full pass is expensive.
- **Validated input.** [`Corpus`](@ref) and [`Document`](@ref) check term ids and counts once,
  at construction; the inner loops then index without bounds checks.
- **Start-up latency.** A precompile workload fits every model once when the package is
  precompiled, so the first `fit` in a session takes milliseconds, not seconds.

## Benchmarks against other software

The Home page shows the LDA comparison with tomotopy (C++), gensim and scikit-learn (Python),
and `lda` and `topicmodels` (R) on the Associated Press corpus. The methodology — same input
files, one scoring function for every implementation, sequential warmed runs — the scripts to
reproduce it (`benchmarks/run_all.sh`), and the checks against results from the literature are
in [`benchmarks/RESULTS.md`](https://github.com/simoneSantoni/TopicModeling.jl/blob/main/benchmarks/RESULTS.md).

Absolute timings are host-specific. Run-to-run noise on the benchmark machine was around ±20%,
and running reference implementations concurrently inflated their times by up to 45%, so the
published runs are strictly sequential.
