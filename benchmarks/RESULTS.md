# Benchmarks

Everything here is produced by [`run_all.sh`](run_all.sh); the tables are copied from
`results/*.md`, which the `bench_*.jl` scripts write.

## Method

- **Same input.** Corpora are exported once ([`data/export_data.R`](data/export_data.R)) to
  Blei's LDA-C format, and one fixed 90/10 document split ([`prepare_splits.jl`](prepare_splits.jl))
  is read by the Julia, Python and R runners alike.
- **Same yardstick.** Each implementation only hands over its topic-word matrix φ. One Julia
  function (`score` in [`common.jl`](common.jl)) then computes, for all of them: held-out
  perplexity by document completion (Wallach et al. 2009) — half of each test document's tokens
  are used to infer its topic proportions with φ fixed, the other half is scored; mean ± sd over
  5 random token splits — and NPMI coherence of the top-10 words on the training corpus.
  The vocabulary is the set of terms seen in training: implementations disagree on whether unseen
  terms get smoothing mass or no column at all, which otherwise changes perplexity by orders of
  magnitude (a first version of this benchmark scored tomotopy at 660,000 for that reason alone).
- **Timing.** Wall-clock time of fitting only (no I/O, no Julia compilation: a small warm-up fit
  precedes each timed fit). Runs are sequential on an otherwise idle 32-thread x86-64 Linux
  machine. Run-to-run noise on this machine is around ±20% (tomotopy LDA, identical settings:
  16.8, 19.6, 20.4, 24.1 s), and running reference implementations concurrently inflated their
  times by up to 45%, so small differences mean nothing.
- Versions: Julia 1.12.6, tomotopy 0.14.0, gensim 4.4.0, scikit-learn 1.9.1, R 4.6.1 with stm,
  topicmodels and lda from CRAN (September 2026).

## Correctness against known ground truth and the literature

These are asserted in the test suite (`test/runtests.jl`).

| Claim | Source | Result |
|---|---|---|
| LDA recovers the ten "bars" topics on a 5×5 pixel vocabulary | Griffiths & Steyvers (2004), Fig. 1 | Gibbs: mean TV distance to the true bars 0.035. VB: 0.026 for the best-ELBO restart |
| Spectral initialisation avoids the poor local modes that LDA/random initialisation falls into | Roberts, Stewart & Tingley (2016) | STM started at the *true* topics ends at bound −5.35139/token, TV 0.060; spectral init reaches the same optimum (−5.35123, TV 0.063); LDA and random init end lower (−5.3557, −5.3553) with one topic lost (max TV 0.89) |
| STM recovers covariate effects on topic prevalence | Roberts, Stewart & Airoldi (2016), §4 | simulated binary covariate: true effect on topic 1 +0.319, estimated +0.334 (se 0.007); sign and magnitude right on all 6 topics |
| A DTM tracks drifting topics where a static model cannot | Blei & Lafferty (2006) | synthetic random-walk topics, 12 slices: DTM error flat at TV ≈ 0.050 in every slice; static LDA U-shaped, 0.104 at the ends and 0.069 mid-period |
| The CTM recovers correlation between topics | Blei & Lafferty (2007) | planted correlations of θ +0.32 and −0.51, estimated +0.31 and −0.50 |
| Analytic gradients/Hessian (STM, CTM, DTM objectives) | — | agree with central finite differences to ~1e-7 |
| Every variational objective is non-decreasing over EM iterations | — | holds for LDA-VB, STM, DTM, CTM on all test problems |

### Local optima: what a single run does and does not tell you

On a synthetic corpus (1500 documents, 8 topics, V = 300) batch variational LDA finds all topics
(TV < 0.05) from **14 of 24** random starts in TopicModeling.jl and **11 of 24** in scikit-learn,
with near-identical error quartiles ([0.027, 0.040, 0.153] vs [0.026, 0.052, 0.161]); the failures
merge two topics, and sit at a visibly lower ELBO (−4.04 to −4.06 vs −3.93 per token). Collapsed
Gibbs fails the same way in about one run in four, serial or threaded, and does not escape in 1000
sweeps. Online VB on a corpus this small is worse still, in both libraries (TV 0.2–0.45, not
improving between 15 and 200 passes). None of this is specific to an implementation — but it
means implementations should be compared over several seeds, or by their best-objective run.

An earlier version of this comparison used 4 seeds per library and showed "1/4 vs 4/4", which
would have been reported as a defect had it not been rerun with 24.
