#!/usr/bin/env bash
# Reproduce every benchmark. Runs are strictly sequential: on the development machine, running
# reference implementations concurrently inflated their wall-clock times by up to ~45%.
#
# Requirements: julia >= 1.10; R with stm, lda, topicmodels (needs GSL), slam; uv (or any Python
# 3.12 env with tomotopy, gensim, scikit-learn in python/.venv).
set -euo pipefail
cd "$(dirname "$0")"
D=data; R=results; PY=python/.venv/bin/python; JL="julia --project=.. -t 8"
mkdir -p $R
[ -d python/.venv ] || (cd python && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python tomotopy gensim scikit-learn numpy scipy)
[ -f $D/ap.ldac ] || Rscript $D/export_data.R $D
[ -f $D/ap.train.ldac ] || julia --project=.. prepare_splits.jl
$JL bench_dtm.jl prepare

# --- LDA, AP, K = 50 -----------------------------------------------------------------------------
AP="--train ../$D/ap.train.ldac --vocab ../$D/ap.vocab"
# (the published table uses the median of three such runs for tomotopy and for Julia)
for th in 1 8; do (cd python && ../$PY run_lda.py --impl tomotopy $AP --out ../$R/lda_ap_tomotopy_t$th -K 50 --iters 1000 --threads $th); done
(cd python && ../$PY run_lda.py --impl sklearn $AP --out ../$R/lda_ap_sklearn -K 50 --iters 100)
(cd python && ../$PY run_lda.py --impl gensim $AP --out ../$R/lda_ap_gensim -K 50 --iters 100)
for impl in lda topicmodels_gibbs; do Rscript R/run_lda.R --impl $impl --train $D/ap.train.ldac --vocab $D/ap.vocab --out $R/lda_ap_$( [ $impl = lda ] && echo Rlda || echo $impl ) -K 50 --iters 1000 --alpha 0.1 --eta 0.01 --seed 1; done
Rscript R/run_lda.R --impl topicmodels_vem --train $D/ap.train.ldac --vocab $D/ap.vocab --out $R/lda_ap_topicmodels_vem -K 50 --iters 100 --alpha 0.1 --eta 0.01 --seed 1
$JL bench_lda.jl

# --- STM, poliblog5k, K = 20 ---------------------------------------------------------------------
Rscript R/run_stm.R --train $D/poliblog5k.train.ldac --vocab $D/poliblog5k.vocab --meta $D/poliblog5k.meta.csv --idx $D/poliblog5k.train.idx \
        --xout $R/stm_poliblog_X.csv --out $R/stm_poliblog_R -K 20 --iters 500 --seed 1
$JL bench_stm.jl

# --- CTM, AP, K = 20 -----------------------------------------------------------------------------
Rscript R/run_ctm.R --train $D/ap.train.ldac --vocab $D/ap.vocab --out $R/ctm_ap_topicmodels -K 20 --iters 500 --seed 1
for th in 1 8; do (cd python && ../$PY run_ctm.py $AP --out ../$R/ctm_ap_tomotopy_t$th -K 20 --iters 1000 --threads $th); done
$JL bench_ctm.jl

# --- DTM -----------------------------------------------------------------------------------------
SY="--train ../$D/dtm_synth.ldac --times ../$D/dtm_synth.times --vocab ../$D/dtm_synth.vocab"
(cd python && ../$PY run_dtm.py --impl gensim $SY --out ../$R/dtm_synth_gensim -K 5 --chain_variance 0.01 --alpha 0.1 --iters 20)
for th in 1 8; do (cd python && ../$PY run_dtm.py --impl tomotopy $SY --out ../$R/dtm_synth_tomotopy_t$th -K 5 --threads $th
  ../$PY run_dtm.py --impl tomotopy --train ../$D/dtm_poliblog.train.ldac --times ../$D/dtm_poliblog.train.times --vocab ../$D/poliblog5k.vocab --out ../$R/dtm_poliblog_tomotopy_t$th -K 10 --threads $th); done
$JL bench_dtm.jl
