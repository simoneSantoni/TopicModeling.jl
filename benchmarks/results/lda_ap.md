| implementation | language | algorithm | threads | time (s) | held-out perplexity | NPMI | diversity |
|---|---|---|---|---|---|---|---|
| **TopicModeling.jl** Gibbs | Julia | collapsed Gibbs, 1000 sweeps | 1 | 14.0 | 2367 ± 20 | 0.232 | 0.71 |
| **TopicModeling.jl** Gibbs | Julia | collapsed Gibbs, 1000 sweeps | 8 | 7.6 | 2375 ± 30 | 0.243 | 0.71 |
| **TopicModeling.jl** VB | Julia | batch VB, 100 passes | 1 | 14.9 | 2844 ± 30 | 0.167 | 0.65 |
| **TopicModeling.jl** VB | Julia | batch VB, 100 passes | 8 | 5.4 | 2844 ± 30 | 0.167 | 0.65 |
| tomotopy | C++ | collapsed Gibbs, 1000 sweeps | 1 | 16.8 | 2462 ± 16 | 0.255 | 0.71 |
| tomotopy | C++ | collapsed Gibbs, 1000 sweeps | 8 | 7.8 | 2472 ± 22 | 0.214 | 0.72 |
| lda | R/C | collapsed Gibbs, 1000 sweeps | 1 | 32.5 | 2556 ± 16 | 0.229 | 0.69 |
| topicmodels | R/C++ | collapsed Gibbs, 1000 sweeps | 1 | 78.2 | 2527 ± 24 | 0.239 | 0.68 |
| scikit-learn | Python/Cython | batch VB, 100 passes | 1 | 68.2 | 2859 ± 40 | 0.144 | 0.64 |
| gensim | Python/NumPy | online VB, 100 passes | 1 | 111.8 | 3675 ± 45 | 0.107 | 0.77 |
| topicmodels | R/C (lda-c) | VEM, α estimated | 1 | 339.7 | 2972 ± 33 | 0.177 | 0.49 |
