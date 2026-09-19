"""
    TopicModeling

Probabilistic topic models in pure Julia:

- [`LDA`](@ref)  Latent Dirichlet Allocation (collapsed Gibbs; batch/online variational Bayes)
- [`STM`](@ref)  Structural Topic Model with prevalence covariates (Laplace variational EM)
- [`DTM`](@ref)  Dynamic Topic Model (variational Kalman smoothing)
- [`CTM`](@ref)  Correlated Topic Model (logistic-normal variational EM)

All models are fitted with `fit(Model, corpus, K; ...)` and share the accessors
[`topicword`](@ref), [`doctopic`](@ref), [`topwords`](@ref) and the evaluation tools
[`coherence`](@ref), [`heldout_perplexity`](@ref) and [`match_topics`](@ref).
"""
module TopicModeling

using LinearAlgebra
using Printf
using Random
using SparseArrays
using Statistics
using SpecialFunctions: digamma, trigamma, loggamma
import StatsAPI
import StatsAPI: fit

export Corpus, Document, tokenize, ndocs, nterms, ntokens, tokens, dtm, docfreq, termfreq, prune,
       read_ldac, write_ldac, read_uci, train_test_split, split_documents
export fit, LDA, STM, DTM, CTM, topic_correlations, transform, estimate_effect, spectral_init
export topicword, doctopic, vocabulary, topwords, coherence, topic_diversity,
       heldout_perplexity, match_topics
export simulate_lda, simulate_logistic_normal, simulate_dtm, bars_topics

"Supertype of all fitted topic models."
abstract type AbstractTopicModel end

"""
    topicword(model)

K×V matrix of topic-word probabilities (rows sum to one).
"""
function topicword end

"""
    doctopic(model)

D×K matrix of document-topic proportions (rows sum to one).
"""
function doctopic end

"Vocabulary the model was fitted on."
vocabulary(m::AbstractTopicModel) = m.vocab

include("utils.jl")
include("optim.jl")
include("corpus.jl")
include("docvb.jl")
include("lda.jl")
include("spectral.jl")
include("stm.jl")
include("dtm.jl")
include("ctm.jl")
include("evaluation.jl")
include("synthetic.jl")

function Base.show(io::IO, m::AbstractTopicModel)
    print(io, nameof(typeof(m)), "(K=", size(topicword(m), 1), ", V=", size(topicword(m), 2),
          ", D=", size(doctopic(m), 1), ")")
end

# Precompilation workload: a tiny fit of every model and the shared tools, run only while the
# package image is generated, so that the first call in a session does not pay for compilation.
# It prints nothing, uses its own random generators, and a failure can never break loading.
function _precompile_workload()
    rng() = Xoshiro(1)
    corpus, phi, _ = simulate_lda(; D=40, K=3, V=50, doclen=30, rng=rng())
    X = reshape(Float64.(1:40) ./ 40, :, 1)
    times = repeat(1:4, 10)
    for nthreads in (1, 2)          # 2: the task-based code paths (they also run on one thread)
        gibbs = fit(LDA, corpus, 3; iters=12, burnin=2, optimize_interval=5, eval_every=6, nthreads, rng=rng())
        vb = fit(LDA, corpus, 3; method=:vb, iters=3, nthreads, rng=rng())
        fit(LDA, corpus, 3; method=:vb, iters=2, batchsize=16, optimize_alpha=true, nthreads, rng=rng())
        stm = fit(STM, corpus, 3; prevalence=X, keep_nu=true, iters=3, nthreads, rng=rng())
        estimate_effect(stm; nsims=2, rng=rng())
        plain = fit(STM, corpus, 3; iters=2, init=:lda, nthreads, rng=rng())
        ctm = fit(CTM, corpus, 3; iters=3, nthreads, rng=rng())
        topic_correlations(ctm)
        dyn = fit(DTM, corpus, times, 3; iters=2, nthreads, rng=rng())
        transform(gibbs, corpus; nthreads); transform(vb, corpus; nthreads)
        transform(stm, corpus; prevalence=X, nthreads); transform(plain, corpus; nthreads)
        transform(ctm, corpus; nthreads); transform(dyn, corpus, times; nthreads)
        heldout_perplexity(phi, 0.1, corpus; nthreads, rng=rng())
        heldout_perplexity(dyn, corpus, times; nthreads, rng=rng())
        for m in (gibbs, vb, stm, ctm)
            heldout_perplexity(m, corpus; nthreads, rng=rng())
            topwords(m; n=5); coherence(m, corpus; n=5); sprint(show, m)
        end
        topwords(dyn, 1; n=5); sprint(show, dyn)
    end
    coherence(phi, corpus; n=5, measure=:umass); coherence(phi, corpus; n=5, measure=:npmi)
    topic_diversity(phi; n=5); match_topics(phi, phi[[2, 3, 1], :])
    train_test_split(corpus; rng=rng()); prune(corpus; min_df=2); dtm(corpus); sprint(show, corpus)
    Corpus(["topic models find topics in text", "models of text and topics"]; min_df=1)
    mktempdir() do dir
        path = write_ldac(joinpath(dir, "c.ldac"), corpus; vocab=joinpath(dir, "c.vocab"))
        read_ldac(path; vocab=joinpath(dir, "c.vocab"))
    end
    return nothing
end

if ccall(:jl_generating_output, Cint, ()) == 1
    let blas_threads = BLAS.get_num_threads()
        try
            Base.CoreLogging.with_logger(_precompile_workload, Base.CoreLogging.NullLogger())
        catch
        finally
            BLAS.set_num_threads(blas_threads)
        end
    end
end

end # module
