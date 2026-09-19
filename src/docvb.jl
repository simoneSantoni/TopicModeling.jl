# Per-document variational inference under a Dirichlet prior with the topics held
# fixed. This is the E-step of LDA (Blei, Ng & Jordan 2003) in the form used by
# Hoffman, Blei & Bach (2010): the token-level responsibilities φ are never stored,
# only their normaliser per term. It is shared by LDA, the DTM document step,
# `transform`, and held-out evaluation.

mutable struct DocVB
    K::Int
    gamma::Vector{Float64}
    last::Vector{Float64}
    Elogtheta::Vector{Float64}
    expElogtheta::Vector{Float64}
    B::Matrix{Float64}          # K × n gather of the topic columns for the current document
    phinorm::Vector{Float64}
    ratio::Vector{Float64}
end

DocVB(K::Int) = DocVB(K, zeros(K), zeros(K), zeros(K), zeros(K), zeros(K, 0), Float64[], Float64[])

function _reserve!(ws::DocVB, n::Int)
    if size(ws.B, 2) < n
        ws.B = Matrix{Float64}(undef, ws.K, max(n, 2 * size(ws.B, 2)))
        resize!(ws.phinorm, size(ws.B, 2))
        resize!(ws.ratio, size(ws.B, 2))
    end
    return ws
end

"""
    infer_doc!(ws, doc, topics, alpha; maxiter=100, tol=1e-3, init=nothing) -> iterations

Coordinate ascent on `q(θ) = Dirichlet(γ)` for one document. `topics` is K×V and
holds either `exp(E[log β])` (when the topics have a variational posterior) or the
topic-word probabilities themselves (point estimate). On return `ws.gamma`,
`ws.Elogtheta`, `ws.expElogtheta`, `ws.B[:, 1:n]` and `ws.phinorm[1:n]` are consistent.

The ascent starts from `γ = α + N/K` (Blei, Ng & Jordan 2003, fig. 6) unless `init`, a
positive length-K vector, is given (only read; may be a view such as a column of a K×D
matrix). Resuming from the γ of an earlier fit to nearby topics needs several times fewer
iterations, but the fixed point then depends on it: a component near `α_k` stays there,
whereas the default start gives every topic the same chance (see `warm_start` in `lda_vb`).
"""
function infer_doc!(ws::DocVB, doc::Document, topics::AbstractMatrix{Float64},
                    alpha::Vector{Float64}; maxiter::Int=100, tol::Float64=1e-3,
                    init::Union{Nothing,AbstractVector{Float64}}=nothing)
    K = ws.K
    n = length(doc.terms)
    _reserve!(ws, n)
    B, phinorm, ratio = ws.B, ws.phinorm, ws.ratio
    gamma, last, Elog, eE = ws.gamma, ws.last, ws.Elogtheta, ws.expElogtheta
    @inbounds for i in 1:n
        w = doc.terms[i]
        @simd for k in 1:K
            B[k, i] = topics[k, w]
        end
    end
    if init === nothing
        N = ntokens(doc)
        @inbounds for k in 1:K
            gamma[k] = alpha[k] + N / K
        end
    else
        length(init) == K || throw(DimensionMismatch("init must have length K"))
        @inbounds for k in 1:K
            gamma[k] = init[k]
        end
    end
    iters = 0
    for it in 1:maxiter
        iters = it
        dirichlet_expectation!(Elog, gamma)
        @inbounds for k in 1:K
            eE[k] = exp(Elog[k])
            last[k] = gamma[k]
            gamma[k] = 0.0
        end
        @inbounds for i in 1:n
            s = 1e-100
            @simd for k in 1:K
                s += eE[k] * B[k, i]
            end
            phinorm[i] = s
            r = doc.counts[i] / s
            @simd for k in 1:K
                gamma[k] += r * B[k, i]
            end
        end
        change = 0.0
        @inbounds for k in 1:K
            gamma[k] = alpha[k] + eE[k] * gamma[k]
            change += abs(gamma[k] - last[k])
        end
        change / K < tol && break
    end
    # Leave E[log θ] and the normalisers consistent with the final γ.
    dirichlet_expectation!(Elog, gamma)
    @inbounds for k in 1:K
        eE[k] = exp(Elog[k])
    end
    @inbounds for i in 1:n
        s = 1e-100
        @simd for k in 1:K
            s += eE[k] * B[k, i]
        end
        phinorm[i] = s
        ratio[i] = doc.counts[i] / s
    end
    return iters
end

"Add the expected topic-term counts `c_w φ_wk` of the current document to `sstats` (K×V)."
function accumulate_sstats!(sstats::AbstractMatrix{Float64}, ws::DocVB, doc::Document)
    K = ws.K
    B, ratio, eE = ws.B, ws.ratio, ws.expElogtheta
    @inbounds for i in eachindex(doc.terms)
        w = doc.terms[i]
        r = ratio[i]
        @simd for k in 1:K
            sstats[k, w] += r * eE[k] * B[k, i]
        end
    end
    return sstats
end

"""
Document contribution to the evidence lower bound (φ collapsed):
`Σ_w c_w log Σ_k exp(E[log θ_k] + E[log β_kw]) − KL(q(θ) ‖ p(θ|α))`.
"""
function doc_bound(ws::DocVB, doc::Document, alpha::Vector{Float64})
    K = ws.K
    ll = 0.0
    @inbounds for i in eachindex(doc.terms)
        ll += doc.counts[i] * log(ws.phinorm[i])
    end
    sa = 0.0; sg = 0.0
    @inbounds for k in 1:K
        a, g = alpha[k], ws.gamma[k]
        ll += (a - g) * ws.Elogtheta[k] + loggamma(g) - loggamma(a)
        sa += a; sg += g
    end
    return ll + loggamma(sa) - loggamma(sg)
end

"""
    infer_theta(corpus, topics, alpha; nthreads, maxiter, tol) -> (theta, gamma)

Posterior-mean topic proportions (D×K) for every document given fixed `topics`.
"""
function infer_theta(c::Corpus, topics::AbstractMatrix{Float64}, alpha::Vector{Float64};
                     nthreads::Int=Threads.nthreads(), maxiter::Int=100, tol::Float64=1e-3)
    check_vocabulary(c, topics)
    K = size(topics, 1)
    D = ndocs(c)
    gamma = Matrix{Float64}(undef, D, K)
    parts = balanced_chunks([length(d.terms) + 1 for d in c.docs], nthreads)
    Threads.@threads for part in parts
        ws = DocVB(K)
        for d in part
            infer_doc!(ws, c.docs[d], topics, alpha; maxiter, tol)
            gamma[d, :] = ws.gamma
        end
    end
    theta = gamma ./ sum(gamma; dims=2)
    return theta, gamma
end
