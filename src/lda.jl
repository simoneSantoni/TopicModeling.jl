# Latent Dirichlet Allocation.
#
#   method = :gibbs  collapsed Gibbs sampling (Griffiths & Steyvers 2004), optionally
#                    multi-threaded by document × word block partitioning (Yan, Xu & Qi
#                    2009), with Minka's fixed-point update for an asymmetric
#                    document-topic prior and posterior averaging of φ and θ.
#   method = :vb     variational Bayes with smoothed topics (Blei, Ng & Jordan 2003),
#                    batch or online/stochastic (Hoffman, Blei & Bach 2010).

"""
    LDA

Fitted Latent Dirichlet Allocation model. Obtain one with
`fit(LDA, corpus, K; method=:gibbs, ...)`.

Fields: `phi` (K×V topic-word probabilities), `theta` (D×K document-topic
proportions), `alpha` (document-topic Dirichlet parameter, length K), `eta`
(topic-word Dirichlet parameter), `vocab`, `trace` (objective per evaluated iteration:
joint log-likelihood per token for Gibbs, ELBO per token for VB), `lambda` (K×V
variational Dirichlet parameters, VB only), `z` (final topic assignments, Gibbs only).

For Gibbs, `phi` and `theta` are posterior means averaged over the last `nsamples`
retained states of the chain (see [`fit`](@ref)), whereas `alpha`, `trace[end]` and `z`
describe its final state. `z[d][i]` is the topic of the `i`-th token of document `d` in
the sampler's token order, `shuffle!(rng, tokens(corpus[d]))` drawn document by document
from the `rng` given to `fit` before anything else, whatever the number of threads. The
tokens of a document are exchangeable, so only the pairing of `z[d]` with that order
carries meaning.
"""
struct LDA <: AbstractTopicModel
    K::Int
    phi::Matrix{Float64}
    theta::Matrix{Float64}
    alpha::Vector{Float64}
    eta::Float64
    vocab::Vector{String}
    method::Symbol
    trace::Vector{Float64}
    lambda::Matrix{Float64}
    z::Vector{Vector{Int32}}
    iterations::Int
    elapsed::Float64
end

topicword(m::LDA) = m.phi
doctopic(m::LDA) = m.theta

"""
    fit(LDA, corpus, K; method=:gibbs, kwargs...)

Common keywords: `alpha=0.1` (scalar or length-K vector), `eta=0.01`, `iters`,
`rng`, `nthreads`, `verbose`.

Gibbs (`method=:gibbs`): `iters=1000`, `burnin=100`, `optimize_alpha=true`,
`optimize_interval=10`, `eval_every=0`, `nsamples=10`, `sample_lag=10`.
`phi` and `theta` are the averages of `(n_kw + η) / (n_k + Vη)` and
`(n_dk + α_k) / (n_d + Σα)` over `nsamples` states of the chain taken every `sample_lag`
sweeps and ending with the last one, i.e. from sweep `iters - (nsamples - 1) * sample_lag`
on; sweeps up to `burnin` are never averaged, so a short run (`iters <= burnin + sample_lag`)
and `nsamples=1` both give the estimate from the final state alone. Averaging lowers
held-out perplexity at no cost in sweeps (Griffiths & Steyvers 2004 average over samples
likewise).

With `nthreads > 1` (and at least 20 000 tokens per thread) the sweep is parallelised by
partitioning (Yan, Xu & Qi 2009): documents and vocabulary are each cut into `T` blocks of
equal token mass, and in each of `T` sub-rounds task `t` samples the tokens of document
block `t` that fall into word block `t + r (mod T)`, so that concurrent tasks touch
disjoint rows and columns of the count tables. Only the `K` topic totals are task-local
for the length of a sub-round. Results are reproducible for a fixed `(rng, nthreads)`
and, for the sampler with one thread, do not depend on the number of Julia threads.

Variational (`method=:vb`): `iters=200` passes, `tol=1e-6` relative ELBO change,
`batchsize=0` (0 = batch VB; >0 = online VB with `tau0=64`, `kappa=0.7`),
`optimize_alpha=false` (Newton updates of `alpha`; in online mode one step per minibatch,
damped by the learning rate), `doc_maxiter=100`, `doc_tol=1e-3` (per-document coordinate
ascent), `init=nothing` (K×V non-negative matrix, e.g. topic-word probabilities, to start
the topics from instead of a random draw: `lambda = eta + init * N / K`), `warm_start=0`
(batch only; `n > 0` resumes every document from its previous variational posterior after
pass `n` instead of restarting it: several times fewer document iterations per pass, but
documents can no longer pick up topics they have dropped, which costs ELBO and held-out
likelihood, the more the smaller `n`). `iters`, `alpha` and `eta` must be positive.
The fit stops at pass `iters` or when the ELBO converges; in the latter case `phi`,
`theta` and `trace[end]` belong to the same pass.
"""
function StatsAPI.fit(::Type{LDA}, corpus::Corpus, K::Integer; method::Symbol=:gibbs, kwargs...)
    K >= 2 || throw(ArgumentError("need at least two topics"))
    method === :gibbs && return lda_gibbs(corpus, Int(K); kwargs...)
    method === :vb && return lda_vb(corpus, Int(K); kwargs...)
    throw(ArgumentError("unknown method $method; use :gibbs or :vb"))
end

_alpha_vector(alpha::Real, K::Int) = fill(Float64(alpha), K)
function _alpha_vector(alpha::AbstractVector{<:Real}, K::Int)
    length(alpha) == K || throw(DimensionMismatch("alpha must have length K"))
    return collect(Float64, alpha)
end

# --- collapsed Gibbs sampling -------------------------------------------------------------

# One sweep over a cell of the document × word-block grid: the tokens of the documents in
# `docs` that belong to word block `b`. Each document stores its tokens grouped by word
# block, those of block `b` at `ptr[b, d]:(ptr[b + 1, d] - 1)`; the serial sampler has one
# block holding everything. `nk` is either the global vector of topic totals or a task-local
# copy of it (see `gibbs_sweep_partitioned!`).
function gibbs_sweep!(z::Vector{Int32}, w::Vector{Int32}, ptr::Matrix{Int}, b::Int,
                      ndk::Matrix{Int32}, nkw::Matrix{Int32}, nk::Vector{Int},
                      alpha::Vector{Float64}, eta::Float64, docs::UnitRange{Int},
                      rng::AbstractRNG, p::Vector{Float64}, adk::Vector{Float64},
                      invnk::Vector{Float64})
    K = length(alpha)
    Veta = size(nkw, 2) * eta
    @inbounds for k in 1:K
        invnk[k] = 1.0 / (nk[k] + Veta)
    end
    @inbounds for d in docs
        lo, hi = ptr[b, d], ptr[b + 1, d] - 1
        lo > hi && continue
        for k in 1:K
            adk[k] = ndk[k, d] + alpha[k]
        end
        for i in lo:hi
            wi = w[i]
            k = z[i]
            ndk[k, d] -= Int32(1); nkw[k, wi] -= Int32(1); nk[k] -= 1
            adk[k] -= 1.0
            invnk[k] = 1.0 / (nk[k] + Veta)
            total = 0.0
            @simd for j in 1:K
                pj = adk[j] * (nkw[j, wi] + eta) * invnk[j]
                p[j] = pj
                total += pj
            end
            u = rand(rng) * total
            knew = K
            acc = 0.0
            for j in 1:K
                acc += p[j]
                if acc >= u
                    knew = j
                    break
                end
            end
            z[i] = knew
            ndk[knew, d] += Int32(1); nkw[knew, wi] += Int32(1); nk[knew] += 1
            adk[knew] += 1.0
            invnk[knew] = 1.0 / (nk[knew] + Veta)
        end
    end
    return nothing
end

# Partitioned parallel sweep (Yan, Xu & Qi 2009; the scheme of tomotopy): with documents and
# vocabulary both cut into T blocks, the T cells (document block t, word block t + r mod T)
# of sub-round r share no row of `ndk` and no column of `nkw`, so T tasks sample them
# concurrently, writing straight into the global tables. Only the topic totals `nk` couple
# the cells: each task works on a private copy, and the copies are reconciled after the
# sub-round as global += Σ_t (local_t − global_old). T sub-rounds visit every token once.
# Unlike approximate distributed LDA (Newman et al. 2009), which keeps a stale private copy
# of all of `nkw` for a whole sweep, this converges per sweep like the serial sampler and
# needs no extra K×V memory.
function gibbs_sweep_partitioned!(z, w, ptr, ndk, nkw, nk::Vector{Int}, local_nk::Vector{Vector{Int}},
                                  alpha, eta, parts, rngs, ps, adks, invs)
    T = length(parts)
    K = length(alpha)
    for r in 0:(T - 1)
        Threads.@threads for t in 1:T
            copyto!(local_nk[t], 1, nk, 1, K)
            gibbs_sweep!(z, w, ptr, mod1(t + r, T), ndk, nkw, local_nk[t], alpha, eta, parts[t],
                         rngs[t], ps[t], adks[t], invs[t])
        end
        @inbounds for k in 1:K
            old = nk[k]
            for t in 1:T
                nk[k] += local_nk[t][k] - old
            end
        end
    end
    return nothing
end

# Split the vocabulary into `T` blocks of near-equal token mass (greedy: most frequent term
# first, each into the currently lightest block) and renumber the terms so that every block
# is a contiguous range of columns of `nkw`, which keeps the tasks of a sub-round off each
# other's cache lines. Returns the block and the new index of every term.
function word_blocks(tf::Vector{Int}, T::Int)
    V = length(tf)
    block = Vector{Int}(undef, V)
    load = zeros(Int, T)
    for v in sortperm(tf; rev=true)
        b = argmin(load)
        block[v] = b
        load[b] += tf[v]
    end
    newid = Vector{Int32}(undef, V)
    newid[sortperm(block)] = 1:V
    return block, newid
end

# Regroup the tokens of every document by word block (a stable counting sort, carrying `z`
# along) and renumber the terms. Returns the (T+1)×D table of block offsets and, for every
# token, its original offset within the document, so that `z` can be handed back in the
# order in which the serial sampler holds it.
function group_tokens!(w::Vector{Int32}, z::Vector{Int32}, docptr::Vector{Int},
                       block::Vector{Int}, newid::Vector{Int32}, T::Int)
    D = length(docptr) - 1
    ptr = Matrix{Int}(undef, T + 1, D)
    origin = Vector{Int32}(undef, length(w))
    w0 = copy(w); z0 = copy(z)
    next = zeros(Int, T)
    @inbounds for d in 1:D
        r = docptr[d]:(docptr[d + 1] - 1)
        fill!(next, 0)
        for i in r
            next[block[w0[i]]] += 1
        end
        ptr[1, d] = docptr[d]
        for b in 1:T
            ptr[b + 1, d] = ptr[b, d] + next[b]
            next[b] = ptr[b, d]
        end
        for i in r
            b = block[w0[i]]
            j = next[b]
            next[b] += 1
            w[j] = newid[w0[i]]; z[j] = z0[i]; origin[j] = i - docptr[d]
        end
    end
    return ptr, origin
end

# The sums over the two count tables are split into a fixed number of partial sums, added up
# in a fixed order, so that the value does not depend on the number of threads or on the
# order in which tasks finish.
const LOGLIK_CHUNKS = 64

# log p(w, z | α, η): the quantity a collapsed sampler explores.
function gibbs_loglik(ndk::Matrix{Int32}, nkw::Matrix{Int32}, nk::Vector{Int},
                      doclen::Vector{Int}, alpha::Vector{Float64}, eta::Float64;
                      nthreads::Int=Threads.nthreads())
    K, V = size(nkw)
    D = size(ndk, 2)
    lg_eta = loggamma(eta)
    ll = K * (loggamma(V * eta) - V * lg_eta)
    @inbounds for k in 1:K
        ll -= loggamma(nk[k] + V * eta)
    end
    wparts = chunks(V, LOGLIK_CHUNKS)
    wsums = zeros(length(wparts))
    Threads.@threads for cs in chunks(length(wparts), nthreads)
        @inbounds for c in cs
            s = 0.0
            for v in wparts[c], k in 1:K
                n = nkw[k, v]
                s += n == 0 ? lg_eta : loggamma(n + eta)
            end
            wsums[c] = s
        end
    end
    ll += sum(wsums)
    sa = sum(alpha)
    lg_sa = loggamma(sa) - sum(loggamma, alpha)
    lg_alpha = loggamma.(alpha)
    dparts = chunks(D, LOGLIK_CHUNKS)
    dsums = zeros(length(dparts))
    Threads.@threads for cs in chunks(length(dparts), nthreads)
        @inbounds for c in cs
            s = 0.0
            for d in dparts[c]
                s += lg_sa - loggamma(doclen[d] + sa)
                for k in 1:K
                    n = ndk[k, d]
                    s += n == 0 ? lg_alpha[k] : loggamma(n + alpha[k])
                end
            end
            dsums[c] = s
        end
    end
    return ll + sum(dsums)
end

# Minka (2000) fixed-point iteration for the Dirichlet-multinomial concentration:
# α_k ← α_k · Σ_d [ψ(n_dk + α_k) − ψ(α_k)] / Σ_d [ψ(n_d + Σα) − ψ(Σα)]. Every α_k is updated
# from the old Σα and its own old value only, so the updates of one iteration are independent
# and the loop over k is threaded; the result is bitwise that of a serial run.
function optimize_alpha_fixedpoint!(alpha::Vector{Float64}, ndk::Matrix{Int32},
                                    doclen::Vector{Int}; iters::Int=10,
                                    nthreads::Int=Threads.nthreads())
    K = size(ndk, 1)
    parts = chunks(K, nthreads)
    for _ in 1:iters
        denom = alpha_denominator(doclen, sum(alpha))
        denom > 0 || return alpha
        update!(ks) = @inbounds for k in ks
            alpha[k] = max(alpha[k] * alpha_numerator(ndk, k, alpha[k]) / denom, 1e-5)
        end
        if length(parts) == 1                  # stay on the calling thread
            update!(parts[1])
        else
            Threads.@threads for ks in parts
                update!(ks)
            end
        end
    end
    return alpha
end

function alpha_denominator(doclen::Vector{Int}, sa::Float64)
    ψsa = digamma(sa)
    denom = 0.0
    @inbounds for d in eachindex(doclen)
        doclen[d] == 0 && continue
        denom += digamma(doclen[d] + sa) - ψsa
    end
    return denom
end

function alpha_numerator(ndk::Matrix{Int32}, k::Int, a::Float64)
    ψa = digamma(a)
    num = 0.0
    @inbounds for d in axes(ndk, 2)
        n = ndk[k, d]
        n == 0 && continue
        num += digamma(n + a) - ψa
    end
    return num
end

# Add the current-state estimates (n_kw + η) / (n_k + Vη) and (n_dk + α_k) / (n_d + Σα) to
# the running sums `phi` (K×V) and `theta` (D×K). Column `newid[v]` of `nkw` holds term `v`.
function accumulate_estimates!(phi::Matrix{Float64}, theta::Matrix{Float64}, ndk::Matrix{Int32},
                               nkw::Matrix{Int32}, nk::Vector{Int}, doclen::Vector{Int},
                               alpha::Vector{Float64}, eta::Float64, newid::Vector{Int32},
                               nthreads::Int)
    K, V = size(nkw)
    D = size(ndk, 2)
    Threads.@threads for cols in chunks(V, nthreads)
        @inbounds for v in cols
            c = newid[v]
            for k in 1:K
                phi[k, v] += (nkw[k, c] + eta) / (nk[k] + V * eta)
            end
        end
    end
    sa = sum(alpha)
    Threads.@threads for docs in chunks(D, nthreads)
        @inbounds for d in docs, k in 1:K
            theta[d, k] += (ndk[k, d] + alpha[k]) / (doclen[d] + sa)
        end
    end
    return nothing
end

function lda_gibbs(corpus::Corpus, K::Int; alpha=0.1, eta::Real=0.01, iters::Int=1000,
                   burnin::Int=100, optimize_alpha::Bool=true, optimize_interval::Int=10,
                   eval_every::Int=0, nsamples::Int=10, sample_lag::Int=10,
                   nthreads::Int=Threads.nthreads(), rng::AbstractRNG=Random.default_rng(),
                   verbose::Bool=false)
    t0 = time()
    nsamples >= 1 || throw(ArgumentError("nsamples must be at least 1"))
    sample_lag >= 1 || throw(ArgumentError("sample_lag must be at least 1"))
    D, V = ndocs(corpus), nterms(corpus)
    α = _alpha_vector(alpha, K)
    η = Float64(eta)
    doclen = [ntokens(d) for d in corpus.docs]
    docptr = cumsum(vcat(1, doclen))
    N = docptr[end] - 1
    w = Vector{Int32}(undef, N)
    for (d, doc) in enumerate(corpus.docs)
        w[docptr[d]:(docptr[d + 1] - 1)] = shuffle!(rng, tokens(doc))
    end
    z = Vector{Int32}(undef, N)
    for i in 1:N
        z[i] = rand(rng, 1:K)
    end

    # Threading only pays off once each thread has a meaningful amount of work.
    parts = balanced_chunks(doclen, clamp(min(nthreads, N ÷ 20_000), 1, max(D, 1)))
    T = length(parts)
    nt = max(nthreads, 1)                      # tasks for the cheap loops over columns
    if T == 1
        newid = collect(Int32, 1:V)
        ptr = permutedims(hcat(docptr[1:D], docptr[2:(D + 1)]))
        origin = Int32[]
    else
        block, newid = word_blocks(termfreq(corpus), T)
        ptr, origin = group_tokens!(w, z, docptr, block, newid, T)
    end
    ndk = zeros(Int32, K, D); nkw = zeros(Int32, K, V); nk = zeros(Int, K)
    for d in 1:D, i in docptr[d]:(docptr[d + 1] - 1)
        k = z[i]
        ndk[k, d] += Int32(1); nkw[k, w[i]] += Int32(1); nk[k] += 1
    end
    rngs = [Xoshiro(rand(rng, UInt64)) for _ in 1:T]
    # Scratch vectors are padded so that different threads never write to one cache line.
    ps = [zeros(K + 16) for _ in 1:T]; adks = [zeros(K + 16) for _ in 1:T]; invs = [zeros(K + 16) for _ in 1:T]
    local_nk = [zeros(Int, K + 16) for _ in 1:(T > 1 ? T : 0)]

    # φ and θ are averaged over the states after sweeps iters, iters − lag, ... that lie
    # beyond the burn-in (the last sweep always counts).
    first_sample = max(iters - (nsamples - 1) * sample_lag, min(burnin, iters - 1) + 1)
    phi = zeros(K, V); theta = zeros(D, K)
    taken = 0

    trace = Float64[]
    for it in 1:iters
        if T == 1
            gibbs_sweep!(z, w, ptr, 1, ndk, nkw, nk, α, η, parts[1], rngs[1], ps[1], adks[1], invs[1])
        else
            gibbs_sweep_partitioned!(z, w, ptr, ndk, nkw, nk, local_nk, α, η, parts, rngs, ps, adks, invs)
        end
        if optimize_alpha && it >= burnin && optimize_interval > 0 && it % optimize_interval == 0
            optimize_alpha_fixedpoint!(α, ndk, doclen; nthreads=nt)
        end
        if eval_every > 0 && (it % eval_every == 0 || it == iters)
            ll = gibbs_loglik(ndk, nkw, nk, doclen, α, η; nthreads=nt) / N
            push!(trace, ll)
            verbose && @printf("iter %5d  log p(w,z)/token = %.5f\n", it, ll)
        end
        if it >= first_sample && (iters - it) % sample_lag == 0
            accumulate_estimates!(phi, theta, ndk, nkw, nk, doclen, α, η, newid, nt)
            taken += 1
        end
    end
    eval_every > 0 || push!(trace, gibbs_loglik(ndk, nkw, nk, doclen, α, η; nthreads=nt) / N)

    taken == 0 && accumulate_estimates!(phi, theta, ndk, nkw, nk, doclen, α, η, newid, nt)   # iters == 0
    if taken > 1
        phi ./= taken; theta ./= taken
    end
    zs = [z[docptr[d]:(docptr[d + 1] - 1)] for d in 1:D]
    if T > 1                                   # back to the token order of the serial sampler
        for d in 1:D, i in docptr[d]:(docptr[d + 1] - 1)
            zs[d][origin[i] + 1] = z[i]
        end
    end
    return LDA(K, phi, theta, α, η, corpus.vocab, :gibbs, trace, zeros(0, 0), zs, iters, time() - t0)
end

# --- variational Bayes ---------------------------------------------------------------------

# Newton step for the Dirichlet parameter using the diagonal-plus-rank-one Hessian
# (Blei, Ng & Jordan 2003, appendix A.4.2). `sumElog[k] = Σ_d E[log θ_dk]`, multiplied by
# `scale` (online VB: D / |minibatch|). `g` and `h` are length-K scratch.
function optimize_alpha_newton!(alpha::Vector{Float64}, sumElog::Vector{Float64}, D::Int;
                                iters::Int=20, scale::Float64=1.0,
                                g::Vector{Float64}=zeros(length(alpha)),
                                h::Vector{Float64}=zeros(length(alpha)))
    K = length(alpha)
    for _ in 1:iters
        sa = sum(alpha)
        ψsa = digamma(sa)
        sgh = 0.0; sh = 0.0
        @inbounds for k in 1:K
            g[k] = D * (ψsa - digamma(alpha[k])) + scale * sumElog[k]
            h[k] = -D * trigamma(alpha[k])
            sgh += g[k] / h[k]; sh += 1 / h[k]
        end
        zc = D * trigamma(sa)
        c = sgh / (1 / zc + sh)
        step = 1.0
        maxdelta = 0.0
        @inbounds for k in 1:K
            g[k] = (g[k] - c) / h[k]                    # Newton direction
            maxdelta = max(maxdelta, abs(g[k]))
            while alpha[k] - step * g[k] <= 0 && step > 1e-8
                step *= 0.5
            end
        end
        step > 1e-8 || break
        @inbounds for k in 1:K
            alpha[k] -= step * g[k]
        end
        step * maxdelta < 1e-6 && break
    end
    return alpha
end

# E[log β] and exp(E[log β]) for the K×V table of Dirichlet parameters in one pass,
# threaded over the column chunks `parts`. The row sums are accumulated serially in
# column order, so every entry is the same whatever the number of chunks; they are left
# in `rowsum` for `topic_bound`.
function topic_expectations!(Elogbeta::Matrix{Float64}, expElogbeta::Matrix{Float64},
                             lambda::Matrix{Float64}, rowsum::Vector{Float64},
                             psisum::Vector{Float64}, parts::Vector{UnitRange{Int}})
    K, V = size(lambda)
    fill!(rowsum, 0.0)
    @inbounds for v in 1:V
        @simd for k in 1:K
            rowsum[k] += lambda[k, v]
        end
    end
    @inbounds for k in 1:K
        psisum[k] = digamma(rowsum[k])
    end
    Threads.@threads for p in 1:length(parts)
        @inbounds for v in parts[p], k in 1:K
            e = digamma(lambda[k, v]) - psisum[k]
            Elogbeta[k, v] = e
            expElogbeta[k, v] = exp(e)
        end
    end
    return Elogbeta
end

# E_q[log p(β | η)] − E_q[log q(β | λ)], with `rowsum[k] = Σ_w λ_kw`. One partial sum per
# column chunk, added in chunk order: the value does not depend on task scheduling.
function topic_bound(lambda::Matrix{Float64}, Elogbeta::Matrix{Float64}, eta::Float64,
                     rowsum::Vector{Float64}, parts::Vector{UnitRange{Int}})
    K, V = size(lambda)
    partial = zeros(length(parts))
    Threads.@threads for p in 1:length(parts)
        s = 0.0
        @inbounds for v in parts[p], k in 1:K
            l = lambda[k, v]
            s += (eta - l) * Elogbeta[k, v] + loggamma(l)
        end
        partial[p] = s
    end
    b = K * (loggamma(V * eta) - V * loggamma(eta))
    for p in 1:length(parts)
        b += partial[p]
    end
    @inbounds for k in 1:K
        b -= loggamma(rowsum[k])
    end
    return b
end

topic_bound(lambda::Matrix{Float64}, Elogbeta::Matrix{Float64}, eta::Float64) =
    topic_bound(lambda, Elogbeta, eta, vec(sum(lambda; dims=2)), chunks(size(lambda, 2), 1))

# sstats[1] += sstats[2] + … + sstats[n]; every entry is summed in that order.
function merge_sstats!(sstats::Vector{Matrix{Float64}}, n::Int, parts::Vector{UnitRange{Int}})
    n > 1 || return sstats[1]
    K = size(sstats[1], 1)
    Threads.@threads for p in 1:length(parts)
        acc = sstats[1]
        @inbounds for t in 2:n
            loc = sstats[t]
            for v in parts[p]
                @simd for k in 1:K
                    acc[k, v] += loc[k, v]
                end
            end
        end
    end
    return sstats[1]
end

# M-step. Batch: λ = η + sstats (Blei et al. 2003, eq. 9). Online, with step ρ and the
# minibatch statistics scaled up to the corpus: λ = (1 − ρ) λ + ρ (η + scale · sstats)
# (Hoffman et al. 2010, eq. 9 and algorithm 2).
function update_lambda!(lambda::Matrix{Float64}, sstats::Matrix{Float64}, eta::Float64,
                        parts::Vector{UnitRange{Int}})
    K = size(lambda, 1)
    Threads.@threads for p in 1:length(parts)
        @inbounds for v in parts[p]
            @simd for k in 1:K
                lambda[k, v] = eta + sstats[k, v]
            end
        end
    end
    return lambda
end

function update_lambda!(lambda::Matrix{Float64}, sstats::Matrix{Float64}, eta::Float64,
                        parts::Vector{UnitRange{Int}}, rho::Float64, scale::Float64)
    K = size(lambda, 1)
    Threads.@threads for p in 1:length(parts)
        @inbounds for v in parts[p]
            @simd for k in 1:K
                lambda[k, v] = (1 - rho) * lambda[k, v] + rho * (eta + scale * sstats[k, v])
            end
        end
    end
    return lambda
end

function lda_vb(corpus::Corpus, K::Int; alpha=0.1, eta::Real=0.01, iters::Int=200,
                tol::Float64=1e-6, batchsize::Int=0, tau0::Real=64.0, kappa::Real=0.7,
                optimize_alpha::Bool=false, doc_maxiter::Int=100, doc_tol::Float64=1e-3,
                warm_start::Int=0, nthreads::Int=Threads.nthreads(),
                rng::AbstractRNG=Random.default_rng(),
                init::Union{Nothing,AbstractMatrix{<:Real}}=nothing, verbose::Bool=false)
    t0 = time()
    iters >= 1 || throw(ArgumentError("iters must be at least 1"))
    warm_start >= 0 || throw(ArgumentError("warm_start must be non-negative"))
    D, V = ndocs(corpus), nterms(corpus)
    N = ntokens(corpus)
    α = _alpha_vector(alpha, K)
    η = Float64(eta)
    all(>(0), α) || throw(ArgumentError("alpha must be positive"))
    η > 0 || throw(ArgumentError("eta must be positive"))
    online = 0 < batchsize < D
    lambda = if init === nothing
        [randgamma(rng, 100.0) / 100.0 for _ in 1:K, _ in 1:V]
    else
        size(init) == (K, V) || throw(DimensionMismatch("init must be K×V"))
        η .+ Float64.(init) .* (N / K)
    end
    Elogbeta = similar(lambda); expElogbeta = similar(lambda)
    T = clamp(nthreads, 1, max(1, D ÷ 8))
    cols = chunks(V, T)
    rowsum = zeros(K); psisum = zeros(K)
    sstats = [zeros(K, V) for _ in 1:T]
    sumElog = [zeros(K) for _ in 1:T]
    bounds = zeros(T)
    inner = zeros(Int, T)
    wss = [DocVB(K) for _ in 1:T]
    gamma = zeros(K, D)                      # one document per column, transposed at the end
    weights = [length(d.terms) + 1 for d in corpus.docs]
    αold = similar(α); gα = similar(α); hα = similar(α)

    # E-step over `docids`. With `warm`, every document resumes the coordinate ascent
    # from its γ of the previous pass instead of α + N/K. The ELBO stays monotone and the
    # ascent needs several times fewer iterations, but a topic that a document has dropped
    # (γ_dk ≈ α_k, weight exp(ψ(α_k)) ≈ 3e-5 at α = 0.1) never returns, whereas the cold
    # start reconsiders all K topics at every pass. On AP (K = 50) warm starts from pass 2
    # end at ELBO/token −8.88 instead of −8.27 and held-out perplexity 3546 instead of
    # 3008; from pass 21, −8.33 and 3031. Hence opt-in, and never in online mode.
    function estep!(docids::AbstractVector{Int}, warm::Bool)
        topic_expectations!(Elogbeta, expElogbeta, lambda, rowsum, psisum, cols)
        parts = balanced_chunks(view(weights, docids), T)
        Threads.@threads for t in 1:length(parts)
            ws = wss[t]; ss = sstats[t]; se = sumElog[t]
            fill!(ss, 0.0); fill!(se, 0.0)
            b = 0.0; ni = 0
            for j in parts[t]
                d = docids[j]
                doc = corpus.docs[d]
                γ = view(gamma, :, d)
                ni += if warm
                    infer_doc!(ws, doc, expElogbeta, α; maxiter=doc_maxiter, tol=doc_tol, init=γ)
                else
                    infer_doc!(ws, doc, expElogbeta, α; maxiter=doc_maxiter, tol=doc_tol)
                end
                accumulate_sstats!(ss, ws, doc)
                b += doc_bound(ws, doc, α)
                se .+= ws.Elogtheta
                copyto!(γ, ws.gamma)
            end
            bounds[t] = b; inner[t] = ni
        end
        merge_sstats!(sstats, length(parts), cols)
        for t in 2:length(parts)
            sumElog[1] .+= sumElog[t]; inner[1] += inner[t]
        end
        return sum(@view bounds[1:length(parts)])
    end

    trace = Float64[]
    updates = 0
    passes = 0
    for it in 1:iters
        passes = it
        if online
            order = randperm(rng, D)
            for lo in 1:batchsize:D
                batch = order[lo:min(lo + batchsize - 1, D)]
                estep!(batch, false)
                ρ = (tau0 + updates)^(-kappa)
                updates += 1
                scale = D / length(batch)
                update_lambda!(lambda, sstats[1], η, cols, ρ, scale)
                if optimize_alpha
                    # One Newton step on the minibatch statistics, damped by ρ like the
                    # λ update (as in Hoffman's onlineldavb.py and in gensim).
                    copyto!(αold, α)
                    optimize_alpha_newton!(α, sumElog[1], D; iters=1, scale, g=gα, h=hα)
                    @. α = (1 - ρ) * αold + ρ * α
                end
            end
        end
        # A full pass gives the ELBO; in batch mode it is also the E-step.
        docbound = estep!(1:D, !online && 0 < warm_start < it)
        elbo = (docbound + topic_bound(lambda, Elogbeta, η, rowsum, cols)) / N
        push!(trace, elbo)
        verbose && @printf("pass %4d  ELBO/token = %.5f  (%.1f inner iterations per document)\n",
                           it, elbo, inner[1] / D)
        # On convergence stop before the M-step: λ (hence φ), θ and the last trace value
        # then belong to the same pass.
        if it > 1 && abs(trace[end] - trace[end - 1]) < tol * abs(trace[end - 1])
            break
        end
        if !online
            update_lambda!(lambda, sstats[1], η, cols)
            optimize_alpha && optimize_alpha_newton!(α, sumElog[1], D; g=gα, h=hα)
        end
    end
    phi = lambda ./ sum(lambda; dims=2)
    theta = permutedims(gamma)
    theta ./= sum(theta; dims=2)
    return LDA(K, phi, theta, α, η, corpus.vocab, :vb, trace, lambda, Vector{Int32}[], passes, time() - t0)
end

"""
    transform(model, corpus; kwargs...) -> D×K matrix

Infer topic proportions for (new) documents with the topics held fixed. A model fitted
by variational Bayes is applied the way it was trained, with `exp(E[log β])` under
`q(β | λ)` rather than with `phi = E[β]`; a Gibbs model with its point estimate `phi`.
"""
function transform(m::LDA, c::Corpus; kwargs...)
    topics = m.phi
    if m.method === :vb && size(m.lambda) == size(m.phi)
        topics = dirichlet_expectation!(similar(m.lambda), m.lambda)
        topics .= exp.(topics)
    end
    return first(infer_theta(c, topics, m.alpha; kwargs...))
end

