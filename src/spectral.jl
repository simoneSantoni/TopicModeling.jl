# Spectral initialisation by anchor words (Arora et al. 2013), following the variant used
# as the default initialisation of the R package stm (Roberts, Stewart & Tingley 2019, §3.4):
# greedy anchor selection on the row-normalised word co-occurrence matrix, then recovery of
# the topics by a simplex-constrained least-squares fit of every word to the anchors.

# Word co-occurrence matrix Q (V×V) with the within-document diagonal correction of
# Arora et al.: E[Q] is the matrix of probabilities that two distinct tokens of a document
# are the words (i, j).
#
# The document-term matrix is ~99% sparse, so each document's outer product c cᵀ / n(n−1) is
# scattered straight into the upper triangle (a dense syrk spends its time multiplying zeros).
# Every task owns a range of columns and scans all documents for the terms in its range: no
# two tasks write the same entry and each entry is summed in document order, so the result
# does not depend on the number of threads.
function cooccurrence_gram(c::Corpus, terms::Vector{Int}; nthreads::Int=Threads.nthreads())
    V = length(terms)
    pos = zeros(Int, nterms(c)); pos[terms] = 1:V
    # CSR copy of the corpus restricted to `terms`, positions increasing within a document.
    ptr = Int[1]; idx = Int[]; val = Float64[]; scale = Float64[]
    diagcorr = zeros(V)
    colwork = zeros(V)                              # inner-loop length per column
    for doc in c.docs
        n = ntokens(doc)
        n >= 2 || continue
        div = n * (n - 1.0)
        lo = length(idx) + 1
        for (t, cnt) in zip(doc.terms, doc.counts)
            j = pos[t]
            j == 0 && continue
            push!(idx, j); push!(val, cnt)
            diagcorr[j] += cnt / div
        end
        hi = length(idx)
        hi >= lo || continue
        if !issorted(view(idx, lo:hi))
            p = sortperm(view(idx, lo:hi))
            idx[lo:hi] = idx[lo:hi][p]; val[lo:hi] = val[lo:hi][p]
        end
        for b in lo:hi
            colwork[idx[b]] += b - lo + 1
        end
        push!(ptr, hi + 1); push!(scale, 1 / div)
    end
    Q = Matrix{Float64}(undef, V, V)               # zeroed by its owners: parallel first touch
    Threads.@threads for cols in balanced_chunks(colwork .+ (1:V), nthreads)
        jlo, jhi = first(cols), last(cols)
        @inbounds for j in cols, i in 1:j
            Q[i, j] = 0.0
        end
        @inbounds for d in 1:length(scale)
            lo, hi = ptr[d], ptr[d + 1] - 1
            idx[hi] >= jlo || continue
            sc = scale[d]
            for b in searchsortedfirst(idx, jlo, lo, hi, Base.Order.Forward):hi
                jb = idx[b]
                jb > jhi && break
                vb = val[b] * sc
                for a in lo:b
                    Q[idx[a], jb] += val[a] * vb
                end
            end
        end
    end
    @inbounds for j in 1:V
        Q[j, j] -= diagcorr[j]
    end
    return _copy_upper!(Q, nthreads)
end

# Q[j, i] = Q[i, j] for j > i, in cache-sized blocks; a task owns the destination columns.
function _copy_upper!(Q::Matrix{Float64}, nthreads::Int; block::Int=64)
    V = size(Q, 1)
    Threads.@threads for cols in balanced_chunks(V:-1:1, nthreads)
        @inbounds for ilo in first(cols):block:last(cols)
            ihi = min(ilo + block - 1, last(cols))
            for jlo in ilo:block:V
                jhi = min(jlo + block - 1, V)
                for i in ilo:ihi, j in max(jlo, i + 1):jhi
                    Q[j, i] = Q[i, j]
                end
            end
        end
    end
    return Q
end

function _colsums(Q::Matrix{Float64}, nthreads::Int)
    out = zeros(size(Q, 2))
    Threads.@threads for cols in chunks(size(Q, 2), nthreads)
        for j in cols
            out[j] = sum(view(Q, :, j))
        end
    end
    return out
end

function _divide_rows!(Q::Matrix{Float64}, r::Vector{Float64}, nthreads::Int)
    Threads.@threads for cols in chunks(size(Q, 2), nthreads)
        @inbounds for j in cols
            @simd for i in axes(Q, 1)
                Q[i, j] /= r[i]
            end
        end
    end
    return Q
end

# Greedy anchor selection: repeatedly take the row of Q̄ farthest from the span of the rows
# chosen so far. The span is tracked with an orthonormal basis instead of deflating Q̄ itself.
function find_anchors(Qbar::Matrix{Float64}, K::Int, candidates::AbstractVector{Bool})
    V = size(Qbar, 1)
    resid = vec(sum(abs2, Qbar; dims=2))
    resid[.!candidates] .= -Inf
    anchors = Int[]
    basis = zeros(V, K)
    proj = zeros(V)
    for i in 1:K
        a = argmax(resid)
        push!(anchors, a)
        b = view(basis, :, i)
        b .= view(Qbar, a, :)
        for j in 1:(i - 1)
            bj = view(basis, :, j)
            b .-= dot(bj, b) .* bj
        end
        nb = norm(b)
        nb > 0 || error("could not find $K linearly independent anchor words; reduce K")
        b ./= nb
        mul!(proj, Qbar, b)
        @inbounds for w in 1:V
            resid[w] -= proj[w]^2
        end
        resid[anchors] .= -Inf
    end
    return anchors
end

# min_c ‖y − Xᵀc‖² over the simplex by exponentiated gradient with backtracking, written in
# terms of G = XXᵀ (K×K) and h = Xy (K), so each iteration is O(K²).
function simplex_ls!(c::Vector{Float64}, G::Matrix{Float64}, h::AbstractVector{Float64},
                     grad::Vector{Float64}, cnew::Vector{Float64}, Gc::Vector{Float64};
                     maxiter::Int=500, tol::Float64=1e-7)
    K = length(c)
    fill!(c, 1 / K)
    mul!(Gc, G, c)
    f = dot(c, Gc) - 2 * dot(c, h)
    step = 1.0
    for _ in 1:maxiter
        @inbounds for k in 1:K
            grad[k] = 2 * (Gc[k] - h[k])
        end
        gmin = minimum(grad)
        # Duality gap of the linearised problem: zero at the optimum.
        gap = dot(c, grad) - gmin
        gap < tol && break
        improved = false
        for _ in 1:30
            s = 0.0
            @inbounds for k in 1:K
                cnew[k] = c[k] * exp(-step * (grad[k] - gmin))
                s += cnew[k]
            end
            cnew ./= s
            mul!(Gc, G, cnew)
            fnew = dot(cnew, Gc) - 2 * dot(cnew, h)
            if fnew <= f
                f = fnew
                copyto!(c, cnew)
                step *= 2.0
                improved = true
                break
            end
            step *= 0.5
        end
        if !improved
            mul!(Gc, G, c)
            break
        end
    end
    return c
end

"""
    spectral_init(corpus, K; max_terms=10_000, anchor_min_df=..., nthreads) -> (phi, anchors)

Deterministic anchor-word initialisation: `phi` is the K×V topic matrix and `anchors` the
vocabulary indices of the K anchor words (topic `k` is built around `anchors[k]`). Only the
`max_terms` most frequent terms enter the co-occurrence matrix; the result does not depend on
`nthreads`.

`anchor_min_df` is the minimum document frequency for a word to be eligible as an anchor.
Rare words have noisy co-occurrence rows that look extreme to the greedy selector, so several
anchors can land in one topic and EM then ends in a poor optimum. The default
`max(10, D / 4K)` is a heuristic: the expected document share of a topic is 1/K, and in
simulations the quality of the initialisation rose steadily with the anchors' document
frequency. Lower it if you expect very rare topics.
"""
function spectral_init(c::Corpus, K::Int; max_terms::Int=10_000,
                       anchor_min_df::Int=max(10, ceil(Int, ndocs(c) / 4K)),
                       nthreads::Int=Threads.nthreads())
    V = nterms(c)
    tf = termfreq(c)
    terms = findall(>(0), tf)
    if length(terms) > max_terms
        terms = sort!(sort(terms; by=t -> -tf[t])[1:max_terms])
    end
    Q = cooccurrence_gram(c, terms; nthreads)
    rowsums = _colsums(Q, nthreads)                 # Q is symmetric
    ok = findall(>(0), rowsums)
    if length(ok) < length(terms)
        terms = terms[ok]; Q = Q[ok, ok]; rowsums = rowsums[ok]
    end
    Vs = length(terms)
    Vs >= K || error("vocabulary too small for $K topics")
    _divide_rows!(Q, rowsums, nthreads)             # row-normalise: Q̄
    df = docfreq(c)[terms]
    candidates = df .>= anchor_min_df
    count(candidates) >= K || (candidates = trues(Vs))
    anchors = find_anchors(Q, K, candidates)

    X = Q[anchors, :]                               # K × Vs
    G = X * X'
    H = X * Q'                                      # K × Vs, column w = X q̄_w
    wprob = rowsums ./ sum(rowsums)
    A = zeros(K, Vs)
    Threads.@threads for part in chunks(Vs, nthreads)
        cw = zeros(K); grad = zeros(K); cnew = zeros(K); Gc = zeros(K)
        for w in part
            simplex_ls!(cw, G, view(H, :, w), grad, cnew, Gc)
            @inbounds for k in 1:K
                A[k, w] = cw[k] * wprob[w]
            end
        end
    end
    for (k, a) in enumerate(anchors)               # an anchor belongs to its own topic
        A[:, a] .= 0.0
        A[k, a] = wprob[a]
    end
    phi = fill(0.0, K, V)
    phi[:, terms] = A
    # Words outside the spectral vocabulary get a small uniform mass so no term has zero
    # probability under every topic.
    phi .= max.(phi, 1e-12)
    phi[:, setdiff(1:V, terms)] .= 1e-6 / V
    normalize_rows!(phi)
    return phi, terms[anchors]
end
