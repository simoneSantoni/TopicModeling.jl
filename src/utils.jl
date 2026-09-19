# Small numerical helpers shared by all models.

"""
    logsumexp(x)

Numerically stable `log(sum(exp, x))`.
"""
function logsumexp(x::AbstractVector{<:Real})
    m = maximum(x)
    isfinite(m) || return float(m)
    s = 0.0
    @inbounds @simd for i in eachindex(x)
        s += exp(x[i] - m)
    end
    return m + log(s)
end

"""
    softmax!(out, x)

Write `exp.(x) ./ sum(exp, x)` into `out` without overflow.
"""
function softmax!(out::AbstractVector{Float64}, x::AbstractVector{<:Real})
    m = maximum(x)
    s = 0.0
    @inbounds for i in eachindex(x)
        e = exp(x[i] - m)
        out[i] = e
        s += e
    end
    out ./= s
    return out
end

softmax(x::AbstractVector{<:Real}) = softmax!(Vector{Float64}(undef, length(x)), x)

"""
    dirichlet_expectation!(out, a)

`E[log θ]` for `θ ~ Dirichlet(a)`, i.e. `ψ(a) .- ψ(sum(a))`.
"""
function dirichlet_expectation!(out::AbstractVector{Float64}, a::AbstractVector{Float64})
    ψ0 = digamma(sum(a))
    @inbounds for i in eachindex(a)
        out[i] = digamma(a[i]) - ψ0
    end
    return out
end

# Row-wise version for a K×V matrix of Dirichlet parameters (one topic per row).
function dirichlet_expectation!(out::AbstractMatrix{Float64}, a::AbstractMatrix{Float64})
    K = size(a, 1)
    rowsum = vec(sum(a; dims=2))
    @inbounds for k in 1:K
        rowsum[k] = digamma(rowsum[k])
    end
    @inbounds for w in axes(a, 2), k in 1:K
        out[k, w] = digamma(a[k, w]) - rowsum[k]
    end
    return out
end

function normalize_rows!(A::AbstractMatrix{Float64})
    @inbounds for i in axes(A, 1)
        s = 0.0
        for j in axes(A, 2)
            s += A[i, j]
        end
        s > 0 || continue
        for j in axes(A, 2)
            A[i, j] /= s
        end
    end
    return A
end

# Split `1:n` into at most `parts` contiguous ranges of near-equal size.
function chunks(n::Int, parts::Int)
    parts = clamp(parts, 1, max(n, 1))
    q, r = divrem(n, parts)
    out = Vector{UnitRange{Int}}(undef, parts)
    lo = 1
    for p in 1:parts
        hi = lo + q - 1 + (p <= r ? 1 : 0)
        out[p] = lo:hi
        lo = hi + 1
    end
    return out
end

# Split `1:n` into ranges holding roughly equal total `weight`, so that threads
# working on documents of very different lengths finish at about the same time.
function balanced_chunks(weights::AbstractVector{<:Real}, parts::Int)
    n = length(weights)
    parts = clamp(parts, 1, max(n, 1))
    target = sum(weights) / parts
    out = UnitRange{Int}[]
    lo = 1
    acc = 0.0
    for i in 1:n
        acc += weights[i]
        remaining_parts = parts - length(out) - 1
        if (acc >= target && remaining_parts > 0 && n - i >= remaining_parts) || i == n
            push!(out, lo:i)
            lo = i + 1
            acc = 0.0
        end
    end
    return out
end

# Run `f(item, p)` for every item with `nworkers` tasks; `p ∈ 1:nworkers` identifies the task,
# for task-owned workspaces. Tasks pull the next item when they finish one, so put expensive
# items first. Which task runs an item is not deterministic: `f` must write only to storage
# owned by the item, and its result must not depend on the history of workspace `p`.
function _foreach_pooled(f, items::AbstractVector, nworkers::Int)
    nworkers = min(nworkers, length(items))
    if nworkers <= 1
        foreach(i -> f(i, 1), items)
        return nothing
    end
    next = Threads.Atomic{Int}(0)
    @sync for p in 1:nworkers
        Threads.@spawn while true
            j = Threads.atomic_add!(next, 1) + 1      # atomic_add! returns the old value
            j > length(items) && break
            f(items[j], p)
        end
    end
    return nothing
end

# `Corpus` guarantees term ids in `1:nterms(c)`, so this check is what makes the `@inbounds`
# gathers of topic columns safe for corpora that were not used in training.
function check_vocabulary(c, topics::AbstractMatrix)
    nterms(c) == size(topics, 2) ||
        throw(DimensionMismatch("the corpus has $(nterms(c)) terms but the topics have $(size(topics, 2))"))
    return nothing
end

"""
    hungarian(cost) -> assignment

Minimum-cost assignment for a square cost matrix. `assignment[i]` is the column
matched to row `i`. O(n³) shortest augmenting path formulation.
"""
function hungarian(cost::AbstractMatrix{<:Real})
    n = size(cost, 1)
    n == size(cost, 2) || throw(DimensionMismatch("cost matrix must be square"))
    u = zeros(n + 1)
    v = zeros(n + 1)
    p = zeros(Int, n + 1)
    way = zeros(Int, n + 1)
    for i in 1:n
        p[1] = i
        j0 = 1
        minv = fill(Inf, n + 1)
        used = falses(n + 1)
        while true
            used[j0] = true
            i0 = p[j0]
            delta = Inf
            j1 = 1
            for j in 2:(n + 1)
                used[j] && continue
                cur = cost[i0, j - 1] - u[i0 + 1] - v[j]
                if cur < minv[j]
                    minv[j] = cur
                    way[j] = j0
                end
                if minv[j] < delta
                    delta = minv[j]
                    j1 = j
                end
            end
            for j in 1:(n + 1)
                if used[j]
                    u[p[j] + 1] += delta
                    v[j] -= delta
                else
                    minv[j] -= delta
                end
            end
            j0 = j1
            p[j0] == 0 && break
        end
        while true
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            j0 == 1 && break
        end
    end
    assignment = zeros(Int, n)
    for j in 2:(n + 1)
        assignment[p[j]] = j - 1
    end
    return assignment
end

"""
    randgamma(rng, a)

Draw from Gamma(shape=a, scale=1) (Marsaglia & Tsang 2000).
"""
function randgamma(rng::AbstractRNG, a::Real)
    a > 0 || throw(DomainError(a, "shape must be positive"))
    a < 1 && return randgamma(rng, a + 1) * rand(rng)^(1 / a)
    d = a - 1 / 3
    c = 1 / sqrt(9d)
    while true
        x = randn(rng)
        v = 1 + c * x
        v <= 0 && continue
        v = v^3
        u = rand(rng)
        if u < 1 - 0.0331 * x^4 || log(u) < 0.5 * x^2 + d * (1 - v + log(v))
            return d * v
        end
    end
end

"Draw from a Dirichlet distribution with parameter vector `a`."
function randdirichlet!(rng::AbstractRNG, out::AbstractVector{Float64}, a::AbstractVector{<:Real})
    @inbounds for i in eachindex(a)
        out[i] = max(randgamma(rng, a[i]), 1e-300)
    end
    out ./= sum(out)
    return out
end

randdirichlet(rng::AbstractRNG, a::AbstractVector{<:Real}) =
    randdirichlet!(rng, Vector{Float64}(undef, length(a)), a)

# Draw an index from unnormalised weights.
function randcat(rng::AbstractRNG, p::AbstractVector{Float64})
    u = rand(rng) * sum(p)
    acc = 0.0
    @inbounds for i in eachindex(p)
        acc += p[i]
        acc >= u && return i
    end
    return lastindex(p)
end
