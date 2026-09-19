# Shared helpers for the benchmark scripts.
using TopicModeling, Random, Statistics, Printf

const DATA = joinpath(@__DIR__, "data")
const RESULTS = joinpath(@__DIR__, "results")

load(name) = read_ldac(joinpath(DATA, "$name.ldac"); vocab=joinpath(DATA, "$(split(name, '.')[1]).vocab"))

"Read a K×V topic matrix written row-major as little-endian Float64 (numpy / R `writeBin` of t(phi))."
function read_phi(prefix)
    meta = read(prefix * ".json", String)
    K = parse(Int, match(r"\"K\":\s*(\d+)", meta)[1]); V = parse(Int, match(r"\"V\":\s*(\d+)", meta)[1])
    raw = Vector{Float64}(undef, K * V); read!(prefix * ".phi.bin", raw)
    return permutedims(reshape(raw, V, K)), meta
end

metafield(meta, key) = (m = match(Regex("\"$key\":\\s*([-0-9.eE+]+)"), meta); m === nothing ? NaN : parse(Float64, m[1]))
function metaalpha(meta, K, default)
    m = match(r"\"alpha\":\s*\[([^\]]*)\]", meta)
    m === nothing && return fill(default, K)
    a = parse.(Float64, split(m[1], ','))
    return length(a) == K ? a : fill(a[1], K)
end

function write_phi(prefix, phi; kwargs...)
    write(prefix * ".phi.bin", permutedims(phi))
    open(prefix * ".json", "w") do io
        pairs = ["\"K\": $(size(phi, 1))", "\"V\": $(size(phi, 2))"]
        for (k, v) in kwargs
            push!(pairs, v isa AbstractVector ? "\"$k\": [$(join(v, ", "))]" : v isa AbstractString ? "\"$k\": \"$v\"" : "\"$k\": $v")
        end
        print(io, "{", join(pairs, ", "), "}")
    end
end

"""
Score a topic matrix with the evaluator common to all implementations. The vocabulary is
the set of terms seen in training: implementations differ in whether unseen terms get
smoothing mass or no column at all, so φ is renormalised over the seen terms and test
tokens of unseen terms are dropped — for every implementation alike.
"""
function score(phi, alpha, train, test; seeds=1:5)
    seen = findall(>(0), docfreq(train))
    remap = zeros(Int32, nterms(train)); remap[seen] = 1:length(seen)
    restrict(c) = Corpus([(keep = findall(t -> remap[t] != 0, d.terms); Document(remap[d.terms[keep]], d.counts[keep]))
                          for d in c.docs], c.vocab[seen])
    φ = phi[:, seen]; φ ./= sum(φ; dims=2)
    tr, te = restrict(train), restrict(test)
    pp = [first(heldout_perplexity(φ, alpha, te; rng=Xoshiro(s))) for s in seeds]
    return (perplexity=mean(pp), perplexity_sd=std(pp), npmi=mean(coherence(φ, tr; n=10)),
            umass=mean(coherence(φ, tr; n=10, measure=:umass)), diversity=topic_diversity(φ; n=25))
end

topword_sets(phi; n=10) = [Set(partialsortperm(view(phi, k, :), 1:n; rev=true)) for k in axes(phi, 1)]
