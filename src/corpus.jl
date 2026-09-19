# Bag-of-words corpus representation and I/O.

"""
    Document(terms, counts)

Sparse bag of words: `terms` are 1-based vocabulary indices (strictly increasing, hence sorted
and unique) and `counts[i] > 0` is the number of occurrences of `terms[i]`. Anything else
throws an `ArgumentError`: the inference kernels index the topics with these ids unchecked,
so they are validated once, here and in [`Corpus`](@ref). Use `Document(tokens)` to build a
document from an unsorted token sequence.
"""
struct Document
    terms::Vector{Int32}
    counts::Vector{Int32}
    function Document(terms::Vector{Int32}, counts::Vector{Int32})
        length(terms) == length(counts) ||
            throw(DimensionMismatch("terms and counts must have equal length"))
        prev = Int32(0)
        @inbounds for i in eachindex(terms)
            (terms[i] > prev && counts[i] > 0) || _invalid_document(terms[i], counts[i], prev)
            prev = terms[i]
        end
        return new(terms, counts)
    end
end

@noinline function _invalid_document(t, n, prev)
    t < 1 && throw(ArgumentError("term ids must be positive, got $t"))
    t > prev || throw(ArgumentError("terms must be sorted and unique (term $t follows term $prev); " *
                                    "use Document(tokens) for a raw token sequence"))
    throw(ArgumentError("counts must be positive, got $n for term $t"))
end

Document(terms::AbstractVector{<:Integer}, counts::AbstractVector{<:Integer}) =
    Document(convert(Vector{Int32}, terms), convert(Vector{Int32}, counts))

# Run-length encode sorted term ids. `toks` is sorted in place.
function _tally!(toks::Vector{Int32})
    sort!(toks)
    terms = Int32[]; counts = Int32[]
    @inbounds for t in toks
        if !isempty(terms) && terms[end] == t
            counts[end] += Int32(1)
        else
            push!(terms, t); push!(counts, Int32(1))
        end
    end
    return Document(terms, counts)
end

"Build a `Document` from a sequence of token ids (repeats allowed, any order)."
Document(tokens::AbstractVector{<:Integer}) = _tally!(collect(Int32, tokens))

"""
    ntokens(document) -> Int
    ntokens(corpus) -> Int

Number of tokens (word occurrences, counting repeats) in a document or in the whole corpus.

# Examples
```jldoctest
julia> c = Corpus([["a", "b", "a"], ["b", "c"]]);

julia> ntokens(c), ntokens(c[1])
(5, 3)
```
"""
ntokens(d::Document) = Int(sum(d.counts))
Base.isempty(d::Document) = isempty(d.terms)

"Expand a document into its token sequence (term ids repeated by count)."
function tokens(d::Document)
    out = Vector{Int32}(undef, ntokens(d))
    i = 0
    @inbounds for (t, c) in zip(d.terms, d.counts), _ in 1:c
        out[i += 1] = t
    end
    return out
end

"""
    Corpus(docs, vocab)

A collection of [`Document`](@ref)s over a shared vocabulary.

Other constructors:

- `Corpus(texts::Vector{String}; kwargs...)` tokenises raw strings.
- `Corpus(tokenized::Vector{Vector{String}}; min_df, max_df, stopwords)`.
- `Corpus(counts::AbstractMatrix, vocab)` from a documents × terms count matrix.

`corpus[i]` is the `i`-th document and `corpus[idx]` the sub-corpus of the documents `idx`
(same vocabulary).

Every term id must lie in `1:length(vocab)`; construction throws an `ArgumentError` otherwise
(the inference kernels index the topics with these ids unchecked).

# Examples
```jldoctest
julia> c = Corpus(["Topic models find topics.", "Models of text."]; minlength=4)
Corpus(2 documents, 5 terms, 6 tokens)

julia> c.vocab
5-element Vector{String}:
 "find"
 "models"
 "text"
 "topic"
 "topics"

julia> Corpus([2 0 1; 0 3 0], ["x", "y", "z"])
Corpus(2 documents, 3 terms, 6 tokens)
```
"""
struct Corpus
    docs::Vector{Document}
    vocab::Vector{String}
    function Corpus(docs::Vector{Document}, vocab::Vector{String})
        V = length(vocab)
        # Terms are sorted and positive (see `Document`), so only the last one can exceed V.
        for (d, doc) in enumerate(docs)
            (isempty(doc.terms) || doc.terms[end] <= V) ||
                throw(ArgumentError("document $d refers to term $(doc.terms[end]) but the vocabulary has $V terms"))
        end
        return new(docs, vocab)
    end
    # Internal: `docs` are known to be valid for `vocab` (a subset of a valid corpus).
    Corpus(docs::Vector{Document}, vocab::Vector{String}, ::Nothing) = new(docs, vocab)
end

Corpus(docs::AbstractVector{<:Document}, vocab::AbstractVector{<:AbstractString}) =
    Corpus(convert(Vector{Document}, docs), convert(Vector{String}, vocab))

"""
    ndocs(corpus) -> Int

Number of documents `D`. `length(corpus)` is the same.
"""
ndocs(c::Corpus) = length(c.docs)

"""
    nterms(corpus) -> Int

Size `V` of the vocabulary.
"""
nterms(c::Corpus) = length(c.vocab)
ntokens(c::Corpus) = sum(ntokens, c.docs; init=0)
Base.length(c::Corpus) = ndocs(c)
Base.firstindex(c::Corpus) = 1
Base.lastindex(c::Corpus) = ndocs(c)
Base.getindex(c::Corpus, i::Integer) = c.docs[i]
Base.getindex(c::Corpus, idx::AbstractVector) = Corpus(c.docs[idx], c.vocab, nothing)

function Base.show(io::IO, c::Corpus)
    print(io, "Corpus(", ndocs(c), " documents, ", nterms(c), " terms, ", ntokens(c), " tokens)")
end

const DEFAULT_TOKEN_PATTERN = r"[\p{L}][\p{L}\p{N}_'-]*[\p{L}\p{N}]|[\p{L}]"

"""
    tokenize(text; pattern, lowercase=true, minlength=2)

Minimal regex tokenizer. Use your own pipeline for anything language-specific and
pass the result to `Corpus(::Vector{Vector{String}})`.

# Examples
```jldoctest
julia> tokenize("The state-of-the-art, in 2024!")
3-element Vector{String}:
 "the"
 "state-of-the-art"
 "in"
```
"""
function tokenize(text::AbstractString; pattern::Regex=DEFAULT_TOKEN_PATTERN,
                  lowercase::Bool=true, minlength::Int=2)
    s = lowercase ? Base.lowercase(text) : text
    return String[m.match for m in eachmatch(pattern, s) if length(m.match) >= minlength]
end

function Corpus(texts::AbstractVector{<:AbstractString}; pattern::Regex=DEFAULT_TOKEN_PATTERN,
                lowercase::Bool=true, minlength::Int=2, kwargs...)
    tokenized = [tokenize(t; pattern, lowercase, minlength) for t in texts]
    return Corpus(tokenized; kwargs...)
end

"""
    Corpus(tokenized; min_df=1, max_df=1.0, stopwords=(), keep_empty=true)

Build a corpus from pre-tokenised documents. Terms appearing in fewer than `min_df`
documents, or in more than a `max_df` share of them, are dropped. Documents left
empty are kept by default so that document indices stay aligned with metadata.
"""
function Corpus(tokenized::AbstractVector{<:AbstractVector{<:AbstractString}};
                min_df::Int=1, max_df::Real=1.0, stopwords=(), keep_empty::Bool=true)
    D = length(tokenized)
    stop = Set{String}(stopwords)
    df = Dict{String,Int}()
    for doc in tokenized, w in Set(doc)
        df[w] = get(df, w, 0) + 1
    end
    maxdocs = max_df * D
    vocab = sort!([w for (w, n) in df if n >= min_df && n <= maxdocs && !(w in stop)])
    index = Dict{String,Int32}(w => Int32(i) for (i, w) in enumerate(vocab))
    docs = Document[]
    for doc in tokenized
        ids = Int32[index[w] for w in doc if haskey(index, w)]
        d = Document(ids)
        (keep_empty || !isempty(d)) && push!(docs, d)
    end
    return Corpus(docs, vocab)
end

function Corpus(counts::SparseMatrixCSC{<:Integer}, vocab::AbstractVector{<:AbstractString}=String[])
    D, V = size(counts)
    vocab = isempty(vocab) ? ["w$i" for i in 1:V] : collect(String, vocab)
    length(vocab) == V || throw(DimensionMismatch("vocab length must match number of columns"))
    ct = sparse(counts')                      # V×D, so each column is one document
    rows, vals = rowvals(ct), nonzeros(ct)
    docs = Vector{Document}(undef, D)
    for d in 1:D
        r = nzrange(ct, d)
        keep = [i for i in r if vals[i] > 0]
        docs[d] = Document(Int32[rows[i] for i in keep], Int32[vals[i] for i in keep])
    end
    return Corpus(docs, vocab)
end

Corpus(counts::AbstractMatrix{<:Integer}, vocab::AbstractVector{<:AbstractString}=String[]) =
    Corpus(sparse(counts), vocab)

"""
    dtm(corpus) -> SparseMatrixCSC{Int}

Documents × terms count matrix.
"""
function dtm(c::Corpus)
    I = Int[]; J = Int[]; X = Int[]
    for (d, doc) in enumerate(c.docs), (t, n) in zip(doc.terms, doc.counts)
        push!(I, d); push!(J, t); push!(X, n)
    end
    return sparse(I, J, X, ndocs(c), nterms(c))
end

"Number of documents each term occurs in."
function docfreq(c::Corpus)
    df = zeros(Int, nterms(c))
    for doc in c.docs, t in doc.terms
        df[t] += 1
    end
    return df
end

"Total occurrences of each term."
function termfreq(c::Corpus)
    tf = zeros(Int, nterms(c))
    for doc in c.docs, (t, n) in zip(doc.terms, doc.counts)
        tf[t] += n
    end
    return tf
end

"""
    prune(corpus; min_df=1, max_df=1.0, max_terms=typemax(Int)) -> (corpus, kept)

Drop rare/ubiquitous terms and re-index the vocabulary. `kept` holds the old
indices of the surviving terms. `max_terms` keeps the most frequent terms.
"""
function prune(c::Corpus; min_df::Int=1, max_df::Real=1.0, max_terms::Int=typemax(Int))
    df = docfreq(c)
    limit = max_df * ndocs(c)
    kept = [t for t in 1:nterms(c) if df[t] >= min_df && df[t] <= limit]
    if length(kept) > max_terms
        tf = termfreq(c)
        kept = sort!(sort(kept; by=t -> -tf[t])[1:max_terms])
    end
    remap = zeros(Int32, nterms(c))
    remap[kept] = 1:length(kept)
    docs = map(c.docs) do doc
        sel = [i for i in eachindex(doc.terms) if remap[doc.terms[i]] != 0]
        Document(remap[doc.terms[sel]], doc.counts[sel])
    end
    return Corpus(docs, c.vocab[kept]), kept
end

# --- I/O -------------------------------------------------------------------------------

# Sort (term, count) pairs by term and merge repeated terms. Counts must be positive.
function _document_from_pairs(terms::Vector{Int32}, counts::Vector{Int32})
    issorted(terms; lt=<=) && return Document(terms, counts)      # strictly increasing already
    p = sortperm(terms)
    ts = Int32[]; ns = Int32[]
    for i in p
        if !isempty(ts) && ts[end] == terms[i]
            ns[end] += counts[i]
        else
            push!(ts, terms[i]); push!(ns, counts[i])
        end
    end
    return Document(ts, ns)
end

# One term per line; blank lines at the end of the file are not terms.
function _read_vocab(path::AbstractString)
    words = readlines(path)
    while !isempty(words) && isempty(strip(words[end]))
        pop!(words)
    end
    return words
end

"""
    read_ldac(path; vocab=nothing) -> Corpus

Read Blei's LDA-C format (`N term:count term:count ...`, 0-based term ids), the
format used by lda-c, ctm-c and dtm. `vocab` is an optional path to a file with one
term per line (blank lines at its end are ignored).

Every line is a document: a blank line (or `0`) is an empty document, so that documents stay
aligned with their metadata; only blank lines at the very end of the file are dropped. A term
repeated on a line has its counts added and pairs with a zero count are dropped. Malformed
input throws an `ArgumentError` naming the line: a pair that is not `term:count`, a negative
id or count, or a declared `N` different from the number of pairs on the line.

Without `vocab` the terms are named `w1`, `w2`, … up to the largest id in the file, so a
[`write_ldac`](@ref)/`read_ldac` round trip has a smaller vocabulary when the last terms of the
original one are unused; pass the vocabulary file to keep `nterms` unchanged.
"""
function read_ldac(path::AbstractString; vocab::Union{Nothing,AbstractString}=nothing)
    docs = Document[]
    maxterm = 0
    lastdoc = 0                                     # docs[1:lastdoc] ends with a non-blank line
    for (lineno, line) in enumerate(eachline(path))
        fields = split(line)
        bad(msg) = throw(ArgumentError("$path, line $lineno: $msg"))
        terms = Int32[]; counts = Int32[]
        if !isempty(fields)
            N = tryparse(Int, fields[1])
            N === nothing && bad("expected the number of distinct terms, got \"$(fields[1])\"")
            N == length(fields) - 1 || bad("declares $N terms but has $(length(fields) - 1) term:count pairs")
            for f in @view fields[2:end]
                i = findfirst(':', f)
                t = i === nothing ? nothing : tryparse(Int32, SubString(f, 1, prevind(f, i)))
                n = i === nothing ? nothing : tryparse(Int32, SubString(f, nextind(f, i)))
                (t === nothing || n === nothing) && bad("malformed term:count pair \"$f\"")
                (t < 0 || t == typemax(Int32)) && bad("term id $t out of range (ids are 0-based)")
                n < 0 && bad("negative count in \"$f\"")
                n == 0 && continue
                push!(terms, t + Int32(1)); push!(counts, n)
            end
            lastdoc = lineno
        end
        doc = _document_from_pairs(terms, counts)
        push!(docs, doc)
        isempty(doc) || (maxterm = max(maxterm, Int(doc.terms[end])))
    end
    resize!(docs, lastdoc)
    words = vocab === nothing ? ["w$i" for i in 1:maxterm] : _read_vocab(vocab)
    length(words) >= maxterm ||
        throw(ArgumentError("$vocab has $(length(words)) terms but $path refers to term $maxterm (0-based id $(maxterm - 1))"))
    return Corpus(docs, words)
end

"""
    write_ldac(path, corpus; vocab=nothing) -> path

Write a corpus in LDA-C format; optionally write the vocabulary to the file `vocab`. The
LDA-C file does not record the vocabulary size: read it back with the vocabulary file (see
[`read_ldac`](@ref)).
"""
function write_ldac(path::AbstractString, c::Corpus; vocab::Union{Nothing,AbstractString}=nothing)
    open(path, "w") do io
        for doc in c.docs
            print(io, length(doc.terms))
            for (t, n) in zip(doc.terms, doc.counts)
                print(io, ' ', t - 1, ':', n)
            end
            println(io)
        end
    end
    vocab === nothing || open(vocab, "w") do io
        foreach(w -> println(io, w), c.vocab)
    end
    return path
end

"""
    read_uci(docword; vocab=nothing) -> Corpus

Read the UCI "Bag of Words" format (`D`, `W`, `NNZ` header lines, then
`docID wordID count` triples, 1-based). `vocab` is an optional path to a file with one term
per line; it must have exactly `W` terms.

Repeated `(docID, wordID)` triples have their counts added and zero counts are dropped. An
`ArgumentError` naming the line is thrown for a malformed header or triple, a `docID` outside
`1:D`, a `wordID` outside `1:W`, a negative count, or a number of triples different from `NNZ`.
"""
function read_uci(docword::AbstractString; vocab::Union{Nothing,AbstractString}=nothing)
    open(docword) do io
        header = map(("D", "W", "NNZ")) do name
            v = tryparse(Int, strip(readline(io)))
            (v === nothing || v < 0) && throw(ArgumentError("$docword: the first three lines must be the counts D, W and NNZ; could not read $name"))
            v
        end
        D, V, NNZ = header
        V <= typemax(Int32) || throw(ArgumentError("$docword: W = $V is too large"))
        terms = [Int32[] for _ in 1:D]; counts = [Int32[] for _ in 1:D]
        nnz = 0
        for (i, line) in enumerate(eachline(io))
            f = split(line)
            isempty(f) && continue
            bad(msg) = throw(ArgumentError("$docword, line $(i + 3): $msg"))
            vals = length(f) == 3 ? map(x -> tryparse(Int, x), f) : nothing
            (vals === nothing || any(isnothing, vals)) && bad("expected \"docID wordID count\", got \"$line\"")
            d, w, n = vals
            1 <= d <= D || bad("docID $d outside 1:$D")
            1 <= w <= V || bad("wordID $w outside 1:$V")
            0 <= n <= typemax(Int32) || bad("count $n out of range")
            nnz += 1
            n == 0 && continue
            push!(terms[d], Int32(w)); push!(counts[d], Int32(n))
        end
        nnz == NNZ || throw(ArgumentError("$docword: header declares NNZ = $NNZ triples but the file has $nnz"))
        docs = Document[_document_from_pairs(terms[d], counts[d]) for d in 1:D]
        words = vocab === nothing ? ["w$i" for i in 1:V] : _read_vocab(vocab)
        length(words) == V ||
            throw(ArgumentError("$vocab has $(length(words)) terms but $docword declares W = $V"))
        return Corpus(docs, words)
    end
end

# --- splitting -------------------------------------------------------------------------

"""
    train_test_split(corpus; test=0.1, rng) -> (train, test, train_idx, test_idx)

Hold out a random share of documents. Both parts have at least one document, so the corpus
needs at least two.
"""
function train_test_split(c::Corpus; test::Real=0.1, rng::AbstractRNG=Random.default_rng())
    D = ndocs(c)
    D >= 2 || throw(ArgumentError("need at least two documents to split, got $D"))
    0 <= test <= 1 || throw(ArgumentError("test must be a share in [0, 1], got $test"))
    perm = randperm(rng, D)
    ntest = clamp(round(Int, test * D), 1, D - 1)
    test_idx = sort!(perm[1:ntest]); train_idx = sort!(perm[(ntest + 1):end])
    return c[train_idx], c[test_idx], train_idx, test_idx
end

"""
    split_documents(corpus; frac=0.5, rng) -> (observed, heldout)

Split the *tokens* of every document into two corpora, for document-completion
evaluation: estimate topic proportions on `observed`, score `heldout`. A share `frac` of each
document's tokens (at least one, and at most all but one) is observed; the split is a
deterministic function of `rng`.
"""
function split_documents(c::Corpus; frac::Real=0.5, rng::AbstractRNG=Random.default_rng())
    0 <= frac <= 1 || throw(ArgumentError("frac must be a share in [0, 1], got $frac"))
    obs = Vector{Document}(undef, ndocs(c)); held = similar(obs)
    for (i, doc) in enumerate(c.docs)
        toks = shuffle!(rng, tokens(doc))
        n = length(toks)
        cut = n <= 1 ? n : clamp(round(Int, frac * n), 1, n - 1)
        obs[i] = _tally!(toks[1:cut]); held[i] = _tally!(toks[(cut + 1):end])
    end
    return Corpus(obs, c.vocab, nothing), Corpus(held, c.vocab, nothing)
end
