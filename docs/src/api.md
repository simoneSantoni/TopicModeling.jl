# API Reference

```@meta
CurrentModule = TopicModeling
```

## Module

```@docs
TopicModeling
```

## Model Types

```@docs
AbstractTopicModel
LDA
STM
DTM
CTM
```

## Fitting

```@docs
fit(::Type{LDA}, ::Corpus, ::Integer)
fit(::Type{STM}, ::Corpus, ::Integer)
fit(::Type{DTM}, ::Corpus, ::AbstractVector, ::Integer)
fit(::Type{CTM}, ::Corpus, ::Integer)
spectral_init
```

## Accessors and Inference

```@docs
topicword
doctopic
vocabulary
topwords
transform
estimate_effect
topic_correlations
```

## Corpora

### Types and Construction

```@docs
Document
Corpus
tokenize
prune
```

### Inspection

```@docs
ndocs
nterms
ntokens
tokens
dtm
docfreq
termfreq
```

### Input and Output

```@docs
read_ldac
write_ldac
read_uci
```

### Splitting

```@docs
train_test_split
split_documents
```

## Evaluation

```@docs
heldout_perplexity
coherence
topic_diversity
match_topics
```

## Simulation

```@docs
simulate_lda
simulate_logistic_normal
simulate_dtm
bars_topics
```

## Index

```@index
```
