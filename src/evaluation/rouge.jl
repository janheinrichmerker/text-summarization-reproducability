using TextAnalysis
using Statistics:mean
using Transformers
using Transformers.BidirectionalEncoder:WordPiece
using PreSumm.Data
using PreSumm.Model

Sample = NamedTuple{(:reference, :candidate),Tuple{String,String}}

function evaluate(
    samples::Channel{SummaryPair},
    model::Translator,
    wordpiece::WordPiece,
    tokenizer::Function,
    evaluation::Function
)
    vocabulary = Vocabulary(wordpiece)
    translated = Channel{Sample}() do channel
        for sample ∈ samples
            candidate_tokens = model(
                ["[CLS]"; wordpiece(tokenizer(sample.source)); "[SEP]"],
                vocabulary
            )
            candidate = join(candidate_tokens[2:end - 1], " ")
            reference = sample.target
            push!(channel, (reference = reference, candidate = candidate))
        end
    end
    return evaluation(translated)
end

function mean_rouge_n(samples::Channel{Sample}, n::Integer)
    return mean(
        rouge_n([sample.reference], sample.candidate, n)
        for sample ∈ samples
    )
end

function mean_rouge_n(
    samples::Channel{SummaryPair},
    model::Translator,
    wordpiece::WordPiece,
    tokenizer::Function,
    n::Integer
)
    return evaluate(
        samples,
        model,
        wordpiece,
        tokenizer,
        samples -> mean_rouge_n(samples, n)
    )
end

function mean_rouge_l(samples::Channel{Sample}, β::AbstractFloat)
    return mean(
        rouge_l_summary([sample.reference], sample.candidate, β)
        for sample ∈ samples
    )
end

function mean_rouge_l(
    samples::Channel{SummaryPair},
    model::Translator,
    wordpiece::WordPiece,
    tokenizer::Function,
    β::AbstractFloat
)
    return evaluate(
        samples,
        model,
        wordpiece,
        tokenizer,
        samples -> mean_rouge_l(samples, β)
    )
end
