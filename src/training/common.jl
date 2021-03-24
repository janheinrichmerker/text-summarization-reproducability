using Transformers.Basic
using Transformers.BidirectionalEncoder:WordPiece
using PreSumm.Model

include("loss.jl")

function preprocess(text::String, wordpiece::WordPiece, tokenizer::Function)::AbstractVector{String}
    tokens = text |> tokenizer |> wordpiece
    max_length = min(4096, length(tokens)) # Truncate to 4096 tokens
    tokens = tokens[1:max_length]
    return ["[CLS]"; tokens; "[SEP]"]
end

label_smoothing_α = 0.0 # Label smoothing doesn't work yet.
function loss(
    inputs::AbstractVector{String},
    outputs::AbstractVector{String},
    ground_truth::AbstractMatrix{<:Number},
    model::Translator,
    vocabulary::Vocabulary,
)::AbstractFloat
    prediction = model.transformers(vocabulary, inputs, outputs)
    loss = logtranslationloss(prediction, ground_truth, α=label_smoothing_α)
    return loss
end
