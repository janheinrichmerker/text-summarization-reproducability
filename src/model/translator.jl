using Transformers
using Transformers.Basic
using Flux
using Flux: @functor

include("transformers.jl")
include("beam_search.jl")

struct Translator
    transformers::TransformersModel
    beam_search::BeamSearch
end

@functor Translator

function (translator::Translator)(
    inputs::AbstractVector{T},
    vocabulary::Vocabulary{T};
    start_token::T="[CLS]",
    end_token::T="[SEP]"
)::AbstractVector{T} where T
    predict(outputs) = translator.transformers(vocabulary, inputs, outputs)
    return translator.beam_search(
        vocabulary,
        predict,
        start_token=start_token,
        end_token=end_token,
    )
end

function Translator(
    transformers::TransformersModel,
    beam_width::Integer,
    max_length::Integer;
    length_normalization::AbstractFloat=0.0
)
    return Translator(
        transformers,
        BeamSearch(beam_width, max_length, length_normalization)
    )
end

