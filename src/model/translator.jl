using Flux
using Transformers
using Transformers.Basic

struct Translator
    transformers::TransformersModel
    beam_search::BeamSearch
end

Flux.@functor Translator
Flux.trainable(translator::Translator) = (translator.transformers,)

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

