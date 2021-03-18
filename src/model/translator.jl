using Transformers
using Transformers.Basic
using Flux
using Flux: @functor, onecold

include("encoder.jl")
include("decoder.jl")
include("classifier.jl")
include("beam_search.jl")

struct Translator
    embed::AbstractEmbed
    encoder::Union{Encoder,Bert}
    decoder::Decoder
    classifier::Classifier
    beam_search::BeamSearch
end

@functor Translator

function (translator::Translator)(
    tokens::AbstractVector{T};
    vocabulary::Vocabulary{T},
    start_token::T="[CLS]",
    end_token::T="[SEP]"
)::AbstractVector{T} where T
    function prepare(tokens::AbstractVector{T})
        indices = translator.vocabulary(tokens)
        if typeof(translator.embed) <: CompositeEmbedding
            indices = (tok = indices, segment = fill(1, length(indices)))
        end
        return translator.embed(indices)
    end

    embeddings = prepare(tokens)
    encoded = translator.encoder(embeddings)

    function predict(sequence::AbstractVector{T})
        target = prepare(sequence)
        decoded = translator.decoder(target, encoded)
        return translator.classifier(decoded)[:,end]
    end

    sequence = translator.beam_search(
        translator.vocabulary.list,
        predict,
        start_token,
        end_token,
    )
    return sequence
end

function Translator(embed::AbstractEmbed, vocab_size::Int, beam_width::Int, max_length::Int, size::Int, head::Int, hs::Int, ps::Int, layer::Int; act=relu,pdrop=0.1)
    return Translator(
        embed,
        Encoder(size, head, hs, ps, layer; act=act,pdrop=pdrop),
        Decoder(size, head, hs, ps, layer; act=act,pdrop=pdrop),
        Classifier(length(vocabulary), size),
        BeamSearch(beam_width, max_length)
    )
end

