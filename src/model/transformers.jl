using Transformers
using Transformers.Basic
using Transformers.Basic:AbstractEmbed
using Flux
using Flux: @functor

include("encoder.jl")
include("decoder.jl")
include("generator.jl")

struct TransformersModel
    embed::AbstractEmbed
    encoder::Union{Encoder,Bert}
    decoder::Decoder
    generator::Generator
end

@functor TransformersModel

function _embed(
    transformers::TransformersModel,
    vocabulary::Vocabulary{T},
    sequence::AbstractVector{T}
) where T
    indices = vocabulary(sequence)
    if typeof(transformers.embed) <: CompositeEmbedding
        indices = (tok = indices, segment = fill(1, length(indices)))
    end
    return transformers.embed(indices)
end

function (transformers::TransformersModel)(
    vocabulary::Vocabulary{T},
    inputs::AbstractVector{T},
    outputs::AbstractVector{T}
)::AbstractMatrix{AbstractFloat} where T
    # Encode inputs.
    input_embedding = _embed(transformers, vocabulary, inputs)
    encoded_embedding = transformers.encoder(input_embedding)

    # Decode outputs.
    output_embedding = _embed(transformers, vocabulary, outputs)
    decoded_embedding = transformers.decoder(output_embedding, encoded_embedding)

    # Calculate probabilities for next token.
    log_probabilities = transformers.generator(decoded_embedding)
    return log_probabilities
end

function TransformersModel(
    embed::AbstractEmbed,
    size::Int,
    head::Int,
    hs::Int,
    ps::Int,
    layer::Int,
    vocab_size::Int;
    act=relu,
    pdrop=0.1,
    length_normalization::Number=0.0
)
    return TransformersModel(
        embed,
        Encoder(size, head, hs, ps, layer; act=act,pdrop=pdrop),
        Decoder(size, head, hs, ps, layer; act=act,pdrop=pdrop),
        Generator(size, vocab_size)
    )
end