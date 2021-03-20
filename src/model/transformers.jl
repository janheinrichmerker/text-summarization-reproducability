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

function (transformers::TransformersModel)(
    vocabulary::Vocabulary{T},
    inputs::Vector{T},
    outputs::Vector{T}
)::AbstractMatrix{<:AbstractFloat} where T
    # Encode inputs.
    input_indices = vocabulary(inputs)
    # if typeof(transformers.embed) <: CompositeEmbedding
    #     input_indices = (
    #         tok = input_indices,
    #         segment = fill(1, length(input_indices))
    #     )
    # end
    input_embedding = transformers.embed(input_indices)
    encoded_embedding = transformers.encoder(input_embedding)

    # Decode outputs.
    output_indices = vocabulary(outputs)
    # if typeof(transformers.embed) <: CompositeEmbedding
    #     output_indices = (
    #         tok = output_indices,
    #         segment = fill(1, length(output_indices))
    #     )
    # end
    output_embedding = transformers.embed(output_indices)
    decoded_embedding = transformers.decoder(output_embedding, encoded_embedding)

    # Calculate probabilities for next token.
    return transformers.generator(decoded_embedding)
end

function TransformersModel(
    embed::AbstractEmbed,
    size::Integer,
    head::Integer,
    hs::Integer,
    ps::Integer,
    layer::Integer,
    vocab_size::Integer;
    act=relu,
    pdrop=0.1,
    length_normalization::AbstractFloat=0.0
)
    return TransformersModel(
        embed,
        Encoder(size, head, hs, ps, layer; act=act,pdrop=pdrop),
        Decoder(size, head, hs, ps, layer; act=act,pdrop=pdrop),
        Generator(size, vocab_size)
    )
end