using Transformers.Basic
using Transformers.Basic:AbstractEmbed
using Flux:@functor

struct Decoder
    embed::AbstractEmbed
    layers::Array{<:TransformerDecoder}
end

@functor Decoder

function (decoder::Decoder)(target_indices, encoded_embedding)
    if typeof(decoder.embed) <: CompositeEmbedding
        target_indices = (
            tok = target_indices,
            segment = fill(1, length(target_indices))
        )
    end

    embedding = decoder.embed(target_indices)
    for transformer ∈ decoder.layers
        embedding = transformer(embedding, encoded_embedding)
    end
    return embedding
end

function Decoder(embed::AbstractEmbed, size::Int, head::Int, hs::Int, ps::Int, layer::Int; act=relu, pdrop=0.1)
    return Decoder(
        embed,
        [TransformerDecoder(size, head, hs, ps, act=act, pdrop=pdrop)
        for i ∈ 1:layer]
    )
end
