using Transformers.Basic
using Flux:@functor

struct Decoder
    layers::Array{<:TransformerDecoder}
end

@functor Decoder

function (decoder::Decoder)(target_embedding, encoded_embedding)
    embedding = target_embedding
    for transformer ∈ decoder.layers
        embedding = transformer(embedding, encoded_embedding)
    end
    return embedding
end

function Decoder(
    size::Integer,
    head::Integer,
    hs::Integer,
    ps::Integer,
    layer::Integer;
    act=relu,
    pdrop=0.1
)::Decoder
    return Decoder([
        TransformerDecoder(size, head, hs, ps, act=act, pdrop=pdrop)
        for i ∈ 1:layer
    ])
end
