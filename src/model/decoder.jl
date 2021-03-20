using Transformers.Basic
using Flux:@functor

struct Decoder
    layers::Array{<:TransformerDecoder}
end

@functor Decoder

function (decoder::Decoder)(target_embedding, encoded_embedding)
    return foldl(
        (embedding, transformer) -> transformer(embedding, encoded_embedding),
        decoder.layers,
        init=target_embedding
    )
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
