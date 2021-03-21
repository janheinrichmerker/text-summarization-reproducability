using  Flux
using Transformers.Basic

struct WordPositionEmbed
    word::Embed
    position::PositionEmbedding
end

Flux.@functor WordPositionEmbed
Flux.trainable(embed::WordPositionEmbed) = (embed.word, embed.position)

function (embed::WordPositionEmbed)(indices::AbstractArray{<:Number})
    word_embedding = embed.word(indices) 
    embedding = word_embedding .+ embed.position(word_embedding)
    return embedding
end

function WordPositionEmbed(
    size::Int, 
    vocab_size::Int, 
    max_len::Int=1024;
    trainable::Bool=false,
    scale=inv(sqrt(size))
)::WordPositionEmbed
    return WordPositionEmbed(
        Embed(size, vocab_size, scale=scale),
        PositionEmbedding(size, max_len, trainable=trainable)
    )
end
