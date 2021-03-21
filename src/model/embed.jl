using Transformers.Basic
using Flux:@functor

struct WordPositionEmbed
    word::Embed
    position::PositionEmbedding
end

@functor WordPositionEmbed

function (embed::WordPositionEmbed)(x)
    word_embedding = embed.word(x) 
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
        Embed(size, vocab_size),
        PositionEmbedding(size, max_len, trainable=trainable)
    )
end
