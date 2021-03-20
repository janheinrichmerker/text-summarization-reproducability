using Transformers.Basic
using Flux
using Flux:@functor

struct Generator
    chain::Positionwise
end

@functor Generator

function (generator::Generator)(embedding)
    return generator.chain(embedding)
end

function Generator(size::Integer, vocab_size::Integer)
    return Generator(
        Positionwise(
            Dense(size, vocab_size),
            logsoftmax
        )
    )
end
