using Transformers.Basic
using Transformers.Basic:AbstractEmbed
using Flux
using Flux:@functor

struct Classifier
    classifier::Positionwise{Dense,typeof(logsoftmax)}
end

@functor Classifier

function (classifier::Classifier)(embedding)
    return classifier.classifier(embedding)
end

function Classifier(vocab_size::Int, size::Int)
    return Classifier(
        Positionwise(
            Dense(size, vocab_size),
            logsoftmax
        )
    )
end
