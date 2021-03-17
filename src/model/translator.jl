using Transformers
using Transformers.Basic
using Flux
using Flux: @functor, onecold

include("encoder.jl")
include("decoder.jl")
include("classifier.jl")
include("../search/beam_search.jl")

struct Translator
    encoder::Union{Encoder,Bert}
    decoder::Decoder
    classifier::Classifier
    vocabulary::Vocabulary
end

@functor Translator

function (translator::Translator)(tokens, start_token, end_token)
    indices = tokens |> translator.vocabulary
    encoded = translator.encoder(indices)    
    sequence = beam_search(
        5,
        translator.vocabulary.list,
        function (sequence::Array)
            target = translator.vocabulary(sequence)
            decoded = translator.decoder(target, encoded)
            classified = translator.classifier(decoded)[:,end]
            return classified
        end,
        sequence -> last(sequence) != end_token,
        initial_sequence=[start_token]
    )
    return sequence[2:end - 1]
end

function Translator(embed::AbstractEmbed, vocabulary::Vocabulary, size::Int, head::Int, hs::Int, ps::Int, layer::Int; act=relu,pdrop=0.1)
    return Translator(
        Encoder(embed, size, head, hs, ps, layer; act=act,pdrop=pdrop),
        Decoder(embed, size, head, hs, ps, layer; act=act,pdrop=pdrop),
        Classifier(length(vocabulary), size),
        vocabulary
    )
end
