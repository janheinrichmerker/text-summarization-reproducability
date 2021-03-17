using Transformers
using Transformers.Basic
using Flux
using Flux:@functor, onecold

include("decoder.jl")
include("classifier.jl")
include("translator.jl")

# TransformerAbs model from the "Text Summarization with Pretrained Encoders" paper by Liu et al. (2019) as described on pages 6-7.
function TransformerAbs(
    embed::AbstractEmbed,
    vocabulary::Vocabulary,
)
    return Translator(embed, vocabulary, 768, 8, 96, 2048, 6, pdrop=0.1)
end

# BertAbs model from the "Text Summarization with Pretrained Encoders" paper by Liu et al. (2019) as described on page 6.
function BertAbs(bert_model::TransformerModel{<:AbstractEmbed,<:Bert,<:Any}, vocabulary::Vocabulary)
    return Translator(
        bert_model.transformers,
        Decoder(bert_model.embed, 768, 8, 96, 2048, 6, pdrop=0.1),
        Classifier(length(vocabulary), 768),
        vocabulary
    )
end
