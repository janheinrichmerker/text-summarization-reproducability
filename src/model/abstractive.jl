using Transformers
using Transformers.Basic

# BertAbs transformer model for abstractive summarization 
# from the "Text Summarization with Pretrained Encoders" paper 
# by Liu et al. (2019) as described on page 6.
function BertAbs(
    bert_model::TransformerModel{<:CompositeEmbedding,<:Bert,<:Any},
    vocab_size::Integer
)::Translator
    return Translator(
        TransformersModel(
            AbsEmbed(768, vocab_size),
            bert_model.transformers,
            Decoder(768, 8, 96, 2048, 6, pdrop=0.1),
            Generator(768, vocab_size)
        ),
        AbsSearch()
    )
    return 
end

# TransformerAbs transformer model for abstractive summarization
# from the "Text Summarization with Pretrained Encoders" paper 
# by Liu et al. (2019) as described on pages 6-7.
function TransformerAbs(vocab_size::Integer)::Translator
    return Translator(
        TransformersModel(
            AbsEmbed(768, vocab_size),
            768, 8, 96, 2048, 6, 
            vocab_size, 
            pdrop=0.1,
        ),
        AbsSearch()
    )
end

# Tiny version of `TransformerAbs` for testing training on devices with
# less GPU memory.
function TransformerAbsTiny(vocab_size::Integer)::Translator
    return Translator(
        TransformersModel(
            AbsEmbed(8, vocab_size),
            8, 4, 2, 16, 2, 
            vocab_size, 
            pdrop=0.1,
        ),
        AbsSearch()
    )
end

# Beam search used in the abstractive summarization models 
# from the "Text Summarization with Pretrained Encoders" paper 
# by Liu et al. (2019) as described on page 6.
function AbsSearch()
    return BeamSearch(
        5,
        # This parameter is not described in the paper.
        32,
        # The length normalization parameter α should be 
        # tuned on the development set from 0.6 to 1.0.
        length_normalization=0.8
    )
end

# Word embeddings used in the abstractive summarization models 
# from the "Text Summarization with Pretrained Encoders" paper 
# by Liu et al. (2019) as described on page 4.
# TODO It's not clear how the random embedding is 
# "added to" BERT embeddings.
function AbsEmbed(size::Integer, vocab_size::Integer)
    return WordPositionEmbed(
        size,
        vocab_size,
        # Not mentioned in the paper, but CNN and Daily Mail 
        # documents are limited to 2000 tokens, XSum documents
        # are shorter then CNN / Daily Mail on average.
        # Our guess is that no document would be longer
        # than 4096 tokens.
        4096,
        trainable=true
    )
end
