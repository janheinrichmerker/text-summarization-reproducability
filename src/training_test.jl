@info "Loading Flux."
using Flux
using Flux:onecold
@info "Loading CUDA."
using CUDA
@info "Loading Transformers."
using Transformers
using Transformers.Basic
using Transformers.Pretrain


# Enable GPU for Transformers.
enable_gpu(true)
gpu = todevice


@info "Loading pretrained BERT model."
bert_model, wordpiece, tokenizer = pretrain"bert-uncased_L-12_H-768_A-12"
vocabulary = Vocabulary(wordpiece)
@info "Pretrained BERT model loaded successfully."

sample_text = "Peter Piper picked a peck of pickled peppers"
sample = sample_text |> tokenizer |> wordpiece |> t -> ["[CLS]", t..., "[SEP]"]

# include("src/model/abstractive.jl")
include("model/abstractive.jl")

model::Translator = BertAbs(
    bert_model,
    length(vocabulary),
    5 # TODO this parameter doesn't appear in the paper.
) # |> gpu

sample
translated = model(sample, vocabulary)
