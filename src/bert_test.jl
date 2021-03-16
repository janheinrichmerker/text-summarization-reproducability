@info "Loading Transformers."
using Transformers
@info "Loading Transformers.Basic."
using Transformers.Basic
@info "Loading Transformers.Pretrain."
using Transformers.Pretrain

@info "Loading pretrained BERT model."
bert_model, wordpiece, tokenizer = pretrain"bert-uncased_L-12_H-768_A-12"
@info "Pretrained BERT model loaded successfully."

vocab = Vocabulary(wordpiece)

text1 = "Peter Piper picked a peck of pickled peppers" |> tokenizer |> wordpiece
text2 = "Fuzzy Wuzzy was a bear" |> tokenizer |> wordpiece

text = ["[CLS]"; text1; "[SEP]"; text2; "[SEP]"]
@assert text == [
    "[CLS]", "peter", "piper", "picked", "a", "peck", "of", "pick", "##led", "peppers", "[SEP]", 
    "fuzzy", "wu", "##zzy", "was", "a", "bear", "[SEP]"
]

token_indices = vocab(text)
segment_indices = [fill(1, length(text1) + 2); fill(2, length(text2) + 1)]

sample = (tok = token_indices, segment = segment_indices)
@info sample

@info "Embedding and applying transformers."
feature_tensors = sample |> bert_model.embed |> bert_model.transformers
@info length(feature_tensors)
