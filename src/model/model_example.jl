@info "Loading Flux."
using Flux
using Flux:onehotbatch
@info "Loading CUDA."
using CUDA
@info "Loading Transformers."
using Transformers
using Transformers.Basic
using Transformers.Pretrain
using PreSumm.Model

# Enable GPU for Transformers.
enable_gpu(true)

@info "Loading pretrained BERT model."
bert_model, wordpiece, tokenizer = pretrain"bert-uncased_L-12_H-768_A-12"
vocabulary = Vocabulary(wordpiece) |> todevice
@info "Pretrained BERT model loaded successfully."

preprocess(text::String) = ["[CLS]"; wordpiece(tokenizer(text)); "[SEP]"]
inputs = "Peter Piper picked a peck of pickled peppers." |> preprocess |> todevice
@show inputs
outputs = "Peter picked pickled peppers." |> preprocess |> todevice
@show outputs

model = BertAbs(bert_model, length(vocabulary)) |> todevice
@show model

# Predict new word probabilities for target.
prediction = model.transformers(vocabulary, inputs, outputs)
@show size(prediction)
# Use original one-hot word probabilities from target for comparison.
ground_truth = onehotbatch(outputs, vocabulary.list)
@show size(ground_truth)

# How good is the prediction?
@show loss = Flux.Losses.logitcrossentropy(prediction, ground_truth)

# Generate a new sequence from the sample.
# The output sequence is likely to be nonsense, 
# because the decoder has not been trained yet.
@show translated = model("Peter picked peppers.", vocabulary)
