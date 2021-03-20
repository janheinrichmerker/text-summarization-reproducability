@info "Load DataDeps package."
using DataDeps
@info "Load Flux package."
using Flux
using Flux:update!,reset!,onehotbatch
using Flux.Losses:logitcrossentropy
@info "Load CUDA package."
using CUDA
@info "Load Transformers package."
using Transformers
using Transformers.Basic
using Transformers.Pretrain

if !CUDA.functional(true)
    @warn "You're training the model without GPU support."
end
enable_gpu(true)


@info "Load preprocessed data (CNN/Dailymail)."
include("data/datasets.jl")
include("data/loader.jl")
include("data/model.jl")
function cnndm_loader(corpus_type::String)::Channel{SummaryPair}
    data_dir = joinpath(
        datadep"CNN-Dailymail-Preprocessed-BERT",
        "bert_data_cnndm_final"
    )
    return data_loader(data_dir, corpus_type)
end

# @show first(cnndm_loader("train"))
# @show first(cnndm_loader("test"))
# @show first(cnndm_loader("valid"))

cnndm_train = cnndm_loader("train")
cnndm_test = cnndm_loader("test")
cnndm_valid = cnndm_loader("valid")


@info "Load pretrained BERT model."
bert_model, wordpiece, tokenizer = pretrain"bert-uncased_L-12_H-768_A-12"
vocabulary = Vocabulary(wordpiece)


@info "Create summarization model from BERT model."
include("model/abstractive.jl")
model = BertAbs(bert_model, length(vocabulary))
@show model


function preprocess(text::String)::AbstractVector{String}
    tokens = text |> tokenizer |> wordpiece
    return ["[CLS]"; tokens; "[SEP]"]
end

function loss(inputs::AbstractVector{String}, outputs::AbstractVector{String})
    @info "Predict new distribution."
    prediction = model.transformers(vocabulary, inputs, outputs)
    @info "Compute ground truth distribution."
    ground_truth = onehotbatch(outputs, vocabulary.list)
    # TODO Replace with label smoothing loss and KL divergence.
    @info "Calculate loss."
    return logitcrossentropy(prediction, ground_truth)
end


include("training/optimizers.jl")
optimizer_encoder = WarmupADAM(2ℯ^(-3), 20_000, (0.9, 0.99))
optimizer_decoder = WarmupADAM(0.1, 10_000, (0.9, 0.99))
parameters_encoder = params(model.transformers.encoder)
@show length(parameters_encoder)
parameters_decoder = params(
    model.transformers.embed, 
    model.transformers.decoder, 
    model.transformers.generator
)
@show length(parameters_decoder)
max_epochs = 200_000


reset!(model)
for (epoch, summary) ∈ zip(1:200_000, cnndm_train)
    @info "Training epoch $epoch."
    inputs = summary.source |> preprocess
    outputs = summary.target |> preprocess

    @info "Take gradients."
    # gradients_encoder = gradient(parameters_encoder) do
    #     loss(inputs, outputs)
    # end
    gradients_decoder = gradient(parameters_decoder) do
        loss(inputs, outputs)
    end

    @info "Update model."
    # update!(optimizer_encoder, parameters_encoder, gradients_encoder)
    update!(optimizer_decoder, parameters_decoder, gradients_decoder)

    if step % 2500
        @info "Save model snapshot."
        # Save model snapshot, evaluuate on validation set.
    end
end