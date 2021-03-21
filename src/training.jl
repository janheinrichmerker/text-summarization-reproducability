@info "Load DataDeps package."
using DataDeps
@info "Load CUDA package."
using CUDA
@info "Load Flux package."
using Flux
using Flux:update!,reset!,onehotbatch
using Flux.Losses:logitcrossentropy
@info "Load Transformers package."
using Transformers
using Transformers.Basic
using Transformers.Pretrain
using BSON: @save


if !CUDA.functional(true)
    @warn "You're training the model without GPU support."
end
enable_gpu(true)


@info "Load preprocessed data (CNN / Daily Mail)."
include("data/datasets.jl")
include("data/loader.jl")
include("data/model.jl")
function cnndm_loader(corpus_type::String)::Channel{SummaryPair}
    data_dir = joinpath(
        datadep"CNN-Daily-Mail-Preprocessed-BERT",
        "bert_data_cnndm_final"
    )
    return data_loader(data_dir, corpus_type)
end

# @show first(cnndm_loader("train"))
# @show first(cnndm_loader("test"))
# @show first(cnndm_loader("valid"))

cnndm_train = cnndm_loader("train")
# cnndm_test = cnndm_loader("test")
# cnndm_valid = cnndm_loader("valid")


@info "Load pretrained BERT model."
bert_model, wordpiece, tokenizer = pretrain"bert-uncased_L-12_H-768_A-12"
vocabulary = Vocabulary(wordpiece) |> gpu


@info "Create summarization model from BERT model."
include("model/abstractive.jl")
model = BertAbs(bert_model, length(vocabulary)) |> gpu
@show model


function preprocess(text::String)::AbstractVector{String}
    tokens = text |> tokenizer |> wordpiece
    return ["[CLS]"; tokens; "[SEP]"]
end

function loss(
    inputs::AbstractVector{String},
    outputs::AbstractVector{String}
)::AbstractFloat
    prediction = model.transformers(vocabulary, inputs, outputs)
    ground_truth = onehotbatch(outputs, vocabulary.list) |> gpu
    # TODO Replace with label smoothing loss and KL divergence.
    loss = logitcrossentropy(prediction, ground_truth)
    return loss
end


include("training/optimizers.jl")
optimizer_encoder = WarmupADAM(2ℯ^(-3), 20_000, (0.9, 0.99))
optimizer_decoder = WarmupADAM(0.1, 10_000, (0.9, 0.99))


parameters_encoder = params(model.transformers.encoder)
@show length(parameters_encoder)
parameters_decoder = params(
    # model.transformers.embed, 
    # model.transformers.decoder, 
    model.transformers.generator
)
@show length(parameters_decoder)
@info "Found $(length(parameters_encoder)) trainable parameters for encoder and $(length(parameters_decoder)) parameters for decoder."


reset!(model)
max_steps = 200_000
for (step, summary) ∈ zip(1:max_steps, cnndm_train)
    @info "Training step $step/$max_steps."
    inputs = summary.source |> preprocess |> gpu
    outputs = summary.target |> preprocess |> gpu

    @info "Take gradients."
    # local loss_encoder
    # @timed gradients_encoder = gradient(parameters_encoder) do
    #     loss_encoder = loss(prediction, ground_truth)
    #     return loss_encoder
    # end
    # @show loss_encoder
    local loss_decoder
    @timed gradients_decoder = gradient(parameters_decoder) do
        loss_decoder = loss(inputs, outputs)
        return loss_decoder
    end
    @show loss_decoder

    @info "Update model."
    # update!(optimizer_encoder, parameters_encoder, gradients_encoder)
    @timed update!(optimizer_decoder, parameters_decoder, gradients_decoder)

    if step % 2500 == 0 || step % 100 == 0
        @info "Save model snapshot."
        @save "../out/bert-abs-$step-$(now()).bson" model optimizer_decoder # optimizer_encoder
        # Evaluate on validation set.
    end
end
