### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 06f8595a-8c42-11eb-3b1c-011d0706850e
using BertAbs

# ╔═╡ 2f26e06e-8c08-11eb-18c0-2f8e5b0cf47a
using CUDA

# ╔═╡ 590fdac2-8c08-11eb-1c42-93ff500817c1
using GPUArrays: allowscalar

# ╔═╡ 4068759c-8c08-11eb-1d2f-d79efe98c2b0
using Flux

# ╔═╡ 8297be82-8c08-11eb-39fb-eb37f75d09e4
using Flux: update!, reset!, loadparams!, onehot

# ╔═╡ 91e66b2c-8c08-11eb-1a3e-891e6401eb8b
using Transformers

# ╔═╡ 92baa6f8-8c08-11eb-3321-233e7bf14afe
using Transformers.Basic

# ╔═╡ 95f72c2e-8c08-11eb-2e18-0b166f648de6
using Transformers.Pretrain

# ╔═╡ 98c4351e-8c08-11eb-14fe-7df792e4b7cf
using BSON: @save, @load

# ╔═╡ 44df9b3a-8c16-11eb-17a7-3d6038911fc5
using Dates

# ╔═╡ 2ed19a74-8c18-11eb-0c06-95c930bbc8c1
using Statistics

# ╔═╡ 74207314-8c47-11eb-0989-f3dab6bcffd2
using PlutoUI

# ╔═╡ 30edfac8-8c18-11eb-1d59-a5042a177b20
using Plots

# ╔═╡ 702d3820-84d9-11eb-1895-1d00242e5363
md"""
# Reproducing Text Summarization with Pretrained Encoders

This is a reproducability study on the ["Text Summarization with Pretrained Encoders"](https://doi.org/10.18653/v1/D19-1387) paper by Yang Liu and Mirella Lapata.
"""

# ╔═╡ 176ccb48-8c08-11eb-3068-3b684ff378b5
md"""
## Setup
"""

# ╔═╡ 400c92b8-8c17-11eb-170d-1f6b3fe06a43
md"""
Should we train the model before evaluating?

Yes $(@bind TRAIN CheckBox(default=false))

Keep in mind that training requires a lot of hardware resources and takes a long while! See the _Training loop_ section for more details.
"""

# ╔═╡ 19fec164-8c43-11eb-3e55-1dc0c30bf8c1
md"""
Should evaluate the trained model?

Yes $(@bind EVALUATE CheckBox(default=true))
"""

# ╔═╡ 6847039e-8c17-11eb-15b9-e70a268a56fb
md"""
Should we use a very simple model for less powerfull machines?

Yes $(@bind DEBUG CheckBox(default=true))

If this checkbox is ticked, we'll drastically over-simplify the model and data so that you should be able to train a tiny variant of the actual model on your own machine. See the _Model_ section for more details.
"""

# ╔═╡ 3351b1ca-8c08-11eb-14c3-8f61900721f4
md"""
### Load packages
"""

# ╔═╡ 1be31330-8c08-11eb-296f-6f5df8bf6e41
md"""
### Setup GPU with CUDA
"""

# ╔═╡ 2646eafe-8c08-11eb-2a25-c338673a3d2a
if CUDA.functional(true)
    # CUDA.device!(1)
    allowscalar(false)
    Flux.use_cuda[] = true
    enable_gpu(true)
else
    Flux.use_cuda[] = false
    enable_gpu(false)
end;

# ╔═╡ dbc654f4-8c18-11eb-059e-f91749357883
if CUDA.functional(true)
    md"Alright, you're running in GPU-accellerated mode using CUDA."
else
    md"""
	**⚠️ You're running the notebook without GPU support.**
	
	It is likely that you won't be able to train our model.
	At least try setting `DEBUG = true`, better yet skip training with `TRAIN = false` and use pretrained model snapshots
	"""
end

# ╔═╡ 39e1ae40-8c18-11eb-3f12-b566f7f516e2
md"""
### Setup plots
"""

# ╔═╡ 48c4e332-8c18-11eb-05c8-f15576c14fcb
gr()

# ╔═╡ 493f6f90-8c26-11eb-0627-fb59af397184
theme(:vibrant)

# ╔═╡ 22e29d1a-84e2-11eb-3e35-f7050cfd6cc4
md"""
## Datasets
We're loading datasets with DataDeps.jl.
This way each dataset is cached and we only have to download once.
If you haven't previously downloaded the datasets, _this process will take a while_.
"""

# ╔═╡ 0d86a904-84e3-11eb-1c0c-9d37637c8420
ENV["DATADEPS_ALWAYS_ACCEPT"] = true; # Don't ask for downloading.

# ╔═╡ 29348670-84e4-11eb-076b-b78a92e94642
ENV["DATADEPS_PROGRESS_UPDATE_PERIOD"] = 5; # Log process to the console, though.

# ╔═╡ 98474e70-8c06-11eb-28c4-35bd54c8a558
md"""
### Data loading
"""

# ╔═╡ 44848686-8c0b-11eb-064c-7f56c697e69c
md"""
#### Training data
Load article-summary pairs from the CNN / Daily Mail dataset's train split.
We can get a fresh `Channel` (iterator) with training data each time we call `cnndm_train()`.
"""

# ╔═╡ 9109e4e8-8c46-11eb-296d-154cfbd4db73
data(split) = cnn_dm(split) 

# ╔═╡ 32af5d7e-8c0a-11eb-1784-17984db6e6fc
data_train() = data("train")

# ╔═╡ 0d56f78e-8c0b-11eb-3d74-218c28817970
first(data_train())

# ╔═╡ d4b77c36-8c0b-11eb-3a32-b3fc1b7e80af
md"""
#### Validation data
Get a `Channel` of article-summary pairs from the dataset's development/validation split.
"""

# ╔═╡ 2a4a03e0-8c0b-11eb-0bea-2d96d67712fc
data_dev() = data("valid")

# ╔═╡ 3a4f4ce4-8c0b-11eb-3518-0907e3d2a62d
first(data_dev())

# ╔═╡ 95b154c8-8c0b-11eb-293f-f15c59a529e8
md"""
#### Test data
Get a `Channel` of article-summary pairs from the dataset's test split.
"""

# ╔═╡ 1c697a62-8c0b-11eb-0814-0d02d802f6fc
data_test() = data("test")

# ╔═╡ 395e445e-8c0b-11eb-0193-17ea21763de6
first(data_test())

# ╔═╡ 6462ab2e-84e3-11eb-33dd-13e5000c78da
md"Load preprocessed CNN/Dailymail data."

# ╔═╡ 58544bfa-84d9-11eb-3426-b3b5cc2a19a8
md"""
### Preprocessing
"""

# ╔═╡ a4f8ec0a-8c06-11eb-042e-290b3405e5e6
md"""
## Pretrained BERT model
Load the pretrained BERT model from Google using Transformers.jl.
We load the uncased BERT-base variant with 12 layers of hidden size 768.
The model is also loaded internally with DataDeps.jl.
"""

# ╔═╡ 0303fb1e-8c0c-11eb-28d1-1d9e631cb8af
bert_model, wordpiece, tokenizer = pretrain"bert-uncased_L-12_H-768_A-12"

# ╔═╡ ff141dee-8c0c-11eb-35e3-c7a7fe623dc3
md"""
### Vocabulary
When debugging, to keep the model trainable on smaller machines, limit the vocabulary to 100 words.
"""

# ╔═╡ 58ae6888-8c0c-11eb-359c-ed1a2cafbfde
function load_vocabulary()::Vocabulary
	if !DEBUG
		Vocabulary(wordpiece) |> gpu
	else
		Vocabulary(wordpiece.vocab[1:100], wordpiece.vocab[100]) |> gpu
	end
end

# ╔═╡ b1ac4c9e-8c0d-11eb-035a-4b075ffeb8f5
vocabulary = load_vocabulary()

# ╔═╡ fc930278-8c0d-11eb-1c70-2176689ce05c
md"""
## Model
"""

# ╔═╡ aab3df80-8c0e-11eb-37bd-bd40e5a498c7
md"""
When debugging on smaller machines, use a tiny transformer encoder-decoder variant instead of the large BERT-based model.
"""

# ╔═╡ 0cb47e48-8c0e-11eb-0a4d-8f1bd42be2b8
function load_model()::Translator
	if !DEBUG
		BertAbs(bert_model, length(vocabulary)) |> gpu
	else
		TransformerAbsTiny(length(vocabulary)) |> gpu
	end
end

# ╔═╡ e6db6ba4-8c0e-11eb-1bdb-3bc0b02e2b4f
model = load_model()

# ╔═╡ 3f4853dc-8c06-11eb-3a59-cdb9fc854879
md"""
## Training
"""

# ╔═╡ 29f82622-8c0f-11eb-211b-1d275e80ebbf
md"""
### Preprocessing
Preprocess a `String` to a sequence of tokens (not necessarily words), tokenized by the BERT tokenizer.
"""

# ╔═╡ 0326189c-8c0f-11eb-38d1-595283980478
function preprocess(text::String)::AbstractVector{String}
    tokens = text |> tokenizer |> wordpiece
    max_length = min(4096, length(tokens)) # Truncate to 4096 tokens
    tokens = tokens[1:max_length]
    return ["[CLS]"; tokens; "[SEP]"]
end

# ╔═╡ e1a94230-8c10-11eb-3a01-e765b58c2c29
md"""
### Prediction & loss calculation
"""

# ╔═╡ 5144a378-8c0f-11eb-3b32-6f7661781a0a
md"""
Label smoothing factor $\alpha$ to apply to the one-hot ground-truth labels when calculating the model loss.
"""

# ╔═╡ 619a2256-8c08-11eb-3544-2b20fba8a71d
# FIXME Label smoothing doesn't work yet.
label_smoothing_α = 0.0

# ╔═╡ c8cf0d90-8c0f-11eb-3a11-4bded8f5be4f
md"""
Calculate the model loss for given input, output, and ground truth token probabilities.
If $\alpha > 0$, return Kullback-Leibler divergence between the model's predictions and smoothed ground truth labels. Otherwise cross entropy between predictions and unsmoothed ground truth.
"""

# ╔═╡ 811777ce-8c0f-11eb-2ac4-b51b85cd6bc9
function loss(
    inputs::AbstractVector{String},
    outputs::AbstractVector{String},
    ground_truth::AbstractMatrix{<:Number},
    translator::Translator
)::AbstractFloat
    prediction = translator.transformers(vocabulary, inputs, outputs)
    loss = logtranslationloss(prediction, ground_truth, α=label_smoothing_α)
    return loss
end

# ╔═╡ a46bcb12-8c11-11eb-0493-1505139bb2bf
md"""
### Model parameters
Parameters are separated for the encoder and the rest of the model (i.e., embeddings, decoder, and generator).
This accounts for the former being pretrained while the latter have to be learned from scratch.
"""

# ╔═╡ 390e1432-8c12-11eb-2697-13d8f3c0e2bf
parameters_encoder = params(model.transformers.encoder)

# ╔═╡ 3ddf05d4-8c12-11eb-0a80-e50ef061dc33
parameters_decoder = params(
    model.transformers.embed, 
    model.transformers.decoder, 
    model.transformers.generator
)

# ╔═╡ ca3a1fe4-8c3c-11eb-243e-b58cc96b2d1d
md"""
We're going to tune
$(params_count(parameters_encoder))
parameters from the encoder and train 
$(params_count(parameters_decoder))
parameters for the decoder, embeddings, and generator from scratch.
"""

# ╔═╡ bf061ff0-8c10-11eb-108d-cf008ae94342
md"""
### Optimizers
We use ADAM optimizers with a custom warmup schedule for the learning rate $\eta$.
The model's encoder is optimized slower, because it has already bee pretrained.
"""

# ╔═╡ 486cc03c-8c11-11eb-03cb-bb640ac822f5
md"""
Encoder schedule:
$\eta = 2\mathcal{e}^{-3} \cdot \min(\text{step}^{-0.5},\ \text{step} \cdot 20\,000)$
"""

# ╔═╡ 2c0590d8-8c11-11eb-1ac0-41a24f84ed58
optimizer_encoder = WarmupADAM(η=2ℯ^(-3), w=20_000, β=(0.9, 0.99)) |> gpu

# ╔═╡ 940c2c58-8c11-11eb-39b6-cf4dc61bd550
md"""
Decoder schedule:
$\eta = 0.1 \cdot \min(\text{step}^{-0.5},\ \text{step} \cdot 10\,000)$
"""

# ╔═╡ 1cd3487e-8c11-11eb-173a-af13bddc850f
optimizer_decoder = WarmupADAM(η=0.1, w=10_000, β=(0.9, 0.99)) |> gpu

# ╔═╡ 03a78c44-8c13-11eb-27da-e11097968bc7
max_steps = !DEBUG ? 200_000 : 10

# ╔═╡ d32d3994-8c12-11eb-1848-d9bd515709a5
md"""
### Training loop
Here we define the training loop that is iterated over for at most $max_steps steps.
"""

# ╔═╡ 07e53e8e-8c13-11eb-1188-5309e353c56a
snapshot_steps = !DEBUG ? 2500 : 10

# ╔═╡ 79b63cd0-8c1c-11eb-3b31-1395f64e979d
start_time = now()

# ╔═╡ ef68c91e-8c3c-11eb-393a-5145b8cff911
md"""
The actual training loop extracts tokens for each row's source and target text (original article vs. short summary).
Then the loss between the model's predicted token probabilities and the ground truth probability is compared.
The model is then updated with gradients for the encoder and gradients for decoder, embeddings, and generator.

Losses are saved from every step and snapshots of the model weights and losses/optimizers for both parameter sets are saved in BSON format to the output folder:
$(out_dir())
"""

# ╔═╡ b4dbc124-8c15-11eb-009d-7b3e0a26ba4d
function save_snapshot(
		time::DateTime,
		step::Int,
		model::Translator,
		optimizer_encoder::WarmupADAM,
		optimizer_decoder::WarmupADAM,
		losses_encoder::AbstractVector{<:AbstractFloat},
		losses_decoder::AbstractVector{<:AbstractFloat})
	@info "Save model snapshot."
	weights = params(model)
	snapfile(time, step, name) = snapshot_file(time, step, name)
	@save snapfile(time, step, "model.bson") model
	@save snapfile(time, step, "weights.bson") weights
	@save snapfile(time, step, "optimizer-encoder.bson") optimizer_encoder
	@save snapfile(time, step, "optimizer-decoder.bson") optimizer_decoder
	@save snapfile(time, step, "losses-encoder.bson") losses_encoder
	@save snapfile(time, step, "losses-decoder.bson") losses_decoder
end

# ╔═╡ 3f117812-8c13-11eb-3b34-3988290737a7
function train!(model::Translator)
	losses_encoder = []
	losses_decoder = []
	
	data = data_train()
	for (step, summary) ∈ zip(1:max_steps, data)
   		@info "Training step $step/$max_steps."
		inputs = summary.source |> preprocess |> gpu
		outputs = summary.target |> preprocess |> gpu
		ground_truth = onehot(vocabulary, outputs) |> gpu
		
		
		@info "Train encoder."
		local loss_encoder
		gradients_encoder = gradient(parameters_encoder) do
			loss_encoder = loss(inputs, outputs, ground_truth)
			return loss_encoder
		end
		push!(losses_encoder, loss_encoder)
		@info "Updating encoder parameters." loss_encoder
		update!(optimizer_encoder, parameters_encoder, gradients_encoder)

		
		@info "Train decoder, embeddings, and generator."
		local loss_decoder
		gradients_decoder = gradient(parameters_decoder) do
			loss_decoder = loss(inputs, outputs, ground_truth)
			return loss_decoder
		end
		push!(losses_decoder, loss_decoder)
		@info "Updating decoder, embeddings, and generator parameters." loss_decoder
		update!(optimizer_decoder, parameters_decoder, gradients_decoder)
		
		
		if step % snapshot_steps == 0
			save_snapshot(
				model |> cpu,
				optimizer_encoder |> cpu,
				optimizer_decoder |> cpu,
				losses_encoder,
				losses_decoder,
			)
		end
	end
end

# ╔═╡ be5c7b3c-8c16-11eb-1a3f-01eadf6bc1bb
md"""
Now train the model. Lean back and take a sip of coffee, as this takes a long while. ☕

_Tip: If you've already trained the model, untick the training checkbox at the beginning of the notebook and the evaluation will continue without training the model first._
"""

# ╔═╡ ded97e7a-8c16-11eb-2b04-23a5dec4405e
if TRAIN
	train!(model)
end

# ╔═╡ 4e3ead50-8c06-11eb-39ff-a3834ba819c2
md"""
## Evaluation
"""

# ╔═╡ 566a15e4-8c44-11eb-3d27-a1b28fc3057d
if EVALUATE
	md"""
	Evaluate training loss, select best model by development/validation loss, 
	and measure summary quality.
	"""
else
	md"""
	**⚠️ Skipping evaluation is not yet fully supported.**
	"""
end

# ╔═╡ 35878a0a-8c1c-11eb-0e18-d1964db60a67
md"""
### Configuration
Define the training run we want to evaluate.
"""

# ╔═╡ 9bcff242-8c1d-11eb-007f-6b3ae5b2ef6a
md"""
When did the chosen model start training?
"""

# ╔═╡ 546c963e-8c1c-11eb-0762-cbecc179fdf0
custom_start_time = DateTime(2021, 03, 22, 17, 50)

# ╔═╡ 655f933a-8c1c-11eb-1fb0-636b021637b6
eval_start_time = TRAIN ? start_time : custom_start_time

# ╔═╡ 16441198-8c1e-11eb-05af-99619660cdd8
md"""
What was the rate at which snapshots were saved for that model?
"""

# ╔═╡ 87ceb29c-8c1d-11eb-2534-39649a087b4f
custom_snapshot_steps = 2500

# ╔═╡ bd5cde20-8c1d-11eb-25f4-c1d10590778a
eval_snapshot_steps = TRAIN ? snapshot_steps : custom_snapshot_steps

# ╔═╡ 110108b4-8c3d-11eb-230e-f1f8ef3985ff
md"""
### Model loading
At this point we've either trained the model from the previous sections or copied pretrained training snapshots to the output folder:
$(out_dir())
"""

# ╔═╡ dc238b06-8c1d-11eb-3cf2-495db70ac90a
weights_snapshots, max_step = EVALUATE ? snapshot_files(
	eval_start_time,
	eval_snapshot_steps,
	"weights.bson"
) : ([],0)

# ╔═╡ c44ed108-8c49-11eb-30d2-73e41b46aed7
has_weights_snapshot = !isempty(weights_snapshots)

# ╔═╡ 873bb7ca-8c1e-11eb-08f7-755484003009
if !has_weights_snapshot
	md"""
	**⚠️ Could not find any weights snapshot for training 
	started at $(Dates.format(eval_start_time, "yyyy-mm-dd (at HH:MM)")) 
	with snapshotting every $eval_snapshot_steps steps.**
	"""
else
	md"""
	Found $(length(weights_snapshots)) snapshots for training 
	started at $(Dates.format(eval_start_time, "yyyy-mm-dd (at HH:MM)"))  
	with snapshotting every $eval_snapshot_steps steps.
	The last snapshot is from step $max_step.
	"""
end

# ╔═╡ 9fcbdd2e-8c21-11eb-0e59-63649715d380
eval_snapshot_file(name) = snapshot_file(eval_start_time, max_step, name)

# ╔═╡ 216e718c-8c1c-11eb-066f-f705805cb70e
md"""
### Training loss
Show loss on the training set at each iteration
"""

# ╔═╡ 7e4da548-8c22-11eb-1b3c-81269c4f5f5a
function plot_losses()
	if !has_weights_snapshot
		return
	end
	
	@load eval_snapshot_file("losses-encoder.bson") losses_encoder
	@load eval_snapshot_file("losses-decoder.bson") losses_decoder
	loss_plot = plot(
		title = "Generator classification loss during training",
		xlabel = "steps",
		ylabel = "cross entropy loss",
    	xticks = 0:eval_snapshot_steps:max_step,
	)
	plot!(loss_plot, losses_encoder, label="encoder")
	plot!(loss_plot, losses_decoder, label="dcoder, embeddings, generator")
	savefig(loss_plot, joinpath(out_dir(), "training-loss.pdf"))
	savefig(loss_plot, joinpath(out_dir(), "training-loss.png"))
	loss_plot
end;

# ╔═╡ a39ba53e-8c22-11eb-04af-95d7d112f67d
plot_losses()

# ╔═╡ 16ccaf1a-8c2a-11eb-274a-5d8831732be1
md"""
### Best model selection
Next, we'll select the best model by classification loss on the development/validation set.
"""

# ╔═╡ b3692f86-8c1b-11eb-3425-35027febcec0
md"""
#### Development data
For evaluating loss on the development/validation set, we load its summary pairs.
Loading and caching to the GPU might take a short moment.
"""

# ╔═╡ 97423bde-8c18-11eb-2b01-8526b32161d4
dev_summaries = collect(data_dev()) |> gpu

# ╔═╡ 40f5d3de-8c2a-11eb-2e1b-d39b430a6550
function dev_loss(translator::Translator; agg=mean)::AbstractFloat
    losses = []
    for summary ∈ dev_summaries
        inputs = summary.source |> preprocess |> gpu
        outputs = summary.target |> preprocess |> gpu
        ground_truth = onehot(vocabulary, outputs) |> gpu
        loss = loss(inputs, outputs, ground_truth, translator)
        push!(losses, loss)
    end
    return agg(losses)
end

# ╔═╡ 7cb9dfd2-8c2a-11eb-27ac-811c01ba9368
function find_best_model(;agg=mean)::Translator
    best_loss = Inf
    model = load_model()
    for snapshot ∈ weights_snapshots
        @info "Load model snapshot $snapshot."
		@load snapshot weights
		loadparams!(model, weights)
        loss = dev_loss(model, agg=agg)
        if loss < best_loss
            best_loss = loss
        end
    end
	if best_loss == Inf
		throw(ErrorException("Could not find model."))
	end
    return model
end

# ╔═╡ c33a3092-8c2a-11eb-03c3-d74e98a3867c
best_model = find_best_model()

# ╔═╡ e2449708-8c06-11eb-0d09-991e4917b9d3
md"""
### Automatic evaluation
"""

# ╔═╡ 26d83e30-8c28-11eb-3107-a3df1788bc2f
md"""
#### Test data
For evaluating quality on the test set, we load its summary pairs.
Loading and caching to the GPU might take a short moment.
"""

# ╔═╡ 3e25c08a-8c28-11eb-143b-b996590c22d4
test_summaries = collect(data_test()) |> gpu

# ╔═╡ a2ef84f2-8c07-11eb-0a94-21ebe1ed471a
md"""
#### ROUGE benchmark
Measure ROUGE-1, ROUGE-2, and ROUGE-L on the summaries generated for the test dataset.
"""

# ╔═╡ f67d71ea-8c06-11eb-06a9-550a8d54c139
md"""
### Manual evaluation
We generate summaries for the first 100 articles from the CNN / Daily Mail dataset's test split.
These summaries are maually examined and scored on the following scale:
- 0 = unreadable summary or unrelated to original article
- 1 = readable and related to original article, some redundancy
- 2 = captures exactly the article's main points, no redundancy
"""

# ╔═╡ Cell order:
# ╟─702d3820-84d9-11eb-1895-1d00242e5363
# ╟─176ccb48-8c08-11eb-3068-3b684ff378b5
# ╟─400c92b8-8c17-11eb-170d-1f6b3fe06a43
# ╟─19fec164-8c43-11eb-3e55-1dc0c30bf8c1
# ╟─6847039e-8c17-11eb-15b9-e70a268a56fb
# ╟─3351b1ca-8c08-11eb-14c3-8f61900721f4
# ╠═06f8595a-8c42-11eb-3b1c-011d0706850e
# ╠═2f26e06e-8c08-11eb-18c0-2f8e5b0cf47a
# ╠═590fdac2-8c08-11eb-1c42-93ff500817c1
# ╠═4068759c-8c08-11eb-1d2f-d79efe98c2b0
# ╠═8297be82-8c08-11eb-39fb-eb37f75d09e4
# ╠═91e66b2c-8c08-11eb-1a3e-891e6401eb8b
# ╠═92baa6f8-8c08-11eb-3321-233e7bf14afe
# ╠═95f72c2e-8c08-11eb-2e18-0b166f648de6
# ╠═98c4351e-8c08-11eb-14fe-7df792e4b7cf
# ╠═44df9b3a-8c16-11eb-17a7-3d6038911fc5
# ╠═2ed19a74-8c18-11eb-0c06-95c930bbc8c1
# ╠═74207314-8c47-11eb-0989-f3dab6bcffd2
# ╠═30edfac8-8c18-11eb-1d59-a5042a177b20
# ╟─1be31330-8c08-11eb-296f-6f5df8bf6e41
# ╠═2646eafe-8c08-11eb-2a25-c338673a3d2a
# ╟─dbc654f4-8c18-11eb-059e-f91749357883
# ╟─39e1ae40-8c18-11eb-3f12-b566f7f516e2
# ╠═48c4e332-8c18-11eb-05c8-f15576c14fcb
# ╠═493f6f90-8c26-11eb-0627-fb59af397184
# ╟─22e29d1a-84e2-11eb-3e35-f7050cfd6cc4
# ╠═0d86a904-84e3-11eb-1c0c-9d37637c8420
# ╠═29348670-84e4-11eb-076b-b78a92e94642
# ╟─98474e70-8c06-11eb-28c4-35bd54c8a558
# ╟─44848686-8c0b-11eb-064c-7f56c697e69c
# ╠═9109e4e8-8c46-11eb-296d-154cfbd4db73
# ╠═32af5d7e-8c0a-11eb-1784-17984db6e6fc
# ╠═0d56f78e-8c0b-11eb-3d74-218c28817970
# ╟─d4b77c36-8c0b-11eb-3a32-b3fc1b7e80af
# ╠═2a4a03e0-8c0b-11eb-0bea-2d96d67712fc
# ╠═3a4f4ce4-8c0b-11eb-3518-0907e3d2a62d
# ╟─95b154c8-8c0b-11eb-293f-f15c59a529e8
# ╠═1c697a62-8c0b-11eb-0814-0d02d802f6fc
# ╠═395e445e-8c0b-11eb-0193-17ea21763de6
# ╟─6462ab2e-84e3-11eb-33dd-13e5000c78da
# ╟─58544bfa-84d9-11eb-3426-b3b5cc2a19a8
# ╟─a4f8ec0a-8c06-11eb-042e-290b3405e5e6
# ╠═0303fb1e-8c0c-11eb-28d1-1d9e631cb8af
# ╟─ff141dee-8c0c-11eb-35e3-c7a7fe623dc3
# ╠═58ae6888-8c0c-11eb-359c-ed1a2cafbfde
# ╠═b1ac4c9e-8c0d-11eb-035a-4b075ffeb8f5
# ╟─fc930278-8c0d-11eb-1c70-2176689ce05c
# ╟─aab3df80-8c0e-11eb-37bd-bd40e5a498c7
# ╠═0cb47e48-8c0e-11eb-0a4d-8f1bd42be2b8
# ╠═e6db6ba4-8c0e-11eb-1bdb-3bc0b02e2b4f
# ╟─3f4853dc-8c06-11eb-3a59-cdb9fc854879
# ╟─29f82622-8c0f-11eb-211b-1d275e80ebbf
# ╠═0326189c-8c0f-11eb-38d1-595283980478
# ╟─e1a94230-8c10-11eb-3a01-e765b58c2c29
# ╟─5144a378-8c0f-11eb-3b32-6f7661781a0a
# ╠═619a2256-8c08-11eb-3544-2b20fba8a71d
# ╟─c8cf0d90-8c0f-11eb-3a11-4bded8f5be4f
# ╠═811777ce-8c0f-11eb-2ac4-b51b85cd6bc9
# ╟─a46bcb12-8c11-11eb-0493-1505139bb2bf
# ╠═390e1432-8c12-11eb-2697-13d8f3c0e2bf
# ╠═3ddf05d4-8c12-11eb-0a80-e50ef061dc33
# ╟─ca3a1fe4-8c3c-11eb-243e-b58cc96b2d1d
# ╟─bf061ff0-8c10-11eb-108d-cf008ae94342
# ╟─486cc03c-8c11-11eb-03cb-bb640ac822f5
# ╠═2c0590d8-8c11-11eb-1ac0-41a24f84ed58
# ╟─940c2c58-8c11-11eb-39b6-cf4dc61bd550
# ╠═1cd3487e-8c11-11eb-173a-af13bddc850f
# ╟─d32d3994-8c12-11eb-1848-d9bd515709a5
# ╠═03a78c44-8c13-11eb-27da-e11097968bc7
# ╠═07e53e8e-8c13-11eb-1188-5309e353c56a
# ╠═79b63cd0-8c1c-11eb-3b31-1395f64e979d
# ╟─ef68c91e-8c3c-11eb-393a-5145b8cff911
# ╠═3f117812-8c13-11eb-3b34-3988290737a7
# ╠═b4dbc124-8c15-11eb-009d-7b3e0a26ba4d
# ╟─be5c7b3c-8c16-11eb-1a3f-01eadf6bc1bb
# ╠═ded97e7a-8c16-11eb-2b04-23a5dec4405e
# ╟─4e3ead50-8c06-11eb-39ff-a3834ba819c2
# ╟─566a15e4-8c44-11eb-3d27-a1b28fc3057d
# ╟─35878a0a-8c1c-11eb-0e18-d1964db60a67
# ╟─9bcff242-8c1d-11eb-007f-6b3ae5b2ef6a
# ╠═546c963e-8c1c-11eb-0762-cbecc179fdf0
# ╠═655f933a-8c1c-11eb-1fb0-636b021637b6
# ╟─16441198-8c1e-11eb-05af-99619660cdd8
# ╠═87ceb29c-8c1d-11eb-2534-39649a087b4f
# ╠═bd5cde20-8c1d-11eb-25f4-c1d10590778a
# ╟─110108b4-8c3d-11eb-230e-f1f8ef3985ff
# ╠═dc238b06-8c1d-11eb-3cf2-495db70ac90a
# ╠═c44ed108-8c49-11eb-30d2-73e41b46aed7
# ╟─873bb7ca-8c1e-11eb-08f7-755484003009
# ╠═9fcbdd2e-8c21-11eb-0e59-63649715d380
# ╟─216e718c-8c1c-11eb-066f-f705805cb70e
# ╠═7e4da548-8c22-11eb-1b3c-81269c4f5f5a
# ╠═a39ba53e-8c22-11eb-04af-95d7d112f67d
# ╟─16ccaf1a-8c2a-11eb-274a-5d8831732be1
# ╟─b3692f86-8c1b-11eb-3425-35027febcec0
# ╠═97423bde-8c18-11eb-2b01-8526b32161d4
# ╠═40f5d3de-8c2a-11eb-2e1b-d39b430a6550
# ╠═7cb9dfd2-8c2a-11eb-27ac-811c01ba9368
# ╠═c33a3092-8c2a-11eb-03c3-d74e98a3867c
# ╟─e2449708-8c06-11eb-0d09-991e4917b9d3
# ╟─26d83e30-8c28-11eb-3107-a3df1788bc2f
# ╠═3e25c08a-8c28-11eb-143b-b996590c22d4
# ╟─a2ef84f2-8c07-11eb-0a94-21ebe1ed471a
# ╟─f67d71ea-8c06-11eb-06a9-550a8d54c139
