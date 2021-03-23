### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 2f26e06e-8c08-11eb-18c0-2f8e5b0cf47a
using CUDA

# ╔═╡ 590fdac2-8c08-11eb-1c42-93ff500817c1
using GPUArrays: allowscalar

# ╔═╡ 4068759c-8c08-11eb-1d2f-d79efe98c2b0
using Flux

# ╔═╡ 8297be82-8c08-11eb-39fb-eb37f75d09e4
using Flux: update!, reset!, onehot

# ╔═╡ 91e66b2c-8c08-11eb-1a3e-891e6401eb8b
using Transformers

# ╔═╡ 92baa6f8-8c08-11eb-3321-233e7bf14afe
using Transformers.Basic

# ╔═╡ 95f72c2e-8c08-11eb-2e18-0b166f648de6
using Transformers.Pretrain

# ╔═╡ 98c4351e-8c08-11eb-14fe-7df792e4b7cf
using BSON: @save, @load

# ╔═╡ 38f9317a-84e2-11eb-117b-9f62916abd75
using DataDeps

# ╔═╡ 702d3820-84d9-11eb-1895-1d00242e5363
md"""
# Reproducing Text Summarization with Pretrained Encoders

This is a reproducability study on the ["Text Summarization with Pretrained Encoders"](https://doi.org/10.18653/v1/D19-1387) paper by Yang Liu and Mirella Lapata.
"""

# ╔═╡ 176ccb48-8c08-11eb-3068-3b684ff378b5
md"""
## Setup
"""

# ╔═╡ f2ef12a6-8c09-11eb-10ae-0177f2691fdf
DEBUG=true

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
    @warn "You're training the model without GPU support."
    Flux.use_cuda[] = false
    enable_gpu(false)
end

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

# ╔═╡ 2e4da306-85d6-11eb-0957-1182af2f9302
md"""
Include data dependency definitions.
To keep the notebook readable, data loading is included from a separate file.
(Note the usage of `ingredients()` instead of `include()`: This is needed to avoid namespace clashes and allow for re-executing the cell.)
"""

# ╔═╡ 44848686-8c0b-11eb-064c-7f56c697e69c
md"""
#### Training data
Load article-summary pairs from the CNN / Daily Mail dataset's train split.
We can get a fresh `Channel` (iterator) with training data each time we call `cnndm_train()`.
"""

# ╔═╡ d4b77c36-8c0b-11eb-3a32-b3fc1b7e80af
md"""
#### Validation data
Get a `Channel` of article-summary pairs from the dataset's development/validation split.
"""

# ╔═╡ 95b154c8-8c0b-11eb-293f-f15c59a529e8
md"""
#### Test data
Get a `Channel` of article-summary pairs from the dataset's test split.
"""

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

# ╔═╡ eadeab8e-8c0d-11eb-34ee-77a32f6005ba


# ╔═╡ 3f4853dc-8c06-11eb-3a59-cdb9fc854879
md"""
## Training
"""

# ╔═╡ 619a2256-8c08-11eb-3544-2b20fba8a71d


# ╔═╡ 4e3ead50-8c06-11eb-39ff-a3834ba819c2
md"""
## Evaluation
"""

# ╔═╡ d7e44134-8c06-11eb-150f-0b37210cff88
md"""
### Model selection
"""

# ╔═╡ e2449708-8c06-11eb-0d09-991e4917b9d3
md"""
### Automatic evaluation
"""

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

# ╔═╡ da824c02-85e4-11eb-37c4-552a37e00739
md"""
## Utilities
"""

# ╔═╡ bee4c866-8c06-11eb-1728-c90b666bc20d
md"""
### Pluto notebook utilities
"""

# ╔═╡ e28f95aa-85e4-11eb-0334-d14173dd479e
md"""
Helper function to include files into the notebook similar to `include()`, 
but the module is returned, so that we can use the included functions with a namespace
and also notebook cells would update automatically.
"""

# ╔═╡ 92d6f834-85e2-11eb-261e-5da9f2c88300
function ingredients(path::String) # See 
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	m
end;

# ╔═╡ 2ca30702-8c08-11eb-31aa-b55c7ce561fb
data = ingredients("data/datasets.jl");

# ╔═╡ 32af5d7e-8c0a-11eb-1784-17984db6e6fc
cnndm_train() = data.cnndm_loader(data.train_type)

# ╔═╡ 0d56f78e-8c0b-11eb-3d74-218c28817970
first(cnndm_train())

# ╔═╡ 2a4a03e0-8c0b-11eb-0bea-2d96d67712fc
cnndm_dev() = data.cnndm_loader(data.dev_type)

# ╔═╡ 3a4f4ce4-8c0b-11eb-3518-0907e3d2a62d
first(cnndm_dev())

# ╔═╡ 1c697a62-8c0b-11eb-0814-0d02d802f6fc
cnndm_test() = data.cnndm_loader(data.test_type)

# ╔═╡ 395e445e-8c0b-11eb-0193-17ea21763de6
first(cnndm_test())

# ╔═╡ Cell order:
# ╟─702d3820-84d9-11eb-1895-1d00242e5363
# ╟─176ccb48-8c08-11eb-3068-3b684ff378b5
# ╠═f2ef12a6-8c09-11eb-10ae-0177f2691fdf
# ╟─3351b1ca-8c08-11eb-14c3-8f61900721f4
# ╠═2f26e06e-8c08-11eb-18c0-2f8e5b0cf47a
# ╠═590fdac2-8c08-11eb-1c42-93ff500817c1
# ╠═4068759c-8c08-11eb-1d2f-d79efe98c2b0
# ╠═8297be82-8c08-11eb-39fb-eb37f75d09e4
# ╠═91e66b2c-8c08-11eb-1a3e-891e6401eb8b
# ╠═92baa6f8-8c08-11eb-3321-233e7bf14afe
# ╠═95f72c2e-8c08-11eb-2e18-0b166f648de6
# ╠═98c4351e-8c08-11eb-14fe-7df792e4b7cf
# ╟─1be31330-8c08-11eb-296f-6f5df8bf6e41
# ╠═2646eafe-8c08-11eb-2a25-c338673a3d2a
# ╟─22e29d1a-84e2-11eb-3e35-f7050cfd6cc4
# ╠═38f9317a-84e2-11eb-117b-9f62916abd75
# ╠═0d86a904-84e3-11eb-1c0c-9d37637c8420
# ╠═29348670-84e4-11eb-076b-b78a92e94642
# ╟─98474e70-8c06-11eb-28c4-35bd54c8a558
# ╟─2e4da306-85d6-11eb-0957-1182af2f9302
# ╠═2ca30702-8c08-11eb-31aa-b55c7ce561fb
# ╟─44848686-8c0b-11eb-064c-7f56c697e69c
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
# ╠═eadeab8e-8c0d-11eb-34ee-77a32f6005ba
# ╠═3f4853dc-8c06-11eb-3a59-cdb9fc854879
# ╠═619a2256-8c08-11eb-3544-2b20fba8a71d
# ╟─4e3ead50-8c06-11eb-39ff-a3834ba819c2
# ╟─d7e44134-8c06-11eb-150f-0b37210cff88
# ╟─e2449708-8c06-11eb-0d09-991e4917b9d3
# ╟─a2ef84f2-8c07-11eb-0a94-21ebe1ed471a
# ╟─f67d71ea-8c06-11eb-06a9-550a8d54c139
# ╟─da824c02-85e4-11eb-37c4-552a37e00739
# ╟─bee4c866-8c06-11eb-1728-c90b666bc20d
# ╟─e28f95aa-85e4-11eb-0334-d14173dd479e
# ╠═92d6f834-85e2-11eb-261e-5da9f2c88300
