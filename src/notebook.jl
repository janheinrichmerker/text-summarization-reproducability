### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 38f9317a-84e2-11eb-117b-9f62916abd75
using DataDeps

# ╔═╡ ccbdbf7e-84d9-11eb-39b5-b9dd61cc9544
include("data/datasets.jl");

# ╔═╡ 702d3820-84d9-11eb-1895-1d00242e5363
md"""
# Reproducing Text Summarization with Pretrained Encoders

This is a reproducability study on the ["Text Summarization with Pretrained Encoders"](https://doi.org/10.18653/v1/D19-1387) paper by Yang Liu and Mirella Lapata.
"""

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

# ╔═╡ 2e4da306-85d6-11eb-0957-1182af2f9302
md"Include data dependency definitions."

# ╔═╡ 98474e70-8c06-11eb-28c4-35bd54c8a558
md"""
### Data loading
"""

# ╔═╡ 430bfb54-84e3-11eb-382d-0b842b8c1e06
md"Load preprocessed XSum data."

# ╔═╡ 554263ca-84e1-11eb-1dfa-a346284f5891
data_xsum = joinpath(
	datadep"XSum-Preprocessed-BERT",
	"bert_data_xsum_new"
)

# ╔═╡ e0c16654-8526-11eb-054e-0796d0c4cc43
readdir(data_xsum)

# ╔═╡ 6462ab2e-84e3-11eb-33dd-13e5000c78da
md"Load preprocessed CNN/Dailymail data."

# ╔═╡ 41c97f76-84e2-11eb-38d3-d5a96f75c59a
data_cnndm = joinpath(
	datadep"CNN-Dailymail-Preprocessed-BERT",
	"bert_data_cnndm_final"
)

# ╔═╡ 25e67bca-8527-11eb-2736-5362c6ce3725
readdir(data_cnndm)

# ╔═╡ ee8b8128-85d5-11eb-03b4-4153ea6de940
md"Include data loading utils for loading preprocessed data."

# ╔═╡ 58544bfa-84d9-11eb-3426-b3b5cc2a19a8
md"""
### Preprocessing
"""

# ╔═╡ a4f8ec0a-8c06-11eb-042e-290b3405e5e6
md"""
## Pretrained BERT model
We'll now load the pretrained BERT model.
"""

# ╔═╡ 3f4853dc-8c06-11eb-3a59-cdb9fc854879
md"""
## Training
"""

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

# ╔═╡ f8ea1b8a-8536-11eb-3f98-3b8e28ac16ff
loader = ingredients("data/loader.jl");

# ╔═╡ Cell order:
# ╟─702d3820-84d9-11eb-1895-1d00242e5363
# ╟─22e29d1a-84e2-11eb-3e35-f7050cfd6cc4
# ╠═38f9317a-84e2-11eb-117b-9f62916abd75
# ╠═0d86a904-84e3-11eb-1c0c-9d37637c8420
# ╠═29348670-84e4-11eb-076b-b78a92e94642
# ╟─2e4da306-85d6-11eb-0957-1182af2f9302
# ╠═ccbdbf7e-84d9-11eb-39b5-b9dd61cc9544
# ╟─98474e70-8c06-11eb-28c4-35bd54c8a558
# ╟─430bfb54-84e3-11eb-382d-0b842b8c1e06
# ╠═554263ca-84e1-11eb-1dfa-a346284f5891
# ╟─e0c16654-8526-11eb-054e-0796d0c4cc43
# ╟─6462ab2e-84e3-11eb-33dd-13e5000c78da
# ╠═41c97f76-84e2-11eb-38d3-d5a96f75c59a
# ╟─25e67bca-8527-11eb-2736-5362c6ce3725
# ╟─ee8b8128-85d5-11eb-03b4-4153ea6de940
# ╠═f8ea1b8a-8536-11eb-3f98-3b8e28ac16ff
# ╟─58544bfa-84d9-11eb-3426-b3b5cc2a19a8
# ╟─a4f8ec0a-8c06-11eb-042e-290b3405e5e6
# ╟─3f4853dc-8c06-11eb-3a59-cdb9fc854879
# ╟─4e3ead50-8c06-11eb-39ff-a3834ba819c2
# ╟─d7e44134-8c06-11eb-150f-0b37210cff88
# ╟─e2449708-8c06-11eb-0d09-991e4917b9d3
# ╟─a2ef84f2-8c07-11eb-0a94-21ebe1ed471a
# ╟─f67d71ea-8c06-11eb-06a9-550a8d54c139
# ╟─da824c02-85e4-11eb-37c4-552a37e00739
# ╟─bee4c866-8c06-11eb-1728-c90b666bc20d
# ╟─e28f95aa-85e4-11eb-0334-d14173dd479e
# ╠═92d6f834-85e2-11eb-261e-5da9f2c88300
