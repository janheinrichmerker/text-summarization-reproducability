### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 38f9317a-84e2-11eb-117b-9f62916abd75
using DataDeps

# ╔═╡ aaf3f94e-84d9-11eb-325e-3d162a5a95a2
using Transformers

# ╔═╡ ccbdbf7e-84d9-11eb-39b5-b9dd61cc9544
include("data/datasets.jl");

# ╔═╡ 702d3820-84d9-11eb-1895-1d00242e5363
md"""
# Reproducing Text Summarization with Pretrained Encoders

This is a reproducability study on the ["Text Summarization with Pretrained Encoders"](https://doi.org/10.18653/v1/D19-1387) paper by Yang Liu and Mirella Lapata.
"""

# ╔═╡ 22e29d1a-84e2-11eb-3e35-f7050cfd6cc4
md"""
## Preproces training data
"""

# ╔═╡ a906d7e6-84e3-11eb-2c10-8b1aba2552a6
md"""
We're loading datasets with DataDeps.jl.
This way each dataset is cached and we only have to download once.
If you haven't previously downloaded the datasets, _this step will take a while_.
"""

# ╔═╡ 0d86a904-84e3-11eb-1c0c-9d37637c8420
ENV["DATADEPS_ALWAYS_ACCEPT"] = true; # Don't ask for downloading.

# ╔═╡ 29348670-84e4-11eb-076b-b78a92e94642
ENV["DATADEPS_PROGRESS_UPDATE_PERIOD"] = 5; # Log process to the console, though.

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

# ╔═╡ 58544bfa-84d9-11eb-3426-b3b5cc2a19a8
md"""
## BERT Test
Here we just test if BERT works.
"""

# ╔═╡ Cell order:
# ╟─702d3820-84d9-11eb-1895-1d00242e5363
# ╟─22e29d1a-84e2-11eb-3e35-f7050cfd6cc4
# ╟─a906d7e6-84e3-11eb-2c10-8b1aba2552a6
# ╠═38f9317a-84e2-11eb-117b-9f62916abd75
# ╠═0d86a904-84e3-11eb-1c0c-9d37637c8420
# ╠═29348670-84e4-11eb-076b-b78a92e94642
# ╠═ccbdbf7e-84d9-11eb-39b5-b9dd61cc9544
# ╟─430bfb54-84e3-11eb-382d-0b842b8c1e06
# ╠═554263ca-84e1-11eb-1dfa-a346284f5891
# ╠═e0c16654-8526-11eb-054e-0796d0c4cc43
# ╟─6462ab2e-84e3-11eb-33dd-13e5000c78da
# ╠═41c97f76-84e2-11eb-38d3-d5a96f75c59a
# ╠═25e67bca-8527-11eb-2736-5362c6ce3725
# ╠═58544bfa-84d9-11eb-3426-b3b5cc2a19a8
# ╠═aaf3f94e-84d9-11eb-325e-3d162a5a95a2
