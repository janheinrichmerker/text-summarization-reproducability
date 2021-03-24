using PyCall
using Conda

torch = pyimport_conda("torch", "pytorch")

# Load preprocessed data from a single PyTorch Pickle file.
# As Julia does not support parsing PyTorch files itself,
# delegate to the Python package instead. 
function load_torch(path::String)
    @info "Loading PyTorch file '$path'."
    dataset = torch.load(abspath(path))
    return dataset
end

# Create an iterator for rows of the preprocessed data.
function data_loader(paths::Array{String})::Channel{SummaryPair}
	return Channel{SummaryPair}() do channel
		for path ∈ paths
			data = load_torch(path)
			@info "Loaded $(length(data)) rows."
			for dict ∈ data
				push!(channel, SummaryPair(dict))
			end
		end
	end
end

# Check if the file is of the given corpus type.
function is_corpus_type(path::String, type::CorpusType)
	isfile(path) && occursin(corpus_type_name(type), basename(path))
end

function filter_corpus_type!(paths::Array{String}, type::CorpusType)
    filter!(path -> is_corpus_type(path, type), paths)
end

# Create an iterator for rows of the preprocessed training/test/validation data.
function data_loader(path::String, type::CorpusType)::Channel{SummaryPair}
	paths = readdir(path, join=true, sort=true)
	filter_corpus_type!(paths, type)

	return Channel{SummaryPair}() do channel
		loader = data_loader(paths)
		for row ∈ loader
			push!(channel, row)
		end
	end
end