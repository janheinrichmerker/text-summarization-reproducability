using PyCall

# Load preprocessed data from a single PyTorch Pickle file.
# As Julia does not support parsing PyTorch files itself,
# delegate to the Python package instead. 
function load_torch(path::String)
	torch = pyimport_conda("torch", "pytorch")
    @info "Loading PyTorch file '$path'."
    dataset = torch.load(abspath(path))
    return dataset
end

# Create an iterator for rows of the preprocessed data.
function load_summaries(paths::Array{String})::Channel{SummaryPair}
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
function is_corpus_type(path::String, type::String)
	@assert type ∈ ["train", "valid", "test"]
	isfile(path) && occursin(type, basename(path))
end

function filter_corpus_type!(paths::Array{String}, type::String)
    filter!(path -> is_corpus_type(path, type), paths)
end

# Create an iterator for rows of the preprocessed training/test/validation data.
function summaries(path::String, type::String)::Channel{SummaryPair}
	paths = readdir(path, join=true, sort=true)
	filter_corpus_type!(paths, type)

	return Channel{SummaryPair}() do channel
		loader = load_summaries(paths)
		for row ∈ loader
			push!(channel, row)
		end
	end
end
