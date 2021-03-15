using PyCall
using Conda

Conda.add("pytorch")
torch = pyimport("torch")

# Load preprocessed data from a single PyTorch file.
function load_torch(path::String)
    @info "Loading PyTorch file '$path'."
    dataset = torch.load(abspath(path))
    return dataset
end


using ResumableFunctions

# Create an iterator for rows of the preprocessed data.
@resumable function data_loader(paths::Array{String})::Dict{String,Any}
	for path ∈ paths
		data = load_torch(path)
		@info "Loaded $(length(data)) rows."
		for row ∈ data
    		@yield row
		end
	end
end

# Check if the file is of the given corpus type.
function is_corpus_type(path::String, corpus_type::String)
	@assert corpus_type ∈ ["train", "valid", "test"]
	isfile(path) && occursin(corpus_type, basename(path))
end

function filter_corpus_type!(paths::Array{String}, corpus_type::String)
    filter!(path -> is_corpus_type(path, corpus_type), paths)
end

# Create an iterator for rows of the preprocessed training/test/validation data.
@resumable function data_loader(path::String, corpus_type::String)::Dict{String,Any}
	paths = readdir(path, join=true, sort=true)
    filter_corpus_type!(paths, corpus_type)

    loader = data_loader(paths)
    for row ∈ loader
        @yield row
    end
end