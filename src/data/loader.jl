using PyCall
using Conda

Conda.add("pytorch")
torch = pyimport("torch")

# Load preprocessed data from a single PyTorch file.
function load_torch(path::String)
    @debug "Loading PyTorch file '$file'."
    dataset = torch.load(abspath(path))
    return dataset
end


using ResumableFunctions

# Create an iterator for rows of the preprocessed training/test/validation data.
@resumable function data_loader(
    path::String, 
    corpus_type::String
)::Dict{String,Any}
    @assert corpus_type ∈ ["train", "valid", "test"]

    # List all files in directory.
	files = readdir(path, join=true, sort=true)
    # Filter by corpus type.
    filter!(file -> occursin(corpus_type, file), files)

	for file ∈ files
        # Load single file.
		data = load_torch(cnn_path)
		println("Loaded $(length(data)) rows.")

        # Yield all rows.
		for row ∈ data
    		@yield row
		end
	end
end