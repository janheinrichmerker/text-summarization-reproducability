module Data

include("model.jl")
export SummaryPair
include("datasets.jl")
export xsum, cnn_dm
include("utils.jl")
export out_dir, snapshot_file, snapshot_files

function __init__()
    register(xsum_dependency)
    register(cnn_dm_dependency)
end

end # module
