module Data

include("model.jl")
export SummaryPair, CorpusType
include("datasets.jl")
export xsum_loader, cnndm_loader
include("utils.jl")
export out_dir, snapshot_file, snapshot_files

function __init__()
    register(xsum_dependency)
    register(cnndm_dependency)
end

end # module
