struct SummaryPair
	source::String
	target::String
end

# Opinionated replacement/joining of preprocessed 
# source and target sequences.
function SummaryPair(dict::Dict{Any,Any})
    source = join(dict["src_txt"], " ")
    target = replace(dict["tgt_txt"], "<q>" => " . ")
    SummaryPair(source, target)
end

@enum CorpusType begin
    train_type
    test_type
    dev_type
end

function corpus_type(name::String)::CorpusType
	@assert name ∈ ["train", "valid", "test"]
    if name == "train"
        return train_type
    elseif name == "test"
        return test_type
    elseif name == "valid"
        return dev_type
    else
        throw(ErrorException("Unknown corpus type."))
    end
end

function corpus_type_name(type::CorpusType)::String
	@assert type ∈ [train_type, test_type, dev_type]
    if type == train_type
        return "train"
    elseif type == test_type
        return "test"
    elseif type == dev_type
        return "valid"
    else
        throw(ErrorException("Unknown corpus type."))
    end
end