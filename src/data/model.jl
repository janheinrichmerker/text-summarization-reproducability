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
