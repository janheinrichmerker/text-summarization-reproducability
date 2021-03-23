using Dates

out_dir() = mkpath(normpath(joinpath(@__FILE__, "..", "..", "..", "out")))

snapshot_name(time::DateTime, step::Int) = "bert-abs-$(Dates.format(time, "yyyy-mm-dd-HH-MM"))-step-$(string(step, pad=6))"

snapshot_file(time::DateTime, step::Int, name::String) = joinpath(out_dir(), "$(snapshot_name(time, step))-$name")

function snapshot_files(time::DateTime, inc_steps::Int, name::String)::Tuple{Vector{String}, Int}
    step = inc_steps
    files = []
    push!(files, snapshot_file(time, step, name))
    while isfile(files[end])
        step += inc_steps
        push!(files, snapshot_file(time, step, name))
    end
    return files[1:end-1], step - inc_steps
end
