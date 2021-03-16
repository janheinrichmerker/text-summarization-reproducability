using Pluto

host = ENV["HOST"]
notebook = "./src/notebook.jl"
if !isempty(host)
    Pluto.run(notebook = notebook, host = host)
else
    Pluto.run(notebook = notebook)
end
