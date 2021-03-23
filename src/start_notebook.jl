using Pluto

host = get(ENV, "HOST", "127.0.0.1")
notebook = "./src/notebook.jl"
if !isempty(host)
    Pluto.run(notebook = notebook, host = host)
else
    Pluto.run(notebook = notebook)
end
