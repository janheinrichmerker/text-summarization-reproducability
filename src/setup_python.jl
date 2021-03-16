using Pkg

# Reset Julia to use Conda environment.
ENV["PYTHON"] = ""

# Re-build PyCall.jl with updated Python version.
Pkg.build("PyCall")
