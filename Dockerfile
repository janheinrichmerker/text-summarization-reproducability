FROM julia:1.10.1

WORKDIR /app

# Install and pre-compile dependencies.
COPY Project.toml Manifest.toml ./
RUN julia --project=./ --eval="using Pkg; Pkg.instantiate(); Pkg.precompile()"

# Setup PyCall to use a Conda environment.
COPY src/setup_python.jl ./src/
RUN julia --project=./ ./src/setup_python.jl

# Copy remaining source files.
COPY src/ ./src/

EXPOSE 1234

# Start notebook.
ENV HOST=0.0.0.0
CMD julia --project=./ ./src/start_notebook.jl | sed "s/0.0.0.0/localhost/"
