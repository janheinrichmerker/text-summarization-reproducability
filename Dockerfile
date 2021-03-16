FROM julia:1.5.3

WORKDIR /app

# Install dependencies.
COPY Project.toml Manifest.toml ./
RUN julia --project=./ --eval="using Pkg; Pkg.instantiate()"

# Setup PyCall to use a Conda environment.
COPY src/setup_python.jl ./src/
RUN julia --project=./ ./src/setup_python.jl

# Pre-compile Pluto.
RUN julia --project=./ --eval="using Pluto"

# Copy remaining source files.
COPY src/ ./src/

EXPOSE 1234

# Start notebook.
ENV HOST=0.0.0.0
CMD julia --project=./ ./src/start_notebook.jl | sed "s/0.0.0.0/localhost/"
