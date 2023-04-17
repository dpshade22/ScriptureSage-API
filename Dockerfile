FROM julia:latest

ENV OPENAI_API_KEY="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"

# Install the required Julia packages
RUN julia -e 'using Pkg; \
    Pkg.add(["CSV", "DataFrames", "HTTP", "URIs", "JSON", "Distances", "OpenAI", "LinearAlgebra", "DotEnv"])'

# Copy the server script into the container
COPY routes.jl /app/server.jl

# Set the working directory
WORKDIR /app

# Expose the port on which the server will run
EXPOSE 8080

# Start the server
CMD ["julia", "/app/server.jl"]
