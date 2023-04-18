docker build -t genie .
docker tag genie:latest dpshade22/genie:julia
docker push dpshade22/genie:julia
docker run -p 8080:8080 dpshade22/genie:julia