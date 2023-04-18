docker build -t scripture-sage-api .
docker tag scripture-sage-api:latest dpshade22/scripture-sage-api:julia
docker push dpshade22/scripture-sage-api:julia
docker run -p 8080:8080 dpshade22/scripture-sage-api:julia