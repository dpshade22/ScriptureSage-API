using CSV, DataFrames, HTTP, URIs, JSON, Distances, OpenAI, LinearAlgebra, DotEnv
DotEnv.config()

api_key = ENV["OPENAI_API_KEY"]

function find_similarities(api_key, num_search_terms, query_params, embeddingsDF)
    search_term = query_params["query"]
    search_term_vector = Vector{Float64}(create_embeddings(api_key, search_term, engine="text-embedding-ada-002").response.data[1].embedding)

    if num_search_terms == 2
        search_term_2 = query_params["query_2"]
        # Get the embeddings for the second search term
        search_term_vector_2 = JSON.parse(create_embeddings(api_key, search_term_2, engine="text-embedding-ada-002"))
        # Calculate the combined embeddings and similarities
        embeddingsDF.similarities = cosine_similarity.(eachrow(embeddingsDF.embedding), combined_vector)
    else
        embeddingsDF.similarities = [cosine_similarity(x[1], search_term_vector) for x in eachrow(embeddingsDF.embedding)]
    end

    # Sort and get the top 5 results
    return sort(embeddingsDF, :similarities, rev=true)[1:5, Not([:embedding, :Column1])]
end

function load_embeddings(embeddingByChapterCSV, embeddingByVerseNTCSV)
    embeddingsByChapter = CSV.read(embeddingByChapterCSV, DataFrame)
    embeddingsByChapter.embedding = [Vector{Float64}(JSON.parse(x)) for x in embeddingsByChapter.embedding]
    embeddingsByChapter.location = [embeddingsByChapter.Book[i] * " " * string(embeddingsByChapter.Chapter[i]) for i in 1:length(embeddingsByChapter.Book)]
    embeddingsByChapter.verse = [embeddingsByChapter.Verses[i] for i in 1:length(embeddingsByChapter.Book)]

    embeddingsByVerseNT = CSV.read(embeddingByVerseNTCSV, DataFrame)
    embeddingsByVerseNT.embedding = [Vector{Float64}(JSON.parse(x)) for x in embeddingsByVerseNT.embedding]
    return embeddingsByChapter, embeddingsByVerseNT
end

function cosine_similarity(a, b)
    return dot(a, b) / (norm(a) * norm(b))
end

# Define a function to handle incoming HTTP requests
function server(req)
    # Extract the request URI
    uri = req.target
    parsed_uri = URIs.URI(uri)

    # Set the OpenAI API key
    embeddingsByChapter, embeddingsByVerseNT = load_embeddings("embeddings/chapter/KJV_Bible_Embeddings_by_Chapter.csv", "embeddings/verse/nt/KJV_NT_Bible_Embeddings_by_Verse.csv")

    # Handle the /hello endpoint
    if uri == "/hello"
        # Respond with a greeting
        return HTTP.Response(200, "Hello, welcome to the API!")
    end

    # Handle the /add endpoint
    if startswith(parsed_uri.path, "/search")
        query_params = URIs.queryparams(parsed_uri.query)

        # Check if the required query parameters are present
        if haskey(query_params, "search_by") && haskey(query_params, "query")
            # Parse the query parameters as integers
            search_terms = haskey(query_params, "query_2") ? 2 : 1
            embeddingsDF = query_params["search_by"] == "chapter" ? embeddingsByChapter : embeddingsByVerseNT
        else
            # Respond with an error message if the query parameters are missing
            return HTTP.Response(400, "Missing query parameters 'search_by' and 'query'")
        end

        foundDF = find_similarities(api_key, search_terms, query_params, embeddingsDF)

        json_array = []
        for i in eachindex(foundDF[!, "location"])
            println(foundDF[!, "location"][i])
            push!(json_array, Dict("index" => i - 1, "location" => foundDF[!, "location"][i], "verse" => foundDF[!, "verse"][i], "similarities" => foundDF[!, "similarities"][i]))
        end

        # Convert the found data to a JSON payload and return it
        return HTTP.Response(200, JSON.json(json_array))
    end

    # Respond with a 404 Not Found status for unrecognized endpoints
    return HTTP.Response(404, "Not Found")
end

# Start the HTTP server on port 8080
HTTP.serve(server, host="127.0.0.1", port=8080)