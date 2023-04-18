using Genie, Genie.Router, Genie.Requests, Genie.Responses
using CSV, DataFrames, JSON, Distances, OpenAI, LinearAlgebra, DotEnv, Revise

DotEnv.config()

function load_embeddings(embeddingByChapterCSV, embeddingByVerseCSV)
    embeddingsByChapter = DataFrame(CSV.File(embeddingByChapterCSV))
    embeddingsByVerse = DataFrame(CSV.File(embeddingByVerseCSV))

    embeddingsByChapter.embedding = Vector{Float64}.(JSON.parse.(embeddingsByChapter.embedding))
    embeddingsByChapter.location = [embeddingsByChapter.Book[i] * " " * string(embeddingsByChapter.Chapter[i]) for i in 1:length(embeddingsByChapter.Book)]
    embeddingsByChapter.verse = [embeddingsByChapter.Verses[i] for i in 1:length(embeddingsByChapter.Book)]

    embeddingsByVerse.embedding = Vector{Float64}.(JSON.parse.(embeddingsByVerse.embedding))

    return embeddingsByChapter, embeddingsByVerse
end

function find_similarities(api_key, query, embeddingsDF)
    search_term_vector = Vector{Float64}(create_embeddings(api_key, query, engine="text-embedding-ada-002").response.data[1].embedding)

    embeddingsDF.similarities = [cosine_similarity(x[1], search_term_vector) for x in eachrow(embeddingsDF.embedding)]

    # Sort and get the top 5 results
    return sort!(embeddingsDF, :similarities, rev=true)[1:50, Not([:embedding])]
end

function cosine_similarity(a, b)
    return dot(a, b) / (norm(a) * norm(b))
end

# Route handlers
function index()
    return json(Dict("message" => "Hello World"))
end

function search()
    println(params())
    if haskey(params(), :search_by) && haskey(params(), :query)
        search_by = params(:search_by)
        query = params(:query)
    else
        return json(Dict("error" => "Missing query parameters 'search_by' and 'query'"))
    end

    println(query)
    println(search_by)
    try
        println("Trying to find similarities")
        embeddingsDF = search_by == "chapter" ? embeddingsByChapter : embeddingsByVerse

        foundDF = find_similarities(api_key, query, embeddingsDF)
        println(foundDF)

        json_array = [
            Dict("index" => i - 1, "location" => foundDF[!, "location"][i], "verse" => foundDF[!, "verse"][i], "similarities" => foundDF[!, "similarities"][i])
            for i in eachindex(foundDF[!, "location"])
        ]

        println("json_array: ", json_array)
        println("typeof(json_array): ", typeof(json_array))

        return json(json_array)
    catch e
        return json(Dict("error" => "Not Found: $e"))
    end
end

api_key = ENV["OPENAI_API_KEY"]
embeddingsByChapter, embeddingsByVerse = load_embeddings("embeddings/chapter/KJV_Bible_Embeddings_by_Chapter.csv", "embeddings/verse/KJV_Bible_Embeddings.csv")

function app()


    # Route registration
    route("/", index, method=GET)
    route("/search", search, method=GET)

    up(port=8000, host="0.0.0.0")
end