import os
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_limiter import Limiter
import openai
from openai.embeddings_utils import cosine_similarity, get_embedding

load_dotenv()


def load_embeddings(embedding_by_chapter_csv, embedding_by_verse_csv):
    embeddings_by_chapter = pd.read_csv(embedding_by_chapter_csv)
    embeddings_by_verse = pd.read_csv(embedding_by_verse_csv)

    embeddings_by_chapter['embedding'] = embeddings_by_chapter['embedding'].apply(
        eval).apply(np.array)
    embeddings_by_chapter['location'] = embeddings_by_chapter['Book'] + \
        " " + embeddings_by_chapter['Chapter'].astype(str)
    embeddings_by_chapter['verse'] = embeddings_by_chapter['Verses']

    embeddings_by_verse['embedding'] = embeddings_by_verse['embedding'].apply(
        eval).apply(np.array)

    return embeddings_by_chapter, embeddings_by_verse


def find_similarities(query_params, embeddings_df):
    search_term = query_params["query"]
    search_term_vector = get_embedding(
        search_term, engine="text-embedding-ada-002")
    embeddings_df['similarities'] = embeddings_df['embedding'].apply(
        lambda x: cosine_similarity(x, search_term_vector))

    return embeddings_df.nlargest(50, 'similarities').drop(columns=['embedding'])


app = Flask(__name__)
limiter = Limiter(app)

embeddings_by_chapter, embeddings_by_verse = load_embeddings(
    "embeddings/chapter/KJV_Bible_Embeddings_by_Chapter.csv", "embeddings/verse/KJV_Bible_Embeddings.csv")


@app.route('/', methods=['GET'])
@limiter.limit("1/second")
def index():
    return jsonify({"message": "Hello World"})


@app.route('/search', methods=['GET'])
@limiter.limit("1/second")
def search():
    search_by = request.args.get("search_by")
    query = request.args.get("query")

    if search_by and query:
        embeddings_df = embeddings_by_chapter if search_by == "chapter" else embeddings_by_verse

        api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = api_key
        found_df = find_similarities(request.args, embeddings_df)

        json_array = [
            {"index": i - 1, "location": row["location"],
                "verse": row["verse"], "similarities": row["similarities"]}
            for i, row in found_df.iterrows()
        ]

        return jsonify(json_array)
    else:
        return "Missing query parameters 'search_by' and 'query'", 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
