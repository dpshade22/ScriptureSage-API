import pandas as pd
import numpy as np
import openai

from openai.embeddings_utils import cosine_similarity, get_embedding

openai.api_key = "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"


df = pd.read_csv("embeddings/chapter/KJV_Bible_Embeddings_by_Chapter.csv")
df["embedding"] = df["embedding"].apply(eval).apply(np.array)


while True:
    search_term = input("Enter a search term: ")
    search_term_vector = get_embedding(search_term, engine="text-embedding-ada-002")

    # search_term_2 = input("Enter a second search term: ")
    # search_term_vector_2 = get_embedding(search_term, engine="text-embedding-ada-002")

    df["similarities"] = df["embedding"].apply(
        lambda x: cosine_similarity(
            x, search_term_vector
        )
    )

    foundDF = df.sort_values("similarities", ascending=False).head(5)

    print(foundDF[["Book", "Chapter", "similarities"]])
