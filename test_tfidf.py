from pipeline import HandbookPipeline
from sklearn.metrics.pairwise import cosine_similarity
import sys

def test_tfidf():
    pl = HandbookPipeline("handbook-1.0")
    query_vec = pl.tfidf_vectorizer.transform(["What is the minimum GPA requirement?"])
    cosine_sim = cosine_similarity(query_vec, pl.tfidf_matrix).flatten()
    
    # Get top 5 by TF-IDF alone:
    top_indices = cosine_sim.argsort()[::-1][:5]
    print("Top 5 by pure TF-IDF cosine similarity:")
    for i in top_indices:
        print(f"IDX {i}: Score {cosine_sim[i]:.4f} | Page: {pl.chunks[i].get('page')}")
        print(pl.chunks[i]["text"][:150], "\n")

if __name__ == "__main__":
    test_tfidf()
