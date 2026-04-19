from pipeline import HandbookPipeline

def test_query():
    pl = HandbookPipeline("handbook-1.0")
    docs = pl.retrieve("What is the minimum GPA requirement?")
    print("Top 3 retrieved chunks for 'What is the minimum GPA requirement?':\n")
    for i, d in enumerate(docs[:3]):
        print(f"[{i+1}] Score: {d['score']:.4f} | Source: {d['source']} | Page: {d['page']}")
        print(f"Preview: {d['text'][:250]}...\n")

if __name__ == "__main__":
    test_query()
