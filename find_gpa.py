import pickle

def find_gpa_chunks():
    with open('handbook-1.0/chunks.pkl', 'rb') as f:
        chunks = pickle.load(f)
        
    print(f"Total chunks: {len(chunks)}")
    
    matches = []
    for i, c in enumerate(chunks):
        text = c.get('text', '').lower()
        if 'cgpa' in text or 'gpa' in text:
            if 'minimum' in text or 'requirement' in text:
                matches.append((i, c))
                
    print(f"Found {len(matches)} chunks mentioning GPA/CGPA and minimum/requirement")
    for i, (idx, c) in enumerate(matches):
        print(f"\n--- MATCH {i+1} (idx {idx}, page {c.get('page')}, source {c.get('source')}) ---")
        print(c.get('text')[:300])

if __name__ == "__main__":
    find_gpa_chunks()
