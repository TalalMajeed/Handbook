import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pickle

with open('handbook-1.0/chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)

manual_chunks = [c for c in chunks if c.get('source') == 'manual.pdf']
print(f'Total chunks from manual.pdf: {len(manual_chunks)}')
for i, c in enumerate(manual_chunks):
    print(f'\n--- manual chunk {i+1} (page {c["page"]}) ---')
    print(c['text'][:400])
