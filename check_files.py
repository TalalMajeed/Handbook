"""
check_files.py
Run this from your project root:  python check_files.py
It tells you exactly which files are outdated and what to copy.
"""
import re, sys, os

OK   = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

root      = os.path.dirname(os.path.abspath(__file__))
iface_dir = os.path.join(root, "interface")

checks = [
    # (filepath, search_string, description)
    (
        os.path.join(root, "pipeline.py"),
        "score_tfidf",
        "pipeline.py — individual score fields in retrieve()"
    ),
    (
        os.path.join(root, "pipeline.py"),
        "q_sig = self.minhaser.signature(cleaned)",
        "pipeline.py — MinHash Jaccard computed in hybrid mode"
    ),
    (
        os.path.join(root, "server.py"),
        "score_tfidf",
        "server.py — score fields passed through format_chunk_for_client()"
    ),
    (
        os.path.join(iface_dir, "logic.js"),
        "score_tfidf",
        "interface/logic.js — score fields read in openModal()"
    ),
    (
        os.path.join(iface_dir, "logic.js"),
        "scoreBar",
        "interface/logic.js — scoreBar() function present"
    ),
    (
        os.path.join(iface_dir, "logic.js"),
        "Score Breakdown",
        "interface/logic.js — breakdown panel rendered"
    ),
    (
        os.path.join(iface_dir, "logic.js"),
        "MinHash Jaccard",
        "interface/logic.js — MinHash row rendered"
    ),
]

all_ok = True
print("\nFile diagnostic for Handbook QA score breakdown\n" + "="*52)

for filepath, needle, desc in checks:
    if not os.path.exists(filepath):
        print(f"{FAIL}  FILE NOT FOUND: {filepath}")
        all_ok = False
        continue
    with open(filepath, encoding="utf-8", errors="replace") as f:
        content = f.read()
    if needle in content:
        print(f"{OK}  {desc}")
    else:
        print(f"{FAIL}  MISSING — {desc}")
        print(f"       → File exists but does NOT contain: '{needle}'")
        print(f"       → Replace: {filepath}")
        all_ok = False

print("="*52)
if all_ok:
    print("\n✓ All files are up to date. If scores still don't show:")
    print("  1. Hard-refresh the browser:  Ctrl+Shift+R  (Windows)")
    print("  2. Open DevTools (F12) → Network tab → check 'Disable cache'")
    print("  3. Ask a question and click a chunk — check the modal")
else:
    print("\n✗ Some files are outdated. Copy the files marked above")
    print("  from your downloaded outputs into the correct locations.")
    print("\n  Correct locations:")
    print(f"  pipeline.py  →  {os.path.join(root, 'pipeline.py')}")
    print(f"  server.py    →  {os.path.join(root, 'server.py')}")
    print(f"  logic.js     →  {os.path.join(iface_dir, 'logic.js')}")