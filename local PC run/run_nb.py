"""
Run a Jupytext percent-format .py notebook as a plain Python script.
Strips # %% markers and executes the rest.
"""
import sys
import re
from pathlib import Path

nb_path = Path(sys.argv[1])
src = nb_path.read_text(encoding="utf-8")

# Remove jupytext header (--- ... ---)
src = re.sub(r'^# ---\n.*?# ---\n', '', src, flags=re.DOTALL)

# Remove cell markers: # %% [markdown] blocks (keep as comments) and # %% code markers
# Convert markdown cells to pass (they're just comments anyway)
lines = src.splitlines()
out = []
in_markdown = False
for line in lines:
    if re.match(r'^# %% \[markdown\]', line):
        in_markdown = True
        continue
    elif re.match(r'^# %%', line):
        in_markdown = False
        continue
    out.append(line)

clean = "\n".join(out)
g = {"__file__": str(nb_path), "__name__": "__main__"}
exec(compile(clean, str(nb_path), 'exec'), g)
