# strip_prints.py
import sys, nbformat

fname = sys.argv[1]
nb    = nbformat.read(fname, as_version=4)
for cell in nb.cells:
    if cell.get('outputs'):
        cell['outputs'] = [
            out for out in cell['outputs']
            if out.get('output_type') != 'stream'
        ]
nbformat.write(nb, fname)
print(f"✔️ Stripped stream outputs from {fname}")