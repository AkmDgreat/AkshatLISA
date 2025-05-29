# Removes the print statements from the Jupyter notebooks, but keeps the plots 
# Useful because some notebooks had lots of print statements 

import sys, glob, nbformat, os

notebooks_directory = os.path.join(os.getcwd(), "..", "notebooks")

def strip_file(fname):
    nb = nbformat.read(fname, as_version=4)
    for cell in nb.cells:
        if cell.get("outputs"):
            cell["outputs"] = [
                out for out in cell["outputs"]
                if out.get("output_type") != "stream"
            ]
    nbformat.write(nb, fname)
    print(f"✔️ Stripped stream outputs from {fname}")

if len(sys.argv) == 2:
    # single‐file mode
    strip_file(os.path.join(notebooks_directory, sys.argv[1]))
else:
    # no‐arg mode: do them all
    for path in glob.glob(os.path.join(notebooks_directory, "*.ipynb")):
        strip_file(path)