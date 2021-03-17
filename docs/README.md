# Building Readthedocs-style documentation locally

These instructions assume a Python 3.7 virtualenv in which `snc` has already been installed (see instructions for installation in `../README.md`)

To compile and view the documentation locally:

1) Change to this directory (e.g. `cd docs` if you are in the Git repository's base directory). 

2) Install docs dependencies:

   `pip install -r docs_requirements.txt`
   
   If pandoc does not install via pip, or step 3) fails with a 'Pandoc' error, download and install Pandoc separately from `https://github.com/jgm/pandoc/releases/` (e.g. `pandoc-<version>-amd64.deb` for Ubuntu), and try running step 2) again.

3) Compile the documentation:

   `make html`

4) Run a web server:

   `python -m http.server`

5) Check documentation locally by opening (in a browser):

   http://localhost:8000/_build/html/
   
If you want to create or edit a Jupyter Notebook for inclusion in this doc set, see `notebooks/README.md`.
