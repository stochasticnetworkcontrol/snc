## Guaranteed Service Model

### Overview

Current directory has an implementation of GSM inventory optimization algorithm
based on a collection of papers:
* Graves and Willems, 2000
* Humair and Willems, 2006
* Schoenmeyr and Graves, 2016

### Examples of use

Two Jupyter notebooks using most of GSM functionality can be found in notebooks folder under GSM category:
* Demonstration of cascading stockouts
* Numerical replication of experiments from GSM with capacity constraints paper (Schoenmeyr and Graves, 2016)

### Datasets

GSM algorithms can be run on the supply chain dataset described in 
[[1](https://seanwillems.com/wp-content/uploads/2020/11/Willems_MSOM_v10n1_Winter2008.pdf)],
which can be downloaded [here](https://pubsonline.informs.org/doi/suppl/10.1287/msom.1070.0176). 

Once the dataset has been downloaded, it has to be converted to csv, which can be done with the script at `snc/sandbox/meio/gsm/convert_excel_to_csv.py`.

[[1](https://seanwillems.com/wp-content/uploads/2020/11/Willems_MSOM_v10n1_Winter2008.pdf)] Willems, S. P., “Data Set:  Real-World Multi-Echelon Supply Chains Used for Inventory Optimization,” Manufacturing & Service Operations Management, Winter 2008, Vol. 10, No. 1, pp. 19-23.
