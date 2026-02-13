# Incompatibilities2026
Data and code to reproduce Silva, González & Laje 2026, "Dynamical incompatibilities in paced finger tapping experiments"


### Execution

1. `script1_load_and_fit_data.py`  
Run this script once for every condition:
	- PSpure, separate fitting
	- SCpure, separate fitting
	- PScomb, separate fitting
	- SCcomb, separate fitting
	- PSpure-SCpure, joint fitting
	- PScomb-SCcomb, joint fitting

2. `script2_filter_divergent_solutions.py`

3. `script3_subpopulation_analysis_and_figures.py`

4. `script4_compute_overlap.py`

5. `script5_mann_whitney.py`


### Dependencies

python=3.10.19
numpy=1.26.4
scipy=1.15.2
pandas=2.3.3
matplotlib=3.10.8
plotnine=0.12.3
patchworklib=0.6.3


