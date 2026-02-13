#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 13:30:14 2025

@author: ariel


'Dynamical incompatibilities in paced finger tapping experiments'
Silva, González, & Laje (2026)

"""

#%%

import pandas as pd
import numpy as np
from scipy import stats

datafolder = '../data/'


#%%










#%% Cargar el archivo CSV
df = pd.read_csv(datafolder + "params_PSSCpurecomb_df.csv")


#%% Filtrar los dos grupos según perturb_type_x_context
grupo_pure = df.loc[df["perturb_type_x_context"] == "PSSCpure", "dist"].dropna()
grupo_comb = df.loc[df["perturb_type_x_context"] == "PSSCcomb", "dist"].dropna()

# Verificar tamaños
print(f"Tamaño grupo PSSCpure: {len(grupo_pure)}")
print(f"Tamaño grupo PSSCcomb: {len(grupo_comb)}")


#%%

# Test de Mann–Whitney U (dos colas)
u_stat, p_value = stats.mannwhitneyu(grupo_pure, grupo_comb, alternative='two-sided')

# Calcular Cliff’s delta (tamaño de efecto)
n1, n2 = len(grupo_pure), len(grupo_comb)
cliffs_delta = (2*u_stat)/(n1*n2) - 1

print("\n--- Resultados Mann–Whitney U ---")
print(f"U = {u_stat:.3f}")
print(f"p-value = {p_value:.5f}")
print(f"Cliff’s delta = {cliffs_delta:.3f}")


#%% Interpretación rápida
alpha = 0.05
if p_value <= alpha:
    print("\nConclusión: se RECHAZA H0 — las distribuciones de 'dist' difieren entre PSSCpure y PSSCcomb.")
else:
    print("\nConclusión: no hay evidencia suficiente para rechazar H0 — las distribuciones parecen similares.")
	

#%%










#%% Delta bootstrap
def cliffs_delta_bootstrap(x, y, n_boot=10000, alpha=0.05, seed=None):
    rng = np.random.default_rng(seed)
    deltas = []
    n1, n2 = len(x), len(y)
    for _ in range(n_boot):
        xb = rng.choice(x, size=n1, replace=True)
        yb = rng.choice(y, size=n2, replace=True)
        u, _ = stats.mannwhitneyu(xb, yb, alternative='two-sided')
        delta = (2*u)/(n1*n2) - 1
        deltas.append(delta)
    low = np.percentile(deltas, 100*alpha/2)
    high = np.percentile(deltas, 100*(1-alpha/2))
    return np.mean(deltas), (low, high)

mean_delta, ci_delta = cliffs_delta_bootstrap(grupo_pure, grupo_comb, n_boot=10000)
print(f"\nCliff's delta bootstrap mean = {mean_delta:.3f}, IC95% = [{ci_delta[0]:.3f}, {ci_delta[1]:.3f}]")


#%%

