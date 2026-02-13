#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 11:19:06 2023

@author: ariel


'Dynamical incompatibilities in paced finger tapping experiments'
Silva, González, & Laje (2026)

"""

#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

datafolder = '../data/'


#%%










#%% Load data
params_comb_df = pd.read_csv(datafolder + 'params_comb_df.csv', index_col=0).reset_index(drop=True)
params_pure2_df = pd.read_csv(datafolder + 'params_pure2_df.csv', index_col=0).reset_index(drop=True)


#%%










#%% FUNCIÓN AUXILIAR
def bin_overlap_percentage(A, B, bins):
    """Calcula el porcentaje de datos en bines comunes entre A y B."""
    A_bins = np.digitize(A, bins, right=False)
    B_bins = np.digitize(B, bins, right=False)

    A_grouped = {i: A[A_bins == i] for i in range(1, len(bins))}
    B_grouped = {i: B[B_bins == i] for i in range(1, len(bins))}

    common_bins = [i for i in A_grouped if len(A_grouped[i]) > 0 and len(B_grouped[i]) > 0]

    total_common = sum(len(A_grouped[i]) + len(B_grouped[i]) for i in common_bins)
    total_data = len(A) + len(B)

    return (total_common / total_data) * 100


#%%










#%% Datos
# params_pure2_aux_df = params_pure2_df[(params_pure2_df['perturb_type_x_subpopulation'] == 'SC1')].reset_index(drop=True)
params_pure2_aux_df = params_pure2_df[(params_pure2_df['perturb_type_x_subpopulation'] == 'SC')].reset_index(drop=True)
A = params_pure2_aux_df['a'].to_numpy()

# params_pure2_aux_df = params_pure2_df[(params_pure2_df['perturb_type_x_subpopulation'] == 'PS0')].reset_index(drop=True)
params_pure2_aux_df = params_pure2_df[(params_pure2_df['perturb_type_x_subpopulation'] == 'PS')].reset_index(drop=True)
B = params_pure2_aux_df['a'].to_numpy()

# Número de bines
num_bins = 30

# Determinamos el rango común de valores
min_val = min(A.min(), B.min())
max_val = max(A.max(), B.max())
bins = np.linspace(min_val, max_val, num_bins + 1)

# CÁLCULO OBSERVADO
observed = bin_overlap_percentage(A, B, bins)
print(f"Porcentaje observado: {observed:.2f}%")


#%% Datos
# params_pure2_aux_df = params_pure2_df[(params_pure2_df['perturb_type_x_subpopulation'] == 'SC1')].reset_index(drop=True)
params_pure2_aux_df = params_pure2_df[(params_pure2_df['perturb_type_x_subpopulation'] == 'SC')].reset_index(drop=True)
A = params_pure2_aux_df['dseta1'].to_numpy()

# params_pure2_aux_df = params_pure2_df[(params_pure2_df['perturb_type_x_subpopulation'] == 'PS0')].reset_index(drop=True)
params_pure2_aux_df = params_pure2_df[(params_pure2_df['perturb_type_x_subpopulation'] == 'PS')].reset_index(drop=True)
B = params_pure2_aux_df['dseta1'].to_numpy()

# Número de bines
num_bins = 30

# Determinamos el rango común de valores
min_val = min(A.min(), B.min())
max_val = max(A.max(), B.max())
bins = np.linspace(min_val, max_val, num_bins + 1)

# CÁLCULO OBSERVADO
observed = bin_overlap_percentage(A, B, bins)
print(f"Porcentaje observado: {observed:.2f}%")


#%%










#%% Datos
# params_comb_aux_df = params_comb_df[(params_comb_df['perturb_type_x_subpopulation'] == 'SC0')].reset_index(drop=True)
params_comb_aux_df = params_comb_df[(params_comb_df['perturb_type_x_subpopulation'] == 'SC')].reset_index(drop=True)
A = params_comb_aux_df['a'].to_numpy()

# params_comb_aux_df = params_comb_df[(params_comb_df['perturb_type_x_subpopulation'] == 'PS0')].reset_index(drop=True)
params_comb_aux_df = params_comb_df[(params_comb_df['perturb_type_x_subpopulation'] == 'PS')].reset_index(drop=True)
B = params_comb_aux_df['a'].to_numpy()

# Número de bines
num_bins = 30

# Determinamos el rango común de valores
min_val = min(A.min(), B.min())
max_val = max(A.max(), B.max())
bins = np.linspace(min_val, max_val, num_bins + 1)

# CÁLCULO OBSERVADO
observed = bin_overlap_percentage(A, B, bins)
print(f"Porcentaje observado: {observed:.2f}%")


#%% Datos
# params_comb_aux_df = params_comb_df[(params_comb_df['perturb_type_x_subpopulation'] == 'SC0')].reset_index(drop=True)
params_comb_aux_df = params_comb_df[(params_comb_df['perturb_type_x_subpopulation'] == 'SC')].reset_index(drop=True)
A = params_comb_aux_df['b'].to_numpy()

# params_comb_aux_df = params_comb_df[(params_comb_df['perturb_type_x_subpopulation'] == 'PS0')].reset_index(drop=True)
params_comb_aux_df = params_comb_df[(params_comb_df['perturb_type_x_subpopulation'] == 'PS')].reset_index(drop=True)
B = params_comb_aux_df['b'].to_numpy()

# Número de bines
num_bins = 30

# Determinamos el rango común de valores
min_val = min(A.min(), B.min())
max_val = max(A.max(), B.max())
bins = np.linspace(min_val, max_val, num_bins + 1)

# CÁLCULO OBSERVADO
observed = bin_overlap_percentage(A, B, bins)
print(f"Porcentaje observado: {observed:.2f}%")


#%% Datos
# params_comb_aux_df = params_comb_df[(params_comb_df['perturb_type_x_subpopulation'] == 'SC0')].reset_index(drop=True)
params_comb_aux_df = params_comb_df[(params_comb_df['perturb_type_x_subpopulation'] == 'SC')].reset_index(drop=True)
A = params_comb_aux_df['c'].to_numpy()

# params_comb_aux_df = params_comb_df[(params_comb_df['perturb_type_x_subpopulation'] == 'PS0')].reset_index(drop=True)
params_comb_aux_df = params_comb_df[(params_comb_df['perturb_type_x_subpopulation'] == 'PS')].reset_index(drop=True)
B = params_comb_aux_df['c'].to_numpy()

# Número de bines
num_bins = 30

# Determinamos el rango común de valores
min_val = min(A.min(), B.min())
max_val = max(A.max(), B.max())
bins = np.linspace(min_val, max_val, num_bins + 1)

# CÁLCULO OBSERVADO
observed = bin_overlap_percentage(A, B, bins)
print(f"Porcentaje observado: {observed:.2f}%")


#%% Datos
# params_comb_aux_df = params_comb_df[(params_comb_df['perturb_type_x_subpopulation'] == 'SC0')].reset_index(drop=True)
params_comb_aux_df = params_comb_df[(params_comb_df['perturb_type_x_subpopulation'] == 'SC')].reset_index(drop=True)
A = params_comb_aux_df['d'].to_numpy()

# params_comb_aux_df = params_comb_df[(params_comb_df['perturb_type_x_subpopulation'] == 'PS0')].reset_index(drop=True)
params_comb_aux_df = params_comb_df[(params_comb_df['perturb_type_x_subpopulation'] == 'PS')].reset_index(drop=True)
B = params_comb_aux_df['d'].to_numpy()

# Número de bines
num_bins = 30

# Determinamos el rango común de valores
min_val = min(A.min(), B.min())
max_val = max(A.max(), B.max())
bins = np.linspace(min_val, max_val, num_bins + 1)

# CÁLCULO OBSERVADO
observed = bin_overlap_percentage(A, B, bins)
print(f"Porcentaje observado: {observed:.2f}%")


#%% Datos
params_comb_aux_df = params_comb_df[(params_comb_df['perturb_type_x_subpopulation'] == 'SC')].reset_index(drop=True)
A = params_comb_aux_df['dseta1'].to_numpy()

params_comb_aux_df = params_comb_df[(params_comb_df['perturb_type_x_subpopulation'] == 'PS')].reset_index(drop=True)
B = params_comb_aux_df['dseta1'].to_numpy()

# Número de bines
num_bins = 30

# Determinamos el rango común de valores
min_val = min(A.min(), B.min())
max_val = max(A.max(), B.max())
bins = np.linspace(min_val, max_val, num_bins + 1)

# CÁLCULO OBSERVADO
observed = bin_overlap_percentage(A, B, bins)
print(f"Porcentaje observado: {observed:.2f}%")


#%% Datos
params_comb_aux_df = params_comb_df[(params_comb_df['perturb_type_x_subpopulation'] == 'SC')].reset_index(drop=True)
A = params_comb_aux_df['alpha2'].to_numpy()

params_comb_aux_df = params_comb_df[(params_comb_df['perturb_type_x_subpopulation'] == 'PS')].reset_index(drop=True)
B = params_comb_aux_df['alpha2'].to_numpy()

# Número de bines
num_bins = 30

# Determinamos el rango común de valores
min_val = min(A.min(), B.min())
max_val = max(A.max(), B.max())
bins = np.linspace(min_val, max_val, num_bins + 1)

# CÁLCULO OBSERVADO
observed = bin_overlap_percentage(A, B, bins)
print(f"Porcentaje observado: {observed:.2f}%")


#%%

