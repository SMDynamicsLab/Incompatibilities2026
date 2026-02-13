#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday Apr 28 9:20:00 2025

@author: ariel


'Dynamical incompatibilities in paced finger tapping experiments'
Silva, González, & Laje (2026)

"""

#%%

import pandas as pd
import numpy as np
from plotnine import *
from plotnine import themes

import patchworklib as pw

import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import rgb2hex
from matplotlib.ticker import MaxNLocator

datafolder = '../data/'


#%%










#%% Function definitions and Model

def iqr(lst):
	Q1 = np.percentile(lst, 25, interpolation = 'midpoint') 
	Q2 = np.percentile(lst, 50, interpolation = 'midpoint') 
	Q3 = np.percentile(lst, 75, interpolation = 'midpoint') 
	IQR = Q3 - Q1
	low_lim = Q1 - 1.5 * IQR
	up_lim = Q3 + 1.5 * IQR
	return low_lim, up_lim


def model(y, t, params, baseline):
	# p = predicted asynchrony
	# x = auxiliary variable
	# t = stimulus period T(n)
	# s = auxiliary variable
	p, x, s = y
	a,b,c,d,alpha1,beta1,gamma1,delta1,eps1,dseta1,eta1, \
		alpha2,beta2,gamma2,delta2,eps2,dseta2,eta2 = params

	DeltaT = t - s 	# DeltaT = T_n - T_{n-1} = T_n - s_n
	e = p - DeltaT 	# observed asynchrony = predicted asynchrony - DeltaT

	f = (                              # Esto es p n+1
		a*(e-baseline) + b*(x-t)
		+ alpha1*(e-baseline)**2
		+ beta1*(e-baseline)*(x-t)
		+ gamma1*(x-t)**2
		+ delta1*(e-baseline)**3
		+ eps1*(e-baseline)**2*(x-t)
		+ dseta1*(e-baseline)*(x-t)**2
		+ eta1*(x-t)**3
		+ baseline
		)
	g = (                              # Esto es x n+1
		c*(e-baseline) + d*(x-t)
		+ alpha2*(e-baseline)**2
		+ beta2*(e-baseline)*(x-t)
		+ gamma2*(x-t)**2
		+ delta2*(e-baseline)**3
		+ eps2*(e-baseline)**2*(x-t)
		+ dseta2*(e-baseline)*(x-t)**2
		+ eta2*(x-t)**3
		+ t
		)
	h = t
	return np.array([f, g, h])


# Wrapper
def integrate(model, y_ini, params, hyper_params):
	baseline_pre, baseline_post, perturb_type, perturb_size = hyper_params

	# create sequence of stimulus periods
	t = np.repeat(ISI,n_beeps)	# no perturbation (isochronous)
	if perturb_type == 'SC':	# step change
		t[perturb_beep-1:] = np.repeat(ISI+perturb_size, n_beeps-perturb_beep+1)
	elif perturb_type == 'PS':	# phase shift
		t[perturb_beep-1] = ISI+perturb_size

	baseline = np.concatenate([np.repeat(baseline_pre,perturb_beep-1), np.repeat(baseline_post, n_beeps-perturb_beep+1)])
	# number of variables
	n_vars = len(y_ini)
	y = np.empty((n_beeps, n_vars))
	#y[:] = np.NaN "removed in the Numpy 2.0 release"
	y[:] = np.nan
	# initial condition should be the first value returned
	y[0,:] = y_ini
	for i in range(0, n_beeps-1):
		# take a step
		y[i+1,:] = model(y[i,:], t[i], params, baseline[i])  # Por qué envía los 18 beeps si sólo necesita el valor del siguiente?
	p = y[:,0]
	s = y[:,2]
	DeltaT = t - s
	asyn = p - DeltaT
	return [t, asyn, y]


#%%










#%% Load data
params_SCpure_df = pd.read_csv(datafolder + 'params_SCpure_df.csv', index_col=0).reset_index(drop=True)
data3_fit_SCpure_df = pd.read_csv(datafolder + 'data3_fit_SCpure_df.csv', index_col=0).reset_index(drop=True)
params_PSpure_df = pd.read_csv(datafolder + 'params_PSpure_df.csv', index_col=0).reset_index(drop=True)
data3_fit_PSpure_df = pd.read_csv(datafolder + 'data3_fit_PSpure_df.csv', index_col=0).reset_index(drop=True)
data3_fit_PSSCpure_df = pd.concat([data3_fit_SCpure_df, data3_fit_PSpure_df], axis=0).reset_index(drop=True)
#data3_fit_PSSCpure_df.to_csv('data3_fit_PSSCpure_df.csv')

params_PSSCcomb_df = pd.read_csv(datafolder + 'params_PSSCcomb_df.csv', index_col=0).reset_index(drop=True)
data3_fit_SCcomb_df = pd.read_csv(datafolder + 'data3_fit_SCcomb_df.csv', index_col=0).reset_index(drop=True)
data3_fit_PScomb_df = pd.read_csv(datafolder + 'data3_fit_PScomb_df.csv', index_col=0).reset_index(drop=True)
data3_fit_PSSCcomb_df = pd.concat([data3_fit_SCcomb_df, data3_fit_PScomb_df], axis=0).reset_index(drop=True)
#data3_fit_PSSCcomb_df.to_csv('data3_fit_PSSCcomb_df.csv')

params_SCcomb_df = pd.read_csv(datafolder + 'params_SCcomb_df.csv', index_col=0).reset_index(drop=True)
params_PScomb_df = pd.read_csv(datafolder + 'params_PScomb_df.csv', index_col=0).reset_index(drop=True)

params_PSSCpure_df = pd.read_csv(datafolder + 'params_PSSCpure_df.csv', index_col=0).reset_index(drop=True)
data3_fit_PSSCpure_df = pd.concat([data3_fit_SCpure_df, data3_fit_PSpure_df], axis=0).reset_index(drop=True)
#data3_fit_PSSCpure_df.to_csv('data3_fit_PSSCpure_df.csv')

params_SCpure_df = params_SCpure_df[(params_SCpure_df['divergent']==False)].reset_index(drop=True)
params_SCpure_df = params_SCpure_df.drop(['divergent'], axis=1)
#params_SCpure_df.to_csv('params_SCpure_df.csv')

params_PSpure_df = params_PSpure_df[(params_PSpure_df['divergent']==False)].reset_index(drop=True)
params_PSpure_df = params_PSpure_df.drop(['divergent'], axis=1)
#params_PSpure_df.to_csv('params_PSpure_df.csv')

params_PSSCpure_df = params_PSSCpure_df[(params_PSSCpure_df['divergent']==False)].reset_index(drop=True)
params_PSSCpure_df = params_PSSCpure_df.drop(['divergent'], axis=1)
#params_PSSCpure_df.to_csv('params_PSSCpure_df.csv')

params_SCcomb_df = params_SCcomb_df[(params_SCcomb_df['divergent']==False)].reset_index(drop=True)
params_SCcomb_df = params_SCcomb_df.drop(['divergent'], axis=1)
#params_SCcomb_df.to_csv('params_SCcomb_df.csv')

params_PScomb_df = params_PScomb_df[(params_PScomb_df['divergent']==False)].reset_index(drop=True)
params_PScomb_df = params_PScomb_df.drop(['divergent'], axis=1)
#params_PScomb_df.to_csv('params_PScomb_df.csv')

params_PSSCcomb_df = params_PSSCcomb_df[(params_PSSCcomb_df['divergent']==False)].reset_index(drop=True)
params_PSSCcomb_df = params_PSSCcomb_df.drop(['divergent'], axis=1)
#params_PSSCcomb_df.to_csv('params_PSSCcomb_df.csv')


#%%










#%% Subpopulation analysis

# SCpure
a_limit = 0.4

params_SCpure2_df = (params_SCpure_df.
					 assign(subpopulation = np.select([params_SCpure_df['a']<=a_limit,
													   params_SCpure_df['a']>a_limit],
												      [0,1])))
#params_SCpure2_df.to_csv('params_SCpure2_df.csv')

subpopulation0_df = params_SCpure2_df[(params_SCpure2_df['subpopulation']==0)]
a_lst = subpopulation0_df['a'].tolist()
low_lim, up_lim = iqr(a_lst)
subpopulation0_df = (subpopulation0_df.
					 assign(outlier = np.select([subpopulation0_df['a']<=low_lim,
											     subpopulation0_df['a']>up_lim],
											    [1,1],0)))
subpopulation0_df = subpopulation0_df[(subpopulation0_df['outlier']==0)]
#subpopulation0_df.to_csv('subpopulation0_df.csv')

subpopulation1_df = params_SCpure2_df[(params_SCpure2_df['subpopulation']==1)]
a_lst = subpopulation1_df['a'].tolist()
low_lim, up_lim = iqr(a_lst)
subpopulation1_df = (subpopulation1_df.
					 assign(outlier = np.select([subpopulation1_df['a']<=low_lim,
											     subpopulation1_df['a']>up_lim],
											    [1,1],0)))
subpopulation1_df = subpopulation1_df[(subpopulation1_df['outlier']==0)]
#subpopulation1_df.to_csv('subpopulation1_df.csv')

params_SCpure3_df = pd.concat([subpopulation0_df, subpopulation1_df], axis=0).reset_index(drop=True)
params_SCpure3_df = params_SCpure3_df.drop(['outlier'], axis=1)
#params_SCpure3_df.to_csv('params_SCpure3_df.csv')




# PSpure
params_PSpure2_df = params_PSpure_df.reset_index(drop = True)
params_PSpure2_df['subpopulation'] = 0

subpopulation0_df = params_PSpure2_df[(params_PSpure2_df['subpopulation']==0)]
a_lst = subpopulation0_df['a'].tolist()
low_lim, up_lim = iqr(a_lst)
subpopulation0_df = (subpopulation0_df.
					 assign(outlier = np.select([subpopulation0_df['a']<=low_lim,
											     subpopulation0_df['a']>up_lim],
											    [1,1],0)))
subpopulation0_df = subpopulation0_df[(subpopulation0_df['outlier']==0)]
#subpopulation0_df.to_csv('subpopulation0_df.csv')

params_PSpure3_df = subpopulation0_df.reset_index(drop = True)
params_PSpure3_df = params_PSpure3_df.drop(['outlier'], axis=1)
#params_PSpure3_df.to_csv('params_PSpure3_df.csv')




# PSSCcomb
a_limit = 0.8

params_PSSCcomb2_df = (params_PSSCcomb_df.
					   assign(subpopulation = np.select([params_PSSCcomb_df['a']<=a_limit,
														  params_PSSCcomb_df['a']>a_limit],
														 [0,1])))
#params_PSSCcomb2_df.to_csv('params_PSSCcomb2_df.csv')

subpopulation0_df = params_PSSCcomb2_df[(params_PSSCcomb2_df['subpopulation']==0)]
a_lst = subpopulation0_df['a'].tolist()
low_lim, up_lim = iqr(a_lst)
subpopulation0_df = (subpopulation0_df.
					 assign(outlier = np.select([subpopulation0_df['a']<=low_lim,
											     subpopulation0_df['a']>up_lim],
											    [1,1],0)))
subpopulation0_df = subpopulation0_df[(subpopulation0_df['outlier']==0)]
#subpopulation0_df.to_csv('subpopulation0_df.csv')

subpopulation1_df = params_PSSCcomb2_df[(params_PSSCcomb2_df['subpopulation']==1)]
a_lst = subpopulation1_df['a'].tolist()
low_lim, up_lim = iqr(a_lst)
subpopulation1_df = (subpopulation1_df.
					 assign(outlier = np.select([subpopulation1_df['a']<=low_lim,
											     subpopulation1_df['a']>up_lim],
											    [1,1],0)))
subpopulation1_df = subpopulation1_df[(subpopulation1_df['outlier']==0)]
#subpopulation1_df.to_csv('subpopulation1_df.csv')

params_PSSCcomb3_df = pd.concat([subpopulation0_df, subpopulation1_df], axis=0).reset_index(drop=True)
params_PSSCcomb3_df = params_PSSCcomb3_df.drop(['outlier'], axis=1)
#params_PSSCcomb3_df.to_csv('params_PSSCcomb3_df.csv')




# SCcomb
params_SCcomb2_df = params_SCcomb_df.reset_index(drop = True)
params_SCcomb2_df['subpopulation'] = 0

subpopulation0_df = params_SCcomb2_df[(params_SCcomb2_df['subpopulation']==0)]
a_lst = subpopulation0_df['a'].tolist()
low_lim, up_lim = iqr(a_lst)
subpopulation0_df = (subpopulation0_df.
					 assign(outlier = np.select([subpopulation0_df['a']<=low_lim,
											     subpopulation0_df['a']>up_lim],
											    [1,1],0)))
subpopulation0_df = subpopulation0_df[(subpopulation0_df['outlier']==0)]
#subpopulation0_df.to_csv('subpopulation0_df.csv')

params_SCcomb3_df = subpopulation0_df.reset_index(drop = True)
params_SCcomb3_df = params_SCcomb3_df.drop(['outlier'], axis=1)
#params_SCcomb3_df.to_csv('params_SCcomb3_df.csv')




# PScomb
params_PScomb2_df = params_PScomb_df.reset_index(drop = True)
params_PScomb2_df['subpopulation'] = 0

subpopulation0_df = params_PScomb2_df[(params_PScomb2_df['subpopulation']==0)]
a_lst = subpopulation0_df['a'].tolist()
low_lim, up_lim = iqr(a_lst)
subpopulation0_df = (subpopulation0_df.
					 assign(outlier = np.select([subpopulation0_df['a']<=low_lim,
											     subpopulation0_df['a']>up_lim],
											    [1,1],0)))
subpopulation0_df = subpopulation0_df[(subpopulation0_df['outlier']==0)]
#subpopulation0_df.to_csv('subpopulation0_df.csv')

params_PScomb3_df = subpopulation0_df.reset_index(drop = True) 
params_PScomb3_df = params_PScomb3_df.drop(['outlier'], axis=1)
#params_PScomb3_df.to_csv('params_PScomb3_df.csv')




# PSSCpure
a_limit = 0.75

params_PSSCpure2_df = (params_PSSCpure_df.
					   assign(subpopulation = np.select([params_PSSCpure_df['a']<=a_limit,
														  params_PSSCpure_df['a']>a_limit],
														 [0,1])))
#params_PSSCpure2_df.to_csv('params_PSSCpure2_df.csv')

subpopulation0_df = params_PSSCpure2_df[(params_PSSCpure2_df['subpopulation']==0)]
a_lst = subpopulation0_df['a'].tolist()
low_lim, up_lim = iqr(a_lst)
subpopulation0_df = (subpopulation0_df.
					 assign(outlier = np.select([subpopulation0_df['a']<=low_lim,
											     subpopulation0_df['a']>up_lim],
											    [1,1],0)))
subpopulation0_df = subpopulation0_df[(subpopulation0_df['outlier']==0)]
#subpopulation0_df.to_csv('subpopulation0_df.csv')

subpopulation1_df = params_PSSCpure2_df[(params_PSSCpure2_df['subpopulation']==1)]
a_lst = subpopulation1_df['a'].tolist()
low_lim, up_lim = iqr(a_lst)
subpopulation1_df = (subpopulation1_df.
					 assign(outlier = np.select([subpopulation1_df['a']<=low_lim,
											     subpopulation1_df['a']>up_lim],
											    [1,1],0)))
subpopulation1_df = subpopulation1_df[(subpopulation1_df['outlier']==0)]
#subpopulation1_df.to_csv('subpopulation1_df.csv')

params_PSSCpure3_df = pd.concat([subpopulation0_df, subpopulation1_df], axis=0).reset_index(drop=True)
params_PSSCpure3_df = params_PSSCpure3_df.drop(['outlier'], axis=1)
#params_PSSCpure3_df.to_csv('params_PSSCpure3_df.csv')


#%%










#%% Histogram dist PSSCcomb and PSSCpure together
violet_dark = rgb2hex(cm.get_cmap('plasma')(0.0))
violet_light = rgb2hex(cm.get_cmap('plasma')(0.3))
custom_palette_PS = {0: violet_dark, 1: violet_light}

orange_dark = rgb2hex(cm.get_cmap('plasma')(0.70))
orange_light = rgb2hex(cm.get_cmap('plasma')(0.85))
custom_palette_SC = {0: orange_dark, 1: orange_light}


params_PSSCcomb3_df['perturb_type_x_context'] = 'PSSCcomb' 
#params_PSSCcomb3_df.to_csv('params_PSSCcomb3_df.csv')
params_PSSCpure3_df['perturb_type_x_context'] = 'PSSCpure' 
#params_PSSCpure3_df.to_csv('params_PSSCpure3_df.csv')
params_PSSCpurecomb_df = pd.concat([params_PSSCpure3_df, params_PSSCcomb3_df], axis=0).reset_index(drop=True)
#params_PSSCpurecomb_df.to_csv('params_PSSCpurecomb_df.csv')

params_PSSCpurecomb_df = params_PSSCpurecomb_df.astype({'perturb_type_x_context':str})
params_PSSCpurecomb_df = params_PSSCpurecomb_df.astype({'subpopulation':str})
params_PSSCpurecomb_df['perturb_type_x_context_x_subpopulation'] = params_PSSCpurecomb_df[['perturb_type_x_context','subpopulation']].agg(''.join, axis=1)
params_PSSCpurecomb_df.to_csv(datafolder + 'params_PSSCpurecomb_df.csv')

fig_xsize = 10 * 0.393701   # centimeter to inch
fig_ysize = 8 * 0.393701   # centimeter to inch

plt.figure(figsize=(fig_xsize, fig_ysize))
sns.set_theme(style="ticks")
custom_palette = {
	'PSSCcomb': violet_dark,    
    'PSSCpure': orange_dark
}

sns.set_context("notebook", font_scale=1, rc={"xtick.labelsize":6, "ytick.labelsize":6, "axes.labelsize":7, "legend.fontsize":6, "legend.title_fontsize":6})
sns.histplot(data=params_PSSCpurecomb_df, x="dist", bins=50, kde=False, hue='perturb_type_x_context', palette=custom_palette)

ax = plt.gca()
legend = ax.get_legend()

if legend:
	legend.set_title("Context")
	legend.set_bbox_to_anchor((0.5, 0.8))
	legend._loc = 10
	
	new_labels = {
        "PSSCpure": "Pure",
        "PSSCcomb": "Combined"
    }
	for text in legend.get_texts():
		old = text.get_text()
		if old in new_labels:
			text.set_text(new_labels[old])

ax.tick_params(axis='both', which='both', length=3, width=0.5, colors='black') 
	
for spine in ax.spines.values():
	spine.set_linewidth(0.5)

ax.set_xlabel("Fitting loss function (ms)")

plt.tight_layout()
plt.savefig("Fig_Distance.pdf")
plt.savefig("Fig_Distance.png", dpi=300, bbox_inches="tight")
plt.show()

summary = params_PSSCpurecomb_df.groupby('perturb_type_x_context')[['dist']].agg(['median', 'min', 'max'])
summary.columns = ['_'.join(col) for col in summary.columns]
summary = summary.reset_index()
print(summary)
#summary.to_csv('summary_PSSCpurecomb_df.csv')


#%%










#%% Model simulations with fitted parameter values
n_beeps = 40
perturb_beep = 6
ISI = 500
p_start, x_start, s_start = 0, ISI, ISI
y_ini = [p_start, x_start, s_start]         # p(asincronía predicha)   x(variable que explica el overshoot en step change)   s(ISI en n-1)




#SCpure

# Select fitting   
params_SCpure_fit_0_df = params_SCpure3_df[(params_SCpure3_df['subpopulation'] == 0)].reset_index(drop=True)
#params_SCpure_fit_0_df.to_csv('params_SCpure_fit_0_df.csv')
params_SCpure_fit_0_df = params_SCpure_fit_0_df[(params_SCpure_fit_0_df['dist'] == params_SCpure_fit_0_df['dist'].min())].reset_index(drop=True)
#params_SCpure_fit_0_df = params_SCpure_fit_0_df.query('index==0').reset_index(drop=True)

# Parameters values
a = params_SCpure_fit_0_df['a'][0]
b = params_SCpure_fit_0_df['b'][0]
c = params_SCpure_fit_0_df['c'][0]
d = params_SCpure_fit_0_df['d'][0]
alpha1 = params_SCpure_fit_0_df['alpha1'][0]
beta1 = params_SCpure_fit_0_df['beta1'][0]
gamma1 = params_SCpure_fit_0_df['gamma1'][0]
delta1 = params_SCpure_fit_0_df['delta1'][0]
eps1 = params_SCpure_fit_0_df['eps1'][0]
dseta1 = params_SCpure_fit_0_df['dseta1'][0]
eta1 = params_SCpure_fit_0_df['eta1'][0]
alpha2 = params_SCpure_fit_0_df['alpha2'][0]
beta2 = params_SCpure_fit_0_df['beta2'][0]
gamma2 = params_SCpure_fit_0_df['gamma2'][0]
delta2 = params_SCpure_fit_0_df['delta2'][0]
eps2 = params_SCpure_fit_0_df['eps2'][0]
dseta2 = params_SCpure_fit_0_df['dseta2'][0]
eta2 = params_SCpure_fit_0_df['eta2'][0]
params = [a, b, c, d, alpha1, beta1, gamma1, delta1, eps1, dseta1, eta1, alpha2, beta2, gamma2, delta2, eps2, dseta2, eta2]

data3_fit_SCpure_aux_df = data3_fit_SCpure_df[(data3_fit_SCpure_df['perturb_sign']=='neg') & (data3_fit_SCpure_df['perturb_size']==20)].reset_index(drop=True)
baseline_pre_neg_20 = data3_fit_SCpure_aux_df['baseline_pre'][0]
baseline_post_neg_20 = data3_fit_SCpure_aux_df['baseline_post'][0]
data3_fit_SCpure_aux_df = data3_fit_SCpure_df[(data3_fit_SCpure_df['perturb_sign']=='neg') & (data3_fit_SCpure_df['perturb_size']==50)].reset_index(drop=True)
baseline_pre_neg_50 = data3_fit_SCpure_aux_df['baseline_pre'][0]
baseline_post_neg_50 = data3_fit_SCpure_aux_df['baseline_post'][0]
data3_fit_SCpure_aux_df = data3_fit_SCpure_df[(data3_fit_SCpure_df['perturb_sign']=='pos') & (data3_fit_SCpure_df['perturb_size']==20)].reset_index(drop=True)
baseline_pre_pos_20 = data3_fit_SCpure_aux_df['baseline_pre'][0]
baseline_post_pos_20 = data3_fit_SCpure_aux_df['baseline_post'][0]
data3_fit_SCpure_aux_df = data3_fit_SCpure_df[(data3_fit_SCpure_df['perturb_sign']=='pos') & (data3_fit_SCpure_df['perturb_size']==50)].reset_index(drop=True)
baseline_pre_pos_50 = data3_fit_SCpure_aux_df['baseline_pre'][0]
baseline_post_pos_50 = data3_fit_SCpure_aux_df['baseline_post'][0]

# Simulation
col_names = ['origin','perturb_size','n','baseline','asyn','p','x','s']
model_data_df = pd.DataFrame(columns=col_names)

perturb_size_lst = [50, 50, -50, -50, 40, -40, 30, -30, 20, 20, -20, -20]
baseline_pre_lst = [baseline_pre_pos_50, 0, baseline_pre_neg_50, 0, 0, 0, 0, 0, baseline_pre_pos_20, 0, baseline_pre_neg_20, 0]
baseline_post_lst = [baseline_post_pos_50, 0, baseline_post_neg_50, 0, 0, 0, 0, 0, baseline_post_pos_20, 0, baseline_post_neg_20, 0]

i = 0
for perturb_size in perturb_size_lst:
	hyper_params = [baseline_pre_lst[i], baseline_post_lst[i], 'SC', perturb_size]
	t, model_asyn, y = integrate(model, y_ini, params, hyper_params)
	origin_list = np.repeat('SCpure',len(t))[:,np.newaxis].T
	perturb_size_list = np.repeat(perturb_size,len(t))[:,np.newaxis].T
	step_n = np.array(range(0,n_beeps))[:,np.newaxis].T-perturb_beep+1

	if	(baseline_pre_lst[i] == 0):
		baseline_list = np.repeat(0,len(t))[:,np.newaxis].T
	else:
		baseline_list = np.repeat(1,len(t))[:,np.newaxis].T
	
	model_array = np.concatenate([ \
							   origin_list, \
							   perturb_size_list, \
							   step_n, \
							   baseline_list, \
							   model_asyn[:,np.newaxis].T, \
							   y.T], axis=0).T
	model_data_df = pd.concat([model_data_df, pd.DataFrame(model_array, columns=col_names)], axis=0)
	i = i+1
model_SCpure_0_df = model_data_df.reset_index(drop=True)
#model_SCpure_0_df.to_csv('model_SCpure_0_df.csv')




# Select fitting   
params_SCpure_fit_1_df = params_SCpure3_df[(params_SCpure3_df['subpopulation'] == 1)].reset_index(drop=True)
#params_SCpure_fit_1_df.to_csv('params_SCpure_fit_1_df.csv')
params_SCpure_fit_1_df = params_SCpure_fit_1_df[(params_SCpure_fit_1_df['dist'] == params_SCpure_fit_1_df['dist'].min())].reset_index(drop=True)
#params_SCpure_fit_1_df = params_SCpure_fit_1_df.query('index==0').reset_index(drop=True)

# Parameters values
a = params_SCpure_fit_1_df['a'][0]
b = params_SCpure_fit_1_df['b'][0]
c = params_SCpure_fit_1_df['c'][0]
d = params_SCpure_fit_1_df['d'][0]
alpha1 = params_SCpure_fit_1_df['alpha1'][0]
beta1 = params_SCpure_fit_1_df['beta1'][0]
gamma1 = params_SCpure_fit_1_df['gamma1'][0]
delta1 = params_SCpure_fit_1_df['delta1'][0]
eps1 = params_SCpure_fit_1_df['eps1'][0]
dseta1 = params_SCpure_fit_1_df['dseta1'][0]
eta1 = params_SCpure_fit_1_df['eta1'][0]
alpha2 = params_SCpure_fit_1_df['alpha2'][0]
beta2 = params_SCpure_fit_1_df['beta2'][0]
gamma2 = params_SCpure_fit_1_df['gamma2'][0]
delta2 = params_SCpure_fit_1_df['delta2'][0]
eps2 = params_SCpure_fit_1_df['eps2'][0]
dseta2 = params_SCpure_fit_1_df['dseta2'][0]
eta2 = params_SCpure_fit_1_df['eta2'][0]
params = [a, b, c, d, alpha1, beta1, gamma1, delta1, eps1, dseta1, eta1, alpha2, beta2, gamma2, delta2, eps2, dseta2, eta2]

# Simulation
model_data_df = pd.DataFrame(columns=col_names)

i = 0
for perturb_size in perturb_size_lst:
	hyper_params = [baseline_pre_lst[i], baseline_post_lst[i], 'SC', perturb_size]
	t, model_asyn, y = integrate(model, y_ini, params, hyper_params)
	origin_list = np.repeat('SCpure',len(t))[:,np.newaxis].T
	perturb_size_list = np.repeat(perturb_size,len(t))[:,np.newaxis].T
	step_n = np.array(range(0,n_beeps))[:,np.newaxis].T-perturb_beep+1
	
	if	(baseline_pre_lst[i] == 0):
		baseline_list = np.repeat(0,len(t))[:,np.newaxis].T
	else:
		baseline_list = np.repeat(1,len(t))[:,np.newaxis].T
	
	model_array = np.concatenate([ \
							   origin_list, \
							   perturb_size_list, \
							   step_n, \
							   baseline_list, \
							   model_asyn[:,np.newaxis].T, \
							   y.T], axis=0).T
	model_data_df = pd.concat([model_data_df, pd.DataFrame(model_array, columns=col_names)], axis=0)
	i = i+1
model_SCpure_1_df = model_data_df.reset_index(drop=True) 
#model_SCpure_1_df.to_csv('model_SCpure_1_df.csv')




# PSpure

# Select fitting   
params_PSpure_fit_0_df = params_PSpure3_df[(params_PSpure3_df['subpopulation'] == 0)].reset_index(drop=True)
#params_PSpure_fit_0_df.to_csv('params_PSpure_fit_0_df.csv')
params_PSpure_fit_0_df = params_PSpure_fit_0_df[(params_PSpure_fit_0_df['dist'] == params_PSpure_fit_0_df['dist'].min())].reset_index(drop=True)
#params_PSpure_fit_0_df = params_PSpure_fit_0_df.query('index==0').reset_index(drop=True)

# Parameters values
a = params_PSpure_fit_0_df['a'][0]
b = params_PSpure_fit_0_df['b'][0]
c = params_PSpure_fit_0_df['c'][0]
d = params_PSpure_fit_0_df['d'][0]
alpha1 = params_PSpure_fit_0_df['alpha1'][0]
beta1 = params_PSpure_fit_0_df['beta1'][0]
gamma1 = params_PSpure_fit_0_df['gamma1'][0]
delta1 = params_PSpure_fit_0_df['delta1'][0]
eps1 = params_PSpure_fit_0_df['eps1'][0]
dseta1 = params_PSpure_fit_0_df['dseta1'][0]
eta1 = params_PSpure_fit_0_df['eta1'][0]
alpha2 = params_PSpure_fit_0_df['alpha2'][0]
beta2 = params_PSpure_fit_0_df['beta2'][0]
gamma2 = params_PSpure_fit_0_df['gamma2'][0]
delta2 = params_PSpure_fit_0_df['delta2'][0]
eps2 = params_PSpure_fit_0_df['eps2'][0]
dseta2 = params_PSpure_fit_0_df['dseta2'][0]
eta2 = params_PSpure_fit_0_df['eta2'][0]
params = [a, b, c, d, alpha1, beta1, gamma1, delta1, eps1, dseta1, eta1, alpha2, beta2, gamma2, delta2, eps2, dseta2, eta2]

data3_fit_PSpure_aux_df = data3_fit_PSpure_df[(data3_fit_PSpure_df['perturb_sign']=='neg') & (data3_fit_PSpure_df['perturb_size']==20)].reset_index(drop=True)
baseline_pre_neg_20 = data3_fit_PSpure_aux_df['baseline_pre'][0]
baseline_post_neg_20 = data3_fit_PSpure_aux_df['baseline_post'][0]
data3_fit_PSpure_aux_df = data3_fit_PSpure_df[(data3_fit_PSpure_df['perturb_sign']=='neg') & (data3_fit_PSpure_df['perturb_size']==50)].reset_index(drop=True)
baseline_pre_neg_50 = data3_fit_PSpure_aux_df['baseline_pre'][0]
baseline_post_neg_50 = data3_fit_PSpure_aux_df['baseline_post'][0]
data3_fit_PSpure_aux_df = data3_fit_PSpure_df[(data3_fit_PSpure_df['perturb_sign']=='pos') & (data3_fit_PSpure_df['perturb_size']==20)].reset_index(drop=True)
baseline_pre_pos_20 = data3_fit_PSpure_aux_df['baseline_pre'][0]
baseline_post_pos_20 = data3_fit_PSpure_aux_df['baseline_post'][0]
data3_fit_PSpure_aux_df = data3_fit_PSpure_df[(data3_fit_PSpure_df['perturb_sign']=='pos') & (data3_fit_PSpure_df['perturb_size']==50)].reset_index(drop=True)
baseline_pre_pos_50 = data3_fit_PSpure_aux_df['baseline_pre'][0]
baseline_post_pos_50 = data3_fit_PSpure_aux_df['baseline_post'][0]

# Simulation
model_data_df = pd.DataFrame(columns=col_names)

perturb_size_lst = [50, 50, -50, -50, 40, -40, 30, -30, 20, 20, -20, -20]
baseline_pre_lst = [baseline_pre_pos_50, 0, baseline_pre_neg_50, 0, 0, 0, 0, 0, baseline_pre_pos_20, 0, baseline_pre_neg_20, 0]
baseline_post_lst = [baseline_post_pos_50, 0, baseline_post_neg_50, 0, 0, 0, 0, 0, baseline_post_pos_20, 0, baseline_post_neg_20, 0]

i = 0
for perturb_size in perturb_size_lst:
	hyper_params = [baseline_pre_lst[i], baseline_post_lst[i], 'PS', perturb_size]
	t, model_asyn, y = integrate(model, y_ini, params, hyper_params)
	origin_list = np.repeat('PSpure',len(t))[:,np.newaxis].T
	perturb_size_list = np.repeat(perturb_size,len(t))[:,np.newaxis].T
	step_n = np.array(range(0,n_beeps))[:,np.newaxis].T-perturb_beep+1	
	
	if	(baseline_pre_lst[i] == 0):
		baseline_list = np.repeat(0,len(t))[:,np.newaxis].T
	else:
		baseline_list = np.repeat(1,len(t))[:,np.newaxis].T
	
	model_array = np.concatenate([ \
							   origin_list, \
							   perturb_size_list, \
							   step_n, \
							   baseline_list, \
							   model_asyn[:,np.newaxis].T, \
							   y.T], axis=0).T
	model_data_df = pd.concat([model_data_df, pd.DataFrame(model_array, columns=col_names)], axis=0)
	i = i+1
model_PSpure_0_df = model_data_df.reset_index(drop=True) 
#model_PSpure_0_df.to_csv('model_PSpure_0_df.csv')




# PSSCcomb

# Select fitting
params_PSSCcomb_fit_0_df = params_PSSCcomb3_df[(params_PSSCcomb3_df['subpopulation'] == 0)].reset_index(drop=True)
#params_PSSCcomb_fit_0_df.to_csv('params_PSSCcomb_fit_0_df.csv')
params_PSSCcomb_fit_0_df = params_PSSCcomb_fit_0_df[(params_PSSCcomb_fit_0_df['dist'] == params_PSSCcomb_fit_0_df['dist'].min())].reset_index(drop=True)
#params_PSSCcomb_fit_0_df = params_PSSCcomb_fit_0_df.query('index==0').reset_index(drop=True)

# Parameters values
a = params_PSSCcomb_fit_0_df['a'][0]
b = params_PSSCcomb_fit_0_df['b'][0]
c = params_PSSCcomb_fit_0_df['c'][0]
d = params_PSSCcomb_fit_0_df['d'][0]
alpha1 = params_PSSCcomb_fit_0_df['alpha1'][0]
beta1 = params_PSSCcomb_fit_0_df['beta1'][0]
gamma1 = params_PSSCcomb_fit_0_df['gamma1'][0]
delta1 = params_PSSCcomb_fit_0_df['delta1'][0]
eps1 = params_PSSCcomb_fit_0_df['eps1'][0]
dseta1 = params_PSSCcomb_fit_0_df['dseta1'][0]
eta1 = params_PSSCcomb_fit_0_df['eta1'][0]
alpha2 = params_PSSCcomb_fit_0_df['alpha2'][0]
beta2 = params_PSSCcomb_fit_0_df['beta2'][0]
gamma2 = params_PSSCcomb_fit_0_df['gamma2'][0]
delta2 = params_PSSCcomb_fit_0_df['delta2'][0]
eps2 = params_PSSCcomb_fit_0_df['eps2'][0]
dseta2 = params_PSSCcomb_fit_0_df['dseta2'][0]
eta2 = params_PSSCcomb_fit_0_df['eta2'][0]
params = [a, b, c, d, alpha1, beta1, gamma1, delta1, eps1, dseta1, eta1, alpha2, beta2, gamma2, delta2, eps2, dseta2, eta2]

data3_fit_PSSCcomb_aux_df = data3_fit_PSSCcomb_df[(data3_fit_PSSCcomb_df['perturb_type']=='SC') & (data3_fit_PSSCcomb_df['perturb_sign']=='neg') & (data3_fit_PSSCcomb_df['perturb_size']==20)].reset_index(drop=True)
baseline_pre_SC_neg_20 = data3_fit_PSSCcomb_aux_df['baseline_pre'][0]
baseline_post_SC_neg_20 = data3_fit_PSSCcomb_aux_df['baseline_post'][0]
data3_fit_PSSCcomb_aux_df = data3_fit_PSSCcomb_df[(data3_fit_PSSCcomb_df['perturb_type']=='SC') & (data3_fit_PSSCcomb_df['perturb_sign']=='neg') & (data3_fit_PSSCcomb_df['perturb_size']==50)].reset_index(drop=True)
baseline_pre_SC_neg_50 = data3_fit_PSSCcomb_aux_df['baseline_pre'][0]
baseline_post_SC_neg_50 = data3_fit_PSSCcomb_aux_df['baseline_post'][0]
data3_fit_PSSCcomb_aux_df = data3_fit_PSSCcomb_df[(data3_fit_PSSCcomb_df['perturb_type']=='SC') & (data3_fit_PSSCcomb_df['perturb_sign']=='pos') & (data3_fit_PSSCcomb_df['perturb_size']==20)].reset_index(drop=True)
baseline_pre_SC_pos_20 = data3_fit_PSSCcomb_aux_df['baseline_pre'][0]
baseline_post_SC_pos_20 = data3_fit_PSSCcomb_aux_df['baseline_post'][0]
data3_fit_PSSCcomb_aux_df = data3_fit_PSSCcomb_df[(data3_fit_PSSCcomb_df['perturb_type']=='SC') & (data3_fit_PSSCcomb_df['perturb_sign']=='pos') & (data3_fit_PSSCcomb_df['perturb_size']==50)].reset_index(drop=True)
baseline_pre_SC_pos_50 = data3_fit_PSSCcomb_aux_df['baseline_pre'][0]
baseline_post_SC_pos_50 = data3_fit_PSSCcomb_aux_df['baseline_post'][0]
data3_fit_PSSCcomb_aux_df = data3_fit_PSSCcomb_df[(data3_fit_PSSCcomb_df['perturb_type']=='PS') & (data3_fit_PSSCcomb_df['perturb_sign']=='neg') & (data3_fit_PSSCcomb_df['perturb_size']==20)].reset_index(drop=True)
baseline_pre_PS_neg_20 = data3_fit_PSSCcomb_aux_df['baseline_pre'][0]
baseline_post_PS_neg_20 = data3_fit_PSSCcomb_aux_df['baseline_post'][0]
data3_fit_PSSCcomb_aux_df = data3_fit_PSSCcomb_df[(data3_fit_PSSCcomb_df['perturb_type']=='PS') & (data3_fit_PSSCcomb_df['perturb_sign']=='neg') & (data3_fit_PSSCcomb_df['perturb_size']==50)].reset_index(drop=True)
baseline_pre_PS_neg_50 = data3_fit_PSSCcomb_aux_df['baseline_pre'][0]
baseline_post_PS_neg_50 = data3_fit_PSSCcomb_aux_df['baseline_post'][0]
data3_fit_PSSCcomb_aux_df = data3_fit_PSSCcomb_df[(data3_fit_PSSCcomb_df['perturb_type']=='PS') & (data3_fit_PSSCcomb_df['perturb_sign']=='pos') & (data3_fit_PSSCcomb_df['perturb_size']==20)].reset_index(drop=True)
baseline_pre_PS_pos_20 = data3_fit_PSSCcomb_aux_df['baseline_pre'][0]
baseline_post_PS_pos_20 = data3_fit_PSSCcomb_aux_df['baseline_post'][0]
data3_fit_PSSCcomb_aux_df = data3_fit_PSSCcomb_df[(data3_fit_PSSCcomb_df['perturb_type']=='PS') & (data3_fit_PSSCcomb_df['perturb_sign']=='pos') & (data3_fit_PSSCcomb_df['perturb_size']==50)].reset_index(drop=True)
baseline_pre_PS_pos_50 = data3_fit_PSSCcomb_aux_df['baseline_pre'][0]
baseline_post_PS_pos_50 = data3_fit_PSSCcomb_aux_df['baseline_post'][0]

# Simulation
col_names = ['origin','perturb_size','perturb_type','n','baseline','asyn','p','x','s']
model_data_df = pd.DataFrame(columns=col_names)

perturb_size_lst = [50, 50, -50, -50, 40, -40, 30, -30, 20, 20, -20, -20,
					50, 50, -50, -50, 40, -40, 30, -30, 20, 20, -20, -20]
perturb_type_lst = ['SC', 'SC', 'SC', 'SC', 'SC', 'SC', 'SC', 'SC', 'SC', 'SC', 'SC', 'SC',
					'PS', 'PS', 'PS', 'PS', 'PS', 'PS', 'PS', 'PS', 'PS', 'PS', 'PS', 'PS']
baseline_pre_lst = [baseline_pre_SC_pos_50, 0, baseline_pre_SC_neg_50, 0, 0, 0, 0, 0, baseline_pre_SC_pos_20, 0, baseline_pre_SC_neg_20, 0,
					baseline_pre_PS_pos_50, 0, baseline_pre_PS_neg_50, 0, 0, 0, 0, 0, baseline_pre_PS_pos_20, 0, baseline_pre_PS_neg_20, 0]
baseline_post_lst = [baseline_post_SC_pos_50, 0, baseline_post_SC_neg_50, 0, 0, 0, 0, 0, baseline_post_SC_pos_20, 0, baseline_post_SC_neg_20, 0,
					 baseline_post_PS_pos_50, 0, baseline_post_PS_neg_50, 0, 0, 0, 0, 0, baseline_post_PS_pos_20, 0, baseline_post_PS_neg_20, 0]

i = 0
for perturb_size in perturb_size_lst:
	hyper_params = [baseline_pre_lst[i], baseline_post_lst[i], perturb_type_lst[i], perturb_size]
	t, model_asyn, y = integrate(model, y_ini, params, hyper_params)
	origin_list = np.repeat('PSSCcomb',len(t))[:,np.newaxis].T
	perturb_size_list = np.repeat(perturb_size,len(t))[:,np.newaxis].T
	perturb_type_list = np.repeat(perturb_type_lst[i],len(t))[:,np.newaxis].T
	step_n = np.array(range(0,n_beeps))[:,np.newaxis].T-perturb_beep+1

	if	(baseline_pre_lst[i] == 0):
		baseline_list = np.repeat(0,len(t))[:,np.newaxis].T
	else:
		baseline_list = np.repeat(1,len(t))[:,np.newaxis].T
	
	model_array = np.concatenate([ \
							   origin_list, \
							   perturb_size_list, \
							   perturb_type_list, \
							   step_n, \
							   baseline_list, \
							   model_asyn[:,np.newaxis].T, \
							   y.T], axis=0).T
	model_data_df = pd.concat([model_data_df, pd.DataFrame(model_array, columns=col_names)], axis=0)
	i = i+1
model_PSSCcomb_0_df = model_data_df.reset_index(drop=True)
#model_PSSCcomb_0_df.to_csv('model_PSSCcomb_0_df.csv')




# Select fitting
params_PSSCcomb_fit_1_df = params_PSSCcomb3_df[(params_PSSCcomb3_df['subpopulation'] == 1)].reset_index(drop=True)
#params_PSSCcomb_fit_1_df.to_csv('params_PSSCcomb_fit_1_df.csv')
params_PSSCcomb_fit_1_df = params_PSSCcomb_fit_1_df[(params_PSSCcomb_fit_1_df['dist'] == params_PSSCcomb_fit_1_df['dist'].min())].reset_index(drop=True)
#params_PSSCcomb_fit_1_df = params_PSSCcomb_fit_1_df.query('index==0').reset_index(drop=True)

# Parameters values
a = params_PSSCcomb_fit_1_df['a'][0]
b = params_PSSCcomb_fit_1_df['b'][0]
c = params_PSSCcomb_fit_1_df['c'][0]
d = params_PSSCcomb_fit_1_df['d'][0]
alpha1 = params_PSSCcomb_fit_1_df['alpha1'][0]
beta1 = params_PSSCcomb_fit_1_df['beta1'][0]
gamma1 = params_PSSCcomb_fit_1_df['gamma1'][0]
delta1 = params_PSSCcomb_fit_1_df['delta1'][0]
eps1 = params_PSSCcomb_fit_1_df['eps1'][0]
dseta1 = params_PSSCcomb_fit_1_df['dseta1'][0]
eta1 = params_PSSCcomb_fit_1_df['eta1'][0]
alpha2 = params_PSSCcomb_fit_1_df['alpha2'][0]
beta2 = params_PSSCcomb_fit_1_df['beta2'][0]
gamma2 = params_PSSCcomb_fit_1_df['gamma2'][0]
delta2 = params_PSSCcomb_fit_1_df['delta2'][0]
eps2 = params_PSSCcomb_fit_1_df['eps2'][0]
dseta2 = params_PSSCcomb_fit_1_df['dseta2'][0]
eta2 = params_PSSCcomb_fit_1_df['eta2'][0]
params = [a, b, c, d, alpha1, beta1, gamma1, delta1, eps1, dseta1, eta1, alpha2, beta2, gamma2, delta2, eps2, dseta2, eta2]

# Simulation
model_data_df = pd.DataFrame(columns=col_names)

i = 0
for perturb_size in perturb_size_lst:
	hyper_params = [baseline_pre_lst[i], baseline_post_lst[i], perturb_type_lst[i], perturb_size]
	t, model_asyn, y = integrate(model, y_ini, params, hyper_params)
	origin_list = np.repeat('PSSCcomb',len(t))[:,np.newaxis].T
	perturb_size_list = np.repeat(perturb_size,len(t))[:,np.newaxis].T
	perturb_type_list = np.repeat(perturb_type_lst[i],len(t))[:,np.newaxis].T
	step_n = np.array(range(0,n_beeps))[:,np.newaxis].T-perturb_beep+1

	if	(baseline_pre_lst[i] == 0):
		baseline_list = np.repeat(0,len(t))[:,np.newaxis].T
	else:
		baseline_list = np.repeat(1,len(t))[:,np.newaxis].T
	
	model_array = np.concatenate([ \
							   origin_list, \
							   perturb_size_list, \
							   perturb_type_list, \
							   step_n, \
							   baseline_list, \
							   model_asyn[:,np.newaxis].T, \
							   y.T], axis=0).T
	model_data_df = pd.concat([model_data_df, pd.DataFrame(model_array, columns=col_names)], axis=0)
	i = i+1
model_PSSCcomb_1_df = model_data_df.reset_index(drop=True)
#model_PSSCcomb_1_df.to_csv('model_PSSCcomb_1_df.csv')




# SCcomb

# Select fitting   
params_SCcomb_fit_0_df = params_SCcomb3_df[(params_SCcomb3_df['subpopulation'] == 0)].reset_index(drop=True)
#params_SCcomb_fit_0_df.to_csv('params_SCcomb_fit_0_df.csv')
params_SCcomb_fit_0_df = params_SCcomb_fit_0_df[(params_SCcomb_fit_0_df['dist'] == params_SCcomb_fit_0_df['dist'].min())].reset_index(drop=True)
#params_SCcomb_fit_0_df = params_SCcomb_fit_0_df.query('index==0').reset_index(drop=True)

# Parameters values
a = params_SCcomb_fit_0_df['a'][0]
b = params_SCcomb_fit_0_df['b'][0]
c = params_SCcomb_fit_0_df['c'][0]
d = params_SCcomb_fit_0_df['d'][0]
alpha1 = params_SCcomb_fit_0_df['alpha1'][0]
beta1 = params_SCcomb_fit_0_df['beta1'][0]
gamma1 = params_SCcomb_fit_0_df['gamma1'][0]
delta1 = params_SCcomb_fit_0_df['delta1'][0]
eps1 = params_SCcomb_fit_0_df['eps1'][0]
dseta1 = params_SCcomb_fit_0_df['dseta1'][0]
eta1 = params_SCcomb_fit_0_df['eta1'][0]
alpha2 = params_SCcomb_fit_0_df['alpha2'][0]
beta2 = params_SCcomb_fit_0_df['beta2'][0]
gamma2 = params_SCcomb_fit_0_df['gamma2'][0]
delta2 = params_SCcomb_fit_0_df['delta2'][0]
eps2 = params_SCcomb_fit_0_df['eps2'][0]
dseta2 = params_SCcomb_fit_0_df['dseta2'][0]
eta2 = params_SCcomb_fit_0_df['eta2'][0]
params = [a, b, c, d, alpha1, beta1, gamma1, delta1, eps1, dseta1, eta1, alpha2, beta2, gamma2, delta2, eps2, dseta2, eta2]

data3_fit_SCcomb_aux_df = data3_fit_SCcomb_df[(data3_fit_SCcomb_df['perturb_sign']=='neg') & (data3_fit_SCcomb_df['perturb_size']==20)].reset_index(drop=True)
baseline_pre_neg_20 = data3_fit_SCcomb_aux_df['baseline_pre'][0]
baseline_post_neg_20 = data3_fit_SCcomb_aux_df['baseline_post'][0]
data3_fit_SCcomb_aux_df = data3_fit_SCcomb_df[(data3_fit_SCcomb_df['perturb_sign']=='neg') & (data3_fit_SCcomb_df['perturb_size']==50)].reset_index(drop=True)
baseline_pre_neg_50 = data3_fit_SCcomb_aux_df['baseline_pre'][0]
baseline_post_neg_50 = data3_fit_SCcomb_aux_df['baseline_post'][0]
data3_fit_SCcomb_aux_df = data3_fit_SCcomb_df[(data3_fit_SCcomb_df['perturb_sign']=='pos') & (data3_fit_SCcomb_df['perturb_size']==20)].reset_index(drop=True)
baseline_pre_pos_20 = data3_fit_SCcomb_aux_df['baseline_pre'][0]
baseline_post_pos_20 = data3_fit_SCcomb_aux_df['baseline_post'][0]
data3_fit_SCcomb_aux_df = data3_fit_SCcomb_df[(data3_fit_SCcomb_df['perturb_sign']=='pos') & (data3_fit_SCcomb_df['perturb_size']==50)].reset_index(drop=True)
baseline_pre_pos_50 = data3_fit_SCcomb_aux_df['baseline_pre'][0]
baseline_post_pos_50 = data3_fit_SCcomb_aux_df['baseline_post'][0]

# Simulation
col_names = ['origin','perturb_size','n','baseline','asyn','p','x','s']
model_data_df = pd.DataFrame(columns=col_names)

perturb_size_lst = [50, 50, -50, -50, 40, -40, 30, -30, 20, 20, -20, -20]
baseline_pre_lst = [baseline_pre_pos_50, 0, baseline_pre_neg_50, 0, 0, 0, 0, 0, baseline_pre_pos_20, 0, baseline_pre_neg_20, 0]
baseline_post_lst = [baseline_post_pos_50, 0, baseline_post_neg_50, 0, 0, 0, 0, 0, baseline_post_pos_20, 0, baseline_post_neg_20, 0]

i = 0
for perturb_size in perturb_size_lst:
	hyper_params = [baseline_pre_lst[i], baseline_post_lst[i], 'SC', perturb_size]
	t, model_asyn, y = integrate(model, y_ini, params, hyper_params)
	origin_list = np.repeat('SCcomb',len(t))[:,np.newaxis].T
	perturb_size_list = np.repeat(perturb_size,len(t))[:,np.newaxis].T
	step_n = np.array(range(0,n_beeps))[:,np.newaxis].T-perturb_beep+1

	if	(baseline_pre_lst[i] == 0):
		baseline_list = np.repeat(0,len(t))[:,np.newaxis].T
	else:
		baseline_list = np.repeat(1,len(t))[:,np.newaxis].T
	
	model_array = np.concatenate([ \
							   origin_list, \
							   perturb_size_list, \
							   step_n, \
							   baseline_list, \
							   model_asyn[:,np.newaxis].T, \
							   y.T], axis=0).T
	model_data_df = pd.concat([model_data_df, pd.DataFrame(model_array, columns=col_names)], axis=0)
	i = i+1
model_SCcomb_0_df = model_data_df.reset_index(drop=True)
#model_SCcomb_0_df.to_csv('model_SCcomb_0_df.csv')




# PScomb

# Select fitting   
params_PScomb_fit_0_df = params_PScomb3_df[(params_PScomb3_df['subpopulation'] == 0)].reset_index(drop=True)
#params_PScomb_fit_0_df.to_csv('params_PScomb_fit_0_df.csv')
params_PScomb_fit_0_df = params_PScomb_fit_0_df[(params_PScomb_fit_0_df['dist'] == params_PScomb_fit_0_df['dist'].min())].reset_index(drop=True)
#params_PScomb_fit_0_df = params_PScomb_fit_0_df.query('index==0').reset_index(drop=True)

# Parameters values
a = params_PScomb_fit_0_df['a'][0]
b = params_PScomb_fit_0_df['b'][0]
c = params_PScomb_fit_0_df['c'][0]
d = params_PScomb_fit_0_df['d'][0]
alpha1 = params_PScomb_fit_0_df['alpha1'][0]
beta1 = params_PScomb_fit_0_df['beta1'][0]
gamma1 = params_PScomb_fit_0_df['gamma1'][0]
delta1 = params_PScomb_fit_0_df['delta1'][0]
eps1 = params_PScomb_fit_0_df['eps1'][0]
dseta1 = params_PScomb_fit_0_df['dseta1'][0]
eta1 = params_PScomb_fit_0_df['eta1'][0]
alpha2 = params_PScomb_fit_0_df['alpha2'][0]
beta2 = params_PScomb_fit_0_df['beta2'][0]
gamma2 = params_PScomb_fit_0_df['gamma2'][0]
delta2 = params_PScomb_fit_0_df['delta2'][0]
eps2 = params_PScomb_fit_0_df['eps2'][0]
dseta2 = params_PScomb_fit_0_df['dseta2'][0]
eta2 = params_PScomb_fit_0_df['eta2'][0]
params = [a, b, c, d, alpha1, beta1, gamma1, delta1, eps1, dseta1, eta1, alpha2, beta2, gamma2, delta2, eps2, dseta2, eta2]

# Simulation
model_data_df = pd.DataFrame(columns=col_names)

i = 0
for perturb_size in perturb_size_lst:
	hyper_params = [baseline_pre_lst[i], baseline_post_lst[i], 'PS', perturb_size]
	t, model_asyn, y = integrate(model, y_ini, params, hyper_params)
	origin_list = np.repeat('PScomb',len(t))[:,np.newaxis].T
	perturb_size_list = np.repeat(perturb_size,len(t))[:,np.newaxis].T
	step_n = np.array(range(0,n_beeps))[:,np.newaxis].T-perturb_beep+1
	
	if	(baseline_pre_lst[i] == 0):
		baseline_list = np.repeat(0,len(t))[:,np.newaxis].T
	else:
		baseline_list = np.repeat(1,len(t))[:,np.newaxis].T
	
	model_array = np.concatenate([ \
							   origin_list, \
							   perturb_size_list, \
							   step_n, \
							   baseline_list, \
							   model_asyn[:,np.newaxis].T, \
							   y.T], axis=0).T
	model_data_df = pd.concat([model_data_df, pd.DataFrame(model_array, columns=col_names)], axis=0)
	i = i+1
model_PScomb_0_df = model_data_df.reset_index(drop=True) 
#model_PScomb_0_df.to_csv('model_PScomb_0_df.csv')




# PSSCpure

# Select fitting
params_PSSCpure_fit_0_df = params_PSSCpure3_df[(params_PSSCpure3_df['subpopulation'] == 0)].reset_index(drop=True)
#params_PSSCpure_fit_0_df.to_csv('params_PSSCpure_fit_0_df.csv')
params_PSSCpure_fit_0_df = params_PSSCpure_fit_0_df[(params_PSSCpure_fit_0_df['dist'] == params_PSSCpure_fit_0_df['dist'].min())].reset_index(drop=True)
#params_PSSCpure_fit_0_df = params_PSSCpure_fit_0_df.query('index==0').reset_index(drop=True)

# Parameters values
a = params_PSSCpure_fit_0_df['a'][0]
b = params_PSSCpure_fit_0_df['b'][0]
c = params_PSSCpure_fit_0_df['c'][0]
d = params_PSSCpure_fit_0_df['d'][0]
alpha1 = params_PSSCpure_fit_0_df['alpha1'][0]
beta1 = params_PSSCpure_fit_0_df['beta1'][0]
gamma1 = params_PSSCpure_fit_0_df['gamma1'][0]
delta1 = params_PSSCpure_fit_0_df['delta1'][0]
eps1 = params_PSSCpure_fit_0_df['eps1'][0]
dseta1 = params_PSSCpure_fit_0_df['dseta1'][0]
eta1 = params_PSSCpure_fit_0_df['eta1'][0]
alpha2 = params_PSSCpure_fit_0_df['alpha2'][0]
beta2 = params_PSSCpure_fit_0_df['beta2'][0]
gamma2 = params_PSSCpure_fit_0_df['gamma2'][0]
delta2 = params_PSSCpure_fit_0_df['delta2'][0]
eps2 = params_PSSCpure_fit_0_df['eps2'][0]
dseta2 = params_PSSCpure_fit_0_df['dseta2'][0]
eta2 = params_PSSCpure_fit_0_df['eta2'][0]
params = [a, b, c, d, alpha1, beta1, gamma1, delta1, eps1, dseta1, eta1, alpha2, beta2, gamma2, delta2, eps2, dseta2, eta2]

data3_fit_PSSCpure_aux_df = data3_fit_PSSCpure_df[(data3_fit_PSSCpure_df['perturb_type']=='SC') & (data3_fit_PSSCpure_df['perturb_sign']=='neg') & (data3_fit_PSSCpure_df['perturb_size']==20)].reset_index(drop=True)
baseline_pre_SC_neg_20 = data3_fit_PSSCpure_aux_df['baseline_pre'][0]
baseline_post_SC_neg_20 = data3_fit_PSSCpure_aux_df['baseline_post'][0]
data3_fit_PSSCpure_aux_df = data3_fit_PSSCpure_df[(data3_fit_PSSCpure_df['perturb_type']=='SC') & (data3_fit_PSSCpure_df['perturb_sign']=='neg') & (data3_fit_PSSCpure_df['perturb_size']==50)].reset_index(drop=True)
baseline_pre_SC_neg_50 = data3_fit_PSSCpure_aux_df['baseline_pre'][0]
baseline_post_SC_neg_50 = data3_fit_PSSCpure_aux_df['baseline_post'][0]
data3_fit_PSSCpure_aux_df = data3_fit_PSSCpure_df[(data3_fit_PSSCpure_df['perturb_type']=='SC') & (data3_fit_PSSCpure_df['perturb_sign']=='pos') & (data3_fit_PSSCpure_df['perturb_size']==20)].reset_index(drop=True)
baseline_pre_SC_pos_20 = data3_fit_PSSCpure_aux_df['baseline_pre'][0]
baseline_post_SC_pos_20 = data3_fit_PSSCpure_aux_df['baseline_post'][0]
data3_fit_PSSCpure_aux_df = data3_fit_PSSCpure_df[(data3_fit_PSSCpure_df['perturb_type']=='SC') & (data3_fit_PSSCpure_df['perturb_sign']=='pos') & (data3_fit_PSSCpure_df['perturb_size']==50)].reset_index(drop=True)
baseline_pre_SC_pos_50 = data3_fit_PSSCpure_aux_df['baseline_pre'][0]
baseline_post_SC_pos_50 = data3_fit_PSSCpure_aux_df['baseline_post'][0]
data3_fit_PSSCpure_aux_df = data3_fit_PSSCpure_df[(data3_fit_PSSCpure_df['perturb_type']=='PS') & (data3_fit_PSSCpure_df['perturb_sign']=='neg') & (data3_fit_PSSCpure_df['perturb_size']==20)].reset_index(drop=True)
baseline_pre_PS_neg_20 = data3_fit_PSSCpure_aux_df['baseline_pre'][0]
baseline_post_PS_neg_20 = data3_fit_PSSCpure_aux_df['baseline_post'][0]
data3_fit_PSSCpure_aux_df = data3_fit_PSSCpure_df[(data3_fit_PSSCpure_df['perturb_type']=='PS') & (data3_fit_PSSCpure_df['perturb_sign']=='neg') & (data3_fit_PSSCpure_df['perturb_size']==50)].reset_index(drop=True)
baseline_pre_PS_neg_50 = data3_fit_PSSCpure_aux_df['baseline_pre'][0]
baseline_post_PS_neg_50 = data3_fit_PSSCpure_aux_df['baseline_post'][0]
data3_fit_PSSCpure_aux_df = data3_fit_PSSCpure_df[(data3_fit_PSSCpure_df['perturb_type']=='PS') & (data3_fit_PSSCpure_df['perturb_sign']=='pos') & (data3_fit_PSSCpure_df['perturb_size']==20)].reset_index(drop=True)
baseline_pre_PS_pos_20 = data3_fit_PSSCpure_aux_df['baseline_pre'][0]
baseline_post_PS_pos_20 = data3_fit_PSSCpure_aux_df['baseline_post'][0]
data3_fit_PSSCpure_aux_df = data3_fit_PSSCpure_df[(data3_fit_PSSCpure_df['perturb_type']=='PS') & (data3_fit_PSSCpure_df['perturb_sign']=='pos') & (data3_fit_PSSCpure_df['perturb_size']==50)].reset_index(drop=True)
baseline_pre_PS_pos_50 = data3_fit_PSSCpure_aux_df['baseline_pre'][0]
baseline_post_PS_pos_50 = data3_fit_PSSCpure_aux_df['baseline_post'][0]

# Simulation
col_names = ['origin','perturb_size','perturb_type','n','baseline','asyn','p','x','s']
model_data_df = pd.DataFrame(columns=col_names)

perturb_size_lst = [50, 50, -50, -50, 40, -40, 30, -30, 20, 20, -20, -20,
					50, 50, -50, -50, 40, -40, 30, -30, 20, 20, -20, -20]
perturb_type_lst = ['SC', 'SC', 'SC', 'SC', 'SC', 'SC', 'SC', 'SC', 'SC', 'SC', 'SC', 'SC',
					'PS', 'PS', 'PS', 'PS', 'PS', 'PS', 'PS', 'PS', 'PS', 'PS', 'PS', 'PS']
baseline_pre_lst = [baseline_pre_SC_pos_50, 0, baseline_pre_SC_neg_50, 0, 0, 0, 0, 0, baseline_pre_SC_pos_20, 0, baseline_pre_SC_neg_20, 0,
					baseline_pre_PS_pos_50, 0, baseline_pre_PS_neg_50, 0, 0, 0, 0, 0, baseline_pre_PS_pos_20, 0, baseline_pre_PS_neg_20, 0]
baseline_post_lst = [baseline_post_SC_pos_50, 0, baseline_post_SC_neg_50, 0, 0, 0, 0, 0, baseline_post_SC_pos_20, 0, baseline_post_SC_neg_20, 0,
					 baseline_post_PS_pos_50, 0, baseline_post_PS_neg_50, 0, 0, 0, 0, 0, baseline_post_PS_pos_20, 0, baseline_post_PS_neg_20, 0]

i = 0
for perturb_size in perturb_size_lst:
	hyper_params = [baseline_pre_lst[i], baseline_post_lst[i], perturb_type_lst[i], perturb_size]
	t, model_asyn, y = integrate(model, y_ini, params, hyper_params)
	origin_list = np.repeat('PSSCpure',len(t))[:,np.newaxis].T
	perturb_size_list = np.repeat(perturb_size,len(t))[:,np.newaxis].T
	perturb_type_list = np.repeat(perturb_type_lst[i],len(t))[:,np.newaxis].T
	step_n = np.array(range(0,n_beeps))[:,np.newaxis].T-perturb_beep+1

	if	(baseline_pre_lst[i] == 0):
		baseline_list = np.repeat(0,len(t))[:,np.newaxis].T
	else:
		baseline_list = np.repeat(1,len(t))[:,np.newaxis].T
	
	model_array = np.concatenate([ \
							   origin_list, \
							   perturb_size_list, \
							   perturb_type_list, \
							   step_n, \
							   baseline_list, \
							   model_asyn[:,np.newaxis].T, \
							   y.T], axis=0).T
	model_data_df = pd.concat([model_data_df, pd.DataFrame(model_array, columns=col_names)], axis=0)
	i = i+1
model_PSSCpure_0_df = model_data_df.reset_index(drop=True)
#model_PSSCpure_0_df.to_csv('model_PSSCpure_0_df.csv')




# Select fitting
params_PSSCpure_fit_1_df = params_PSSCpure3_df[(params_PSSCpure3_df['subpopulation'] == 1)].reset_index(drop=True)
#params_PSSCpure_fit_1_df.to_csv('params_PSSCpure_fit_1_df.csv')
params_PSSCpure_fit_1_df = params_PSSCpure_fit_1_df[(params_PSSCpure_fit_1_df['dist'] == params_PSSCpure_fit_1_df['dist'].min())].reset_index(drop=True)
#params_PSSCpure_fit_1_df = params_PSSCpure_fit_1_df.query('index==0').reset_index(drop=True)

# Parameters values
a = params_PSSCpure_fit_1_df['a'][0]
b = params_PSSCpure_fit_1_df['b'][0]
c = params_PSSCpure_fit_1_df['c'][0]
d = params_PSSCpure_fit_1_df['d'][0]
alpha1 = params_PSSCpure_fit_1_df['alpha1'][0]
beta1 = params_PSSCpure_fit_1_df['beta1'][0]
gamma1 = params_PSSCpure_fit_1_df['gamma1'][0]
delta1 = params_PSSCpure_fit_1_df['delta1'][0]
eps1 = params_PSSCpure_fit_1_df['eps1'][0]
dseta1 = params_PSSCpure_fit_1_df['dseta1'][0]
eta1 = params_PSSCpure_fit_1_df['eta1'][0]
alpha2 = params_PSSCpure_fit_1_df['alpha2'][0]
beta2 = params_PSSCpure_fit_1_df['beta2'][0]
gamma2 = params_PSSCpure_fit_1_df['gamma2'][0]
delta2 = params_PSSCpure_fit_1_df['delta2'][0]
eps2 = params_PSSCpure_fit_1_df['eps2'][0]
dseta2 = params_PSSCpure_fit_1_df['dseta2'][0]
eta2 = params_PSSCpure_fit_1_df['eta2'][0]
params = [a, b, c, d, alpha1, beta1, gamma1, delta1, eps1, dseta1, eta1, alpha2, beta2, gamma2, delta2, eps2, dseta2, eta2]

# Simulation
model_data_df = pd.DataFrame(columns=col_names)

i = 0
for perturb_size in perturb_size_lst:
	hyper_params = [baseline_pre_lst[i], baseline_post_lst[i], perturb_type_lst[i], perturb_size]
	t, model_asyn, y = integrate(model, y_ini, params, hyper_params)
	origin_list = np.repeat('PSSCpure',len(t))[:,np.newaxis].T
	perturb_size_list = np.repeat(perturb_size,len(t))[:,np.newaxis].T
	perturb_type_list = np.repeat(perturb_type_lst[i],len(t))[:,np.newaxis].T
	step_n = np.array(range(0,n_beeps))[:,np.newaxis].T-perturb_beep+1

	if	(baseline_pre_lst[i] == 0):
		baseline_list = np.repeat(0,len(t))[:,np.newaxis].T
	else:
		baseline_list = np.repeat(1,len(t))[:,np.newaxis].T
	
	model_array = np.concatenate([ \
							   origin_list, \
							   perturb_size_list, \
							   perturb_type_list, \
							   step_n, \
							   baseline_list, \
							   model_asyn[:,np.newaxis].T, \
							   y.T], axis=0).T
	model_data_df = pd.concat([model_data_df, pd.DataFrame(model_array, columns=col_names)], axis=0)
	i = i+1
model_PSSCpure_1_df = model_data_df.reset_index(drop=True)
#model_PSSCpure_1_df.to_csv('model_PSSCpure_1_df.csv')


#%%










#%% Plot model time series
color_map = ["blue","magenta","blue","magenta"]
shape_map = ["s","D"]
line_map = ["solid","dashed"]
marker_size = 2
base_Size = 10
x_lims = (-6,12)
y_lims = (-60, 60)
y_lims2 = (-100, 100)

lower_color = 0
upper_color = 3
num_colors = 5
color_map_rgba = cm.get_cmap('plasma')(np.linspace(lower_color,upper_color,num_colors))
color_map_hex = [rgb2hex(color, keep_alpha=True) for color in color_map_rgba.tolist()]
color_map_hex2 = ['#000000','#f89540ff']
color_map_hex3 = ['#000000','#0d0887ff']




# SCpure

data3_fit_SCpure_df.insert(0, column='origin_data', value='exp')
#data3_fit_SCpure_df.to_csv('data3_fit_SCpure_df.csv')

model_SCpure_0_df = model_SCpure_0_df.astype({'perturb_size':int})
model_SCpure_0_df = model_SCpure_0_df.astype({'baseline':int})
model_SCpure_0_50_df = model_SCpure_0_df[(model_SCpure_0_df["baseline"]==1) & ((model_SCpure_0_df["perturb_size"]==50) | (model_SCpure_0_df["perturb_size"]==-50) | (model_SCpure_0_df["perturb_size"]==20) | (model_SCpure_0_df["perturb_size"]==-20))] 
model_SCpure_0_50_df = model_SCpure_0_50_df.assign(perturb_sign = np.select([model_SCpure_0_50_df["perturb_size"]==50, model_SCpure_0_50_df["perturb_size"]==-50, model_SCpure_0_50_df["perturb_size"]==20, model_SCpure_0_50_df["perturb_size"]==-20],['pos','neg','pos','neg']))
model_SCpure_0_50_df.insert(0, column='origin_data', value='model')
model_SCpure_0_50_df.insert(1, column='context', value='pure')
model_SCpure_0_50_df.insert(2, column='perturb_type', value='SC')
model_SCpure_0_50_df["perturb_size"] = abs(model_SCpure_0_50_df["perturb_size"])
model_SCpure_0_50_df = model_SCpure_0_50_df.astype({'perturb_size':str})
model_SCpure_0_50_df['origin'] = model_SCpure_0_50_df[['context','perturb_type','perturb_sign','perturb_size']].agg(''.join, axis=1)
#model_SCpure_0_50_df.to_csv('model_SCpure_0_50_df.csv')

all_data_df = pd.concat([data3_fit_SCpure_df[['origin_data', 'origin', 'perturb_type', 'perturb_size', 'perturb_sign', 'n', 'asyn']], 
						 model_SCpure_0_50_df[['origin_data', 'origin', 'perturb_type', 'perturb_size', 'perturb_sign', 'n', 'asyn']]])
all_data_df['origin_data_x_origin'] = all_data_df[['origin_data','origin']].agg(''.join, axis=1)
all_data_df["n"] = all_data_df["n"].astype('int')
all_data_df = all_data_df[all_data_df['n']<=11]
#all_data_df.to_csv('all_data_df.csv')

all_data_df["asyn"] = all_data_df["asyn"].astype('float')
all_data_df["n"] = all_data_df["n"].astype('int')
all_data_df["perturb_size"] = all_data_df["perturb_size"].astype('string')
plot_model_timeseries = (
	ggplot(all_data_df, aes(x = 'n', y = 'asyn',						    
						 	 group = 'origin_data_x_origin',
							 color = 'origin_data',
							 shape = 'perturb_size',
							 linetype = 'perturb_sign'))
	+ geom_line()
	+ geom_point(size = marker_size)
	+ scale_color_manual(values = color_map_hex2)
	+ scale_shape_manual(values = shape_map)
	+ scale_linetype_manual(values = line_map)
	+ scale_x_continuous(breaks=range(x_lims[0]+1,x_lims[1],1))
	+ scale_y_continuous(limits=y_lims,breaks=range(y_lims[0],y_lims[1],10))
	+ theme_bw(base_size=22)
	+ theme(legend_title = element_text(size=18),
                     legend_text=element_text(size=18),
                     legend_key=element_rect(fill = "white", color = 'white'), 
                     figure_size = (12, 6))
	+ themes.theme(
                axis_title_y = themes.element_text(angle = 90, va = 'center', size = 22),
                axis_title_x = themes.element_text(va = 'center', size = 22))
	+ theme(strip_background = element_blank())
	+ ggtitle("(b) SCpure")
	+ xlab("n")
    + ylab("Asynchrony $e_n$ (ms)")
	)
print(plot_model_timeseries)
plot_model_timeseries_SCpure_0 = pw.load_ggplot(plot_model_timeseries)
#plot_model_timeseries.save('Adjustment_SCpure_0.pdf')




# PSpure

data3_fit_PSpure_df.insert(0, column='origin_data', value='exp')
#data3_fit_PSpure_df.to_csv('data3_fit_PSpure_df')

model_PSpure_0_df = model_PSpure_0_df.astype({'perturb_size':int})
model_PSpure_0_df = model_PSpure_0_df.astype({'baseline':int})
model_PSpure_0_50_df = model_PSpure_0_df[(model_PSpure_0_df["baseline"]==1) & ((model_PSpure_0_df["perturb_size"]==50) | (model_PSpure_0_df["perturb_size"]==-50) | (model_PSpure_0_df["perturb_size"]==20) | (model_PSpure_0_df["perturb_size"]==-20))] 
model_PSpure_0_50_df = model_PSpure_0_50_df.assign(perturb_sign = np.select([model_PSpure_0_50_df["perturb_size"]==50, model_PSpure_0_50_df["perturb_size"]==-50, model_PSpure_0_50_df["perturb_size"]==20, model_PSpure_0_50_df["perturb_size"]==-20],['pos','neg','pos','neg']))
model_PSpure_0_50_df.insert(0, column='origin_data', value='model')
model_PSpure_0_50_df.insert(1, column='context', value='pure')
model_PSpure_0_50_df.insert(2, column='perturb_type', value='PS')
model_PSpure_0_50_df["perturb_size"] = abs(model_PSpure_0_50_df["perturb_size"])
model_PSpure_0_50_df = model_PSpure_0_50_df.astype({'perturb_size':str})
model_PSpure_0_50_df['origin'] = model_PSpure_0_50_df[['context','perturb_type','perturb_sign','perturb_size']].agg(''.join, axis=1)
#model_PSpure_0_50_df.to_csv('model_PSpure_0_50_df')

all_data_df = pd.concat([data3_fit_PSpure_df[['origin_data', 'origin', 'perturb_type', 'perturb_size', 'perturb_sign', 'n', 'asyn']], 
						 model_PSpure_0_50_df[['origin_data', 'origin', 'perturb_type', 'perturb_size', 'perturb_sign', 'n', 'asyn']]])
all_data_df['origin_data_x_origin'] = all_data_df[['origin_data','origin']].agg(''.join, axis=1)
all_data_df["n"] = all_data_df["n"].astype('int')
all_data_df = all_data_df[all_data_df['n']<=11]
#all_data_df.to_csv('all_data_df.csv')

all_data_df["asyn"] = all_data_df["asyn"].astype('float')
all_data_df["n"] = all_data_df["n"].astype('int')
all_data_df["perturb_size"] = all_data_df["perturb_size"].astype('string')
plot_model_timeseries = (
	ggplot(all_data_df, aes(x = 'n', y = 'asyn',						    
						 	 group = 'origin_data_x_origin',
							 color = 'origin_data',
							 shape = 'perturb_size',
							 linetype = 'perturb_sign'))
	+ geom_line()
	+ geom_point(size = marker_size)
	+ scale_color_manual(values = color_map_hex3)
	+ scale_shape_manual(values = shape_map)
	+ scale_linetype_manual(values = line_map)
	+ scale_x_continuous(breaks=range(x_lims[0]+1,x_lims[1],1))
	+ scale_y_continuous(limits=y_lims,breaks=range(y_lims[0],y_lims[1],10))
	+ theme_bw(base_size=22)
	+ theme(legend_title = element_text(size=18),
                     legend_text=element_text(size=18),
                     legend_key=element_rect(fill = "white", color = 'white'), 
                     figure_size = (12, 6))
	+ themes.theme(
                axis_title_y = themes.element_text(angle = 90, va = 'center', size = 22),
                axis_title_x = themes.element_text(va = 'center', size = 22))
	+ theme(strip_background = element_blank())
	+ ggtitle("(a) PSpure")
	+ xlab("n")
    + ylab("Asynchrony $e_n$ (ms)")
	)
print(plot_model_timeseries)
plot_model_timeseries_PSpure_0 = pw.load_ggplot(plot_model_timeseries)
#plot_model_timeseries.save('Adjustment_PSpure_0.pdf')


#%% Plot phase-space diagram for model adjusted only to 50 and 20 ms

# Without baseline
model_SCpure_0_df = model_SCpure_0_df.astype({'n':int})
model_SCpure_0_df = model_SCpure_0_df.astype({'x':float})
model_SCpure_0_df = model_SCpure_0_df.astype({'s':float})
model_SCpure_0_df = model_SCpure_0_df.astype({'asyn':float})
model_SCpure_0_df = model_SCpure_0_df.astype({'perturb_size':int})
model_SCpure_0_PhaseS2_df = model_SCpure_0_df[(model_SCpure_0_df["baseline"]==0) & (model_SCpure_0_df["n"]>=1) & ((model_SCpure_0_df["perturb_size"]==50) | (model_SCpure_0_df["perturb_size"]==-50) | (model_SCpure_0_df["perturb_size"]==20) | (model_SCpure_0_df["perturb_size"]==-20))] 
model_SCpure_0_PhaseS2_df = model_SCpure_0_PhaseS2_df.assign(perturb_sign = np.select([model_SCpure_0_PhaseS2_df["perturb_size"]==50, model_SCpure_0_PhaseS2_df["perturb_size"]==-50, model_SCpure_0_PhaseS2_df["perturb_size"]==20, model_SCpure_0_PhaseS2_df["perturb_size"]==-20],['pos','neg','pos','neg'])) 
model_SCpure_0_PhaseS2_df["perturb_size"] = abs(model_SCpure_0_PhaseS2_df["perturb_size"])
model_SCpure_0_PhaseS2_df["perturb_type"] = 'SC'
model_SCpure_0_PhaseS2_df['x-Tpost'] = model_SCpure_0_PhaseS2_df['x'] - model_SCpure_0_PhaseS2_df['s']
#model_SCpure_0_PhaseS2_df.to_csv('model_SCpure_0_PhaseS2_df.csv')

model_PSpure_0_df = model_PSpure_0_df.astype({'n':int})
model_PSpure_0_df = model_PSpure_0_df.astype({'x':float})
model_PSpure_0_df = model_PSpure_0_df.astype({'s':float})
model_PSpure_0_df = model_PSpure_0_df.astype({'asyn':float})
model_PSpure_0_df = model_PSpure_0_df.astype({'perturb_size':int})
model_PSpure_0_PhaseS2_df = model_PSpure_0_df[(model_PSpure_0_df["baseline"]==0) & (model_PSpure_0_df["n"]>=2) & ((model_PSpure_0_df["perturb_size"]==50) | (model_PSpure_0_df["perturb_size"]==-50) | (model_PSpure_0_df["perturb_size"]==20) | (model_PSpure_0_df["perturb_size"]==-20))] 
model_PSpure_0_PhaseS2_df = model_PSpure_0_PhaseS2_df.assign(perturb_sign = np.select([model_PSpure_0_PhaseS2_df["perturb_size"]==50, model_PSpure_0_PhaseS2_df["perturb_size"]==-50, model_PSpure_0_PhaseS2_df["perturb_size"]==20, model_PSpure_0_PhaseS2_df["perturb_size"]==-20],['pos','neg','pos','neg'])) 
model_PSpure_0_PhaseS2_df["perturb_size"] = abs(model_PSpure_0_PhaseS2_df["perturb_size"])
model_PSpure_0_PhaseS2_df["perturb_type"] = 'PS'
model_PSpure_0_PhaseS2_df['x-Tpost'] = model_PSpure_0_PhaseS2_df['x'] - model_PSpure_0_PhaseS2_df['s']
#model_PSpure_0_PhaseS2_df.to_csv('model_PSpure_0_PhaseS2_df.csv')

model_PSSCpure_0_PhaseS2_df = pd.concat([model_PSpure_0_PhaseS2_df, model_SCpure_0_PhaseS2_df], axis=0).reset_index(drop=True)
model_PSSCpure_0_PhaseS2_df = model_PSSCpure_0_PhaseS2_df.astype({'perturb_size':str})
model_PSSCpure_0_PhaseS2_df['perturb_type_x_perturb_sign_x_perturb_size'] = model_PSSCpure_0_PhaseS2_df[['perturb_type','perturb_sign','perturb_size']].agg(''.join, axis=1)
model_PSSCpure_0_PhaseS2_df = model_PSSCpure_0_PhaseS2_df.astype({'p':float})
#model_PSSCpure_0_PhaseS2_df.to_csv('model_PSSCpure_0_PhaseS2_df.csv')

plot_phase_ps = (
	ggplot(model_PSSCpure_0_PhaseS2_df, aes(x = 'p', y = 'x-Tpost',
										 group = 'perturb_type_x_perturb_sign_x_perturb_size',
										 color = 'perturb_type',
										 shape = 'perturb_size',
										 linetype = 'perturb_sign'
										 ))
	+ geom_path()
	+ scale_color_manual(values = color_map_hex)
	+ scale_shape_manual(values = shape_map)
	+ scale_linetype_manual(values = line_map)
    + geom_point(model_PSSCpure_0_PhaseS2_df, aes(x='p', y='x-Tpost'), size = 4)     
    + theme_bw(base_size=22)
	+ theme(legend_title = element_text(size=18),
	                    legend_text=element_text(size=18),
	                    legend_key=element_rect(fill = "white", color = 'white'), 
	                    figure_size = (6, 6))
	+ themes.theme(
	               axis_title_y = themes.element_text(angle = 90, va = 'center', size = 22),
	               axis_title_x = themes.element_text(va = 'center', size = 22))
	+ theme(strip_background = element_blank())
	+ ggtitle("(c) PSpure vs SCpure")
	+ xlab("Predicted asynchrony $p_n$ (ms)")
    + ylab("$x_n - T_{post}$ (ms)")
	)
#print(plot_phase_ps)
plot_phase_PSSCpure_3 = pw.load_ggplot(plot_phase_ps)
#plot_phase_ps.save("model_PSSCpure_3_Phase-Space.pdf")


#%%
plot_model = (plot_model_timeseries_PSpure_0/plot_model_timeseries_SCpure_0)|plot_phase_PSSCpure_3
plot_model.savefig('Fig_Fit.pdf')
plot_model.savefig('Fig_Fit.png')


#%%










#%% Embedding data
embed_start = 1
embed_end = 8

data3_fit_PSSCcomb_df.insert(0, column='origin_data', value='exp')
data3_fit_PSSCpure_df.insert(0, column='origin_data', value='exp')
data3_fit_PSSCpurecomb_df = pd.concat([data3_fit_PSSCpure_df, data3_fit_PSSCcomb_df], axis=0).reset_index(drop=True)
#data3_fit_PSSCpurecomb_df.to_csv('data3_fit_PSSCpurecomb_df.csv')

data3_fit_PSSCpurecomb_df = (data3_fit_PSSCpurecomb_df
							  .assign(asyn_pred = np.select([(data3_fit_PSSCpurecomb_df['n']==0) & (data3_fit_PSSCpurecomb_df['perturb_sign']=='pos'),
															(data3_fit_PSSCpurecomb_df['n']==0) & (data3_fit_PSSCpurecomb_df['perturb_sign']=='neg'),
															(data3_fit_PSSCpurecomb_df['n']==1) & (data3_fit_PSSCpurecomb_df['perturb_type']=='PS') & (data3_fit_PSSCpurecomb_df['perturb_sign']=='pos'),
															(data3_fit_PSSCpurecomb_df['n']==1) & (data3_fit_PSSCpurecomb_df['perturb_type']=='PS') & (data3_fit_PSSCpurecomb_df['perturb_sign']=='neg')],
				 											[data3_fit_PSSCpurecomb_df['asyn'] + data3_fit_PSSCpurecomb_df['perturb_size'],
															 data3_fit_PSSCpurecomb_df['asyn'] - data3_fit_PSSCpurecomb_df['perturb_size'],
															 data3_fit_PSSCpurecomb_df['asyn'] - data3_fit_PSSCpurecomb_df['perturb_size'],
															 data3_fit_PSSCpurecomb_df['asyn'] + data3_fit_PSSCpurecomb_df['perturb_size']],
															 default = data3_fit_PSSCpurecomb_df['asyn']))
							  )
#data3_fit_PSSCpurecomb_df.to_csv('data3_fit_PSSCpurecomb_df.csv')

data3_fit_PSSCpurecomb_df['asyn_post'] = data3_fit_PSSCpurecomb_df['asyn'] - data3_fit_PSSCpurecomb_df['baseline_post']
data3_fit_PSSCpurecomb_df['asyn_pred_post'] = data3_fit_PSSCpurecomb_df['asyn_pred'] - data3_fit_PSSCpurecomb_df['baseline_post']
#data3_fit_PSSCpurecomb_df.to_csv('data3_fit_PSSCpurecomb_df.csv')

# time-delayed embedding
data3_fit_PSSCpurecomb_df['asyn_post_prev'] = (data3_fit_PSSCpurecomb_df
											   .groupby(['origin','context','perturb_type','perturb_sign','perturb_size'], as_index=False)['asyn_post']
											   .shift(1)	)
data3_fit_PSSCpurecomb_df['asyn_pred_post_prev'] = (data3_fit_PSSCpurecomb_df
													.groupby(['origin','context','perturb_type','perturb_sign','perturb_size'], as_index=False)['asyn_pred_post']
													.shift(1))
data3_fit_PSSCpurecomb_df.dropna(inplace=True)
#data3_fit_PSSCpurecomb_df.to_csv('data3_fit_PSSCpurecomb_df.csv')

# difference embedding
data3_fit_PSSCpurecomb_df['asyn_post_diff'] = data3_fit_PSSCpurecomb_df['asyn_post'] - data3_fit_PSSCpurecomb_df['asyn_post_prev']
data3_fit_PSSCpurecomb_df['asyn_pred_post_diff'] = data3_fit_PSSCpurecomb_df['asyn_pred_post'] - data3_fit_PSSCpurecomb_df['asyn_pred_post_prev']
#data3_fit_PSSCpurecomb_df.to_csv('data3_fit_PSSCpurecomb_df.csv')

# select transient phase only (PS starts one beep later than SC)
data3_fit_PSSCpurecomb_df = data3_fit_PSSCpurecomb_df[(((data3_fit_PSSCpurecomb_df['perturb_type']=='SC') & (data3_fit_PSSCpurecomb_df['n']>=embed_start))
													   | ((data3_fit_PSSCpurecomb_df['perturb_type']=='PS') & (data3_fit_PSSCpurecomb_df['n']>=embed_start+1)))
													  & (data3_fit_PSSCpurecomb_df['n']<=embed_end)]
#data3_fit_PSSCpurecomb_df.to_csv('data3_fit_PSSCpurecomb_df.csv')


#%% Plot embedding, ASYN_PRED difference, perturb_size 20 and 50 together
x_lims = [-60,60]
y_lims = [-60,60]
fig_xsize = 10 * 0.393701   # centimeter to inch
fig_ysize = 10 * 0.393701   # centimeter to inch

data3_fit_pure_20_df = data3_fit_PSSCpurecomb_df[(data3_fit_PSSCpurecomb_df['context'] == 'pure') & (data3_fit_PSSCpurecomb_df['perturb_size'] == 20)]
data3_fit_pure_50_df = data3_fit_PSSCpurecomb_df[(data3_fit_PSSCpurecomb_df['context'] == 'pure') & (data3_fit_PSSCpurecomb_df['perturb_size'] == 50)]
data3_fit_pure_df = pd.concat([data3_fit_pure_20_df, data3_fit_pure_50_df], axis=0).reset_index(drop=True)
data3_fit_pure_df['perturb_size'] = data3_fit_pure_df['perturb_size'].astype('str') 
#data3_fit_pure_df.to_csv('data3_fit_pure_df.csv')
plot_embed_p_diff_facet = (
	ggplot(data3_fit_pure_df, aes(x = 'asyn_pred_post', y = 'asyn_pred_post_diff',
								  group = 'origin',
								  color = 'perturb_type',
								  linetype = 'perturb_sign',
								  shape = 'perturb_size'))
 		 + geom_path()
		 + geom_point(size = marker_size)
		 + scale_color_manual(values = color_map_hex)
		 + scale_linetype_manual(values = line_map)
		 + scale_shape_manual(values = shape_map)
		 + scale_x_continuous(breaks=range(x_lims[0],x_lims[1],20))
		 + scale_y_continuous(breaks=range(y_lims[0],y_lims[1],20))
		 + theme_bw(base_size=14)	 	 
 		 + theme(legend_title = element_text(size=9),
	                     legend_text=element_text(size=9),
	                     legend_key=element_rect(fill = "white", color = 'white'), 
	                     figure_size = (fig_xsize, fig_ysize))
		 + themes.theme(
                 axis_title_y = themes.element_text(angle = 90, va = 'center', size = 12),
                 axis_title_x = themes.element_text(va = 'center', size = 12))
		 + theme(strip_background = element_blank())
		 + xlab("Predicted asynchrony $p_n$ (ms)")
		 + ylab("$p_n - p_{n-1}$ (ms)")
		 + ggtitle("(b)")
		 )
#print(plot_embed_p_diff_facet)
td_emb_asyn_pred_diff_pure = pw.load_ggplot(plot_embed_p_diff_facet)
#plot_embed_p_diff_facet.save("time-delayed_embedding_asyn_pred_difference_pure.pdf")

data3_fit_comb_20_df = data3_fit_PSSCpurecomb_df[(data3_fit_PSSCpurecomb_df['context'] == 'comb') & (data3_fit_PSSCpurecomb_df['perturb_size'] == 20)]
data3_fit_comb_50_df = data3_fit_PSSCpurecomb_df[(data3_fit_PSSCpurecomb_df['context'] == 'comb') & (data3_fit_PSSCpurecomb_df['perturb_size'] == 50)]
data3_fit_comb_df = pd.concat([data3_fit_comb_20_df, data3_fit_comb_50_df], axis=0).reset_index(drop=True)
data3_fit_comb_df['perturb_size'] = data3_fit_comb_df['perturb_size'].astype('str') 
#data3_fit_comb_df.to_csv('data3_fit_comb_df.csv')
plot_embed_p_diff_facet = (
	ggplot(data3_fit_comb_df, aes(x = 'asyn_pred_post', y = 'asyn_pred_post_diff',
								  group = 'origin',
								  color = 'perturb_type',
								  linetype = 'perturb_sign',
								  shape = 'perturb_size'))
 		 + geom_path()
		 + geom_point(size = marker_size)
		 + scale_color_manual(values = color_map_hex)
		 + scale_linetype_manual(values = line_map)
		 + scale_shape_manual(values = shape_map)
		 + scale_x_continuous(breaks=range(x_lims[0],x_lims[1],20))
		 + scale_y_continuous(breaks=range(y_lims[0],y_lims[1],20))
		 + theme_bw(base_size=14)		 	 
 		 + theme(legend_title = element_text(size=9),
	                     legend_text=element_text(size=9),
	                     legend_key=element_rect(fill = "white", color = 'white'), 
	                     figure_size = (fig_xsize, fig_ysize))
		 + themes.theme(
                 axis_title_y = themes.element_text(angle = 90, va = 'center', size = 12),
                 axis_title_x = themes.element_text(va = 'center', size = 12))
		 + theme(strip_background = element_blank())
		 + xlab("Predicted asynchrony $p_n$ (ms)")
		 + ylab("$p_n - p_{n-1}$ (ms)")
		 + ggtitle("(d)")
		 )
#print(plot_embed_p_diff_facet)
td_emb_asyn_pred_diff_comb = pw.load_ggplot(plot_embed_p_diff_facet)
#plot_embed_p_diff_facet.save("time-delayed_embedding_asyn_pred_difference_comb.pdf")


#%% Experimental data timeseries for embeddings
x_lims = [-4,11]
y_lims = [-60, 60]
fig_xsize = 15 * 0.393701   # centimeter to inch
fig_ysize = 10 * 0.393701   # centimeter to inch

exp_PSSCpure_df = data3_fit_PSSCpure_df[['origin_data','origin','context','perturb_type','perturb_sign','perturb_size','n','asyn']]
exp_PSSCpure_df = exp_PSSCpure_df[(exp_PSSCpure_df['n'] >= -3) & (exp_PSSCpure_df['n'] < 11)]
exp_PSSCpure_df['perturb_size'] = exp_PSSCpure_df['perturb_size'].astype('str') 
#exp_PSSCpure_df.to_csv('exp_PSSCpure_df.csv')
plot_model_timeseries = (
	ggplot(exp_PSSCpure_df, aes(x = 'n', y = 'asyn',
								 group = 'origin',
								 color = 'perturb_type',
								 linetype = 'perturb_sign',
								 shape = 'perturb_size'))
	+ geom_line()
	+ geom_point(size = marker_size)
	+ scale_color_manual(values = color_map_hex, guide=False)
	+ scale_linetype_manual(values = line_map, guide=False)
	+ scale_shape_manual(values = shape_map, guide=False)
	+ scale_x_continuous(breaks=range(x_lims[0]+1,x_lims[1],1))
	+ scale_y_continuous(breaks=range(y_lims[0],y_lims[1],10))
	+ theme_bw(base_size=16)
	+ theme(legend_title = element_text(size=16),
                     legend_text=element_text(size=16),
                     legend_key=element_rect(fill = "white", color = 'white'), 
                     figure_size = (fig_xsize, fig_ysize))
	+ themes.theme(
                axis_title_y = themes.element_text(angle = 90, va = 'center', size = 14),
                axis_title_x = themes.element_text(va = 'center', size = 14))
	+ theme(strip_background = element_blank())
	+ xlab("n")
	+ ylab("Asynchrony $e_n$ (ms)")
	+ ggtitle("(a)")
	)
#print(plot_model_timeseries)
plot_model_timeseries_pure = pw.load_ggplot(plot_model_timeseries)
#plot_model_timeseries.save('Exp_data_pure.pdf')

exp_PSSCcomb_df = data3_fit_PSSCcomb_df[['origin_data','origin','context','perturb_type','perturb_sign','perturb_size','n','asyn']]
exp_PSSCcomb_df = exp_PSSCcomb_df[(exp_PSSCcomb_df['n'] >= -3) & (exp_PSSCpure_df['n'] < 11)]
exp_PSSCcomb_df['perturb_size'] = exp_PSSCcomb_df['perturb_size'].astype('str') 
#exp_PSSCcomb_df.to_csv('exp_PSSCcomb_df.csv')
plot_model_timeseries = (
	ggplot(exp_PSSCcomb_df, aes(x = 'n', y = 'asyn',
								 group = 'origin',
								 color = 'perturb_type',
								 linetype = 'perturb_sign',
								 shape = 'perturb_size'))
	+ geom_line()
	+ geom_point(size = marker_size)
	+ scale_color_manual(values = color_map_hex, guide=False)
	+ scale_linetype_manual(values = line_map, guide=False)
	+ scale_shape_manual(values = shape_map, guide=False)
	+ scale_x_continuous(breaks=range(x_lims[0]+1,x_lims[1],1))
	+ scale_y_continuous(breaks=range(y_lims[0],y_lims[1],10))
	+ theme_bw(base_size=16)
	+ theme(legend_title = element_text(size=16),
                     legend_text=element_text(size=16),
                     legend_key=element_rect(fill = "white", color = 'white'), 
                     figure_size = (fig_xsize, fig_ysize))
	+ themes.theme(
                axis_title_y = themes.element_text(angle = 90, va = 'center', size = 14),
                axis_title_x = themes.element_text(va = 'center', size = 14))
	+ theme(strip_background = element_blank())
	+ xlab("n")
	+ ylab("Asynchrony $e_n$ (ms)")
	+ ggtitle("(c)")
	)
#print(plot_model_timeseries)
plot_model_timeseries_comb = pw.load_ggplot(plot_model_timeseries)
#plot_model_timeseries.save('Exp_data_comb.pdf')


#%%
plot_model = (plot_model_timeseries_pure|td_emb_asyn_pred_diff_pure)/(plot_model_timeseries_comb|td_emb_asyn_pred_diff_comb)
plot_model.savefig("Fig_Embeddings.pdf")
plot_model.savefig("Fig_Embeddings.png")


#%%










#%% Correlation between parameters
params2_PSSCcomb_df = params_PSSCcomb_df.drop(columns=['alpha1', 'beta1', 'gamma1', 'delta1', 'eps1', 'eta1', 'beta2',
													   'gamma2', 'delta2', 'eps2', 'dseta2', 'eta2', 'dist']).reset_index(drop=True)
#params2_PSSCcomb_df.to_csv('params2_PSSCcomb_df.csv')
with plt.rc_context({
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
}):
	correlation_parameters = sns.pairplot(
		params2_PSSCcomb_df,
        corner=True,
        height=1.2,
        aspect=1,
		plot_kws=dict(s=10)
	)
	
	for ax in correlation_parameters.axes.flatten():
		if ax is not None:
			ax.ticklabel_format(
				axis='x',
				style='sci',
				scilimits=(-2, 2)
			)
			
	label_map = {
		"a": "a (nondim.)",
		"b": "b (nondim.)",
		"c": "c (nondim.)",
		"d": "d (nondim.)",
		"alpha2": r"$\delta$ (ms$^{-1}$)",
		"dseta1": r"$\beta$ (ms$^{-2}$)"
	}
	for ax in correlation_parameters.axes.flatten():
		if ax is not None:
			xlabel = ax.get_xlabel()
			ylabel = ax.get_ylabel()
			if xlabel in label_map:
				ax.set_xlabel(label_map[xlabel])
			if ylabel in label_map:
				ax.set_ylabel(label_map[ylabel])
				
	for ax in correlation_parameters.axes.flatten():
		if ax is not None:
			ax.xaxis.get_offset_text().set_fontsize(4.5)
			ax.yaxis.get_offset_text().set_fontsize(4.5)
		
	correlation_parameters.savefig("Fig_Corr_params_PSSCcomb.pdf", bbox_inches="tight")
	correlation_parameters.savefig("Fig_Corr_params_PSSCcomb.png", dpi=300, bbox_inches="tight")


params2_PSSCpure_df = params_PSSCpure_df.drop(columns=['alpha1', 'beta1', 'gamma1', 'delta1', 'eps1', 'eta1', 'beta2',
													   'gamma2', 'delta2', 'eps2', 'dseta2', 'eta2', 'dist']).reset_index(drop=True)
#params2_PSSCpure_df.to_csv('params2_PSSCpure_df.csv')
with plt.rc_context({
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
}):
	correlation_parameters = sns.pairplot(
		params2_PSSCpure_df,
        corner=True,
        height=1.2,
        aspect=1,
		plot_kws=dict(s=10)
	)
	for ax in correlation_parameters.axes.flatten():
		if ax is not None:
			ax.ticklabel_format(
				axis='x',
				style='sci',
				scilimits=(-2, 2)
			)
			
	label_map = {
		"a": "a (nondim.)",
		"b": "b (nondim.)",
		"c": "c (nondim.)",
		"d": "d (nondim.)",
		"alpha2": r"$\delta$ (ms$^{-1}$)",
		"dseta1": r"$\beta$ (ms$^{-2}$)"
	}	
	for ax in correlation_parameters.axes.flatten():
		if ax is not None:
			xlabel = ax.get_xlabel()
			ylabel = ax.get_ylabel()
			if xlabel in label_map:
				ax.set_xlabel(label_map[xlabel])
			if ylabel in label_map:
				ax.set_ylabel(label_map[ylabel])
				
	for ax in correlation_parameters.axes.flatten():
		if ax is not None:
			ax.xaxis.get_offset_text().set_fontsize(4.5)
			ax.yaxis.get_offset_text().set_fontsize(4.5)
	
	correlation_parameters.savefig("Fig_Corr_params_PSSCpure.pdf", bbox_inches="tight")
	correlation_parameters.savefig("Fig_Corr_params_PSSCpure.png", dpi=300, bbox_inches="tight")


#%%










#%% Histograms of the parameters SCpure vs PSpure.

# Pure
params_SCpure3_df['perturb_type'] = 'SC'
params_PSpure3_df['perturb_type'] = 'PS'
params_pure_df = pd.concat([params_SCpure3_df, params_PSpure3_df], axis=0).reset_index(drop=True)
params_pure_df = params_pure_df.astype({'perturb_type':str})
params_pure_df = params_pure_df.astype({'subpopulation':str})
params_pure_df['perturb_type_x_subpopulation'] = params_pure_df[['perturb_type','subpopulation']].agg(''.join, axis=1)
params_pure2_df = params_pure_df[(params_pure_df['perturb_type_x_subpopulation']=='PS0') | (params_pure_df['perturb_type_x_subpopulation']=='SC1')]
#params_pure2_df.to_csv('params_pure_df.csv')

params_pure2_df['perturb_type_x_subpopulation'] = params_pure2_df['perturb_type_x_subpopulation'].replace({'SC1': 'SC', 'PS0': 'PS'})
#params_pure2_df.to_csv('params_pure_df.csv')

fig_xsize = 15 * 0.393701   # centimeter to inch
fig_ysize = 10 * 0.393701   # centimeter to inch

fig, axs = plt.subplots(3, 2, figsize=(fig_xsize, fig_ysize))
sns.set_theme(style="ticks")

custom_palette = {
	'SC': orange_dark,
    'PS': violet_dark
}

sns.set_context("notebook", font_scale=1, rc={"xtick.labelsize":5, "ytick.labelsize":5, "axes.labelsize":6, "legend.fontsize":4, "legend.title_fontsize":4})

sns.histplot(data=params_pure2_df, x="a", bins=50, ax=axs[0, 0], kde=False, hue='perturb_type_x_subpopulation', palette=custom_palette, alpha=0.6)
axs[0, 0].set_xlabel("a (nondim.)")
axs[0, 0].get_legend().set_title("")
axs[0, 0].text(
    0.02, 0.97, "(a)",
    transform=axs[0, 0].transAxes,
    fontsize=7,
    va="top"
)
axs[0, 0].yaxis.set_major_locator(MaxNLocator(integer=True))

sns.histplot(data=params_pure2_df, x="b", bins=50, ax=axs[0, 1], kde=False, hue='perturb_type_x_subpopulation', palette=custom_palette, alpha=0.6)
axs[0, 1].set_xlabel("b (nondim.)")
axs[0, 1].get_legend().set_title("")
axs[0, 1].text(
    0.02, 0.97, "(b)",
    transform=axs[0, 1].transAxes,
    fontsize=7,
    va="top"
)
axs[0, 1].yaxis.set_major_locator(MaxNLocator(integer=True))

sns.histplot(data=params_pure2_df, x="c", bins=50, ax=axs[1, 0], kde=False, hue='perturb_type_x_subpopulation', palette=custom_palette, alpha=0.6)
axs[1, 0].set_xlabel("c (nondim.)")
axs[1, 0].get_legend().set_title("")
axs[1, 0].text(
    0.02, 0.97, "(c)",
    transform=axs[1, 0].transAxes,
    fontsize=7,
    va="top"
)
axs[1, 0].yaxis.set_major_locator(MaxNLocator(integer=True))

sns.histplot(data=params_pure2_df, x="d", bins=50, ax=axs[1, 1], kde=False, hue='perturb_type_x_subpopulation', palette=custom_palette, alpha=0.6)
axs[1, 1].set_xlabel("d (nondim.)")
axs[1, 1].get_legend().set_title("")
axs[1, 1].text(
    0.02, 0.97, "(d)",
    transform=axs[1, 1].transAxes,
    fontsize=7,
    va="top"
)
axs[1, 1].yaxis.set_major_locator(MaxNLocator(integer=True))

sns.histplot(data=params_pure2_df, x="dseta1", bins=50, ax=axs[2, 0], kde=False, hue='perturb_type_x_subpopulation', palette=custom_palette, alpha=0.6)
axs[2, 0].set_xlabel("β (ms$^{-2}$)")
axs[2, 0].get_legend().set_title("")
axs[2, 0].text(
    0.02, 0.97, "(e)",
    transform=axs[2, 0].transAxes,
    fontsize=7,
    va="top"
)
axs[2, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
axs[2, 0].ticklabel_format(style='sci', axis='x', scilimits=(-2, 2))

sns.histplot(data=params_pure2_df, x="alpha2", bins=50, ax=axs[2, 1], kde=False, hue='perturb_type_x_subpopulation', palette=custom_palette, alpha=0.6)
axs[2, 1].set_xlabel("δ (ms$^{-1}$)")
axs[2, 1].get_legend().set_title("")
axs[2, 1].text(
    0.02, 0.97, "(f)",
    transform=axs[2, 1].transAxes,
    fontsize=7,
    va="top"
)
axs[2, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
axs[2, 1].ticklabel_format(style='sci', axis='x', scilimits=(-2, 2))

for i, ax in enumerate(axs.flat):
    if i == 0:
        ax.get_legend().set_title("")
    else:
        ax.get_legend().remove()

for i, ax in enumerate(axs.flat):
	ax.tick_params(axis='both', which='both', length=3, width=0.5, colors='black') 

for i, ax in enumerate(axs.flat):
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

plt.tight_layout()
plt.savefig("Fig_Distribution2.pdf")
plt.savefig("Fig_Distribution2.png", dpi=300)
plt.show()


#%% Histograms of the parameters SCcomb vs PScomb.

# Comb
params_SCcomb3_df['perturb_type'] = 'SC'
params_PScomb3_df['perturb_type'] = 'PS'
params_comb_df = pd.concat([params_SCcomb3_df, params_PScomb3_df], axis=0).reset_index(drop=True)
params_comb_df = params_comb_df.astype({'perturb_type':str})
params_comb_df = params_comb_df.astype({'subpopulation':str})
params_comb_df['perturb_type_x_subpopulation'] = params_comb_df[['perturb_type','subpopulation']].agg(''.join, axis=1)
params_comb_df['perturb_type_x_subpopulation'] = params_comb_df['perturb_type_x_subpopulation'].replace({'SC0': 'SC', 'PS0': 'PS'})
params_comb_df.to_csv(datafolder + 'params_comb_df.csv')

fig_xsize = 15 * 0.393701   # centimeter to inch
fig_ysize = 10 * 0.393701   # centimeter to inch

fig, axs = plt.subplots(3, 2, figsize=(fig_xsize, fig_ysize))
sns.set_theme(style="ticks")

custom_palette = {
	'SC': orange_dark,
    'PS': violet_dark
}

sns.set_context("notebook", font_scale=1, rc={"xtick.labelsize":5, "ytick.labelsize":5, "axes.labelsize":6, "legend.fontsize":4, "legend.title_fontsize":4})

sns.histplot(data=params_comb_df, x="a", bins=50, ax=axs[0, 0], kde=False, hue='perturb_type_x_subpopulation', palette=custom_palette, alpha=0.6)
axs[0, 0].set_xlabel("a (nondim.)")
axs[0, 0].get_legend().set_title("")
axs[0, 0].text(
    0.02, 0.97, "(a)",
    transform=axs[0, 0].transAxes,
    fontsize=7,
    va="top"
)
axs[0, 0].yaxis.set_major_locator(MaxNLocator(integer=True))

sns.histplot(data=params_comb_df, x="b", bins=50, ax=axs[0, 1], kde=False, hue='perturb_type_x_subpopulation', palette=custom_palette, alpha=0.6)
axs[0, 1].set_xlabel("b (nondim.)")
axs[0, 1].get_legend().set_title("")
axs[0, 1].text(
    0.02, 0.97, "(b)",
    transform=axs[0, 1].transAxes,
    fontsize=7,
    va="top"
)
axs[0, 1].yaxis.set_major_locator(MaxNLocator(integer=True))

sns.histplot(data=params_comb_df, x="c", bins=50, ax=axs[1, 0], kde=False, hue='perturb_type_x_subpopulation', palette=custom_palette, alpha=0.6)
axs[1, 0].set_xlabel("c (nondim.)")
axs[1, 0].get_legend().set_title("")
axs[1, 0].text(
    0.02, 0.97, "(c)",
    transform=axs[1, 0].transAxes,
    fontsize=7,
    va="top"
)
axs[1, 0].yaxis.set_major_locator(MaxNLocator(integer=True))

sns.histplot(data=params_comb_df, x="d", bins=50, ax=axs[1, 1], kde=False, hue='perturb_type_x_subpopulation', palette=custom_palette, alpha=0.6)
axs[1, 1].set_xlabel("d (nondim.)")
axs[1, 1].get_legend().set_title("")
axs[1, 1].text(
    0.02, 0.97, "(d)",
    transform=axs[1, 1].transAxes,
    fontsize=7,
    va="top"
)
axs[1, 1].yaxis.set_major_locator(MaxNLocator(integer=True))

sns.histplot(data=params_comb_df, x="dseta1", bins=50, ax=axs[2, 0], kde=False, hue='perturb_type_x_subpopulation', palette=custom_palette, alpha=0.6)
axs[2, 0].set_xlabel("β (ms$^{-2}$)")
axs[2, 0].get_legend().set_title("")
axs[2, 0].text(
    0.02, 0.97, "(e)",
    transform=axs[2, 0].transAxes,
    fontsize=7,
    va="top"
)
axs[2, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
axs[2, 0].ticklabel_format(style='sci', axis='x', scilimits=(-2, 2))

sns.histplot(data=params_comb_df, x="alpha2", bins=50, ax=axs[2, 1], kde=False, hue='perturb_type_x_subpopulation', palette=custom_palette, alpha=0.6)
axs[2, 1].set_xlabel("δ (ms$^{-1}$)")
axs[2, 1].get_legend().set_title("")
axs[2, 1].text(
    0.02, 0.97, "(f)",
    transform=axs[2, 1].transAxes,
    fontsize=7,
    va="top"
)
axs[2, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
axs[2, 1].ticklabel_format(style='sci', axis='x', scilimits=(-2, 2))

for i, ax in enumerate(axs.flat):
    if i == 0:
        ax.get_legend().set_title("")
    else:
        ax.get_legend().remove()

for i, ax in enumerate(axs.flat):
	ax.tick_params(axis='both', which='both', length=3, width=0.5, colors='black') 

for i, ax in enumerate(axs.flat):
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

plt.tight_layout()
plt.savefig("Fig_Distribution3.pdf")
plt.savefig("Fig_Distribution3.png", dpi=300)
plt.show()


#%% Histograms of the parameters a and dseta1, SCpure vs PSpure.
params_pure_SC1_df = params_pure_df[(params_pure_df['perturb_type_x_subpopulation'] == 'SC1')]
#params_pure_SC1_df.to_csv('params_pure_SC1_df.csv')
dseta1_lst = params_pure_SC1_df['dseta1'].tolist()
low_lim, up_lim = iqr(dseta1_lst)
params_pure_SC1_df = params_pure_SC1_df[(params_pure_SC1_df['dseta1'] >= low_lim) & (params_pure_SC1_df['dseta1'] <= up_lim)]
#params_pure_SC1_df.to_csv('params_pure_SC1_df.csv')

params_pure_PS0_df = params_pure_df[(params_pure_df['perturb_type_x_subpopulation'] == 'PS0')]
#params_pure_PS0_df.to_csv('params_pure_PS0_df.csv')
dseta1_lst = params_pure_PS0_df['dseta1'].tolist()
low_lim, up_lim = iqr(dseta1_lst)
params_pure_PS0_df = params_pure_PS0_df[(params_pure_PS0_df['dseta1'] >= low_lim) & (params_pure_PS0_df['dseta1'] <= up_lim)]
#params_pure_PS0_df.to_csv('params_pure_PS0_df.csv')

params_pure2_df = pd.concat([params_pure_SC1_df, params_pure_PS0_df], axis=0).reset_index(drop=True)
params_pure2_df['perturb_type_x_subpopulation'] = params_pure2_df['perturb_type_x_subpopulation'].replace({'SC1': 'SC', 'PS0': 'PS'})
params_pure2_df.to_csv(datafolder + 'params_pure2_df.csv')

fig_xsize = 15 * 0.393701   # centimeter to inch
fig_ysize = 7 * 0.393701   # centimeter to inch

fig, axs = plt.subplots(1, 2, figsize=(fig_xsize, fig_ysize))
sns.set_theme(style="ticks")

custom_palette = {  
    'SC': orange_dark,
	'PS': violet_dark
}

sns.set_context("notebook", font_scale=1, rc={"xtick.labelsize":5, "ytick.labelsize":5, "axes.labelsize":6, "legend.fontsize":4, "legend.title_fontsize":4})
ax = axs[0]
sns.histplot(data=params_pure2_df, x="a", bins=50, ax=ax, kde=False, hue='perturb_type_x_subpopulation', palette=custom_palette, alpha=0.6)
axs[0].set_xlabel("a (nondim.)")
ax.get_legend().set_title("")
axs[0].text(
    0.02, 0.97, "(a)",
    transform=axs[0].transAxes,
    fontsize=7,
    va="top"
)
axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))

sns.histplot(data=params_pure2_df, x="dseta1", bins=50, ax=axs[1], kde=False, hue='perturb_type_x_subpopulation', palette=custom_palette, alpha=0.6, legend=False)
axs[1].set_xlabel("β (ms$^{-2}$)")
axs[1].text(
    0.02, 0.97, "(b)",
    transform=axs[1].transAxes,
    fontsize=7,
    va="top"
)
axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))
axs[1].ticklabel_format(style='sci', axis='x', scilimits=(-2, 2))

for ax in axs: 
	ax.tick_params(axis='both', which='both', length=3, width=0.5, colors='black') 
#	ax.minorticks_on()
	
for ax in axs:
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

plt.tight_layout()
plt.savefig("Fig_Distribution.pdf")
plt.savefig("Fig_Distribution.png", dpi=300)
plt.show()


#%%

