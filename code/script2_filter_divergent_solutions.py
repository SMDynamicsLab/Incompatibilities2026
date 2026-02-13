#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 12:25:32 2025

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

datafolder = '../data/'
SHOW_AND_SAVE_DIVERGENT_PLOTS = 0


#%%










#%% Function definitions and Model

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
params_PSpure_df = pd.read_csv(datafolder + 'params_PSpure_df.csv', index_col=0).reset_index(drop=True)
params_PSSCpure_df = pd.read_csv(datafolder + 'params_PSSCpure_df.csv', index_col=0).reset_index(drop=True)
params_SCcomb_df = pd.read_csv(datafolder + 'params_SCcomb_df.csv', index_col=0).reset_index(drop=True)
params_PScomb_df = pd.read_csv(datafolder + 'params_PScomb_df.csv', index_col=0).reset_index(drop=True)
params_PSSCcomb_df = pd.read_csv(datafolder + 'params_PSSCcomb_df.csv', index_col=0).reset_index(drop=True)


#%%










#%% Search for divergent fittings
n_beeps = 40
perturb_beep = 6
ISI = 500
p_start, x_start, s_start = 0, ISI, ISI
y_ini = [p_start, x_start, s_start]         # p(asincronía predicha)   x(variable que explica el overshoot en step change)   s(ISI en n-1)

# Creating parameters list
parameter_family_lst = [params_SCpure_df, params_PSpure_df, params_PSSCpure_df, params_SCcomb_df, params_PScomb_df, params_PSSCcomb_df]
parameter_family_name_lst = ['SCpure', 'PSpure', 'PSSCpure', 'SCcomb', 'PScomb', 'PSSCcomb']
#parameter_family_lst = [params_PSSCpure_df, params_PSSCcomb_df]
#parameter_family_name_lst = ['PSSCpure', 'PSSCcomb']
#parameter_family_lst = [params_PSSCpure_df]
#parameter_family_name_lst = ['PSSCpure']
col_names = ['origin','perturb_size','perturb_type','n','asyn','p','x','s']

# Filter value
limit = 0.5 * ISI

# Parameters for graphs
marker_size = 2
base_Size = 10
y_lims = (-100, 100)

parameter_family_divergent_column_lst = []
i = 0
for parameter_family in parameter_family_lst:
	params_df = parameter_family
	#params_df.to_csv('params_df.csv')
	print('Condition: ' + parameter_family_name_lst[i])
	
	if (i==0 or i==3):
		perturb_type_lst = ['SC', 'SC', 'SC', 'SC', 'SC', 'SC', 'SC', 'SC']
		perturb_size_lst = [50, -50, 40, -40, 30, -30, 20, -20]
	elif (i==1 or i==4):
		perturb_type_lst = ['PS', 'PS', 'PS', 'PS', 'PS', 'PS', 'PS', 'PS']
		perturb_size_lst = [50, -50, 40, -40, 30, -30, 20, -20]
	elif (i==2 or i==5):
		perturb_size_lst = [50, -50, 40, -40, 30, -30, 20, -20,
							50, -50, 40, -40, 30, -30, 20, -20]
		perturb_type_lst = ['SC', 'SC', 'SC', 'SC', 'SC', 'SC', 'SC', 'SC',
							'PS', 'PS', 'PS', 'PS', 'PS', 'PS', 'PS', 'PS']
	
	divergent_parameters_lst = []
	#for index in range(0, len(params_df)):
	# for index in range(162, len(params_df)):
	for index in range(0, len(params_df)):
		param_df = (pd.DataFrame(params_df.iloc[index]).T).reset_index(drop=True)
		#param_df.to_csv('param_df.csv')
				
		# Parameters values
		a = param_df['a'][0]
		b = param_df['b'][0]
		c = param_df['c'][0]
		d = param_df['d'][0]
		alpha1 = param_df['alpha1'][0]
		beta1 = param_df['beta1'][0]
		gamma1 = param_df['gamma1'][0]
		delta1 = param_df['delta1'][0]
		eps1 = param_df['eps1'][0]
		dseta1 = param_df['dseta1'][0]
		eta1 = param_df['eta1'][0]
		alpha2 = param_df['alpha2'][0]
		beta2 = param_df['beta2'][0]
		gamma2 = param_df['gamma2'][0]
		delta2 = param_df['delta2'][0]
		eps2 = param_df['eps2'][0]
		dseta2 = param_df['dseta2'][0]
		eta2 = param_df['eta2'][0]
		params = [a, b, c, d, alpha1, beta1, gamma1, delta1, eps1, dseta1, eta1, alpha2, beta2, gamma2, delta2, eps2, dseta2, eta2]
		
		model_data2_df = pd.DataFrame()
		divergent_timeseries_lst = []
		j = 0
		for perturb_size in perturb_size_lst:
			hyper_params = [0, 0, perturb_type_lst[j], perturb_size]
			t, model_asyn, y = integrate(model, y_ini, params, hyper_params)
			origin_list = np.repeat(parameter_family_name_lst[i],len(t))[:,np.newaxis].T
			perturb_size_list = np.repeat(perturb_size,len(t))[:,np.newaxis].T
			perturb_type_list = np.repeat(perturb_type_lst[j],len(t))[:,np.newaxis].T
			step_n = np.array(range(0,n_beeps))[:,np.newaxis].T-perturb_beep+1
			
			model_array = np.concatenate([ \
									   origin_list, \
									   perturb_size_list, \
									   perturb_type_list, \
									   step_n, \
									   model_asyn[:,np.newaxis].T, \
									   y.T], axis=0).T
			model_data_df = pd.DataFrame(model_array, columns=col_names)
			#model_data_df.to_csv('model_data_df.csv')
			
			# Divergent time series
			model_data_df = model_data_df.astype({'asyn':float})
			model_data2_df = pd.concat([model_data2_df, model_data_df], axis=0)
			result = ( (model_data_df['asyn'] < -limit) | (model_data_df['asyn'] > limit) ).any()
			divergent_timeseries_lst.append(result)  
			j = j+1
			
		# Divergent parameters
		result2 = any(divergent_timeseries_lst)
		divergent_parameters_lst.append(result2)
		
		if result2 == False:
			model_data2_df["asyn"] = model_data2_df["asyn"].astype('float')
			model_data2_df["x"] = model_data2_df["x"].astype('float')
			model_data2_df["s"] = model_data2_df["s"].astype('float')
			model_data2_df["n"] = model_data2_df["n"].astype('int')
			model_data2_df["perturb_size"] = model_data2_df["perturb_size"].astype('string')
			model_data2_df["origin_x_perturb_type_x_perturb_size"] = model_data2_df[['origin','perturb_type','perturb_size']].agg(''.join, axis=1)
			model_data2_df["x_zeroed"] = model_data2_df["x"] - 500
			model_data2_df["x-s"] = model_data2_df["x"] - model_data2_df["s"]
			#model_data2_df.to_csv('model_data2_df.csv')

			if SHOW_AND_SAVE_DIVERGENT_PLOTS==1:
				plot_model_timeseries = (
					ggplot(model_data2_df, aes(x = 'n', y = 'asyn',						    
										 	 group = 'origin_x_perturb_type_x_perturb_size',
											   color = 'origin_x_perturb_type_x_perturb_size'))
					+ geom_path()
					+ geom_point(size = marker_size)
					+ scale_y_continuous(limits=y_lims,breaks=range(y_lims[0],y_lims[1],10))
					+ theme_bw(base_size=base_Size)
					+ theme(legend_key = element_rect(fill = "white", color = 'white'), figure_size = (12, 6))
					)
				plot_model_timeseries_0 = pw.load_ggplot(plot_model_timeseries)
			
				plot_model_timeseries = (
					ggplot(model_data2_df, aes(x = 'n', y = 'x_zeroed',						    
										 	 group = 'origin_x_perturb_type_x_perturb_size',
											   color = 'origin_x_perturb_type_x_perturb_size'))
					+ geom_path()
					+ geom_point(size = marker_size)
					+ scale_y_continuous(limits=y_lims,breaks=range(y_lims[0],y_lims[1],10))
					+ theme_bw(base_size=base_Size)
					+ theme(legend_key = element_rect(fill = "white", color = 'white'), figure_size = (12, 6))
					)
				plot_model_timeseries_1 = pw.load_ggplot(plot_model_timeseries)
			
				model_data2_df = model_data2_df[(model_data2_df['n'] >= 2)]
				plot_model_phase_space = (
					ggplot(model_data2_df, aes(x = 'asyn', y = 'x-s',						    
										 	 group = 'origin_x_perturb_type_x_perturb_size',
											   color = 'origin_x_perturb_type_x_perturb_size'))
					+ geom_path()
					+ geom_point(size = marker_size)
					+ scale_y_continuous(limits=y_lims,breaks=range(y_lims[0],y_lims[1],10))
					+ theme_bw(base_size=base_Size)
					+ theme(legend_key = element_rect(fill = "white", color = 'white'), figure_size = (6, 6))
					)
				plot_model_phase_space = pw.load_ggplot(plot_model_phase_space)
	
				plot_model = (plot_model_timeseries_0/plot_model_timeseries_1)|plot_model_phase_space
				plot_model.savefig( str(parameter_family_name_lst[i]) + "_" + str(index) + ".pdf")

		
	# Diverging column
	params_df['divergent'] = divergent_parameters_lst
	#params_df.to_csv('params_df.csv')
	parameter_family_divergent_column_lst.append(params_df)
	i = i+1

# Parameters with divergent parameters column
parameter_family_divergent_column_lst[0].to_csv(datafolder + 'params_SCpure_df.csv')
parameter_family_divergent_column_lst[1].to_csv(datafolder + 'params_PSpure_df.csv')
parameter_family_divergent_column_lst[2].to_csv(datafolder + 'params_PSSCpure_df.csv')
parameter_family_divergent_column_lst[3].to_csv(datafolder + 'params_SCcomb_df.csv')
parameter_family_divergent_column_lst[4].to_csv(datafolder + 'params_PScomb_df.csv')
parameter_family_divergent_column_lst[5].to_csv(datafolder + 'params_PSSCcomb_df.csv')


#%%










#%% Statistics of divergent fittings
true_count = parameter_family_divergent_column_lst[0]['divergent'].sum()
print("SCpure:", true_count/len(parameter_family_divergent_column_lst[0])*100, "%")

true_count = parameter_family_divergent_column_lst[1]['divergent'].sum()
print("PSpure:", true_count/len(parameter_family_divergent_column_lst[1])*100, "%")

true_count = parameter_family_divergent_column_lst[2]['divergent'].sum()
print("PSSCpure:", true_count/len(parameter_family_divergent_column_lst[2])*100, "%")

true_count = parameter_family_divergent_column_lst[3]['divergent'].sum()
print("SCcomb:", true_count/len(parameter_family_divergent_column_lst[3])*100, "%")

true_count = parameter_family_divergent_column_lst[4]['divergent'].sum()
print("PScomb:", true_count/len(parameter_family_divergent_column_lst[4])*100, "%")

true_count = parameter_family_divergent_column_lst[5]['divergent'].sum()
print("PSSCcomb:", true_count/len(parameter_family_divergent_column_lst[5])*100, "%")


#%%










#%% Plotting example
n_beeps = 40
perturb_beep = 6
ISI = 500
p_start, x_start, s_start = 0, ISI, ISI
y_ini = [p_start, x_start, s_start]         # p(asincronía predicha)   x(variable que explica el overshoot en step change)   s(ISI en n-1)

#param_df = parameter_family_divergent_column_lst[5]
param_df = params_PSSCcomb_df.reset_index(drop=True)
param_df = param_df.query('index==198').reset_index(drop=True)

# SC
#perturb_type_lst = ['SC', 'SC', 'SC', 'SC', 'SC', 'SC', 'SC', 'SC']
#perturb_size_lst = [50, -50, 40, -40, 30, -30, 20, -20]
#parameter_family_name = 'SCpure'
#parameter_family_name = 'SCcomb'

# PS
#perturb_type_lst = ['PS', 'PS', 'PS', 'PS', 'PS', 'PS', 'PS', 'PS']
#perturb_size_lst = [50, -50, 40, -40, 30, -30, 20, -20]
#parameter_family_name = 'PSpure'
#parameter_family_name = 'PScomb'

#PSSC
perturb_size_lst = [50, -50, 40, -40, 30, -30, 20, -20, 50, -50, 40, -40, 30, -30, 20, -20]
perturb_type_lst = ['SC', 'SC', 'SC', 'SC', 'SC', 'SC', 'SC', 'SC', 'PS', 'PS', 'PS', 'PS', 'PS', 'PS', 'PS', 'PS']
#parameter_family_name = 'PSSCpure'
parameter_family_name = 'PSScomb'

# Parameters values
a = param_df['a'][0]
b = param_df['b'][0]
c = param_df['c'][0]
d = param_df['d'][0]
alpha1 = param_df['alpha1'][0]
beta1 = param_df['beta1'][0]
gamma1 = param_df['gamma1'][0]
delta1 = param_df['delta1'][0]
eps1 = param_df['eps1'][0]
dseta1 = param_df['dseta1'][0]
eta1 = param_df['eta1'][0]
alpha2 = param_df['alpha2'][0]
beta2 = param_df['beta2'][0]
gamma2 = param_df['gamma2'][0]
delta2 = param_df['delta2'][0]
eps2 = param_df['eps2'][0]
dseta2 = param_df['dseta2'][0]
eta2 = param_df['eta2'][0]
params = [a, b, c, d, alpha1, beta1, gamma1, delta1, eps1, dseta1, eta1, alpha2, beta2, gamma2, delta2, eps2, dseta2, eta2]

# Simulation
col_names = ['origin','perturb_size','n','asyn','p','x','s']
model_data_df = pd.DataFrame(columns=col_names)

i = 0
for perturb_size in perturb_size_lst:
	hyper_params = [0, 0, perturb_type_lst[i], perturb_size]
	t, model_asyn, y = integrate(model, y_ini, params, hyper_params)
	origin_list = np.repeat(parameter_family_name,len(t))[:,np.newaxis].T
	perturb_size_list = np.repeat(perturb_size,len(t))[:,np.newaxis].T
	step_n = np.array(range(0,n_beeps))[:,np.newaxis].T-perturb_beep+1

	model_array = np.concatenate([ \
							   origin_list, \
							   perturb_size_list, \
							   step_n, \
							   model_asyn[:,np.newaxis].T, \
							   y.T], axis=0).T
	model_data_df = pd.concat([model_data_df, pd.DataFrame(model_array, columns=col_names)], axis=0)
	i = i+1
model_df = model_data_df.reset_index(drop=True)
#model_df.to_csv('model_df.csv')

model_df["asyn"] = model_df["asyn"].astype('float')
model_df["n"] = model_df["n"].astype('int')
model_df["perturb_size"] = model_df["perturb_size"].astype('string')
model_df["origin_x_perturb_size"] = model_df[['origin','perturb_size']].agg(''.join, axis=1)
# model_df.to_csv('model_df.csv')

marker_size = 2
base_Size = 10
y_lims = (-60, 60)

plot_model_timeseries = (
	ggplot(model_df, aes(x = 'n', y = 'asyn',						    
						 	 group = 'origin_x_perturb_size',
							 color = 'origin_x_perturb_size'))
	+ geom_path()
	+ geom_point(size = marker_size)
	+ scale_y_continuous(limits=y_lims,breaks=range(y_lims[0],y_lims[1],10))
	+ theme_bw(base_size=base_Size)
	+ theme(legend_key = element_rect(fill = "white", color = 'white'), figure_size = (12, 6))
	)
print(plot_model_timeseries)
# plot_model_timeseries.save('example_divergent.pdf')


#%%

