#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 11:19:06 2023

@author: ariel


'Dynamical incompatibilities in paced finger tapping experiments'
Silva, González, & Laje (2026)

"""

#%%

import pandas as pd
import numpy as np
from plotnine import *

from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution
from scipy.optimize import dual_annealing
from scipy.optimize import shgo
from scipy.optimize import direct
from time import perf_counter

import matplotlib.pyplot as plt
import patchworklib as pw

datafolder = '../data/'


#%% Load data
data_GroupSubjCond_OS_df = pd.read_csv(datafolder + 'data_GroupSubjCond_OS_df.csv', index_col=0)
data_GroupSubjCond_OS_dict = pd.read_csv(datafolder + 'data_GroupSubjCond_OS_dict.csv')


#%% CHOOSE FITTING
# RUN THIS SCRIPT ONCE FOR EVERY FITTING CONDITION
# COMMENT OUT ACCORDINGLY

# PSpure, separate fitting
context_lst = ['pure']
perturb_type_lst = ['PS']

# # SCpure, separate fitting
# context_lst = ['pure']
# perturb_type_lst = ['SC']

# # PScomb, separate fitting
# context_lst = ['comb']
# perturb_type_lst = ['PS']

# # SCcomb, separate fitting
# context_lst = ['comb']
# perturb_type_lst = ['SC']

# # PSpure and SCpure, joint fitting
# context_lst = ['pure']
# perturb_type_lst = ['PS','SC']

# # PScomb and SCcomb, joint fitting
# context_lst = ['comb']
# perturb_type_lst = ['PS','SC']


fitting_data_filename = 'data3_fit_' + ''.join(perturb_type_lst) + ''.join(context_lst) + '_df.csv'
fitting_params_filename = 'params_' + ''.join(perturb_type_lst) + ''.join(context_lst) + '_df.csv'





#%% Rearrange data
# define resynchronization region (postperturbation baseline: beep>resynch_end)
resynch_start = 1
resynch_end = 6

# points to show in embedding plots
embed_start = 1
embed_end = 8

# add data from dictionary
data_df = (data_GroupSubjCond_OS_df[['Exp_name','General_condition',
                                    'Subject','Relative_beep','mean_asyn']]
           .rename(columns={'Relative_beep':'beep', 'mean_asyn':'asyn'})
		   .merge(data_GroupSubjCond_OS_dict[['General_condition','Name','Perturb_type','Perturb_size']],
					  how='inner', on='General_condition')
 		   .query('Perturb_size!=0') # discard isochronous
           )
#data_df.to_csv('borrar_data_df.csv')

# create new columns
data2_df = (data_df
			# rename perturbation type
			.assign(perturb_type = np.select([data_df['Perturb_type']==0,
											  data_df['Perturb_type']==1],
											 ['SC','PS'],
											 default='Other'))
			# separate perturbation sign and size
 			.assign(perturb_sign = np.select([data_df['Perturb_size']>0,
 											  data_df['Perturb_size']<0],
											 ['pos','neg'],
											 default='Other'))
 			.assign(perturb_size = abs(data_df['Perturb_size']))
			# make subjects unique
			.assign(subject = np.select([data_df['Exp_name']=='Experiment_PS',
 										 data_df['Exp_name']=='Experiment_SC',
 										 data_df['Exp_name']=='Experiment_PS_SC',
 										 data_df['Exp_name']=='Experiment2_PS_SC'],
										[data_df['Subject']+100,
 									      data_df['Subject']+200,
										  data_df['Subject']+300,
										  data_df['Subject']+400]))
			# define context
			.assign(context = np.select([(data_df['Exp_name']=='Experiment_PS_SC') | (data_df['Exp_name']=='Experiment2_PS_SC'),
										(data_df['Exp_name']=='Experiment_PS') | (data_df['Exp_name']=='Experiment_SC')],
										['comb', 'pure'],
										 default='Other'))
			)
#data2_df.to_csv('borrar_data2_df.csv')

# compute predicted asynchrony
data2_df = (data2_df
			.assign(asyn_pred = np.select([(data2_df['beep']==0) & (data2_df['perturb_sign']=='pos'),
											(data2_df['beep']==0) & (data2_df['perturb_sign']=='neg'),
											(data2_df['beep']==1) & (data2_df['perturb_type']=='PS') & (data2_df['perturb_sign']=='pos'),
											(data2_df['beep']==1) & (data2_df['perturb_type']=='PS') & (data2_df['perturb_sign']=='neg')],
 											[data2_df['asyn'] + data2_df['perturb_size'],
 											data2_df['asyn'] - data2_df['perturb_size'],
 											data2_df['asyn'] - data2_df['perturb_size'],
 											data2_df['asyn'] + data2_df['perturb_size']],
											default = data2_df['asyn']))
			)
# reorder columns, drop unused columns
data2_df = data2_df[['context','perturb_type','perturb_sign','perturb_size','subject','beep','asyn','asyn_pred']]
#data2_df.to_csv('borrar_data2B_df.csv')

# dataframe for embedding
data2_embed_df = (data2_df
				 # mean across subjects
 				 .groupby(['context','perturb_type','perturb_sign','perturb_size','beep'], as_index=False)
				 .agg(mean_asyn = ('asyn','mean'),
					  mean_asyn_pred = ('asyn_pred','mean'))
				)
#data2_embed_df.to_csv('borrar_data2_embed_df.csv')

# compute post-perturbation baseline
data2_post_df = (data2_embed_df[data2_embed_df['beep']>resynch_end]
				  .groupby(['context','perturb_type','perturb_sign','perturb_size'], as_index=False)
				  .agg(mean_asyn_postbaseline = ('mean_asyn','mean'),
					     mean_asyn_pred_postbaseline = ('mean_asyn_pred','mean'))
				  )
#data2_post_df.to_csv('borrar_data2_post_df.csv')

data2_embed_df = pd.merge(data2_embed_df,data2_post_df, how='left', on=['context','perturb_type','perturb_sign','perturb_size'])
data2_embed_df['mean_asyn_post'] = data2_embed_df['mean_asyn'] - data2_embed_df['mean_asyn_postbaseline']
data2_embed_df['mean_asyn_pred_post'] = data2_embed_df['mean_asyn_pred'] - data2_embed_df['mean_asyn_pred_postbaseline']
#data2_embed_df.to_csv('borrar_data2B_embed_df.csv')

# dummy column for plot grouping
data2_embed_df['label'] = data2_embed_df[['context','perturb_type','perturb_sign']].agg(''.join, axis=1)
#data2_embed_df.to_csv('borrar_data2C_embed_df.csv')

# time-delayed embedding
data2_embed_df['mean_asyn_post_prev'] = (data2_embed_df
 						 .groupby(['context','perturb_type','perturb_sign','perturb_size'], as_index=False)['mean_asyn_post']
						 .shift(1)
						)
data2_embed_df['mean_asyn_pred_post_prev'] = (data2_embed_df
 						 .groupby(['context','perturb_type','perturb_sign','perturb_size'], as_index=False)['mean_asyn_pred_post']
						 .shift(1)
						)
data2_embed_df.dropna(inplace=True)
#data2_embed_df.to_csv('borrar_data2D_embed_df.csv')

# difference embedding
data2_embed_df['mean_asyn_post_diff'] = data2_embed_df['mean_asyn_post'] - data2_embed_df['mean_asyn_post_prev']
data2_embed_df['mean_asyn_pred_post_diff'] = data2_embed_df['mean_asyn_pred_post'] - data2_embed_df['mean_asyn_pred_post_prev']
#data2_embed_df.to_csv('borrar_data2E_embed_df.csv')

# average-difference embedding
data2_embed_df['mean_asyn_post_ave'] = 0.5*(data2_embed_df['mean_asyn_post'] + data2_embed_df['mean_asyn_post_prev'])
data2_embed_df['mean_asyn_post_semidiff'] = 0.5*(data2_embed_df['mean_asyn_post'] - data2_embed_df['mean_asyn_post_prev'])
data2_embed_df['mean_asyn_pred_post_ave'] = 0.5*(data2_embed_df['mean_asyn_pred_post'] + data2_embed_df['mean_asyn_pred_post_prev'])
data2_embed_df['mean_asyn_pred_post_semidiff'] = 0.5*(data2_embed_df['mean_asyn_pred_post'] - data2_embed_df['mean_asyn_pred_post_prev'])
#data2_embed_df.to_csv('borrar_data2F_embed_df.csv')

# select transient phase only (PS starts one beep later than SC)
data2_embed_red_df = data2_embed_df[(((data2_embed_df['perturb_type']=='SC') & (data2_embed_df['beep']>=embed_start))
											 | ((data2_embed_df['perturb_type']=='PS') & (data2_embed_df['beep']>=embed_start+1)))
									& (data2_embed_df['beep']<=embed_end)]
#data2_embed_red_df.to_csv(datafolder + 'data2_embed_red_df.csv')


#%%










#%% Parameter and hyperparameter values
n_beeps = 17    # Reemplaza a n_steps
perturb_beep = 6 # from beginning of file   # Reemplaza a perturb_step
ISI = 500   # Reeplaza tau
perturb_size_lst = [20,50]
baseline_pre = 0        #Ojo! Actualmente lo calcula al levantar los datos en la seccón "Concatenate all experimental data"
baseline_post = 0       #Ojo! Actualmente lo calcula al levantar los datos en la seccón "Concatenate all experimental data"

p_start, x_start, s_start = 0, ISI, ISI
y_ini = [p_start, x_start, s_start]         # p(asincronía predicha)   x(variable que explica el overshoot en step change)   s(ISI en n-1)


# Mandatory initialization of parameters
#a = 0.981
a = 0
#b = 0.266
b = 0
#c = -0.823
c = 0
#d = 0.0238
d = 0
alpha1 = 0
beta1 = 0
gamma1 = 0
#delta1 = -0.0000221
delta1 = 0
eps1 = 0
#dseta1 = -0.0000784
dseta1 = 0
#eta1 = 0.0000534
eta1 = 0
#alpha2 = 0.00335
alpha2 = 0
beta2 = 0
gamma2 = 0
delta2 = 0
eps2 = 0
dseta2 = 0
eta2 = 0


params = [a,b,c,d,alpha1,beta1,gamma1,delta1,eps1,dseta1,eta1, \
		alpha2,beta2,gamma2,delta2,eps2,dseta2,eta2]


#%%










#%% Model and function definitions

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


def dist_fun(fit_par, exp_data_df, model, y_ini, params, bounds_aux_lst):

	# fit_par is the parameter to be varied (can be a list)
	# replace the corresponding value in params by fit_par
	a,b,c,d,alpha1,beta1,gamma1,delta1,eps1,dseta1,eta1,alpha2,beta2,gamma2,delta2,eps2,dseta2,eta2 = params
	
	a = fit_par[0]
	b = fit_par[1]
	c = fit_par[2]
	d = fit_par[3]
	dseta1 = fit_par[4]
	alpha2 = fit_par[5]
	
#	i = 0
#	for bounds_aux in bounds_aux_lst:
#		if bounds_aux == 'a':
#			a = fit_par[i]
#		if bounds_aux == 'b':
#			b = fit_par[i]
#		if bounds_aux == 'c':
#			c = fit_par[i]
#		if bounds_aux == 'd':
#			d = fit_par[i]
#		if bounds_aux == 'alpha1':
#			alpha1 = fit_par[i]
#		if bounds_aux == 'beta1':
#			beta1 = fit_par[i]
#		if bounds_aux == 'gamma1':
#			gamma1 = fit_par[i]
#		if bounds_aux == 'delta1':
#			delta1 = fit_par[i]
#		if bounds_aux == 'eps1':
#			eps1 = fit_par[i]
#		if bounds_aux == 'dseta1':
#			dseta1 = fit_par[i]
#		if bounds_aux == 'eta1':
#			eta1 = fit_par[i]
#		if bounds_aux == 'alpha2':
#			alpha2 = fit_par[i]
#		if bounds_aux == 'beta2':
#			beta2 = fit_par[i]
#		if bounds_aux == 'gamma2':
#			gamma2 = fit_par[i]
#		if bounds_aux == 'delta2':
#			delta2 = fit_par[i]
#		if bounds_aux == 'eps2':
#			eps2 = fit_par[i]
#		if bounds_aux == 'dseta2':
#			dseta2 = fit_par[i]
#		if bounds_aux == 'eta2':
#			eta2 = fit_par[i]
#		i = i + 1

	params = [a,b,c,d,alpha1,beta1,gamma1,delta1,eps1,dseta1,eta1, \
			alpha2,beta2,gamma2,delta2,eps2,dseta2,eta2]

	exp_asyn_all = []
	model_asyn_all = []
	origins = exp_data_df['origin'].unique()
	for origin in origins:
		# filter a single perturbation size
		exp_asyn = np.array(exp_data_df[exp_data_df['origin']==origin]['asyn'])
		exp_asyn_all = np.concatenate((exp_asyn_all, exp_asyn))
		# simulate model
		baseline_pre = (exp_data_df[exp_data_df['origin']==origin].reset_index(drop = True))['baseline_pre'][0]
		baseline_post = (exp_data_df[exp_data_df['origin']==origin].reset_index(drop = True))['baseline_post'][0]
		perturb_type = (exp_data_df[exp_data_df['origin']==origin].reset_index(drop = True))['perturb_type'][0]
		perturb_sign = (exp_data_df[exp_data_df['origin']==origin].reset_index(drop = True))['perturb_sign'][0]
		if perturb_sign == "pos":
			perturb_size = (exp_data_df[exp_data_df['origin']==origin].reset_index(drop = True))['perturb_size'][0]
		elif perturb_sign == "neg":
			perturb_size = -((exp_data_df[exp_data_df['origin']==origin].reset_index(drop = True))['perturb_size'][0])
		hyper_params = [baseline_pre, baseline_post, perturb_type, perturb_size]
# 		t, model_asyn, y = integrate(model, y_ini, params, hyper_params)
		t, model_asyn, y = integrate(model, y_ini, params, hyper_params)
		model_asyn_all = np.concatenate((model_asyn_all, model_asyn))
	# euclidean distance between data and simulated (number of infected individuals I)
	# https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
	
	dist = np.linalg.norm(np.array(exp_asyn_all) - model_asyn_all)
	if np.isnan(dist):
		dist = np.inf
# 	dist = np.linalg.norm(np.array(exp_data_df['asyn']) - model_asyn_all)

	# history of parameter and distance values
	# dist_history.append([*fit_par, dist])
	return dist


#%%










#%% Add baseline_pre and baseline_post

data_fit_df = data2_embed_df.drop(columns=['mean_asyn_pred', 'mean_asyn_postbaseline', 'mean_asyn_pred_postbaseline', 'mean_asyn_post', 
                                      'mean_asyn_pred_post', 'label', 'mean_asyn_post_prev', 'mean_asyn_pred_post_prev', 
                                      'mean_asyn_post_diff', 'mean_asyn_pred_post_diff', 'mean_asyn_post_ave', 'mean_asyn_post_semidiff', 
                                      'mean_asyn_pred_post_ave', 'mean_asyn_pred_post_semidiff'], axis=1)
#data_fit_df.to_csv('borrar_data_fit_df.csv')

data2_fit_df = (data_fit_df
                # Add baseline_pre
                .assign(baseline_pre = np.select([(data_fit_df['context']=='pure') & (data_fit_df['perturb_type']=='PS') & 
                                                  (data_fit_df['perturb_sign']=='neg') & (data_fit_df['perturb_size']==20),
                                                  (data_fit_df['context']=='pure') & (data_fit_df['perturb_type']=='PS') & 
                                                  (data_fit_df['perturb_sign']=='neg') & (data_fit_df['perturb_size']==50),
                                                  (data_fit_df['context']=='pure') & (data_fit_df['perturb_type']=='PS') & 
                                                  (data_fit_df['perturb_sign']=='pos') & (data_fit_df['perturb_size']==20),
                                                  (data_fit_df['context']=='pure') & (data_fit_df['perturb_type']=='PS') & 
                                                  (data_fit_df['perturb_sign']=='pos') & (data_fit_df['perturb_size']==50),
                                                  (data_fit_df['context']=='pure') & (data_fit_df['perturb_type']=='SC') & 
                                                  (data_fit_df['perturb_sign']=='neg') & (data_fit_df['perturb_size']==20),
                                                  (data_fit_df['context']=='pure') & (data_fit_df['perturb_type']=='SC') & 
                                                  (data_fit_df['perturb_sign']=='neg') & (data_fit_df['perturb_size']==50),
                                                  (data_fit_df['context']=='pure') & (data_fit_df['perturb_type']=='SC') & 
                                                  (data_fit_df['perturb_sign']=='pos') & (data_fit_df['perturb_size']==20),
                                                  (data_fit_df['context']=='pure') & (data_fit_df['perturb_type']=='SC') & 
                                                  (data_fit_df['perturb_sign']=='pos') & (data_fit_df['perturb_size']==50),
                                                  (data_fit_df['context']=='comb') & (data_fit_df['perturb_type']=='PS') & 
                                                  (data_fit_df['perturb_sign']=='neg') & (data_fit_df['perturb_size']==20),
                                                  (data_fit_df['context']=='comb') & (data_fit_df['perturb_type']=='PS') & 
                                                  (data_fit_df['perturb_sign']=='neg') & (data_fit_df['perturb_size']==50),
                                                  (data_fit_df['context']=='comb') & (data_fit_df['perturb_type']=='PS') & 
                                                  (data_fit_df['perturb_sign']=='pos') & (data_fit_df['perturb_size']==20),
                                                  (data_fit_df['context']=='comb') & (data_fit_df['perturb_type']=='PS') & 
                                                  (data_fit_df['perturb_sign']=='pos') & (data_fit_df['perturb_size']==50),
                                                  (data_fit_df['context']=='comb') & (data_fit_df['perturb_type']=='SC') & 
                                                  (data_fit_df['perturb_sign']=='neg') & (data_fit_df['perturb_size']==20),
                                                  (data_fit_df['context']=='comb') & (data_fit_df['perturb_type']=='SC') & 
                                                  (data_fit_df['perturb_sign']=='neg') & (data_fit_df['perturb_size']==50),
                                                  (data_fit_df['context']=='comb') & (data_fit_df['perturb_type']=='SC') & 
                                                  (data_fit_df['perturb_sign']=='pos') & (data_fit_df['perturb_size']==20),
                                                  (data_fit_df['context']=='comb') & (data_fit_df['perturb_type']=='SC') & 
                                                  (data_fit_df['perturb_sign']=='pos') & (data_fit_df['perturb_size']==50)],
                                                 [data_fit_df[(data_fit_df['context']=='pure') & 
                                                              (data_fit_df['perturb_type']=='PS') &
                                                              (data_fit_df['perturb_sign']=='neg') & 
                                                              (data_fit_df['perturb_size']==20) &
                                                              (data_fit_df['beep']<0)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='pure') & 
                                                               (data_fit_df['perturb_type']=='PS') &
                                                               (data_fit_df['perturb_sign']=='neg') & 
                                                               (data_fit_df['perturb_size']==50) &
                                                               (data_fit_df['beep']<0)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='pure') & 
                                                               (data_fit_df['perturb_type']=='PS') &
                                                               (data_fit_df['perturb_sign']=='pos') & 
                                                               (data_fit_df['perturb_size']==20) &
                                                               (data_fit_df['beep']<0)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='pure') & 
                                                               (data_fit_df['perturb_type']=='PS') &
                                                               (data_fit_df['perturb_sign']=='pos') & 
                                                               (data_fit_df['perturb_size']==50) &
                                                               (data_fit_df['beep']<0)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='pure') & 
                                                               (data_fit_df['perturb_type']=='SC') &
                                                               (data_fit_df['perturb_sign']=='neg') & 
                                                               (data_fit_df['perturb_size']==20) &
                                                               (data_fit_df['beep']<0)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='pure') & 
                                                               (data_fit_df['perturb_type']=='SC') &
                                                               (data_fit_df['perturb_sign']=='neg') & 
                                                               (data_fit_df['perturb_size']==50) &
                                                               (data_fit_df['beep']<0)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='pure') & 
                                                              (data_fit_df['perturb_type']=='SC') &
                                                              (data_fit_df['perturb_sign']=='pos') & 
                                                              (data_fit_df['perturb_size']==20) &
                                                              (data_fit_df['beep']<0)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='pure') & 
                                                              (data_fit_df['perturb_type']=='SC') &
                                                              (data_fit_df['perturb_sign']=='pos') & 
                                                              (data_fit_df['perturb_size']==50) &
                                                              (data_fit_df['beep']<0)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='comb') & 
                                                              (data_fit_df['perturb_type']=='PS') &
                                                              (data_fit_df['perturb_sign']=='neg') & 
                                                              (data_fit_df['perturb_size']==20) &
                                                              (data_fit_df['beep']<0)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='comb') & 
                                                              (data_fit_df['perturb_type']=='PS') &
                                                              (data_fit_df['perturb_sign']=='neg') & 
                                                              (data_fit_df['perturb_size']==50) &
                                                              (data_fit_df['beep']<0)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='comb') & 
                                                              (data_fit_df['perturb_type']=='PS') &
                                                              (data_fit_df['perturb_sign']=='pos') & 
                                                              (data_fit_df['perturb_size']==20) &
                                                              (data_fit_df['beep']<0)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='comb') & 
                                                               (data_fit_df['perturb_type']=='PS') &
                                                               (data_fit_df['perturb_sign']=='pos') & 
                                                               (data_fit_df['perturb_size']==50) &
                                                               (data_fit_df['beep']<0)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='comb') & 
                                                              (data_fit_df['perturb_type']=='SC') &
                                                              (data_fit_df['perturb_sign']=='neg') & 
                                                              (data_fit_df['perturb_size']==20) &
                                                              (data_fit_df['beep']<0)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='comb') & 
                                                              (data_fit_df['perturb_type']=='SC') &
                                                              (data_fit_df['perturb_sign']=='neg') & 
                                                              (data_fit_df['perturb_size']==50) &
                                                              (data_fit_df['beep']<0)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='comb') & 
                                                              (data_fit_df['perturb_type']=='SC') &
                                                              (data_fit_df['perturb_sign']=='pos') & 
                                                              (data_fit_df['perturb_size']==20) &
                                                              (data_fit_df['beep']<0)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='comb') & 
                                                              (data_fit_df['perturb_type']=='SC') &
                                                              (data_fit_df['perturb_sign']=='pos') & 
                                                              (data_fit_df['perturb_size']==50) &
                                                              (data_fit_df['beep']<0)]['mean_asyn'].mean()]))
                # Add baseline_post
                .assign(baseline_post = np.select([(data_fit_df['context']=='pure') & (data_fit_df['perturb_type']=='PS') & 
                                                  (data_fit_df['perturb_sign']=='neg') & (data_fit_df['perturb_size']==20),
                                                  (data_fit_df['context']=='pure') & (data_fit_df['perturb_type']=='PS') & 
                                                  (data_fit_df['perturb_sign']=='neg') & (data_fit_df['perturb_size']==50),
                                                  (data_fit_df['context']=='pure') & (data_fit_df['perturb_type']=='PS') & 
                                                  (data_fit_df['perturb_sign']=='pos') & (data_fit_df['perturb_size']==20),
                                                  (data_fit_df['context']=='pure') & (data_fit_df['perturb_type']=='PS') & 
                                                  (data_fit_df['perturb_sign']=='pos') & (data_fit_df['perturb_size']==50),
                                                  (data_fit_df['context']=='pure') & (data_fit_df['perturb_type']=='SC') & 
                                                  (data_fit_df['perturb_sign']=='neg') & (data_fit_df['perturb_size']==20),
                                                  (data_fit_df['context']=='pure') & (data_fit_df['perturb_type']=='SC') & 
                                                  (data_fit_df['perturb_sign']=='neg') & (data_fit_df['perturb_size']==50),
                                                  (data_fit_df['context']=='pure') & (data_fit_df['perturb_type']=='SC') & 
                                                  (data_fit_df['perturb_sign']=='pos') & (data_fit_df['perturb_size']==20),
                                                  (data_fit_df['context']=='pure') & (data_fit_df['perturb_type']=='SC') & 
                                                  (data_fit_df['perturb_sign']=='pos') & (data_fit_df['perturb_size']==50),
                                                  (data_fit_df['context']=='comb') & (data_fit_df['perturb_type']=='PS') & 
                                                  (data_fit_df['perturb_sign']=='neg') & (data_fit_df['perturb_size']==20),
                                                  (data_fit_df['context']=='comb') & (data_fit_df['perturb_type']=='PS') & 
                                                  (data_fit_df['perturb_sign']=='neg') & (data_fit_df['perturb_size']==50),
                                                  (data_fit_df['context']=='comb') & (data_fit_df['perturb_type']=='PS') & 
                                                  (data_fit_df['perturb_sign']=='pos') & (data_fit_df['perturb_size']==20),
                                                  (data_fit_df['context']=='comb') & (data_fit_df['perturb_type']=='PS') & 
                                                  (data_fit_df['perturb_sign']=='pos') & (data_fit_df['perturb_size']==50),
                                                  (data_fit_df['context']=='comb') & (data_fit_df['perturb_type']=='SC') & 
                                                  (data_fit_df['perturb_sign']=='neg') & (data_fit_df['perturb_size']==20),
                                                  (data_fit_df['context']=='comb') & (data_fit_df['perturb_type']=='SC') & 
                                                  (data_fit_df['perturb_sign']=='neg') & (data_fit_df['perturb_size']==50),
                                                  (data_fit_df['context']=='comb') & (data_fit_df['perturb_type']=='SC') & 
                                                  (data_fit_df['perturb_sign']=='pos') & (data_fit_df['perturb_size']==20),
                                                  (data_fit_df['context']=='comb') & (data_fit_df['perturb_type']=='SC') & 
                                                  (data_fit_df['perturb_sign']=='pos') & (data_fit_df['perturb_size']==50)],
                                                 [data_fit_df[(data_fit_df['context']=='pure') & 
                                                              (data_fit_df['perturb_type']=='PS') &
                                                              (data_fit_df['perturb_sign']=='neg') & 
                                                              (data_fit_df['perturb_size']==20) &
                                                              (data_fit_df['beep']>=7)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='pure') & 
                                                               (data_fit_df['perturb_type']=='PS') &
                                                               (data_fit_df['perturb_sign']=='neg') & 
                                                               (data_fit_df['perturb_size']==50) &
                                                               (data_fit_df['beep']>=7)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='pure') & 
                                                               (data_fit_df['perturb_type']=='PS') &
                                                               (data_fit_df['perturb_sign']=='pos') & 
                                                               (data_fit_df['perturb_size']==20) &
                                                               (data_fit_df['beep']>=7)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='pure') & 
                                                               (data_fit_df['perturb_type']=='PS') &
                                                               (data_fit_df['perturb_sign']=='pos') & 
                                                               (data_fit_df['perturb_size']==50) &
                                                               (data_fit_df['beep']>=7)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='pure') & 
                                                               (data_fit_df['perturb_type']=='SC') &
                                                               (data_fit_df['perturb_sign']=='neg') & 
                                                               (data_fit_df['perturb_size']==20) &
                                                               (data_fit_df['beep']>=7)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='pure') & 
                                                               (data_fit_df['perturb_type']=='SC') &
                                                               (data_fit_df['perturb_sign']=='neg') & 
                                                               (data_fit_df['perturb_size']==50) &
                                                               (data_fit_df['beep']>=7)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='pure') & 
                                                              (data_fit_df['perturb_type']=='SC') &
                                                              (data_fit_df['perturb_sign']=='pos') & 
                                                              (data_fit_df['perturb_size']==20) &
                                                              (data_fit_df['beep']>=7)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='pure') & 
                                                              (data_fit_df['perturb_type']=='SC') &
                                                              (data_fit_df['perturb_sign']=='pos') & 
                                                              (data_fit_df['perturb_size']==50) &
                                                              (data_fit_df['beep']>=7)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='comb') & 
                                                              (data_fit_df['perturb_type']=='PS') &
                                                              (data_fit_df['perturb_sign']=='neg') & 
                                                              (data_fit_df['perturb_size']==20) &
                                                              (data_fit_df['beep']>=7)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='comb') & 
                                                              (data_fit_df['perturb_type']=='PS') &
                                                              (data_fit_df['perturb_sign']=='neg') & 
                                                              (data_fit_df['perturb_size']==50) &
                                                              (data_fit_df['beep']>=7)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='comb') & 
                                                              (data_fit_df['perturb_type']=='PS') &
                                                              (data_fit_df['perturb_sign']=='pos') & 
                                                              (data_fit_df['perturb_size']==20) &
                                                              (data_fit_df['beep']>=7)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='comb') & 
                                                               (data_fit_df['perturb_type']=='PS') &
                                                               (data_fit_df['perturb_sign']=='pos') & 
                                                               (data_fit_df['perturb_size']==50) &
                                                               (data_fit_df['beep']>=7)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='comb') & 
                                                              (data_fit_df['perturb_type']=='SC') &
                                                              (data_fit_df['perturb_sign']=='neg') & 
                                                              (data_fit_df['perturb_size']==20) &
                                                              (data_fit_df['beep']>=7)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='comb') & 
                                                              (data_fit_df['perturb_type']=='SC') &
                                                              (data_fit_df['perturb_sign']=='neg') & 
                                                              (data_fit_df['perturb_size']==50) &
                                                              (data_fit_df['beep']>=7)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='comb') & 
                                                              (data_fit_df['perturb_type']=='SC') &
                                                              (data_fit_df['perturb_sign']=='pos') & 
                                                              (data_fit_df['perturb_size']==20) &
                                                              (data_fit_df['beep']>=7)]['mean_asyn'].mean(),
                                                  data_fit_df[(data_fit_df['context']=='comb') & 
                                                              (data_fit_df['perturb_type']=='SC') &
                                                              (data_fit_df['perturb_sign']=='pos') & 
                                                              (data_fit_df['perturb_size']==50) &
                                                              (data_fit_df['beep']>=7)]['mean_asyn'].mean()])))

data2_fit_df = data2_fit_df.assign(origin = data2_fit_df.context + data2_fit_df.perturb_type +
								   data2_fit_df.perturb_sign + data2_fit_df.perturb_size.astype('string'))
data2_fit_df.rename(columns={"beep": "n", "mean_asyn": "asyn"}, inplace = True)
data2_fit_df = data2_fit_df.reindex(['origin', 'context', 'perturb_type', 'perturb_size', 'perturb_sign', 'n', 'asyn', 'baseline_pre', 'baseline_post'], axis=1).reset_index(drop=True)
#data2_fit_df.to_csv('borrar_data2_fit_df.csv')

data3_fit_df = data2_fit_df.reset_index(drop=True)
if len(perturb_type_lst) == 1:
	data3_fit_df = data3_fit_df[(data3_fit_df["perturb_type"] == perturb_type_lst[0])] 
if len(context_lst) == 1:
	data3_fit_df = data3_fit_df[(data3_fit_df["context"] == context_lst[0])] 
if len(perturb_size_lst) == 1:
	data3_fit_df = data3_fit_df[(data3_fit_df["perturb_size"] == perturb_size_lst[0])]
data3_fit_df.to_csv(datafolder + fitting_data_filename)


#%% Plot data to fit
#color_map = ["blue","magenta"]
line_map = ["solid","dashed"]
shape_map = ["s","D"]
marker_size = 2
base_Size = 10

plot_data_to_fit = (
	ggplot(data3_fit_df, aes(x = 'n', y = 'asyn',						    
						 	 group = 'origin',
							 color = 'origin'))
	+ facet_grid(facets="origin~")
	# + facet_grid('origin')
	+ geom_line()
	+ geom_point(size = marker_size)
	#+ scale_color_manual(values = color_map)
	+ theme_bw(base_size=base_Size)
	+ theme(legend_key = element_rect(fill = "white", color = 'white'), figure_size = (12, 6))
	)
print(plot_data_to_fit)
# plot_data_to_fit.save(fitting_data_filename[:-4] + '.pdf')


#%%










#%% Fitting: DIFFERENTIAL EVOLUTION
# TAKES A LONG TIME if n_loops is large

n_loops = 200


a_bounds = (-2, 2)
b_bounds = (-2, 2)
c_bounds = (-2, 2)
d_bounds = (-2, 2)
alpha1_bounds = (-0.02, 0.02)
beta1_bounds = (-0.0002, 0.0002)
gamma1_bounds = (-0.0002, 0.0002)
delta1_bounds = (-0.0002, 0.0002)
eps1_bounds = (-0.0002, 0.0002)
dseta1_bounds = (-0.0002, 0.0002)
eta1_bounds = (-0.0002, 0.0002)
alpha2_bounds = (-0.02, 0.02)
beta2_bounds = (-0.0002, 0.0002)
gamma2_bounds = (-0.0002, 0.0002)
delta2_bounds = (-0.0002, 0.0002)
eps2_bounds = (-0.0002, 0.0002)
dseta2_bounds = (-0.0002, 0.0002)
eta2_bounds = (-0.0002, 0.0002)


# Select parameters to fit
bounds_lst = [a_bounds, b_bounds, c_bounds, d_bounds, dseta1_bounds, alpha2_bounds]
bounds_aux_lst = ['a', 'b', 'c', 'd', 'dseta1', 'alpha2'] 

params_col_names = ['a','b','c','d','alpha1','beta1','gamma1','delta1','eps1','dseta1','eta1',
					'alpha2','beta2','gamma2','delta2','eps2','dseta2','eta2','dist']
params_df = pd.DataFrame(columns=params_col_names)


t_ini = perf_counter()
for i in range(0,n_loops):
	print('loop: ' + str(i))
	res_de2 = differential_evolution(dist_fun, bounds_lst, mutation=(0.5, 1), recombination=0.7, \
								  #polish=False, workers=-1, maxiter=500, popsize = 15,
								  polish=False, workers=-1, maxiter=200, popsize = 100, tol=0.3,
								  disp=True, args=(data3_fit_df, model, y_ini, params, bounds_aux_lst))

	res_de2_lst = res_de2['x']
	dist = res_de2.fun
	
	i = 0
	for bounds_aux in bounds_aux_lst:
		if bounds_aux == 'a':
			a = res_de2_lst[i]
		if bounds_aux == 'b':
			b = res_de2_lst[i]
		if bounds_aux == 'c':
			c = res_de2_lst[i]
		if bounds_aux == 'd':
			d = res_de2_lst[i]
		if bounds_aux == 'alpha1':
			alpha1 = res_de2_lst[i]
		if bounds_aux == 'beta1':
			beta1 = res_de2_lst[i]
		if bounds_aux == 'gamma1':
			gamma1 = res_de2_lst[i]
		if bounds_aux == 'delta1':
			delta1 = res_de2_lst[i]
		if bounds_aux == 'eps1':
			eps1 = res_de2_lst[i]
		if bounds_aux == 'dseta1':
			dseta1 = res_de2_lst[i]
		if bounds_aux == 'eta1':
			eta1 = res_de2_lst[i]
		if bounds_aux == 'alpha2':
			alpha2 = res_de2_lst[i]
		if bounds_aux == 'beta2':
			beta2 = res_de2_lst[i]
		if bounds_aux == 'gamma2':
			gamma2 = res_de2_lst[i]
		if bounds_aux == 'delta2':
			delta2 = res_de2_lst[i]
		if bounds_aux == 'eps2':
			eps2 = res_de2_lst[i]
		if bounds_aux == 'dseta2':
			dseta2 = res_de2_lst[i]
		if bounds_aux == 'eta2':
			eta2 = res_de2_lst[i]
		i = i + 1
	
	params_aux_df = pd.DataFrame({'a':a,'b':b,'c':c,'d':d,
							   'alpha1':alpha1,'beta1':beta1,'gamma1':gamma1,'delta1':delta1,'eps1':eps1,'dseta1':dseta1,'eta1':eta1,
							   'alpha2':alpha2,'beta2':beta2,'gamma2':gamma2,'delta2':delta2,'eps2':eps2,'dseta2':dseta2,'eta2':eta2,'dist':dist},index=[0])
	params_df = pd.concat([params_df, params_aux_df], axis=0)


t_de2 = perf_counter() - t_ini
print('tiempo de corrida: ' + str(t_de2))


params_df.to_csv(datafolder + fitting_params_filename)




#%%

