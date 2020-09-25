import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import os
import argparse
import pandas as pd
import itertools
import seaborn as sns
import ast

sns.set()

parser = argparse.ArgumentParser()

parser.add_argument("--env", type=str, required=True)
parser.add_argument("--lr_act", type=float, required=False, default=1e-3)
parser.add_argument("--lr_crt", type=float, required=False, default=1e-3)
parser.add_argument("--eps", type=int, required=False, default=1)
parser.add_argument("--epochs", type=int, required=False, default=3000)
parser.add_argument("--nn_hidden", type=str, required=False, default='[64,64]')
parser.add_argument("--seeds", type=str,  required=False, default='[0,10,1234]')

args = parser.parse_args()

env_name = args.env
lr_act = args.lr_act
lr_crt = args.lr_crt
epochs = args.epochs
episodes_per_epoch = args.eps
nn_hidden = ast.literal_eval(args.nn_hidden)
seeds = ast.literal_eval(args.seeds)

#### Logs Figure ####
eps_len_fig = plt.figure('Episode length')
eps_len_axes = eps_len_fig.add_subplot(111)

q_estimate_fig = plt.figure('Q estimate')
q_estimate_axes = q_estimate_fig.add_subplot(111)

training_return_fig = plt.figure('Trainining return')
training_return_axes  = training_return_fig.add_subplot(111)

mean_return_fig = plt.figure('Mean return')
mean_return_axes  = mean_return_fig.add_subplot(111)

std_return_fig = plt.figure('Std return')
std_return_axes  = std_return_fig.add_subplot(111)

max_return_fig = plt.figure('Max return')
max_return_axes  = max_return_fig.add_subplot(111)

min_return_fig = plt.figure('Min return')
min_return_axes  = min_return_fig.add_subplot(111)

mean_value_fig = plt.figure('Mean value')
mean_value_axes  = mean_value_fig.add_subplot(111)

policy_loss_fig = plt.figure('Policy loss')
policy_loss_axes  = policy_loss_fig.add_subplot(111)

kl_fig = plt.figure('KL')
kl_axes = kl_fig.add_subplot(111)

entropy_fig = plt.figure('Entropy')
entropy_axes  = entropy_fig.add_subplot(111)

value_loss_fig = plt.figure('Value loss')
value_loss_axes  = value_loss_fig.add_subplot(111)

# policy_grad_norm_fig = plt.figure('Policy grad norm')
# policy_grad_norm_axes  = policy_grad_norm_fig.add_subplot(111)

# value_grad_norm_fig = plt.figure('Value grad norm')
# value_grad_norm_axes  = value_grad_norm_fig.add_subplot(111)

# explained_variance_fig = plt.figure('Explained variance')
# explained_variance_axes  = explained_variance_fig.add_subplot(111)
		  

def plot(axes, data, ylabel, legend, j):
	plot_color = colors[j]
	data_mean = data.mean(axis=0)
	data_std = data.std(axis=0)
	axes.plot(range(len(data_mean)), data_mean, label=legend, color=plot_color)
	axes.fill_between(range(len(data_mean)), (data_mean-data_std), (data_mean+data_std), color=plot_color, alpha=0.3)
	axes.set_xlabel('Iteration')
	axes.set_ylabel(ylabel)
	axes.grid(True)
	axes.set_title(f'{env_name}')
	axes.legend()

colors = ['b', 'r', 'g', 'k', 'c']

runs = itertools.product([env_name], [lr_act], [lr_crt], [episodes_per_epoch], [nn_hidden], [ epochs] ,[seeds])

if __name__ == '__main__':
	for j, run in enumerate(runs):
		seeds = run[6]
		for i,seed in enumerate(seeds):
			env_name = run[0]
			lr_act = run[1]
			lr_crt = run[2]
			episodes_per_epoch = run[3]
			nn_hidden = run[4]
			epochs = run[5]
			log_dir = f'logs_for_seeds-env_name={env_name}-lr_act={lr_act}-lr_crt={lr_crt}-nn_hidden={nn_hidden}-episodes_per_epoch={episodes_per_epoch}' 
			log_file_name =  log_dir + '/' + f'-PG_with_stb-env={env_name}-po_lr={lr_act}-crt_lr={lr_crt}-episodes_per_epoch={episodes_per_epoch}-no.hidden: {nn_hidden}-no.iteration:{epochs}-seed:{seed}.csv'
			logs = pd.read_csv(log_file_name)
			if i == 0:
				eps_length = logs['episode length']
				q_estimate = logs['Q estimate']
				training_return = logs['Training return']
				mean_return = logs['mean return']
				std_return = logs['std return']
				max_return = logs['max return']
				min_return = logs['min return']
				mean_value = logs['mean value']
				policy_loss = logs['policy loss']
				kl = logs['KL']
				entropy = logs['entropy']
				value_loss = logs['value loss']
			else:
				eps_length = np.vstack((eps_length,logs['episode length'].to_numpy()))
				q_estimate = np.vstack((q_estimate,logs['Q estimate'].to_numpy()))
				training_return = np.vstack((training_return,logs['Training return'].to_numpy()))
				mean_return = np.vstack((mean_return,logs['mean return'].to_numpy()))
				std_return = np.vstack((std_return,logs['std return'].to_numpy()))
				max_return = np.vstack((max_return,logs['max return'].to_numpy()))
				min_return = np.vstack((min_return,logs['min return'].to_numpy()))
				mean_value = np.vstack((mean_value,logs['mean value'].to_numpy()))
				policy_loss = np.vstack((policy_loss,logs['policy loss'].to_numpy()))
				kl = np.vstack((kl,logs['KL'].to_numpy()))
				entropy = np.vstack((entropy,logs['entropy'].to_numpy()))
				value_loss = np.vstack((value_loss,logs['value loss'].to_numpy()))
		

		# ####################################       Plot the data        ####################################
		plot(eps_len_axes, eps_length, 'Episode length', f'{lr_act},{lr_crt},{nn_hidden},{episodes_per_epoch}',j)
		plot(q_estimate_axes, q_estimate, 'Q estimate', f'{lr_act},{lr_crt},{nn_hidden},{episodes_per_epoch}' ,j)
		plot(training_return_axes, training_return, 'Training return', f'{lr_act},{lr_crt},{nn_hidden},{episodes_per_epoch}', j)
		plot(mean_return_axes, mean_return, 'Mean return', f'{lr_act},{lr_crt},{nn_hidden},{episodes_per_epoch}', j)
		plot(std_return_axes, std_return, 'Std return', f'{lr_act},{lr_crt},{nn_hidden},{episodes_per_epoch}', j )
		plot(max_return_axes, max_return, 'Max return', f'{lr_act},{lr_crt},{nn_hidden},{episodes_per_epoch}', j )
		plot(min_return_axes, min_return, 'Min return', f'{lr_act},{lr_crt},{nn_hidden},{episodes_per_epoch}', j )
		plot(mean_value_axes, mean_value, 'Mean value', f'{lr_act},{lr_crt},{nn_hidden},{episodes_per_epoch}', j )
		plot(policy_loss_axes, policy_loss, 'Policy loss', f'{lr_act},{lr_crt},{nn_hidden},{episodes_per_epoch}', j)
		plot(kl_axes, kl, 'KL', f'{lr_act},{lr_crt},{nn_hidden},{episodes_per_epoch}', j)
		plot(entropy_axes, entropy, 'Entropy', f'{lr_act},{lr_crt},{nn_hidden},{episodes_per_epoch}', j)
		plot(value_loss_axes, value_loss, 'Value loss', f'{lr_act},{lr_crt},{nn_hidden},{episodes_per_epoch}', j)
		# # plot(policy_grad_norm_axes, policy_grad_norm, 'Policy grad norm', f'{lr_act},{lr_crt},{nn_hidden}' )
		# # plot(value_grad_norm_axes, value_grad_norm, 'value grad norm', f'{lr_act},{lr_crt},{nn_hidden}' )
		# # plot(explained_variance_axes, explained_variance, 'Explained variance', f'{lr_act},{lr_crt},{nn_hidden}')
		# ####################################       Plot the data        ####################################


	##############################  Tight the plot  ###########################
	eps_len_fig.tight_layout()
	q_estimate_fig.tight_layout()
	training_return_fig.tight_layout()
	mean_return_fig.tight_layout()
	std_return_fig.tight_layout()
	max_return_fig.tight_layout()
	min_return_fig.tight_layout()
	mean_return_fig.tight_layout()
	policy_loss_fig.tight_layout()
	kl_fig.tight_layout()
	entropy_fig.tight_layout()
	value_loss_fig.tight_layout()
	# policy_grad_norm_fig.tight_layout()
	# value_grad_norm_fig.tight_layout()
	# explained_variance_fig.tight_layout()
	##############################  Tight the plot  ###########################


	###############################  Save the figures  ########################################
	current_dir = os.getcwd()
	save_path = f'plot_figures--env_name={env_name}-lr_act={lr_act}-lr_crt={lr_crt}-nn_hidden={nn_hidden}-episodes_per_epoch={episodes_per_epoch}-epochs={epochs}'
	save_dir = current_dir + '/' + save_path
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	eps_len_fig.savefig(f'{save_dir}/{env_name} Episode length', dpi=350)
	q_estimate_fig.savefig(f'{save_dir}/{env_name} Q estimate', dpi=350)
	training_return_fig.savefig(f'{save_dir}/{env_name} Training return', dpi=350)
	mean_return_fig.savefig(f'{save_dir}/{env_name} Mean return', dpi=350)
	std_return_fig.savefig(f'{save_dir}/{env_name} Std return', dpi=350)
	max_return_fig.savefig(f'{save_dir}/{env_name} Max return', dpi=350)
	min_return_fig.savefig(f'{save_dir}/{env_name} Min return', dpi=350)
	mean_value_fig.savefig(f'{save_dir}/{env_name} Mean value', dpi=350)
	policy_loss_fig.savefig(f'{save_dir}/{env_name} Policy loss', dpi=350)
	kl_fig.savefig(f'{save_dir}/{env_name} KL', dpi=350)
	entropy_fig.savefig(f'{save_dir}/{env_name} Entropy', dpi=350)
	value_loss_fig.savefig(f'{save_dir}/{env_name} Value loss', dpi=350)
	# # policy_grad_norm_fig.savefig('Policy grad norm', dpi=350)
	# # value_grad_norm_fig.savefig('Value grad norm', dpi=350)
	# # explained_variance_fig.savefig('Explained variance', dpi=350)
	# ##############################  Save the figures  ########################################
	plt.show()
