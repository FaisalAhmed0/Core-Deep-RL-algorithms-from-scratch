import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import os
import argparse
import pandas as pd
import itertools
import seaborn as sns
sns.set()

parser = argparse.ArgumentParser()

parser.add_argument("--env", type=str, required=True)
parser.add_argument("--epochs", type=int, required=False, default=3000)
args = parser.parse_args()

env_name = args.env
epochs = args.epochs

# Hyperparameters should be the same as in the file policy_gradient_sdb_exp.py
episodes_per_epoch = [1,5]
nn_hiddens = [[32,32],[64,64]]
lrs_act = [1e-3,1e-5]
lr_crt = [1e-3]

configs = itertools.product(lrs_act, lr_crt, episodes_per_epoch, nn_hiddens)
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

policy_grad_norm_fig = plt.figure('Policy grad norm')
policy_grad_norm_axes  = policy_grad_norm_fig.add_subplot(111)

value_grad_norm_fig = plt.figure('Value grad norm')
value_grad_norm_axes  = value_grad_norm_fig.add_subplot(111)

explained_variance_fig = plt.figure('Explained variance')
explained_variance_axes  = explained_variance_fig.add_subplot(111)
		  

def plot(axes, data, ylabel, legend):
	data_numpy = data.to_numpy()
	# smoothed_data = gaussian_filter1d(data_numpy, sigma=0)
	smoothed_data = data_numpy
	axes.plot(range(len(data_numpy)), smoothed_data, label=legend, alpha=0.7)
	axes.set_xlabel('Iteration')
	axes.set_ylabel(ylabel)
	axes.grid(True)
	axes.set_title(f'{env_name}-{ylabel}')
	axes.legend()
	# axes.legend(loc=(1.05, 0.5))

if __name__ == '__main__':
	for lr_act, lr_crt, episodes_per_epoch, nn_hidden in configs:
		log_file_name = f'-PG_with_stb-env={env_name}-po_lr={lr_act}-crt_lr={lr_crt}-episodes_per_epoch={episodes_per_epoch}-no.hidden: {nn_hidden}-no.iteration:{epochs}.csv'
		logs = pd.read_csv(log_file_name)

		####################################    Retrive the log data    ####################################
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
		policy_grad_norm = logs['policy grad norm']
		value_grad_norm = logs['value grad norm']
		explained_variance = logs['Explained variance']
		####################################    Retrive the log data    ####################################

		####################################       Plot the data        ####################################
		plot(eps_len_axes, eps_length, 'Episode length', f'{lr_act},{lr_crt},{nn_hidden}' )
		plot(q_estimate_axes, q_estimate, 'Q estimate', f'{lr_act},{lr_crt},{nn_hidden}' )
		plot(training_return_axes, training_return, 'Training return', f'{lr_act},{lr_crt},{nn_hidden}' )
		plot(mean_return_axes, mean_return, 'Mean return', f'{lr_act},{lr_crt},{nn_hidden}' )
		plot(std_return_axes, std_return, 'Std return', f'{lr_act},{lr_crt},{nn_hidden}' )
		plot(max_return_axes, max_return, 'Max return', f'{lr_act},{lr_crt},{nn_hidden}' )
		plot(min_return_axes, min_return, 'Min return', f'{lr_act},{lr_crt},{nn_hidden}' )
		plot(mean_value_axes, mean_value, 'Mean value', f'{lr_act},{lr_crt},{nn_hidden}' )
		plot(policy_loss_axes, policy_loss, 'Policy loss', f'{lr_act},{lr_crt},{nn_hidden}' )
		plot(kl_axes, kl, 'KL', f'{lr_act},{lr_crt},{nn_hidden}' )
		plot(entropy_axes, entropy, 'Entropy', f'{lr_act},{lr_crt},{nn_hidden}' )
		plot(value_loss_axes, value_loss, 'Value loss', f'{lr_act},{lr_crt},{nn_hidden}' )
		plot(policy_grad_norm_axes, policy_grad_norm, 'Policy grad norm', f'{lr_act},{lr_crt},{nn_hidden}' )
		plot(value_grad_norm_axes, value_grad_norm, 'value grad norm', f'{lr_act},{lr_crt},{nn_hidden}' )
		plot(explained_variance_axes, explained_variance, 'Explained variance', f'{lr_act},{lr_crt},{nn_hidden}')
		####################################       Plot the data        ####################################




##############################  Tight the plot for the legend  ###########################
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
policy_grad_norm_fig.tight_layout()
value_grad_norm_fig.tight_layout()
explained_variance_fig.tight_layout()
##############################  Tight the plot for the legend  ###########################

##############################  Save the figures  ########################################
logs_fig_directory = f'{env_name}_logs_figs'
if not os.path.exists(logs_fig_directory):
    os.mkdir(logs_fig_directory)
eps_len_fig.savefig(logs_fig_directory + '/' + f'{env_name} Episode return', dpi=350)
q_estimate_fig.savefig(logs_fig_directory + '/' +f'{env_name} Q estimate', dpi=350)
training_return_fig.savefig(logs_fig_directory + '/' +f'{env_name} Training return', dpi=350)
mean_return_fig.savefig(logs_fig_directory + '/' +f'{env_name} Mean return', dpi=350)
std_return_fig.savefig(logs_fig_directory + '/' +f'{env_name} Std return', dpi=350)
max_return_fig.savefig(logs_fig_directory + '/' +f'{env_name} Max return', dpi=350)
min_return_fig.savefig(logs_fig_directory + '/' +f'{env_name} Min return', dpi=350)
mean_value_fig.savefig(logs_fig_directory + '/' +f'{env_name} Mean value', dpi=350)
policy_loss_fig.savefig(logs_fig_directory + '/' +f'{env_name} Policy loss', dpi=350)
kl_fig.savefig(logs_fig_directory + '/' +f'{env_name} KL', dpi=350)
entropy_fig.savefig(logs_fig_directory + '/' +f'{env_name} Entropy', dpi=350)
value_loss_fig.savefig(logs_fig_directory + '/' +f'{env_name} Value loss', dpi=350)
policy_grad_norm_fig.savefig(logs_fig_directory + '/' +f'{env_name} Policy grad norm', dpi=350)
value_grad_norm_fig.savefig(logs_fig_directory + '/' +f'{env_name} Value grad norm', dpi=350)
explained_variance_fig.savefig(logs_fig_directory + '/' +f'{env_name} Explained variance', dpi=350)
##############################  Save the figures  ########################################
plt.show()
