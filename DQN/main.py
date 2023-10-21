import argparse
import utils
import os
import ast
import agents
import torch


def cmd_args():
    parser = argparse.ArgumentParser()
    # Paths and directories
    parser.add_argument('--working_dir', type=str, default="./DQN_runs")
    parser.add_argument('--checkpoint', type=str, default="none")
    # model layers
    parser.add_argument("--hidden_dims", type=str,
                        default="[512, 512]")
    # learning rate
    parser.add_argument("--lr", type=float, default=3e-4)
    # use target network
    parser.add_argument("--target_network",
                        action=argparse.BooleanOptionalAction)
    # double dqn
    parser.add_argument("--double",
                        action=argparse.BooleanOptionalAction)
    # tau, for target network update
    parser.add_argument("--tau", type=float, default=0.01)
    # size of the replay buffer
    parser.add_argument("--replay_buffer_size", type=int, default=int(1e6))
    # training batch size
    parser.add_argument("--batch_size", type=int, default=512)
    # minimum epsilon for exploration 
    parser.add_argument("--eps_min", type=float, default=0.01)
    # epslilon decay function
    parser.add_argument('--esp_decay_function', type=str, default="linear")
    # epslilon decay rate
    parser.add_argument('--eps_decay_rate', type=float, default=0.00001)
    # environment name
    parser.add_argument('--env_name', type=str, default="CartPole-v0")
    # discount factor
    parser.add_argument('--gamma', type=float, default=0.99)
    # use huber loss
    parser.add_argument('--huber_loss', action=argparse.BooleanOptionalAction)
    # observation type
    parser.add_argument('--obs_type', type=str, default="state")
    # seed
    parser.add_argument('--seed', type=int, default=0)
    # number of training steps
    parser.add_argument("--training_step", type=int, default=int(1e5))
    # logger
    parser.add_argument('--logger', type=str, default="wandb")
    # project_name
    parser.add_argument('--project_name', type=str, default="dqn")


    args = parser.parse_args()
    return args


def setup_experiment(args):
    hidden_dims = ast.literal_eval(args.hidden_dims)
    utils.seed_everything(args.seed)
    # create experiment folder
    exp_name = f"env_name_{args.env_name}, obs_type_{args.obs_type}, hidden_dims_{args.hidden_dims}, gamma_{args.gamma}, training_step{args.training_step}, batch_size:{args.batch_size}, target_network:{args.target_network}"
    exp_dir = f"{args.working_dir}/{exp_name}/seed_{args.seed}"
    os.makedirs(exp_dir, exist_ok=True)
    args_dic = vars(args)
    args_dic["hidden_dims"] = hidden_dims
    args_dic["exper_dir"] = exp_dir
    args_dic["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # create the DQN agent
    agent = agents.DQN_Agent(args_dic)
    # train the agent 
    agent.train(args.training_step)


if __name__ == "__main__":
    args = cmd_args()
    setup_experiment(args)