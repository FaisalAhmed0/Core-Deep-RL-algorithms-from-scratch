## This codebase contains an implementation of the Vanilla policy gradient algorithm with State dependent baseline V(s)
for more details about this form of policy gradient: 
https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#baselines-in-policy-gradients

## pararmeters:
    env: environnment name assuming it is a gym environment
    s: a reference path for saving relevent files such as the network state and tensorboard log files
    lr_act : learning rate for policy network
    lr_crt : learning rate for value network
    epochs: number of training epochs
    eps: number of episodes per epoch
    nn_hidden: number of units in the two hidden layers architecture for both the policy and value nets
    seed: random number generation seed

## Performance and diagnosis:
    KL divergence of the policy
    Loss for both value and policy nets
    Max Episode Return
    Min Episode Return
    Mean Episode Return
    Std Episode Return
    Entropy
    Q estimation
    Value net output

## Default parameters:
    learning rate (for both policy and value networks): 1e-3
    nn_hidden: [64,64]
    eps : 1
    epochs: 3000
## To see the performance curves run tensorboard on the cmd/terminal:
    tensorboard --logdit=<log_files directory>
    open the output local host link in your favorite browser