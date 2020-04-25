## This codebase contains an implementation of the Vanilla policy gradient algorithm with State dependent baseline V(s)
for more details about this form of policy gradient: 
https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#baselines-in-policy-gradients

## pararmeters:
    env: environnment name assuming it is a gym environment
    s: a reference path for saving relevent files such as the network state and tensorboard log files
    lr : learning rate for both actor and the critic
    eps: number of training epoches
    batch: batch size for each epoch
    nn_hidden: number of units in the two hidden layers architecture for both the policy and value nets

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
    learning rate: 1e-3
    nn_hidden: [64,64]
    batch : 10
    eps: 3000
## To see the performance curves run tensorboard on the cmd/terminal:
    tensorboard --logdit=<log_files directory>
    open the output local host link in your favorite browser