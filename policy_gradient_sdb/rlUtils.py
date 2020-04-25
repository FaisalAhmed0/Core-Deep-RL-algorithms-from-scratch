import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym  



class Utils():
    @staticmethod
    def list_to_torch_tensor(List):
        l = []
        for element in List:
            for r in element:
                l.append(r)
        return torch.tensor(l, dtype=torch.float32)
    @staticmethod
    @torch.no_grad()
    def test_policy(network, env_name, tb, eps, n_runs=5,render=False, record=False):
        runs = n_runs
        total_reward = 0.0
        env = gym.make(env_name)
        if record:
            env = gym.wrappers.Monitor(env, "recording")
        for run in range(runs):
            state = env.reset()
            done = False
            while not done:
                if render:
                    env.render()
                state_t = torch.tensor(state, dtype=torch.float32)
                if type(env.action_space) == gym.spaces.Discrete:
                    action_logits = network(state_t)
                    actions_prob =  F.softmax(action_logits, dim=0)
                    action = torch.multinomial(actions_prob, 1).item()
                else:
                    mu,_ = network(state_t)
                    action = mu.cpu().numpy()
                state, reward, done, _ = env.step(action)
                total_reward += reward
        env.close()
        tb.add_scalar('test return', total_reward / runs, eps)
        return total_reward / runs

class SampleGeneration():

    @staticmethod
    @torch.no_grad()
    def generate_samples(net, env, tb, eps, batch_size=40 ):
        states_t = []
        actions_t = []
        rewards_t = []
        dones_t = []
        for trajectory in range(batch_size):
            states = []
            actions = []
            rewards = []
            dones = []
            state = env.reset()
            states.append(state)
            done = False
            total_reward = 0
            while not done:
                state_t = torch.tensor(state.astype(np.float32)).unsqueeze(0)
                if type(env.action_space) == gym.spaces.Discrete:
                    action_logits = net(state_t)
                    actions_prob =  F.softmax(action_logits, dim=1)
                    action = torch.multinomial(actions_prob, 1).item()
                else:
                    mu,var = net(state_t)
                    mu_numpy = mu.cpu().numpy()
                    var = var.cpu().numpy() 
                    std = np.sqrt(var)
                    action = np.random.normal(mu_numpy, std)
                    action = action.squeeze(0)
                actions.append(action)
                state, reward, done, _ = env.step(action)
                state = state.reshape(1,-1).squeeze(0)
                total_reward += reward
                rewards.append(reward)
                dones.append(done)
                if not done:
                    states.append(state)
            tb.add_scalar('training return', total_reward, ((batch_size*eps) + trajectory))
            rewards_t.append(np.array(rewards))
            states_t.append(np.array(states))
            actions_t.append(np.array(actions))
            dones_t.append(np.array(dones))
        return (np.array(states_t),np.array(rewards_t), np.array(actions_t), np.array(dones_t))


class ReturnEstimator():
    @staticmethod
    # reward_to_go
    @torch.no_grad()
    def reward_to_go(rewards, gamma=0.999):
        n = len(rewards)
        rewards_to_go = np.empty(n)
        for i in reversed(range(n)):
            rewards_to_go[i] = rewards[i] + (rewards_to_go[i+1] if i+1  < n else 0 )
        return rewards_to_go

    
    def n_step_batch(self, states, rewards_n, n=4):
        net = self.value_net
        cur_states = states
        res = []
        for i in range(len(rewards_n)):
            if i == 0:
                cur_states = states[i:len(rewards_n[i])]
            else:
                cur_states = states[len(rewards_n[i-1]):len(rewards_n[i])+len(rewards_n[i-1])]
            res.append(self._n_step_TD(net,cur_states,rewards_n[i],n))
        return np.array(res)

    @torch.no_grad()
    def _n_step_TD(self,net,states,rewards,n):
        res = []
        gamma = 0.99
        T = len(rewards)
        for t in range(T):
            sum_r = 0
            taw = t - n
            if taw < 0:
                res.append(rewards[t])
            if taw >= 0:
                for i in range(taw,min(taw+n,T)):
                    sum_r += (gamma**(i-taw) * rewards[i])
                if (taw+n) < T:
                    sum_r += gamma**n * net(torch.tensor(states[taw+n], dtype=torch.float32)).item()
                res.append(sum_r)
        res.reverse()
        return res


    def fit_v(self, states, targets, opt, tb, step):
        net = self.value_net
        states_t = torch.tensor(states,dtype=torch.float32)
        targets_t = Utils.list_to_torch_tensor(targets)
        opt.zero_grad()
        preds = net(states_t)
        loss = F.mse_loss(preds.squeeze(1), targets_t)
        tb.add_scalar('val_loss', loss, step)
#         print(loss)
        loss.backward()
        opt.step()

    @staticmethod
    @torch.no_grad()
    def calc_adv(net, states, rewards, dones, gamma=0.999):
        states_t = torch.FloatTensor(states)
        dones_mask = torch.tensor(dones).reshape(-1,1)
        rewards_t = torch.FloatTensor(rewards[0:-1]).reshape(-1,1)
        # print(states_t.shape)
        # print(dones_mask.shape)
        # print(rewards_t.shape)
        values = net(states_t)
        # print(values.shape)
        adv = (rewards_t + gamma * values[1:] )- (values[0:-1])
        adv_l = adv.data.tolist()
        adv_l.append([0])
        # print(adv_l)
        adv = torch.tensor(adv_l).view(-1)
        # print(adv.shape)
        return adv
    @torch.no_grad()
    def calc_adv_from_reward(self, states, rewards, gamma=0.99):
        value_net = self.value_net
        advs = []
        for i in range(len(rewards)):
            eps_len = len(rewards[i])
            last_state_reward = rewards[i][-1]
            rewards_t = torch.tensor(np.array(rewards[i][:-1]), dtype=torch.float32)

            if i == 0:
                cur_states = states[i:len(rewards[i])]
            else:
                cur_states = states[len(rewards[i-1]):len(rewards[i])+len(rewards[i-1])]

            states_t = torch.FloatTensor(cur_states[0:eps_len-1])
            next_states_t = torch.FloatTensor(cur_states[1:eps_len])

            states_values_t = value_net(states_t)
            next_states_values_t = value_net(next_states_t)
            rewards_t = rewards_t.reshape(-1,1)

            adv_t = (rewards_t + (gamma * next_states_values_t)) - states_values_t
            adv_n = adv_t.detach().numpy().tolist()
            adv_n.append([last_state_reward])
            advs.append(adv_n)

        return advs

class PolicyImprover():
    def __init__(self, policy_net):
        self.policy_net = policy_net

    def improve_policy(self, states, adv_t, actions, optimizer, tb, step):
        optimizer.zero_grad()
        states_t = torch.tensor(states, dtype=torch.float32)
        tb.add_scalar('adv',torch.mean(adv_t),step)
        logits = self.policy_net(states_t)
        actions_log_probs = F.log_softmax(logits, dim=1)
        selected_actions_log_probs_t = actions_log_probs[range(len(actions_log_probs)), actions]
    #     print(q_t)
        loss = selected_actions_log_probs_t * adv_t
        loss = -loss.mean()
        print(selected_actions_log_probs_t.mean())
    #     print(loss)
        tb.add_scalar('loss', loss, step)
        loss.backward()
        optimizer.step()
