#!/usr/bin/env python
# coding: utf-8

# # Coursework Template

# **Dependencies and imports**
# 
# This can take a minute...

# In[1]:


# !pip install swig
# !pip install --upgrade rldurham


# In[2]:


import torch
import rldurham as rld
import copy 
import math
import numpy as np


# **Reinforcement learning agent**
# 
# Replace this with your own agent, I recommend starting with TD3 (lecture 8).

# In[3]:


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, max_size=1000000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        # Storage arrays
        self.obs_buf      = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.act_buf      = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rew_buf      = np.zeros((max_size, 1), dtype=np.float32)
        self.done_buf     = np.zeros((max_size, 1), dtype=np.float32)
    def add(self, obs, act, rew, next_obs, done):
        idx = self.ptr
        self.obs_buf[idx]      = obs
        self.next_obs_buf[idx] = next_obs
        self.act_buf[idx]      = act
        self.rew_buf[idx]      = rew
        self.done_buf[idx]     = done
        self.ptr = (idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    def sample_batch(self, batch_size):
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        obs_batch      = torch.as_tensor(self.obs_buf[indices], dtype=torch.float32)
        act_batch      = torch.as_tensor(self.act_buf[indices], dtype=torch.float32)
        rew_batch      = torch.as_tensor(self.rew_buf[indices], dtype=torch.float32)
        next_obs_batch = torch.as_tensor(self.next_obs_buf[indices], dtype=torch.float32)
        done_batch     = torch.as_tensor(self.done_buf[indices], dtype=torch.float32)
        return obs_batch, act_batch, rew_batch, next_obs_batch, done_batch

# Soft Actor-Critic Agent
class SACAgent(torch.nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(SACAgent, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # Hyperparameters
        self.gamma = 0.99
        self.tau   = 0.005         
        self.alpha = 0.2           
        self.target_entropy = -act_dim  
        self.auto_entropy = True   
        self.batch_size = 256
        self.start_steps = 5000    
        # Actor network: outputs mean and log-std for Gaussian policy
        hidden_size = 256
        self.actor_net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 2 * act_dim)  # outputs [mean, log_std] concatenated
        )
        # Critic networks (twin Q-functions)
        def build_q_net():
            return torch.nn.Sequential(
                torch.nn.Linear(obs_dim + act_dim, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, 1)
            )
        self.q1 = build_q_net()
        self.q2 = build_q_net()
        # Target networks for Q (start as copies of initial Q networks)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for p in self.q1_target.parameters(): p.requires_grad = False
        for p in self.q2_target.parameters(): p.requires_grad = False

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=3e-4)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=3e-4
        )
        # Entropy (alpha) optimizer for automatic tuning
        if self.auto_entropy:
            # Initialize alpha as a learnable parameter (log scale)
            self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        # Replay buffer
        self.buffer = ReplayBuffer(obs_dim, act_dim, max_size=1000000)
        self.total_steps = 0

    def sample_action(self, state, eval_mode=False):
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        if self.total_steps < self.start_steps and not eval_mode:
            # Initial random exploration
            action = 2 * torch.rand((1, self.act_dim)) - 1  # random in [-1,1]
        else:
            with torch.no_grad():
                # Get mean and log_std from actor network
                mean_log_std = self.actor_net(s)
                mean = mean_log_std[..., :self.act_dim]
                log_std = mean_log_std[..., self.act_dim:]
                # Clamp log_std to reasonable range to prevent numerical issues
                log_std = torch.clamp(log_std, -20, 2)
                std = torch.exp(log_std)
                if eval_mode:
                    action = torch.tanh(mean)
                else:
                    eps = torch.randn_like(mean)
                    z = mean + std * eps
                    action = torch.tanh(z)
        self.total_steps += 1
        return action.squeeze(0).cpu().numpy()

    def compute_action_and_logprob(self, state_batch):
        mean_log_std = self.actor_net(state_batch)
        mean = mean_log_std[..., :self.act_dim]
        log_std = mean_log_std[..., self.act_dim:]
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        eps = torch.randn_like(mean)
        z = mean + std * eps
        action = torch.tanh(z)
        log_prob_gauss = -0.5 * (((z - mean) / (std + 1e-8))**2 + 2*log_std + math.log(2*math.pi))
        log_prob_gauss = log_prob_gauss.sum(dim=-1, keepdim=True)
        log_prob = log_prob_gauss - torch.log(1 - action.pow(2) + 1e-8).sum(dim=-1, keepdim=True)
        return action, log_prob

    def train_step(self):
        if self.buffer.size < self.batch_size:
            return
        obs_b, act_b, rew_b, next_obs_b, done_b = self.buffer.sample_batch(self.batch_size)
        # Critic update:
        with torch.no_grad():
            # Sample action from current policy for next state (actor is used directly for next state)
            next_action, next_log_prob = self.compute_action_and_logprob(next_obs_b)
            # Evaluate target Q values
            target_q1 = self.q1_target(torch.cat([next_obs_b, next_action], dim=1))
            target_q2 = self.q2_target(torch.cat([next_obs_b, next_action], dim=1))
            target_min_q = torch.min(target_q1, target_q2)
            alpha = self.alpha if not self.auto_entropy else self.log_alpha.exp()
            target_value = target_min_q - alpha * next_log_prob
            backup = rew_b + self.gamma * (1 - done_b) * target_value
        # Current Q estimates
        current_q1 = self.q1(torch.cat([obs_b, act_b], dim=1))
        current_q2 = self.q2(torch.cat([obs_b, act_b], dim=1))
        # MSE loss for critics
        critic_loss = torch.nn.functional.mse_loss(current_q1, backup) +                       torch.nn.functional.mse_loss(current_q2, backup)
        # Optimize critic
        self.q_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), max_norm=1.0)
        self.q_optimizer.step()
        # Actor update:
        # Sample actions for current state batch and compute log probs and Q-values
        new_action, log_prob = self.compute_action_and_logprob(obs_b)
        q1_val = self.q1(torch.cat([obs_b, new_action], dim=1))
        q2_val = self.q2(torch.cat([obs_b, new_action], dim=1))
        q_val = torch.min(q1_val, q2_val)
        # Actor loss = E[alpha * log_prob - Q]
        alpha = self.alpha if not self.auto_entropy else self.log_alpha.exp()
        actor_loss = (alpha * log_prob - q_val).mean()
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        # Entropy temperature update 
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            # Update alpha value from log_alpha
            self.alpha = self.log_alpha.exp().item()
        # Soft-update target networks
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# **Prepare the environment and wrap it to capture statistics, logs, and videos**

# In[4]:


#env = rld.make("rldurham/Walker", render_mode="rgb_array")
env = rld.make("rldurham/Walker", render_mode="rgb_array", hardcore=True) # only attempt this when your agent has solved the non-hardcore version

# get statistics, logs, and videos
env = rld.Recorder(
    env,
    smoothing=10,                       # track rolling averages (useful for plotting)
    video=True,                         # enable recording videos
    video_folder="videos",              # folder for videos
    video_prefix="wcrg83-agent-hardcore-video",  # prefix for videos (replace xxxx00 with your username)
    logs=True,                          # keep logs
)

# training on CPU recommended
rld.check_device()



# environment info
discrete_act, discrete_obs, act_dim, obs_dim = rld.env_info(env, print_out=True)

# render start image
env.reset(seed=42)
rld.render(env)


# In[ ]:


# in the submission please use seed_everything with seed 42 for verification
seed, observation, info = rld.seed_everything(42, env)

agent = SACAgent(obs_dim, act_dim)
max_episodes = 1500
max_timesteps = 2000
tracker = rld.InfoTracker()   # to track and plot statistics
env.video = False             # disable video recording initially

for episode in range(max_episodes):
    env.info  = True
    if episode < 1200:
        env.video = (episode + 1) % 100 == 0
    else:
        env.video = (episode + 1) % 1 == 0

    obs, info = env.reset()
    for t in range(max_timesteps):
        action = agent.sample_action(obs)           
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.buffer.add(obs, action, reward, next_obs, float(done))
        agent.train_step()                          

        obs = next_obs
        if done:
            break
    
    # Record statistics and plotting
    tracker.track(info)
    if (episode + 1) % 10 == 0:
        tracker.plot(r_mean_=True, r_std_=True, r_sum={'linestyle':':', 'marker':'x'})

# Close environment and save log
env.close()
env.write_log(folder="logs", file="wcrg83-agent-hardcore-log.txt")


# A small demo with a predefined heuristic that is suboptimal and has no notion of balance (and is designed for the orignal BipedalWalker environment)...

# In[ ]:


from gymnasium.envs.box2d.bipedal_walker import BipedalWalkerHeuristics

env = rld.make(
    "rldurham/Walker",
    # "BipedalWalker-v3",
    render_mode="human",
    # render_mode="rgb_array",
    hardcore=False,
    # hardcore=True,
)
_, obs, info = rld.seed_everything(42, env)

heuristics = BipedalWalkerHeuristics()

act = heuristics.step_heuristic(obs)
for _ in range(500):
    obs, rew, terminated, truncated, info = env.step(act)
    act = heuristics.step_heuristic(obs)
    if terminated or truncated:
        break
    if env.render_mode == "rgb_array":
        rld.render(env, clear=True)
env.close()

