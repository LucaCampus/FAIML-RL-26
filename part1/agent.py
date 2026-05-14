import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)


        """
            Critic network
        """
        # TODO: TASK 3: critic network for actor-critic algorithm

        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        # The critic outputs a single scalar value representing the estimated value of the input state
        self.fc3_critic = torch.nn.Linear(self.hidden, 1)

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)


        """
            Critic
        """
        # TASK 3: forward in the critic network

        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        state_value = self.fc3_critic(x_critic)

        
        return normal_dist, state_value


class Agent(object):
    def __init__(self, policy, device='cpu', algorithm = 'reinforce', use_baseline=False):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        #lr=3e-4 for actor-critic, 1e-3 for REINFORCE
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        self.gamma = 0.99
        # If use_baseline is True, the agent will learn a value function baseline to reduce the variance 
        # of the policy gradient estimator
        self.use_baseline = use_baseline
        #algorithm can be 'reinforce' or 'actor_critic'
        self.algorithm = algorithm

        self.state_values = []

        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []


    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)
        
        #
        #REINFORCE
        #

        # Use baseline if enabled
        if self.algorithm == 'reinforce':
            # Compute discounted returns
            returns = discount_rewards(rewards, self.gamma)
            if self.use_baseline:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            loss = - (action_log_probs * returns).sum()
        
        #
        #ACTOR-CRITIC
        #
        
        elif self.algorithm == 'actor_critic':

            # For every state the critic estimates a value, and we store these predictions.
            state_values = torch.stack(self.state_values).to(self.train_device).squeeze(-1)

            #This tells PyTorch: “Do NOT compute gradients for the operations inside this block.”
            #Normally PyTorch builds a computation graph for backpropagation.
            #Without no_grad, every tensor operation is stored so .backward() can later compute gradients.
            #But here we only want to estimate values, not train from them directly.
            with torch.no_grad():
                _, next_state_values = self.policy(next_states)
                next_state_values = next_state_values.squeeze(-1)

            # Compute TD targets bootstraped, because instead of waiting until the end of the episode, 
            # you estimate future rewards using the critic itself. 
            # We multiply the next state values by (1 - done) to ensure that if the episode has ended, 
            # we don't add any future value.
            targets = rewards + self.gamma * next_state_values * (1 - done)
            
            # Compute advantages by subtracting the critic's value estimates from the TD targets.
            # Was this action better or worse than expected?
            advantages = targets - state_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Without .deatch actor loss would also modify critic parameters incorrectly.
            actor_loss = -(action_log_probs * advantages.detach()).sum()


            # The critic tries to learn the actual returns, using mean squared error loss between 
            # its value estimates and the TD targets.
            critic_loss = F.mse_loss(state_values, targets)

            loss = actor_loss + 0.5 *critic_loss

        # Gradient step
        # Clear the gradients
        self.optimizer.zero_grad()
        # Compute gradients
        loss.backward()
        # Step the optimizer
        self.optimizer.step()

        #
        #Clear memory
        #
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        self.state_values = []

        return        


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, state_value = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None, state_value

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob, state_value


    def store_outcome(self, state, next_state, action_log_prob, state_value, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.state_values.append(state_value)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

