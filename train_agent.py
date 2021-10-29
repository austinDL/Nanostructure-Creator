#Imports

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import os

from Optics_Model import Environment
from td3_torch import Agent

import torch
if torch.cuda.is_available():
  print('Working on GPU')
  
  
  
def create_target_structure(resolution, start, stop):
  ## CREATES A LINEAR NANOSTRUCTURE ##
  trgt_structure = np.zeros((resolution, resolution))
  trgt_structure[:] = np.cos(np.linspace(start, stop, resolution))
  
  return trgt_structure


### File saving Functions ###

def save_new_buffer(name):
  mem_size = 1_000_000
  np.save(f'/content/gdrive/My Drive/rmc/Training Ground/Replay Buffer/{name}p/counter.npy', np.array(0))
  np.save(f'/content/gdrive/My Drive/rmc/Training Ground/Replay Buffer/{name}p/states.npy', np.zeros((mem_size, 2)))
  np.save(f'/content/gdrive/My Drive/rmc/Training Ground/Replay Buffer/{name}p/new_states.npy', np.zeros((mem_size, 2)))
  np.save(f'/content/gdrive/My Drive/rmc/Training Ground/Replay Buffer/{name}p/actions.npy', np.zeros((mem_size, 3)))
  np.save(f'/content/gdrive/My Drive/rmc/Training Ground/Replay Buffer/{name}p/rewards.npy', np.zeros(mem_size))
  np.save(f'/content/gdrive/My Drive/rmc/Training Ground/Replay Buffer/{name}p/dones.npy', np.zeros(mem_size, dtype=np.bool))
  
def create_folders(name):
  os.makedirs(f'/content/gdrive/My Drive/rmc/Training Ground/Agent Weights/{name}p')
  os.makedirs(f'/content/gdrive/My Drive/rmc/Training Ground/Progress Images/{name}p')
  os.makedirs(f'/content/gdrive/My Drive/rmc/Training Ground/Replay Buffer/{name}p')
  save_new_buffer(name)




### Agent training functions ###
def create_noise(min_noise, max_noise, end_ratio, t):
  range_ = np.linspace(0,1,num=n_exploit*t)
  scale = np.log(max_noise/min_noise)/end_ratio
  exploit_schedule = np.clip(np.e**(-scale*range_)*max_noise, min_noise, max_noise)
  
  explore_schedule = np.ones(n_explore*t)*explore_noise

  n_schedule = n_explore + n_exploit
  noise_schedule = np.zeros(n_schedule*t + 1_000_000)
  noise_schedule[:n_explore*t] = explore_schedule
  noise_schedule[n_explore*t:n_schedule*t] = exploit_schedule
  
  return noise_schedule

def train():
    
  #Keep track of the scores
  
  best_score = best_score_0
  score_history = []
  
  #Initialize general parameters
  is_exploit = False
  n_episodes = n_explore + n_exploit
  p_num = p_num_0
  
  #Initialize the environment and agent
  env = Environment(max_val, array_num,                                   # Env
                    action_max, action_min, TF_,                          # Action
                    trgt_DP, reward_power, old_reward_scale)              # Reward
                    
  
  t = env.arrayNum**2
  agent = Agent(alpha=0.001, beta=0.001, input_dims=2, tau=0.005, env=env,
                noise_schedule=create_noise(exploit_noise_min, exploit_noise_max, end_ratio, t), bound_scale=bound_scale,
                gamma=1, update_actor_interval=2, warmup=n_explore*t, 
                n_actions=3, layer1_size=400, layer2_size=300, batch_size=300, name=f_name)
    
    
  if load_agent:
    agent.load_models()
  if reset_buffer:
    agent.reset_buffer()
        
  i = 0
  for e in tqdm(range(n_episodes)):
    #Initialize episode paramaters
    state, done = env.reset()
    score = 0
    #print(i)
    i=0
    while not done:
      i += 1
      #Step through the environment by choosing an action
      action = agent.choose_action(state)
      action = np.array(action).flatten()
      next_state, reward, done = env.step(action)
      
      #Store the experience tuple
      agent.remember(state, action, reward, next_state, done)
      #Learn from the experiences
      agent.learn()
      
      #Increment the score
      score += reward
      #Set the new state
      state = next_state

  
    #Append the average of the scores to the score history
    avrg_episode_score = score/(env.arrayNum**2)
    score_history.append(avrg_episode_score)
    avrg_score = np.mean(score_history[-25:])
      
    #Save the results
    if avrg_score > best_score:
      best_score = avrg_score
      print(best_score)
      #Save agent parameters
      agent.save_models()
      agent.save_buffer()
      plt.title(best_score)
      plt.imshow(env.render(), cmap='gray')
      plt.show()
        
        
    #Display the progress
    if (e+1) % render_every == 0:
      progress_IMG, progress_score = agent.eval_()

      title_score = round(progress_score*100,2)
      plt.title(f'Score: {title_score}%')
      plt.imshow(progress_IMG, cmap='gray', vmin=-1, vmax=1)
      plt.axis('off')
      plt.savefig(save_fig_name + f'p_{p_num}.jpg')
      #Increment
      p_num += 1
      #plt.show()

  agent.save_buffer()    
  #Plot the Agent's progress through the episodes
  plt.show()
  plot_name = save_fig_name + 'Agent Progress.jpg'
  plot_learning_curve(score_history, plot_name)
  plt.show()
  
  
### Setup the training parameters and train ###
#Exploration-Exploitation
n_explore = 50
render_every = 5

n_episodes = 2000
n_exploit = n_episodes - n_explore

#Initial conditions
best_score_0 = np.float('-inf')
p_num_0 = 0

#Noise Schedule
explore_noise = 0.5
exploit_noise_max = 0.5
exploit_noise_min = 0.025
end_ratio = 0.9
bound_scale = 1.75

#Determine if we are loading an agent and/or resetting the buffer
load_agent = False
reset_buffer = True
new_agent = True

#Environment Parameters
max_val = 0.1
array_num = 30
trgt_DP = ref_IMG

action_max = np.array([2,  .25*np.pi, .75*np.pi])
action_min = np.array([0, -.25*np.pi, .25*np.pi ])

reward_power = 1
old_reward_scale = 0.01
TF_ = False

#Determine the save locations
name = int(old_reward_scale*100)
f_name = f'{name}p'
save_fig_name = f'/content/gdrive/My Drive/rmc/Training Ground/Progress Images/{name}p/'
if new_agent:
  create_folders(name)
  load_agent = False

train()
