import os
import aprel
from aprel import Belief, Preference

import numpy as np
import gymnasium as gym
from gymnasium import RewardWrapper, ObservationWrapper, Wrapper
from aprel.basics import Trajectory
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

# Taha:
# # Classic Control
# # https://gymnasium.farama.org/environments/classic_control/
# env_name = 'CartPole-v1'
env_name = 'MountainCar-v0'
# env_name = 'Acrobot-v1'

# # BOX2D
# # https://gymnasium.farama.org/environments/box2d/
# env_name = 'LunarLander-v2'
# env_name = 'BipedalWalker-v3'

# # Atari
# # https://gymnasium.farama.org/environments/atari/
# env_name = 'Pong-v0'
# env_name = 'SpaceInvaders-v0'

# # Mujoco # Some sort of rendering issue. TODO: Use Wrappers (RecordVideo, RenderCollection, etc.) [https://gymnasium.farama.org/api/wrappers/misc_wrappers/]
# # https://gymnasium.farama.org/environments/mujoco/
# env_name = 'HalfCheetah-v4' 
# env_name = 'Ant-v4'

# # Toy Text
# # https://gymnasium.farama.org/environments/toy_text/ # IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed (line 50 simple.py). This is a issue with the code for the feature function, not the aprel library.
# env_name = 'CliffWalking-v0'
# env_name = 'Taxi-v3'
# env_name = 'FrozenLake-v1'
# env_name = 'Blackjack-v1' # Doesn't even render. Probably due to the nature of the observations.

class CustomRewardWrapper(RewardWrapper):
    def __init__(self, env, belief: Belief):
        super().__init__(env)
        self.env = env
        # Keeping only 5 state action pairs in this trajectory.
        self.trajectory = deque(maxlen=5) # Taha: If a 6th element is added, the first element is removed.
        self.prev_obs = None
        self.belief = belief # Taha: The belief distribution over the user parameters.

    def reset(self, seed=None, options=None):
        """Modifies the :attr:`env` after calling :meth:`reset`, returning a modified observation using :meth:`self.observation`."""
        obs, info = self.env.reset(seed=seed, options=options)
        self.trajectory.clear()
        self.prev_obs = obs
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action) 
        self.trajectory.append((self.prev_obs, action)) # Append a tuple of the previous observation and the action taken.
        self.prev_obs = observation
        return observation, self.reward(reward), terminated, truncated, info
    
    def reward(self, reward):
        reward = reward # True reward
        if len(self.trajectory) == 5 and self.belief is not None:
            features = Trajectory(env, list(self.trajectory)).features
            # Taha: The return of the trajectory is the dot product of the features of the trajectory and the user parameters.
            # Using this as a reward for now. The features don't allow for one single state-action pair to be used to calculate the reward.
            reward = self.belief.mean['weights'].dot(features) 
        
        return reward
    
    def update_belief(self, preference: Preference):
        self.belief.update(preference)


def feature_func(traj): # Taha: This is the feature function that works well only with the MountainCar environment.
    """Returns the features of the given MountainCar trajectory, i.e. Phi(traj).
    
    Args:
        traj: List of state-action tuples, e.g. [(state0, action0), (state1, action1), ...]
    
    Returns:
        features: a numpy vector corresponding the features of the trajectory
    """
    states = np.array([pair[0] for pair in traj])
    actions = np.array([pair[1] for pair in traj[:-1]])
    min_pos, max_pos = states[:,0].min(), states[:,0].max() 
    mean_speed = np.abs(states[:,1]).mean()
    mean_vec = [-0.703, -0.344, 0.007]
    std_vec = [0.075, 0.074, 0.003]
    return (np.array([min_pos, max_pos, mean_speed]) - mean_vec) / std_vec

if __name__ == '__main__':
    
    # Taha: Making directories to save the models and logs.
    model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Taha: Create the OpenAI Gym environment
    gym_env = gym.make(env_name, render_mode='rgb_array')

    np.random.seed(0) # Taha: Set the random seed for reproducibility for numpy operations
    gym_env.reset(seed=0) # Taha: Set the random seed for reproducibility for gym environment

    env = aprel.Environment(gym_env, feature_func) # Taha: This is like a wrapper around the gym environment. 

    # Taha: They assume that a real human is going to respond to the queries. 0.5 seconds delay time after each trajectory visualization.
    true_user = aprel.HumanUser(delay=0.5) 
    
    # features_dim = len(trajectory_set[0].features) 
    features_dim = 3 # Taha: The features dimension is 3 for the MountainCar environment. Fixed for now.

    # Taha: Generate a random normalized vector of the same dimension as the features of the trajectory.
    params = {'weights': aprel.util_funs.get_random_normalized_vector(features_dim)}

    # Taha: "We will learn a reward function that is linear in trajectory features by assuming a softmax human response model." Reward = dot product of the features of the trajectory and the user parameters. The system will try to match the softmax user's responses to the true user's responses.
    user_model = aprel.SoftmaxUser(params)

    # Taha: We "create a belief distribution over the parameters we want to learn". The parameters dictionary can be sampled from the belief object. 
    belief = aprel.SamplingBasedBelief(user_model, [], params)
    # print('Estimated user parameters: ' + str(belief.mean))

    env = CustomRewardWrapper(env, belief) # Taha: This is a custom reward wrapper around the environment. It needs a belief distribution to calculate the reward.

    # Taha: This will check the custom environment and output additional warnings if needed. Helped to make the aprel env compatible with stable-baselines.
    check_env(env)

    # Taha: Generate 10 trajectories randomly of maximum timestep length 300.
    trajectory_set = aprel.generate_trajectories_randomly(env, num_trajectories=10,
                                                        max_episode_length=300,
                                                        file_name=env_name, seed=0,
                                                        # restore=True # Taha: Just to move things along faster. Uses the saved trajectories.
                                                        )
    
    # Taha: This helps to select which trajectories to show to the human. This particilar one assumes a discrete set of trajectories is available. The query optimization is then performed over this discrete set. How they will be optimzed is done later.
    query_optimizer = aprel.QueryOptimizerDiscreteTrajectorySet(trajectory_set)

    # Taha: Calculating the average reward of the trajectory set. This is calculated using the belief distribution which is random right now.
    reward = 0
    for traj in trajectory_set:
        reward += belief.mean['weights'].dot(traj.features)
    reward /= trajectory_set.size
    print ('Average reward before learning: ' + str(reward))
                                        
    # Taha: Creates a query object using the first 2 trajectories. This object is not indexable. Can't get the trajectories back from it, I think.
    # This is created as an initial query and used later to ensure that the output query of the optimize function will have the same type.
    query = aprel.PreferenceQuery(trajectory_set[:2]) 

    # Taha: Spoofing the true user responses. The true user will respond with the following responses.
    # fake_human_responses =  [1, 1, 

    number_of_querries = 10
    model = PPO("MlpPolicy", env, verbose=1, device='cuda', tensorboard_log=log_dir)
    TIMESTEPS = 50_000

    # Taha: Ask the user 10 queries.
    for query_no in range(number_of_querries):

        # Taha: Optimizing the query_optimizer object with the 'mutual information' acquisition function.
        # Generates the optimal batch of queries to ask to the user given a belief distribution about them (which ones the user will select or which one the model thinks is tricky). It also returns the acquisition function values of the optimized queries. 
        # IN this case: The default optimization_method is 'exhaustive search' which returns a list of 1 query. It selects the query based on which one maximizes the acquisition function. Optimization method is how optimization is done. Acquisition function is hthe values that the optimization will base its functioning on.
        # The third argument is the query object that is used to ensure that the output query of the optimize function will have the same type.
        queries, acquisition_function_values = query_optimizer.optimize('mutual_information', belief, query)
        print('Acquisition Function Value: ' + str(acquisition_function_values[0])) 
        
        # Taha: Ask the human to pick one of the queries. The human will respond with a list of 1 response.
        responses = true_user.respond(queries[0]) 
        # Taha: Spoofing for now. Will not ask human to respond.
        # responses = [fake_human_responses[query_no]]

        # Taha: Use a Preference object to update the belief distribution. This is the feedback from the human. The belief distribution is updated based on the user's response.
        env.update_belief(aprel.Preference(queries[0], responses[0]))

        # Taha: We can see the belief mean here for each parameter. The belief distribution is being used to generate means of parameters which can then be used to create a reward. 
        print('Estimated user parameters: ' + str(belief.mean))

        # Taha: Calculating the average reward of the trajectory set using the LEARNED belief distribution.
        reward = 0
        for traj in trajectory_set:
            reward += belief.mean['weights'].dot(traj.features)
        reward /= trajectory_set.size
        print (f'Average reward after query_no {query_no}:{str(reward)}')
        
        # Taha: Doing some reinforcement learning with the learned reward function.
        # model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=True)
        model.save(f"{model_dir}/PPO_{TIMESTEPS}")

        # Taha: Update the trajectory set with new trajectories generated by the updated agent. Should be better than previous trajectories.
        trajectory_set = aprel.generate_trajectories_randomly(env, num_trajectories=10,
                                                        max_episode_length=300,
                                                        file_name=env_name, seed=0, 
                                                        model=model,
                                                        )

    # Taha: Load the model and run it. See if we learned anything.
    model = PPO.load(f"{model_dir}/PPO_{TIMESTEPS}", env=env)

    env = gym.make(env_name, render_mode='human')

    obs, _ = env.reset()
    while True:
        env.render()
        action, _ = model.predict(obs)
        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break