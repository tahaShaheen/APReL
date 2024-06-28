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

import torch

# Taha:
# # Classic Control
# # https://gymnasium.farama.org/environments/classic_control/
# ENV_NAME = 'CartPole-v1'
# ENV_NAME = 'MountainCar-v0'
ENV_NAME = 'Acrobot-v1'
# ENV_NAME = 'MountainCarContinuous-v0'

# # BOX2D
# # https://gymnasium.farama.org/environments/box2d/
# ENV_NAME = 'LunarLander-v2'
# ENV_NAME = 'BipedalWalker-v3'

# # Atari
# # https://gymnasium.farama.org/environments/atari/
# ENV_NAME = 'Pong-v0'
# ENV_NAME = 'SpaceInvaders-v0'

# # Mujoco # Some sort of rendering issue. TODO: Use Wrappers (RecordVideo, RenderCollection, etc.) [https://gymnasium.farama.org/api/wrappers/misc_wrappers/]
# # https://gymnasium.farama.org/environments/mujoco/
# ENV_NAME = 'HalfCheetah-v4' 
# ENV_NAME = 'Ant-v4'

# # Toy Text
# # https://gymnasium.farama.org/environments/toy_text/ # IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed (line 50 simple.py). This is a issue with the code for the feature function, not the aprel library.
# ENV_NAME = 'CliffWalking-v0'
# ENV_NAME = 'Taxi-v3'
# ENV_NAME = 'FrozenLake-v1'
# ENV_NAME = 'Blackjack-v1' # Doesn't even render. Probably due to the nature of the observations.

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
        # True reward not used.
        if len(self.trajectory) == 5:
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

    SEED = 0 # Taha: Random seed for reproducibility.
    MAX_EPISODE_LENGTH = 300 # Taha: Maximum length of the trajectory. If the trajectory is not terminated by then, it will be terminated.
    NUMBER_OF_QUERIES = 10 # Taha: Total number of queries to ask the human. 10 is good
    NUM_TRAJECTORIES = 10 # Taha: Number of trajectories to generate randomly. 10 is good.
    TIMESTEPS = 5000 # Worked perfectly once. Then have had some success.
    # TIMESTEPS = 500 # Learned to reach the goal. Did not learn that going back is good. Once. Did not have success again.
    # TIMESTEPS = 1000 
    # TIMESTEPS = 10_000

    # Taha: Making directories to save the models and logs.
    MODEL_DIRECTORY = "models"
    LOG_DIRECTORY = "logs"
    os.makedirs(MODEL_DIRECTORY, exist_ok=True)
    os.makedirs(LOG_DIRECTORY, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)

    # Taha: Create the OpenAI Gym environment
    gym_env = gym.make(ENV_NAME, render_mode='rgb_array')

    np.random.seed(SEED) # Taha: Set the random seed for reproducibility for numpy operations
    gym_env.reset(seed=SEED) # Taha: Set the random seed for reproducibility for gym environment
    
    env = aprel.Environment(gym_env, feature_func) # Taha: This is like a wrapper around the gym environment. 

    model = PPO("MlpPolicy", env, verbose=1, device=device, tensorboard_log=LOG_DIRECTORY)
    model.set_random_seed(SEED)

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
    
    model = PPO("MlpPolicy", env, verbose=1, device=device, tensorboard_log=LOG_DIRECTORY, seed=SEED)

    # Taha: Generate 10 trajectories randomly of maximum timestep length 300.
    print('Generating trajectories with random policy ...')
    trajectory_set = aprel.generate_trajectories_randomly(env, num_trajectories=NUM_TRAJECTORIES,
                                                        max_episode_length=MAX_EPISODE_LENGTH,
                                                        file_name=ENV_NAME, seed=SEED,
                                                        # restore=True # Taha: Just to move things along faster. Uses the saved trajectories.
                                                        )
    
    # Taha: This helps to select which trajectories to show to the human. This particilar one assumes a discrete set of trajectories is available. The query optimization is then performed over this discrete set. How they will be optimzed is done later.
    query_optimizer = aprel.QueryOptimizerDiscreteTrajectorySet(trajectory_set)

    # Taha: Calculating the average reward of the trajectory set. This is calculated using the belief distribution which is random right now.
    reward = 0
    for traj in trajectory_set:
        reward += belief.mean['weights'].dot(traj.features)
    reward /= trajectory_set.size
    print('Average reward before learning: ' + str(reward))
                                        
    # Taha: Creates a query object using the first 2 trajectories. This object is not indexable. Can't get the trajectories back from it, I think.
    # This is created as an initial query and used later to ensure that the output query of the optimize function will have the same type.
    query = aprel.PreferenceQuery(trajectory_set[:2]) 

    # Taha: Spoofing the true user responses. The true user will respond with the following responses.
    # Since the seed is set to 0, the trajectories to compare will be the same each time and thus the responses will be the same each time.
    # simulated_human_feedback =  [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  # Subject to randomness.

    all_responses = []

    # Taha: Ask the user 10 queries.
    for query_no in range(NUMBER_OF_QUERIES):
        print('-----------------------------------')
        print(f'Query No: {query_no+1} of {NUMBER_OF_QUERIES} running')
        print('-----------------------------------')

        print('---------------------------------')
        print(f'Query number: {query_no+1} of {NUMBER_OF_QUERIES}')
        print('---------------------------------')

        # Taha: Optimizing the query_optimizer object with the 'mutual information' acquisition function.
        # Generates the optimal batch of queries to ask to the user given a belief distribution about them (which ones the user will select or which one the model thinks is tricky). It also returns the acquisition function values of the optimized queries. 
        # IN this case: The default optimization_method is 'exhaustive search' which returns a list of 1 query. It selects the query based on which one maximizes the acquisition function. Optimization method is how optimization is done. Acquisition function is hthe values that the optimization will base its functioning on.
        # The third argument is the query object that is used to ensure that the output query of the optimize function will have the same type.
        queries, acquisition_function_values = query_optimizer.optimize('mutual_information', belief, query)
        # print('Acquisition Function Value: ' + str(acquisition_function_values[0])) 
        
        # Taha: Ask the human to pick one of the queries from this list of 1 querry object. The human will respond with a list of 1 response.
        responses = true_user.respond(queries[0]) 
        # Taha: Spoofing for now. Will not ask human to respond.
        # responses = [fake_human_responses[query_no]]
        all_responses.append(responses[0])

        # Taha: Use a Preference object to update the belief distribution. This is the feedback from the human. The belief distribution is updated based on the user's response.
        print('Updating belief...')
        env.update_belief(aprel.Preference(queries[0], responses[0]))

        # Taha: We can see the belief mean here for each parameter. The belief distribution is being used to generate means of parameters which can then be used to create a reward. 
        # print('Estimated user parameters: ' + str(belief.mean))

        # Taha: Calculating the average reward of the trajectory set using the LEARNED belief distribution.
        reward = 0
        for traj in trajectory_set:
            reward += belief.mean['weights'].dot(traj.features)
        reward /= trajectory_set.size
        print(f'Average reward after {query_no+1}th query: {str(reward)}')
        
        # Taha: Doing some reinforcement learning with the learned reward function.
        print('Learning...')
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, log_interval=1)

        print('Saving model ...')
        # model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=True)
        model.save(f"{MODEL_DIRECTORY}/PPO_{TIMESTEPS}_{ENV_NAME}_query_{query_no+1}")
        if query_no < NUMBER_OF_QUERIES - 1:
            # Taha: Update the trajectory set with new trajectories generated by the updated agent. Should be better than previous trajectories.
            print('Generating new trajectories ...')
            trajectory_set = aprel.generate_trajectories_randomly(env, num_trajectories=NUM_TRAJECTORIES,
                                                            max_episode_length=MAX_EPISODE_LENGTH,
                                                            file_name=ENV_NAME, seed=SEED, 
                                                            model=model,
                                                            )
            
    print('All responses: ' + str(all_responses))

    # Taha: Load the model and run it. See if we learned anything.
    for query_no in range(NUMBER_OF_QUERIES):
        print(f'Generating video for model from query number: {query_no+1} of {NUMBER_OF_QUERIES}')

        env = gym.make(ENV_NAME, render_mode='rgb_array')
        env = gym.wrappers.RecordVideo(env, video_folder="videos/", name_prefix=f"PPO_{TIMESTEPS}_{ENV_NAME}_query_{query_no+1}", )
        model = PPO.load(f"{MODEL_DIRECTORY}/PPO_{TIMESTEPS}_{ENV_NAME}_query_{query_no+1}", env=env)

        obs, _ = env.reset()
        while True:
            # env.render() # Taha: This is not needed as we are recording the video.
            action, _ = model.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                break