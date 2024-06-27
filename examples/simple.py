import aprel
import numpy as np
import gymnasium as gym
from gymnasium import RewardWrapper, ObservationWrapper, Wrapper
from aprel.basics import Trajectory
from collections import deque


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

class CustomRewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        # Keeping only 5 state action pairs in this trajectory.
        self.trajectory = deque(maxlen=5) # Taha: If a 6th element is added, the first element is removed.
        self.prev_obs = None

    def reset(self, seed=None, options=None):
        """Modifies the :attr:`env` after calling :meth:`reset`, returning a modified observation using :meth:`self.observation`."""
        obs, info = self.env.reset(seed=seed, options=options)
        self.trajectory.clear()
        self.prev_obs = obs
        return obs, info

    def step(self, action, belief=None):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action) 
        self.trajectory.append((self.prev_obs, action)) # Append a tuple of the previous observation and the action taken.
        self.prev_obs = observation
        return observation, self.reward(reward, belief), terminated, truncated, info
    
    def reward(self, reward, belief=None):
        reward = reward # True reward
        if len(self.trajectory) == 5 and belief is not None:
            features = Trajectory(env, list(self.trajectory)).features
            # Taha: The return of the trajectory is the dot product of the features of the trajectory and the user parameters.
            # Using this as a reward for now. The features don't allow for one single state-action pair to be used to calculate the reward.
            reward = belief.mean['weights'].dot(features) 
        
        return reward

gym_env = gym.make(env_name, render_mode='rgb_array')

np.random.seed(0) # Taha: Set the random seed for reproducibility for numpy operations
gym_env.reset(seed=0) # Taha: Set the random seed for reproducibility for gym environment

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


env = aprel.Environment(gym_env, feature_func) # Taha: This is like a wrapper around the gym environment. 
env = CustomRewardWrapper(env) # Taha: This is a custom reward wrapper around the environment.

# Taha: Generate 10 trajectories randomly of maximum timestep length 300.
trajectory_set = aprel.generate_trajectories_randomly(env, num_trajectories=10,
                                                      max_episode_length=300,
                                                      file_name=env_name, seed=0,
                                                    #   restore=True # Taha: Just to move things along faster. Uses the saved trajectories.
                                                      )
features_dim = len(trajectory_set[0].features) 

# Taha: This helps to select which trajectories to show to the human. This particilar one assumes a discrete set of trajectories is available. The query optimization is then performed over this discrete set. How they will be optimzed is done later.
query_optimizer = aprel.QueryOptimizerDiscreteTrajectorySet(trajectory_set)

# Taha: They assume that a real human is going to respond to the queries. 0.5 seconds delay time after each trajectory visualization.
true_user = aprel.HumanUser(delay=0.5) 

# Taha: Generate a random normalized vector of the same dimension as the features of the trajectory.
params = {'weights': aprel.util_funs.get_random_normalized_vector(features_dim)}

# Taha: "We will learn a reward function that is linear in trajectory features by assuming a softmax human response model." Reward = dot product of the features of the trajectory and the user parameters. The system will try to match the softmax user's responses to the true user's responses.
user_model = aprel.SoftmaxUser(params)

# Taha: We "create a belief distribution over the parameters we want to learn". The parameters dictionary can be sampled from the belief object. 
belief = aprel.SamplingBasedBelief(user_model, [], params)
# print('Estimated user parameters: ' + str(belief.mean))

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
resopnses_true =  [0, 0, 0, 0, 1, 0, 1, 0, 0, 0]

# Taha: Ask the user 10 queries.
for query_no in range(10):

    # Taha: Optimizing the query_optimizer object with the 'mutual information' acquisition function.
    # Generates the optimal batch of queries to ask to the user given a belief distribution about them (which ones the user will select or which one the model thinks is tricky). It also returns the acquisition function values of the optimized queries. 
    # IN thsi case: The default optimization_method is 'exhaustive search' which returns a list of 1 query. It selects the query based on which one maximizes the acquisition function. Optimization method is how optimization is done. Acquisition function is hthe values that the optimization will base its functioning on.
    queries, acquisition_function_values = query_optimizer.optimize('mutual_information', belief, query)
    print('Acquisition Function Value: ' + str(acquisition_function_values[0])) 
    
    # Taha: Ask the human to pick one of the queries. The human will respond with a list of 1 response.
    # responses = true_user.respond(queries[0]) 
    # Taha: Spoofing for now. Will not ask human to respond.
    responses = [resopnses_true[query_no]]

    # Taha: Use a Preference object to update the belief distribution. This is the feedback from the human. The belief distribution is updated based on the user's response.
    belief.update(aprel.Preference(queries[0], responses[0]))

    # Taha: We can see the belief mean here for each parameter. The belief distribution is being used to generate means of parameters which can then be used to create a reward. 
    print('Estimated user parameters: ' + str(belief.mean))

# Taha: Calculating the average reward of the trajectory set using the LEARNED belief distribution.
reward = 0
for traj in trajectory_set:
    reward += belief.mean['weights'].dot(traj.features)
reward /= trajectory_set.size
print ('Average reward after learning: ' + str(reward))
