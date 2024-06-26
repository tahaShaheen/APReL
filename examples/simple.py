import aprel
import numpy as np
import gymnasium as gym

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

# Taha: Generate 10 trajectories randomly of maximum timestep length 300.
trajectory_set = aprel.generate_trajectories_randomly(env, num_trajectories=10,
                                                      max_episode_length=300,
                                                      file_name=env_name, seed=0)
features_dim = len(trajectory_set[0].features)

query_optimizer = aprel.QueryOptimizerDiscreteTrajectorySet(trajectory_set)

true_user = aprel.HumanUser(delay=0.5)

params = {'weights': aprel.util_funs.get_random_normalized_vector(features_dim)}
user_model = aprel.SoftmaxUser(params)
belief = aprel.SamplingBasedBelief(user_model, [], params)
print('Estimated user parameters: ' + str(belief.mean))
                                       
query = aprel.PreferenceQuery(trajectory_set[:2])

for query_no in range(10):
    queries, objective_values = query_optimizer.optimize('mutual_information', belief, query)
    print('Objective Value: ' + str(objective_values[0]))
    
    responses = true_user.respond(queries[0])
    belief.update(aprel.Preference(queries[0], responses[0]))
    print('Estimated user parameters: ' + str(belief.mean))
