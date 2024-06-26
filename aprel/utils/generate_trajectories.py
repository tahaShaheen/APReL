"""This module stores the functions for trajectory set generation."""

from typing import List, Union
import pickle
import numpy as np
from moviepy.editor import ImageSequenceClip
import warnings
import os

from gymnasium.wrappers import RecordVideo

from aprel.basics import Environment, Trajectory, TrajectorySet


def generate_trajectories_randomly(env: Environment,
                                   num_trajectories: int,
                                   max_episode_length: int = None,
                                   file_name: str = None,
                                   restore: bool = False,
                                   headless: bool = False,
                                   seed: int = None) -> TrajectorySet:
    """
    Generates :py:attr:`num_trajectories` random trajectories, or loads (some of) them from the given file.

    Args:
        env (Environment): An :class:`.Environment` instance containing the OpenAI Gym environment to be simulated.
        num_trajectories (int): the number of trajectories to generate.
        max_episode_length (int): the maximum number of time steps for the new trajectories. No limit is assumed if None (or not given).
        file_name (str): the file name to save the generated trajectory set and/or restore the trajectory set from.
            :Note: If :py:attr:`restore` is true and so a set is being restored, then the restored file will be overwritten with the new set.
        restore (bool): If true, it will first try to load the trajectories from :py:attr:`file_name`. If the file has fewer trajectories
            than needed, then more trajectories will be generated to compensate the difference.
        headless (bool): If true, the trajectory set will be saved and returned with no visualization. This makes trajectory generation
            faster, but it might be difficult for real humans to compare trajectories only based on the features without any visualization.
        seed (int): Seed for the randomness of action selection.
            :Note: Environment should be separately seeded. This seed is only for the action selection.

    Returns:
        TrajectorySet: A set of :py:attr:`num_trajectories` randomly generated trajectories.
        
    Raises:
        AssertionError: if :py:attr:`restore` is true, but no :py:attr:`file_name` is given.
    """
    assert(not (file_name is None and restore)), 'Trajectory set cannot be restored, because no file_name is given.'
    max_episode_length = np.inf if max_episode_length is None else max_episode_length # Taha: Set infinite max_episode_length if not given.
    if restore: # Taha: If restore is true, then try to load the trajectories from the file.
        try:
            with open('aprel_trajectories/' + file_name + '.pkl', 'rb') as f:
                trajectories = pickle.load(f)
        except:
            warnings.warn('Ignoring restore=True, because \'aprel_trajectories/' + file_name + '.pkl\' is not found.')
            trajectories = TrajectorySet([])
        if not headless:
            for traj_no in range(trajectories.size):
                if trajectories[traj_no].clip_path is None or not os.path.isfile(trajectories[traj_no].clip_path):
                    warnings.warn('Ignoring restore=True, because headless=False and some trajectory clips are missing.')
                    trajectories = TrajectorySet([])
                    break
    else: # Taha: Start with an empty TrajectorySet
        trajectories = TrajectorySet([])
    
    if not os.path.exists('aprel_trajectories'):
        os.makedirs('aprel_trajectories')
    
    if trajectories.size >= num_trajectories: # Taha: If size exceeds, save only first num_trajectories 
        trajectories = TrajectorySet(trajectories[:num_trajectories])
    else:
        env_has_rgb_render = env.render_exists and not headless
        if env_has_rgb_render and not os.path.exists('aprel_trajectories/clips'):
            os.makedirs('aprel_trajectories/clips')
        env.action_space.seed(seed)
        for traj_no in range(trajectories.size, num_trajectories):
            traj = []
            obs, _ = env.reset()
            if env_has_rgb_render:
                try:
                    frames = [np.uint8(env.render())]
                except:
                    env_has_rgb_render = False
            done = False
            t = 0
            while not done and t < max_episode_length:
                act = env.action_space.sample()
                traj.append((obs,act))
                obs, _, terminated, truncated, _ = env.step(act)
                done = terminated or truncated
                t += 1
                if env_has_rgb_render:
                    frames.append(np.uint8(env.render()))
            traj.append((obs, None))
            if env_has_rgb_render:
                clip = ImageSequenceClip(frames, fps=30)
                clip_path = 'aprel_trajectories/clips/' + file_name + '_' + str(traj_no) + '.mp4'
                clip.write_videofile(clip_path, audio=False)
            else:
                clip_path = None
            trajectories.append(Trajectory(env, traj, clip_path))
            if env.close_exists:
                env.close()

    with open('aprel_trajectories/' + file_name + '.pkl', 'wb') as f:
        pickle.dump(trajectories, f)

    if not headless and trajectories[-1].clip_path is None:
        warnings.warn(('headless=False was set, but either the environment is missing '
                       'a render function or the render function does not accept '
                       'mode=\'rgb_array\'. Automatically switching to headless mode.'))
    if headless:
        for traj_no in range(trajectories.size):
            trajectories[traj_no].clip_path = None
    return trajectories
