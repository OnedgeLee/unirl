import numpy as np


def map_action(action, setting, action_key):
    
    if len(action[action_key]) > 1:
        mapped_action = [setting["action_domain"][action_key][action_idx] for action_idx in action[action_key]]
    else:
        mapped_action = setting["action_domain"][action_key][action[action_key][0]]
    return mapped_action


def default_obs(obs_config):
    obs = {}
    for obs_key, obs_info in obs_config["obs"].items():
        obs[obs_key] = obs_info["default"]
    return obs


def generate_act_mask(setting, action_keys=[]):
    act_mask = {}
    for action_key, action_domain in setting["action_domain"].items():
        if action_key in action_keys:
            act_mask[action_key] = np.zeros((1, len(action_domain)))
        else:
            act_mask[action_key] = np.ones((1, len(action_domain)))
    return act_mask


def generate_neglogp_mask(setting, action_keys=[]):
    act_mask = {}
    for action_key, _ in setting["action_domain"].items():
        if action_key in action_keys:
            act_mask[action_key] = np.zeros((1))
        else:
            act_mask[action_key] = np.ones((1))
    return act_mask


def reset(obs_config, setting, unique_code=None):

    nextact_mask = generate_act_mask(setting, action_keys=["dummy_action_1"])
    obs = {}
    done = False

    return obs, done, nextact_mask

def dummy_env_func_1(obs, action, setting):

    neglogp_mask = generate_neglogp_mask(setting, action_keys=["dummy_action_1"])
    nextact_mask = generate_act_mask(setting, action_keys=["dummy_action_2"])

    dummy_action_1 = map_action(action, setting, "dummy_action_1")

    reward = 0.
    done = False
    
    return obs, reward, done, neglogp_mask, nextact_mask