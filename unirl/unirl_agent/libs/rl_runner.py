import numpy as np
import pandas as pd
from unirl_env.libs.envlib.env_func import *
from unirl_agent.libs.rl_preprocess import *

class RlRunner():

    def __init__(self, yaml_config):
        self.yaml_config = yaml_config
        self.env_config = yaml_config["env"]
        self.agent_config = yaml_config["agent"]
        self.env_func_config = self.env_config["env_func"]
        self.env_setting = self.env_config["setting"]
        self.obs_config = yaml_config["obs"]
        self.state_config = yaml_config["state"]
        self.action_config = yaml_config["action"]
        self.action_domain_config = self.action_config["action_domain"]
        
        
    def preprocess(self, obs):
        state = {}
        for state_key, state_infos in self.state_config["state"].items():
            for preprocess in state_infos["preprocess"]:
                for preprocess_key, preprocess_infos in preprocess.items():
                    preprocess_kwargs = {}
                    for kwargs_key, kwargs_value in preprocess_infos["kwargs"].items():
                        if kwargs_value["type"] == "obs":
                            preprocess_kwargs[kwargs_key] = obs[kwargs_value["arg"]]
                        elif kwargs_value["type"] == "obs_list":
                            preprocess_kwargs[kwargs_key] = [obs[obs_key] for obs_key in kwargs_value["arg"]]
                        else:
                            preprocess_kwargs[kwargs_key] = kwargs_value["arg"]
    
                    obs[preprocess_infos["return"]] = globals()[preprocess_key](**preprocess_kwargs)
            
            state_shape = state_infos["spec"]["shape"][1:]
            padding = [[0, x] for x in state_shape]
            slicing = tuple([slice(x) for x in state_shape])
            padded_state = np.pad(obs[state_key], padding)[slicing]
            state[state_key] = np.expand_dims(padded_state, 0)
        
        return state

    def rollout(self, model, batch_steps=1024, gam=0.99, lam=0.95):

        done = True
        while done:
            obs, done, act_mask = globals()["reset"](self.obs_config, self.action_domain_config, self.env_setting)
        state = self.preprocess(obs)
        
        batch_obs, batch_states, batch_actions, batch_rewards, batch_values, batch_logits, batch_neglogps, batch_dones, batch_act_masks, batch_neglogp_masks = [],[],[],[],[],[],[],[],[],[]
    
        env_idx = 0
        for i in range(batch_steps):
            env_key = self.env_func_config[env_idx % len(self.env_func_config)]
            
            action_sample, action_logit, action_neglogp, action_entropy, value = model.step(state, act_mask)
            next_obs, reward, next_done, neglogp_mask, nextact_mask = globals()[env_key](obs, action_sample, self.action_domain_config, self.env_setting)
            next_state = self.preprocess(next_obs)
            
            batch_obs.append(obs)
            batch_states.append(state)
            batch_actions.append(action_sample)
            batch_rewards.append(np.asarray([reward], dtype=np.float32))
            batch_values.append(value)
            batch_logits.append(action_logit)
            batch_neglogps.append(action_neglogp)
            batch_dones.append(np.asarray([done], dtype=bool))
            batch_act_masks.append(act_mask)
            batch_neglogp_masks.append(neglogp_mask)

            obs = next_obs
            state = next_state
            act_mask = nextact_mask
            done = next_done

            env_idx += 1

            if done:
                while done:
                    obs, done, act_mask = globals()["reset"](self.obs_config, self.action_domain_config, self.env_setting)
                state = self.preprocess(obs)
                env_idx = 0

        batch_states, batch_actions, batch_logits, batch_neglogps, batch_act_masks, batch_neglogp_masks = map(
            lambda x: {k:np.concatenate(v) for k, v in pd.DataFrame(x).items()}, (
                batch_states, 
                batch_actions, 
                batch_logits, 
                batch_neglogps, 
                batch_act_masks,
                batch_neglogp_masks,
            )
        )
        batch_rewards, batch_values, batch_dones = map(
            lambda x: np.concatenate(x), (batch_rewards, batch_values, batch_dones)
        )

        last_value = model.value(state)

        batch_returns = np.zeros_like(batch_rewards)
        batch_advs = np.zeros_like(batch_rewards)
        lastgaelam = 0
        for t in reversed(range(batch_steps)):
            if t == batch_steps - 1:
                nextnonterminal = 1.0 - done
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - batch_dones[t+1]
                nextvalues = batch_values[t+1]
            delta = batch_rewards[t] + gam * nextvalues * nextnonterminal - batch_values[t]
            batch_advs[t] = lastgaelam = delta + gam * lam * nextnonterminal * lastgaelam        
        batch_returns = batch_advs + batch_values

        return batch_states, batch_actions, batch_returns, batch_values, batch_logits, batch_neglogps, batch_rewards, batch_act_masks, batch_neglogp_masks
