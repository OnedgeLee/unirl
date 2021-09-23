import os
import numpy as np
import pandas as pd
import torch
from absl import logging
import ray
from .libs.rl_model import RlModel
from .libs.rl_runner import RlRunner
import wandb


def learn(yaml_config):

    learn_args = yaml_config["learn"]["learn_args"]
    wandb.init(
        project="dummy-project",
        notes="experiments for 0.0.0",
        tags=["0.0.0"],
        config=learn_args
    )

    gpu_count = torch.cuda.device_count()
    cpu_count = os.cpu_count()
    gpus = [torch.device("cuda:{}".format(i)) for i in range(gpu_count)]
    ray.init(num_cpus=cpu_count, num_gpus=gpu_count)
        
    RemoteRlModel = ray.remote(RlModel)
    RemoteRlRunner = ray.remote(RlRunner)
    
    if gpu_count > 0:
        models = [RemoteRlModel.options(num_gpus=1).remote(yaml_config) for gpu in gpus]
        n_runners_per_model = cpu_count // gpu_count
    else:
        models = [RemoteRlModel.remote(yaml_config)]
        n_runners_per_model = cpu_count

    for model in models:
        setattr(model, "runners", [RemoteRlRunner.remote(yaml_config) for _ in range(n_runners_per_model)])
        for runner in model.runners:
            setattr(runner, "model_ff", RlModel(yaml_config))

    ray.get([model.optim_set_lr.remote(learn_args["lr"]) for model in models])

    batch_steps = learn_args["batch_steps"]
    n_updates = int(learn_args["total_steps"]) // batch_steps
    model_steps = batch_steps // len(models)
    runner_steps = model_steps // n_runners_per_model
    minibatch_steps = model_steps // learn_args["n_minibatches_per_model"]
    
    os.makedirs(learn_args["save_path"], exist_ok=True)
    ckpt_paths = []
    with os.scandir(learn_args["save_path"]) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            if not entry.name.split(".")[-1] == "tar":
                continue
            if not entry.name.split("_")[0] == "ckpt":
                continue
            ckpt_paths.append(entry.path)

    if ckpt_paths:
        latest_ckpt_path = sorted(ckpt_paths, reverse=True)[0]
        ckpt = torch.load(latest_ckpt_path)
        step = int(ckpt["step"])
        ray.get([model.load_state_dict.remote(ckpt['model_state_dict']) for model in models])
        ray.get([model.optim_load_state_dict.remote(ckpt['optimizer_state_dict']) for model in models])
    else:
        step = 0

    dummy = ray.get([model.cuda.remote() for model in models])

    while step < n_updates:
        
        step += 1
        reward = []

        for model in models:
            [runner.model_ff.load_state_dict(ray.get(model.state_dict.remote())) for runner in model.runners]
            setattr(model, "rollout", [runner.rollout.remote(runner.model_ff, runner_steps, learn_args["gam"], learn_args["lam"]) for runner in model.runners])
        
        for model in models:
            batch_states, batch_actions, batch_returns, batch_values, batch_logits, batch_neglogps, batch_rewards, batch_act_masks, batch_neglogp_masks = zip(*ray.get(model.rollout))
            
            setattr(model, "batch_states", {k: np.concatenate(v) for k, v in pd.DataFrame(batch_states).items()})
            setattr(model, "batch_actions", {k: np.concatenate(v) for k, v in pd.DataFrame(batch_actions).items()})
            setattr(model, "batch_returns", np.concatenate(batch_returns))
            setattr(model, "batch_values", np.concatenate(batch_values))
            setattr(model, "batch_logits", {k: np.concatenate(v) for k, v in pd.DataFrame(batch_logits).items()})
            setattr(model, "batch_neglogps", {k: np.concatenate(v) for k, v in pd.DataFrame(batch_neglogps).items()})
            setattr(model, "batch_rewards", np.concatenate(batch_rewards))
            setattr(model, "batch_act_masks", {k: np.concatenate(v) for k, v in pd.DataFrame(batch_act_masks).items()})
            setattr(model, "batch_neglogp_masks", {k: np.concatenate(v) for k, v in pd.DataFrame(batch_neglogp_masks).items()})

            reward.append(model.batch_rewards)

        loss, policy_loss, entropy_loss, value_loss, entropy = [], [], [], [], []
        for _ in range(learn_args["n_opt_epochs"]):
            batch_indices = np.arange(model_steps)
            np.random.shuffle(batch_indices)

            for start in range(0, model_steps, minibatch_steps):
                end = start + minibatch_steps
                minibatch_indices = batch_indices[start:end]

                ray.get([model.optim_zero_grad.remote() for model in models])

                grads, batch_loss, batch_policy_loss, batch_entropy_loss, batch_value_loss, batch_entropy = zip(*ray.get([getattr(model, "train_{}".format(learn_args["algorithm"])).remote(**{
                    "batch_states": {k:v[minibatch_indices] for k, v in model.batch_states.items()}, 
                    "batch_actions": {k:v[minibatch_indices] for k, v in model.batch_actions.items()}, 
                    "batch_returns": model.batch_returns[minibatch_indices], 
                    "batch_values": model.batch_values[minibatch_indices], 
                    "batch_neglogps": {k:v[minibatch_indices] for k, v in model.batch_neglogps.items()}, 
                    "batch_act_masks": {k:v[minibatch_indices] for k, v in model.batch_act_masks.items()}, 
                    "batch_neglogp_masks": {k:v[minibatch_indices] for k, v in model.batch_neglogp_masks.items()},
                    "max_grad_norm": learn_args["max_grad_norm"], 
                    "cliprange": learn_args["cliprange"], 
                    "coef_el": learn_args["coef_el"], 
                    "coef_vl": learn_args["coef_vl"]
                }) for model in models]))

                master_grad = [sum(filter(lambda x: x is not None, grad_list)) / len(grad_list) for grad_list in zip(*grads)]
                master_grad = [grad if isinstance(grad, torch.Tensor) else None for grad in master_grad]
                
                ray.get(models[0].set_grads.remote(master_grad))
                ray.get(models[0].optim_step.remote())

                state_dict = ray.get(models[0].state_dict.remote())
                ray.get([model.load_state_dict.remote(state_dict) for model in models[1:]])
                [[runner.model_ff.load_state_dict(state_dict) for runner in model.runners] for model in models]

                loss.append(batch_loss)
                policy_loss.append(batch_policy_loss)
                entropy_loss.append(batch_entropy_loss)
                value_loss.append(batch_value_loss)
                entropy.append(batch_entropy)
                    

        loss = np.mean(loss)
        policy_loss = np.mean(policy_loss)
        entropy_loss = np.mean(entropy_loss)
        value_loss = np.mean(value_loss)
        entropy = np.mean(entropy)
        reward = np.mean(reward)
                
        logging.log(logging.INFO, '\t[step:{:10d}]\t[loss:{:10.2f}]\t[p_loss:{:10.2f}]\t[e_loss:{:10.2f}]\t[v_loss:{:10.2f}]\t[rew:{:10.2f}]'.format(step, loss, policy_loss, entropy_loss, value_loss, reward))
        
        if step % learn_args["log_steps"] == 0:
            ckpt = torch.save({
                "step": step,
                "model_state_dict": ray.get(models[0].state_dict.remote()),
                "optimizer_state_dict": ray.get(models[0].optim_state_dict.remote())
            }, os.path.join(learn_args["save_path"], "ckpt_{}.tar".format(str(step).zfill(10))))
            logging.log(logging.INFO, "Saved checkpoint for step {}: {}".format(step, learn_args["save_path"]))
            wandb.log({
                "loss":loss,
                "policy_loss":policy_loss,
                "entropy_loss":entropy_loss,
                "value_loss":value_loss,
                "entropy": entropy,
                "reward":reward
            })
