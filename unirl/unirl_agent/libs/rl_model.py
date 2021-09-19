import sys
import torch
import torch.nn as nn
from .nn_models import *


class RlModel(nn.Module):

    def __init__(self, yaml_config):
        super(RlModel, self).__init__()

        self.yaml_config = yaml_config
        self.agent_config = yaml_config["agent"]
        self.state_encoder_configs = self.agent_config["state_encoder"]
        self.action_head_configs = self.agent_config["action_head"]
        self.core_encoder_config = self.agent_config["core_encoder"]
        self.value_predictor_config = self.agent_config["value_predictor"]
        
        for state_encoder_config in self.state_encoder_configs:
            encoder = globals()[state_encoder_config["encoder"]](**state_encoder_config["kwargs"])
            setattr(self, "_".join([state_encoder_config["state"], "encoder"]), encoder)
            
        self.core = globals()[self.core_encoder_config["encoder"]](**self.core_encoder_config["kwargs"])

        for action_head_config in self.action_head_configs:
            head = globals()[action_head_config["head"]](**action_head_config["kwargs"])
            setattr(self, "_".join([action_head_config["action"], "head"]), head)
        
        self.value_predictor = globals()[self.value_predictor_config["predictor"]](**self.value_predictor_config["kwargs"])

        self.optimizer = torch.optim.RMSprop(self.parameters())

    def optim_set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def optim_load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def optim_zero_grad(self):
        self.optimizer.zero_grad()

    def optim_step(self):
        if all([p.is_cuda for p in self.parameters()]):
            for param in self.optimizer.state.values():
                if isinstance(param, torch.Tensor):
                    param.data = param.data.cuda()
                    if param._grad is not None:
                        param._grad.data = param._grad.data.cuda()
                elif isinstance(param, dict):
                    for subparam in param.values():
                        if isinstance(subparam, torch.Tensor):
                            subparam.data = subparam.data.cuda()
                            if subparam._grad is not None:
                                subparam._grad.data = subparam._grad.data.cuda()
        self.optimizer.step()

    def optim_state_dict(self):
        return self.optimizer.state_dict()

    def get_model(self):
        return self

    def get_grads(self):
        return [x.grad for x in self.parameters()]

    def set_grads(self, grads):
        for param, grad in zip(self.parameters(), grads):
            if grad is not None:
                if param.is_cuda:
                    param._grad.data = grad.data.cuda()
                else:
                    param._grad.data = grad.data
        
    def step(self, state, act_mask):

        state = {state_key: torch.Tensor(state_) for state_key, state_ in state.items()}
        act_mask = {action_key: torch.Tensor(act_mask_) for action_key, act_mask_ in act_mask.items()}
        if all([p.is_cuda for p in self.parameters()]):
            state = {state_key: state_.cuda() for state_key, state_ in state.items()}
            act_mask = {action_key: act_mask_.cuda() for action_key, act_mask_ in act_mask.items()}

        state_emb = {}
        entity_emb = {}
        for state_key, encoder_info in self.state_encoder_config.items():
            encoder = getattr(self, "_".join([state_key, "encoder"]))
            encoded = encoder(state[state_key])
            if isinstance(encoder, globals()["EntityEncoder"]):
                state_emb[state_key] = encoded[1]
                entity_emb[state_key] = encoded[0]
            elif isinstance(encoder, globals()["PolygonEntityEncoder"]):
                state_emb[state_key] = encoded[1]
                entity_emb[state_key] = encoded[0]
            elif isinstance(encoder, globals()["LinestringEntityEncoder"]):
                state_emb[state_key] = encoded[1]
                entity_emb[state_key] = encoded[0]
            else:
                state_emb[state_key] = encoded
                
        state_emb_list = [v for k, v in sorted(state_emb.items())]
        core_emb = self.core(state_emb_list)

        value = self.value_predictor(core_emb).detach().numpy()

        action_sample = {}
        action_logit = {}
        action_neglogp = {}
        action_entropy = {}
        action_emb = {}
        action_emb["none"] = core_emb

        for action_key in self.action_order_config:
            rationale_keys = self.action_sampling_config[action_key]["rationales"]
            rationales = []
            for rationale_key in rationale_keys:
                rationale_key_splitted = rationale_key.split("_")
                rationale_type = rationale_key_splitted[0]
                rationale_name = "_".join(rationale_key_splitted[1:])
                if rationale_type == "action":
                    rationales.append(action_emb[rationale_name])
                elif rationale_type == "state":
                    rationales.append(state_emb[rationale_name])
                elif rationale_type == "entity":
                    rationales.append(entity_emb[rationale_name])
                else:
                    raise ValueError()
            action_sample_, action_logit_, action_neglogp_, action_entropy_, action_emb[action_key] = getattr(self, "_".join([action_key, "head"]))(*rationales, mask=act_mask[action_key])
            action_sample[action_key], action_logit[action_key], action_neglogp[action_key], action_entropy[action_key] = action_sample_.detach().numpy(), action_logit_.detach().numpy(), action_neglogp_.detach().numpy(), action_entropy_.detach().numpy()
            
        return action_sample, action_logit, action_neglogp, action_entropy, value


    def recollection(self, batch_states, batch_actions, batch_act_masks):

        batch_states = {state_key: torch.Tensor(batch_states_) for state_key, batch_states_ in batch_states.items()}
        batch_actions = {action_key: torch.Tensor(batch_actions_) for action_key, batch_actions_ in batch_actions.items()}
        batch_act_masks = {action_key: torch.Tensor(batch_act_masks_) for action_key, batch_act_masks_ in batch_act_masks.items()}
        if all([p.is_cuda for p in self.parameters()]):
            batch_states = {state_key: batch_states_.cuda() for state_key, batch_states_ in batch_states.items()}
            batch_actions = {action_key: batch_actions_.cuda() for action_key, batch_actions_ in batch_actions.items()}
            batch_act_masks = {action_key: batch_act_masks_.cuda() for action_key, batch_act_masks_ in batch_act_masks.items()}

        state_emb = {}
        entity_emb = {}
        for state_key in self.state_order_config:
            encoder = getattr(self, "_".join([state_key, "encoder"]))
            encoded = encoder(batch_states[state_key])
            if isinstance(encoder, globals()[self.agent_cls_config["entity_encoder"]["name"]]):
                state_emb[state_key] = encoded[1]
                entity_emb[state_key] = encoded[0]
            elif isinstance(encoder, globals()[self.agent_cls_config["polygon_entity_encoder"]["name"]]):
                state_emb[state_key] = encoded[1]
                entity_emb[state_key] = encoded[0]
            elif isinstance(encoder, globals()[self.agent_cls_config["linestring_entity_encoder"]["name"]]):
                state_emb[state_key] = encoded[1]
                entity_emb[state_key] = encoded[0]
            else:
                state_emb[state_key] = encoded
                
        state_emb_list = [v for k, v in sorted(state_emb.items())]
        core_emb = self.core(state_emb_list)

        value = self.value_predictor(core_emb)

        action_sample = {}
        action_logit = {}
        action_neglogp = {}
        action_entropy = {}
        action_emb = {}
        action_emb["none"] = core_emb

        for action_key in self.action_order_config:
            rationale_keys = self.action_sampling_config[action_key]["rationales"]
            rationales = []
            for rationale_key in rationale_keys:
                rationale_key_splitted = rationale_key.split("_")
                rationale_type = rationale_key_splitted[0]
                rationale_name = "_".join(rationale_key_splitted[1:])
                if rationale_type == "action":
                    rationales.append(action_emb[rationale_name])
                elif rationale_type == "state":
                    rationales.append(state_emb[rationale_name])
                elif rationale_type == "entity":
                    rationales.append(entity_emb[rationale_name])
                else:
                    raise ValueError()
            action_sample[action_key], action_logit[action_key], action_neglogp[action_key], action_entropy[action_key], action_emb[action_key] = getattr(self, "_".join([action_key, "head"]))(*rationales, action=batch_actions[action_key], mask=batch_act_masks[action_key])
            
        return action_sample, action_logit, action_neglogp, action_entropy, value

    def mode(self, state, action_mask):

        state = {state_key: torch.Tensor(state_) for state_key, state_ in state.items()}
        action_mask = {state_key: torch.Tensor(action_mask_) for state_key, action_mask_ in action_mask.items()}
        if all([p.is_cuda for p in self.parameters()]):
            state = {state_key: state_.cuda() for state_key, state_ in state.items()}
            action_mask = {state_key: action_mask_.cuda() for state_key, action_mask_ in action_mask.items()}

        state_emb = {}
        entity_emb = {}
        for state_key in self.state_order_config:
            encoder = getattr(self, "_".join([state_key, "encoder"]))
            encoded = encoder(state[state_key])
            if isinstance(encoder, globals()[self.agent_cls_config["entity_encoder"]["name"]]):
                state_emb[state_key] = encoded[1]
                entity_emb[state_key] = encoded[0]
            elif isinstance(encoder, globals()[self.agent_cls_config["polygon_entity_encoder"]["name"]]):
                state_emb[state_key] = encoded[1]
                entity_emb[state_key] = encoded[0]
            elif isinstance(encoder, globals()[self.agent_cls_config["linestring_entity_encoder"]["name"]]):
                state_emb[state_key] = encoded[1]
                entity_emb[state_key] = encoded[0]
            else:
                state_emb[state_key] = encoded
                
        state_emb_list = [v for k, v in sorted(state_emb.items())]
        core_emb = self.core(state_emb_list)

        value = self.value_predictor(core_emb)

        action_mode = {}
        action_logit = {}
        action_emb = {}
        action_emb["none"] = core_emb

        for action_key in self.action_order_config:
            rationale_keys = action_sampling_config[action_key].rationales
            rationales = [action_emb[rationale_key] for rationale_key in rationale_keys]
            action_mode_, action_logit_, _, action_emb[action_key] = getattr(self, "_".join([action_key, "head"])).mode(*rationales, mask=action_mask[action_key])
            action_mode[action_key], action_logit[action_key] = action_mode_.numpy(), action_logit_.numpy()
        
        return action_mode, action_logit
        
    def value(self, state):

        state = {state_key: torch.Tensor(state_) for state_key, state_ in state.items()}
        if all([p.is_cuda for p in self.parameters()]):
            state = {state_key: state_.cuda() for state_key, state_ in state.items()}

        state_emb = {}
        entity_emb = {}
        for state_key in self.state_order_config:
            encoder = getattr(self, "_".join([state_key, "encoder"]))
            encoded = encoder(state[state_key])
            if isinstance(encoder, globals()[self.agent_cls_config["entity_encoder"]["name"]]):
                state_emb[state_key] = encoded[1]
                entity_emb[state_key] = encoded[0]
            elif isinstance(encoder, globals()[self.agent_cls_config["polygon_entity_encoder"]["name"]]):
                state_emb[state_key] = encoded[1]
                entity_emb[state_key] = encoded[0]
            elif isinstance(encoder, globals()[self.agent_cls_config["linestring_entity_encoder"]["name"]]):
                state_emb[state_key] = encoded[1]
                entity_emb[state_key] = encoded[0]
            else:
                state_emb[state_key] = encoded
                
        state_emb_list = [v for k, v in sorted(state_emb.items())]
        core_emb = self.core(state_emb_list)

        value = self.value_predictor(core_emb).detach().numpy()

        return value

    
    def train_ppo(
        self, batch_states, batch_actions, batch_returns, batch_values, batch_neglogps, batch_act_masks, batch_neglogp_masks, 
        max_grad_norm=0.5, cliprange=0.5, coef_el=0.0, coef_vl=0.5
    ):

        batch_returns = torch.Tensor(batch_returns)
        batch_values = torch.Tensor(batch_values)
        batch_neglogps = {action_key: torch.Tensor(batch_neglogp) for action_key, batch_neglogp in batch_neglogps.items()}
        batch_neglogp_masks = {action_key: torch.Tensor(batch_neglogp_mask) for action_key, batch_neglogp_mask in batch_neglogp_masks.items()}

        if all([p.is_cuda for p in self.parameters()]):
            batch_returns = batch_returns.cuda()
            batch_values = batch_values.cuda()
            batch_neglogps = {action_key: batch_neglogp.cuda() for action_key, batch_neglogp in batch_neglogps.items()}
            batch_neglogp_masks = {action_key: batch_neglogp_mask.cuda() for action_key, batch_neglogp_mask in batch_neglogp_masks.items()}
        
        batch_advs = batch_returns - batch_values
        batch_advs_std = (batch_advs - torch.mean(batch_advs)) / (torch.std(batch_advs) + 1e-8)
                
        actions, logits, neglogps, entropies, values = self.recollection(batch_states, batch_actions, batch_act_masks)
        
        value_clip = batch_values + torch.clamp(values - batch_values, -cliprange, cliprange)

        value_loss_noclip = (values - batch_returns) ** 2
        value_loss_clip = (value_clip - batch_returns) ** 2
        value_loss = 0.5 * torch.mean(torch.maximum(value_loss_noclip, value_loss_clip)) * coef_vl

        neglogps_diff = {action_key: (batch_neglogps[action_key] - neglogps[action_key]) for action_key in neglogps.keys()}
        neglogps_diff_masked = {action_key: (neglogps_diff[action_key] * (1 - batch_neglogp_masks[action_key])) for action_key in neglogps_diff.keys()}
        neglogps_diff_sum = torch.sum(torch.stack(list(neglogps_diff_masked.values())), axis=0)
        entropies_mean = torch.mean(torch.stack(list(entropies.values())))
        
        policy_loss_noclip = -batch_advs_std * torch.exp(neglogps_diff_sum)
        
        policy_loss_clip = -batch_advs_std * torch.clamp(torch.exp(neglogps_diff_sum), 1.0 - cliprange, 1.0 + cliprange)
        policy_loss = torch.mean(torch.maximum(policy_loss_noclip, policy_loss_clip))

        entropy_loss = -torch.mean(entropies_mean) * coef_el
        
        loss = policy_loss + entropy_loss + value_loss

        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

        grad = [x.grad.detach().cpu() if isinstance(x.grad, torch.Tensor) else None for x in self.parameters()]
        
        return grad, loss.detach().cpu().numpy(), policy_loss.detach().cpu().numpy(), entropy_loss.detach().cpu().numpy(), value_loss.cpu().detach().numpy(), entropies_mean.cpu().detach().numpy()



    def train_reinforce(
        self, batch_states, batch_actions, batch_returns, batch_values, batch_neglogps, batch_act_masks, batch_neglogp_masks, 
        max_grad_norm=0.5, cliprange=0.5, coef_el=0.0, coef_vl=0.5
    ):

        batch_returns = torch.Tensor(batch_returns)
        batch_values = torch.Tensor(batch_values)
        batch_neglogps = {action_key: torch.Tensor(batch_neglogp) for action_key, batch_neglogp in batch_neglogps.items()}
        batch_neglogp_masks = {action_key: torch.Tensor(batch_neglogp_mask) for action_key, batch_neglogp_mask in batch_neglogp_masks.items()}

        if all([p.is_cuda for p in self.parameters()]):
            batch_returns = batch_returns.cuda()
            batch_values = batch_values.cuda()
            batch_neglogps = {action_key: batch_neglogp.cuda() for action_key, batch_neglogp in batch_neglogps.items()}
            batch_neglogp_masks = {action_key: batch_neglogp_mask.cuda() for action_key, batch_neglogp_mask in batch_neglogp_masks.items()}
        
        batch_advs = batch_returns - batch_values
        batch_advs_std = (batch_advs - torch.mean(batch_advs)) / (torch.std(batch_advs) + 1e-8)
                
        actions, logits, neglogps, entropies, values = self.recollection(batch_states, batch_actions, batch_act_masks)
        
        value_loss = torch.mean((values - batch_returns) ** 2) * coef_vl
        
        neglogps_masked = {action_key: neglogps[action_key] * (1 - batch_neglogp_masks[action_key]) for action_key in neglogps.keys()}
        neglogps_sum = torch.sum(torch.stack(list(neglogps_masked.values())), axis=0)
        entropies_mean = torch.mean(torch.stack(list(entropies.values())))
        
        policy_loss = torch.mean(batch_advs_std * neglogps_sum)

        entropy_loss = -torch.mean(entropies_mean) * coef_el
        
        loss = policy_loss + entropy_loss + value_loss

        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

        grad = [x.grad.detach().cpu() if isinstance(x.grad, torch.Tensor) else None for x in self.parameters()]
        
        return grad, loss.detach().cpu().numpy(), policy_loss.detach().cpu().numpy(), entropy_loss.detach().cpu().numpy(), value_loss.cpu().detach().numpy(), entropies_mean.cpu().detach().numpy()