import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from . import nn_layers



class ScalarEncoder(nn.Module):
    
    def __init__(self, d_x, d_encoder=64, ds_hidden=[64]):
        super(ScalarEncoder, self).__init__()

        self.mlp = nn_layers.Mlp(d_x, d_encoder, ds_hidden)

    
    def forward(self, scalar):
        
        embedded_scalar = self.mlp(scalar)

        return embedded_scalar
        

class PolygonEncoder(nn.Module):

    def __init__(self, d_seq, d_encoder=64, ds_hidden=[]):
        super(PolygonEncoder, self).__init__()

        self.conv1d_0 = nn.Conv1d(2, 4, 5, padding=2, padding_mode="circular")
        self.conv1d_1 = nn.Conv1d(4, 8, 5, padding=2, padding_mode="circular")
        self.conv1d_2 = nn.Conv1d(8, 4, 5, padding=2, padding_mode="circular")
        self.conv1d_3 = nn.Conv1d(4, 1, 5, padding=2, padding_mode="circular")
        self.mlp = nn_layers.Mlp(d_seq, d_encoder, ds_hidden)
        self.flatten = nn.Flatten()

    
    def forward(self, polygon):
        
        net = torch.transpose(polygon, 1, 2)
        net = self.conv1d_0(net)
        net = self.conv1d_1(net)
        net = self.conv1d_2(net)
        net = self.conv1d_3(net)
        net = self.flatten(net)
        embedded_polygon = self.mlp(net)

        return embedded_polygon

class LinestringEncoder(nn.Module):

    def __init__(self, d_seq, d_encoder=64, ds_hidden=[]):
        super(LinestringEncoder, self).__init__()

        self.conv1d_0 = nn.Conv1d(2, 4, 5, padding=2)
        self.conv1d_1 = nn.Conv1d(4, 8, 5, padding=2)
        self.conv1d_2 = nn.Conv1d(8, 4, 5, padding=2)
        self.conv1d_3 = nn.Conv1d(4, 1, 5, padding=2)
        self.mlp = nn_layers.Mlp(d_seq, d_encoder, ds_hidden)
        self.flatten = nn.Flatten()

    
    def forward(self, polygon):
        
        net = torch.transpose(polygon, 1, 2)
        net = self.conv1d_0(net)
        net = self.conv1d_1(net)
        net = self.conv1d_2(net)
        net = self.conv1d_3(net)
        net = self.flatten(net)
        embedded_linestring = self.mlp(net)

        return embedded_linestring


class EntityEncoder(nn.Module):

    def __init__(self, d_x, d_encoder=64, num_heads=2, d_ff=256, num_layers=3, rate=0.1):
        super(EntityEncoder, self).__init__()

        self.tfe = nn_layers.TransformerEncoder(d_x, d_encoder, num_heads, d_ff, num_layers, rate)
        self.relu1 = nn.ReLU()
        self.dense1 = nn.Linear(d_encoder, d_encoder)
        self.relu2 = nn.ReLU()
        self.dense2 = nn.Linear(d_encoder, d_encoder)
        self.relu3 = nn.ReLU()

    
    def forward(self, entities, mask=None):

        transformer_output = self.tfe(entities, mask)  # (batch_size, input_seq_len, d_model)
        entity_embeddings = self.relu1(transformer_output)
        entity_embeddings = self.dense1(entity_embeddings)
        entity_embeddings = self.relu2(entity_embeddings)
        
        embedded_entity = torch.mean(entity_embeddings, -2)
        embedded_entity = self.dense2(embedded_entity)
        embedded_entity = self.relu3(embedded_entity)

        return entity_embeddings, embedded_entity

class PolygonEntityEncoder(nn.Module):

    def __init__(self, d_seq, d_encoder=64, num_heads=2, d_ff=256, num_layers=3, rate=0.1):
        super(PolygonEntityEncoder, self).__init__()

        self.pe = PolygonEncoder(d_seq, d_encoder)
        self.tfe = nn_layers.TransformerEncoder(d_encoder, d_encoder, num_heads, d_ff, num_layers, rate)
        self.relu1 = nn.ReLU()
        self.dense1 = nn.Linear(d_encoder, d_encoder)
        self.relu2 = nn.ReLU()
        self.dense2 = nn.Linear(d_encoder, d_encoder)
        self.relu3 = nn.ReLU()

    
    def forward(self, polygons, mask=None):

        batch_size = polygons.shape[0]
        d_entity = polygons.shape[1]
        flattened_polygons = torch.flatten(polygons, 0, 1)
        flattened_entities = self.pe(flattened_polygons)
        entities = torch.reshape(flattened_entities, (batch_size, d_entity, -1))

        transformer_output = self.tfe(entities, mask)  # (batch_size, input_seq_len, d_model)
        entity_embeddings = self.relu1(transformer_output)
        entity_embeddings = self.dense1(entity_embeddings)
        entity_embeddings = self.relu2(entity_embeddings)
        
        embedded_entity = torch.mean(entity_embeddings, -2)
        embedded_entity = self.dense2(embedded_entity)
        embedded_entity = self.relu3(embedded_entity)

        return entity_embeddings, embedded_entity    


class LinestringEntityEncoder(nn.Module):

    def __init__(self, d_seq, d_encoder=64, num_heads=2, d_ff=256, num_layers=3, rate=0.1):
        super(LinestringEntityEncoder, self).__init__()

        self.le = LinestringEncoder(d_seq, d_encoder)
        self.tfe = nn_layers.TransformerEncoder(d_encoder, d_encoder, num_heads, d_ff, num_layers, rate)
        self.relu1 = nn.ReLU()
        self.dense1 = nn.Linear(d_encoder, d_encoder)
        self.relu2 = nn.ReLU()
        self.dense2 = nn.Linear(d_encoder, d_encoder)
        self.relu3 = nn.ReLU()

    
    def forward(self, linestrings, mask=None):

        batch_size = linestrings.shape[0]
        d_entity = linestrings.shape[1]
        flattened_linestrings = torch.flatten(linestrings, 0, 1)
        flattened_entities = self.le(flattened_linestrings)
        entities = torch.reshape(flattened_entities, (batch_size, d_entity, -1))

        transformer_output = self.tfe(entities, mask)  # (batch_size, input_seq_len, d_model)
        entity_embeddings = self.relu1(transformer_output)
        entity_embeddings = self.dense1(entity_embeddings)
        entity_embeddings = self.relu2(entity_embeddings)
        
        embedded_entity = torch.mean(entity_embeddings, -2)
        embedded_entity = self.dense2(embedded_entity)
        embedded_entity = self.relu3(embedded_entity)

        return entity_embeddings, embedded_entity    

class SpatialEncoder(nn.Module):
    def __init__(
        self,
        cv2d_filters=[8, 16, 32, 32],
        cv2d_kernels=[1, 4, 4, 4],
        cv2d_strides=[1, 2, 2, 2],
        res_cv2d_filters=[32, 32, 32, 32],
        res_cv2d_kernels=[3, 3, 3, 3],
        d_core=128):

        super(SpatialEncoder, self).__init__()
        self.cnn = nn_layers.Cnn(
            filters=cv2d_filters,
            kernels=cv2d_kernels,
            strides=cv2d_strides)

        self.res_cnn = nn_layers.ResCnn(
            filters=res_cv2d_filters,
            kernels=res_cv2d_kernels)

        self.gap = nn.AvgPool2d(res_cv2d_kernels[-1], padding=0)
        self.dense = nn.Linear(res_cv2d_filters[-1], d_core)
        self.relu = nn.ReLU()
    
    def forward(self, spatial_feature):

        net = self.cnn(spatial_feature)
        net = self.res_cnn(net)
        net = self.gap(net)
        net = self.dense(net)
        net = self.relu(net)
        
        return net
        

class Core(nn.Module):
    
    def __init__(self, d_x, d_head=96, ds_hidden=[256, 128]):
        super(Core, self).__init__()
        self.mlp = nn_layers.Mlp(d_x, d_head, ds_hidden)

    
    def forward(self, inputs):
        embedded_states = inputs
        net = torch.cat(embedded_states, dim=-1)
        net = self.mlp(net)
        return net


class CategoricalHead(nn.Module):
    def __init__(self, d_out, d_head=96, ds_hidden=[32, 16, 8], ds_ar_hidden=[16, 32]):
        super(CategoricalHead, self).__init__()
        self.d_out = d_out
        self.mlp = nn_layers.Mlp(d_head, d_out, ds_hidden)
        self.mlp_ar = nn_layers.Mlp(d_out, d_head, ds_ar_hidden)

    
    def forward(self, autoregressive_embedding, action=None, mask=None):

        categorical_logits = self.mlp(autoregressive_embedding)
        if mask is not None:
            categorical_logits += (mask[:categorical_logits.shape[-2],:categorical_logits.shape[-1]] * -1e9)
        categorical_dist = Categorical(logits=categorical_logits)
        if isinstance(action, type(None)):
            categorical_sample = categorical_dist.sample()
        else:
            categorical_sample = action
        categorical_sample_neglogp = -categorical_dist.log_prob(categorical_sample)
        categorical_entropy = categorical_dist.entropy()
        one_hot_embedding = F.one_hot(categorical_sample.type(torch.int64), num_classes=self.d_out).type(torch.FloatTensor)
        if all([p.is_cuda for p in self.parameters()]):
            one_hot_embedding = one_hot_embedding.cuda()
        autoregressive_embedding = autoregressive_embedding + self.mlp_ar(one_hot_embedding)

        return categorical_sample, categorical_logits, categorical_sample_neglogp, categorical_entropy, autoregressive_embedding

    def mode(self, inputs, mask=None):

        autoregressive_embedding = inputs

        categorical_logits = self.mlp(autoregressive_embedding)
        if mask is not None:
            categorical_logits += (mask[:categorical_logits.shape[-2],:categorical_logits.shape[-1]] * -1e9)
        mode = torch.argmax(categorical_logits, -1)
        one_hot_embedding = F.one_hot(mode, num_classes=self.d_out).type(torch.FloatTensor)
        if all([p.is_cuda for p in self.parameters()]):
            one_hot_embedding = one_hot_embedding.cuda()
        autoregressive_embedding = autoregressive_embedding + self.mlp_ar(one_hot_embedding)

        return mode, categorical_logits, autoregressive_embedding


class EntityHead(nn.Module):

    def __init__(self, d_encoder=64, d_head=96):
        super(EntityHead, self).__init__()
        
        self.ptr_net = nn_layers.PointerNetwork(d_head, d_encoder, d_head)
        self.dense_ar = nn.Linear(d_encoder, d_head)

    
    def forward(self, autoregressive_embedding, entity_embeddings, action=None, mask=None):
        
        entity_logits = self.ptr_net(autoregressive_embedding, entity_embeddings, mask=None, temp=0.8)
        if mask is not None:
            entity_logits += (mask[:entity_logits.shape[-2],:entity_logits.shape[-1]] * -1e9)
        entity_dist = Categorical(logits=entity_logits)
        if isinstance(action, type(None)):
            entity_sample = entity_dist.sample()
        else:
            entity_sample = action
        entity_sample_neglogp = -entity_dist.log_prob(entity_sample)
        entity_entropy = entity_dist.entropy()
        gather_idx = torch.unsqueeze(torch.unsqueeze(entity_sample, -1), -1).type(torch.int64)
        gather_idx = gather_idx.expand(entity_embeddings.size(0), 1, entity_embeddings.size(-1))
        sampled_entity_embeddings = torch.squeeze(torch.gather(entity_embeddings, 1, gather_idx), 1)
        autoregressive_embedding = autoregressive_embedding + self.dense_ar(sampled_entity_embeddings)

        return entity_sample, entity_logits, entity_sample_neglogp, entity_entropy, autoregressive_embedding

    def mode(self, autoregressive_embedding, entity_embeddings, mask=None):

        entity_logits = self.ptr_net((autoregressive_embedding, entity_embeddings), mask=mask, temp=0.8)
        if mask is not None:
            entity_logits += (mask[:entity_logits.shape[-2],:entity_logits.shape[-1]] * -1e9)
        mode = torch.argmax(entity_logits, -1)

        gather_idx = torch.unsqueeze(torch.unsqueeze(mode, -1), -1).type(torch.int64)
        gather_idx = gather_idx.expand(entity_embeddings.size(0), 1, entity_embeddings.size(-1))
        mode_entity_embeddings = torch.squeeze(torch.gather(entity_embeddings, 1, gather_idx), 1)
        autoregressive_embedding += self.dense_ar(mode_entity_embeddings)

        return mode, entity_logits, autoregressive_embedding


class ScalarPredictor(nn.Module):

    def __init__(self, d_head, d_out=1, ds_hidden=[16, 2]):
        super(ScalarPredictor, self).__init__()

        self.mlp = nn_layers.MlpNum(d_head, d_out, ds_hidden)
        
    
    def forward(self, inputs):
        autoregressive_embedding = inputs

        net = self.mlp(autoregressive_embedding)
        scalar = torch.squeeze(net, axis=-1)

        return scalar