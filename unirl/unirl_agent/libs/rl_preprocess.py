import numpy as np

def concatenate(target):
    result = np.concatenate(target)
    return result
    
def pad(target, d_state, value=0):
    pad_width = [(0, 0) for _ in target.shape]
    pad_width[0] = (0, d_state)
    result = np.pad(target, pad_width, mode='constant', constant_values=value)[:d_state]
    return result

def pad_cycle(target, d_state):
    result = np.concatenate([target for _ in range(d_state // len(target) + 1)])[:d_state]
    return result

def flatten_seq(target, d_seq, d_state):
    shape = list(target.shape)
    shape[0] = d_seq
    shape[1] = d_state
    result = np.zeros(shape)
    for i in range(d_seq):
        if i < len(target):
            result[i] = pad(target[i], d_state)
        else:
            continue
    result = np.reshape(result, [-1] + list(shape[2:]))
    return result

def flatten_seq_cycle(target, d_seq, d_state):
    shape = list(target.shape)
    shape[0] = d_seq
    shape[1] = d_state
    result = np.zeros(shape)
    for i in range(d_seq):
        if i < len(target):
            result[i] = pad_cycle(target[i], d_state)
        else:
            continue
    result = np.reshape(result, [-1] + list(shape[2:]))
    return result