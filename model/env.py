import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

class Env(nn.Module):
    def __init__(self, reader, database, trie, tokenizer, max_len=100, batch_size=32, num_workers=4):
        super().__init__()
        self.reader = reader
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.database = database
        self.trie = trie
    
    def reset(self, data, indices, gt_y):
        current_nodes = [self.trie.root for _ in range(len(data))]
        self.answer = torch.ones_like(data) * self.tokenizer.pad_token_id
        self.a_idx = 0
        self.orig_length = indices.clone()
        self.gt_y = gt_y
        return data, indices, current_nodes
    
    def compute_reward(self, obs, already_done, requires_grad=False):
        # TODO:
        # check whether done or not.
        # if done, compute reward.
        #   re-organize 
        grad_control = torch.enable_grad if requires_grad else torch.no_grad

        with grad_control():
            done = self.answer[:, -1] == self.tokenizer.eos_token_id
            B = self.answer.shape[0]
            reward = torch.zeros(B, device=obs.device)
            if done.any():
                # compute reward
                # decode answer
                ids = self.tokenizer.batch_decode(self.answer, skip_special_tokens=True)
                ids = [id.replace(' ', '') for id in ids]

                # parse if possible
                data = [[0 for _ in range(10)] if not id in self.database else self.database[id] for id in ids]
                data = [''.join(map(str, d)) for d in data]
                data = self.tokenizer(data, return_tensors='pt').input_ids

                # construct input of reader
                # format: [full_expression, <pad>, x, <pad>, y0, y1, ..., y9, <eos>]
                inputs = torch.zeros_like(obs)
                targets = torch.zeros_like(obs)
                masks = torch.zeros_like(obs).bool()
                for b in range(B):
                    inputs[b, :self.orig_length[b]] = obs[b, :self.orig_length[b]]
                    # convert eos to pad
                    assert inputs[b, self.orig_length[b]-1] == self.tokenizer.eos_token_id
                    inputs[b, self.orig_length[b]-1] = self.tokenizer.pad_token_id
                    inputs[b, self.orig_length[b]:self.orig_length[b]+10] = data[b]
                    inputs[b, self.orig_length[b]+10] = self.tokenizer.eos_token_id
                    masks[b, self.orig_length[b]+10] = True
                    targets[b, self.orig_length[b]+10] = self.gt_y[b]
                # compute reward
                with torch.no_grad():
                    reward = self.reader(inputs, masks=masks, targets=targets, reduce=False)[1]
                reward = -reward
                # re-organize reward
                reward = reward * done.float()
            reward[already_done] = 0.
            return reward, done
    
    @torch.no_grad()
    def step(self, obs, indices, current_nodes, action, already_done=False):
        # TODO:
        #   if action == eos and not already done: done = True, compute reward.
        #       decode obs, and parse database. construct input of reader.
        #       compute reward.
        #   else: concat action to obs. reward is 0.
        B = obs.shape[0]

        # update node
        current_nodes = [current_nodes[b].children[action[b].item()] for b in range(B)]
        # update obs
        obs = torch.scatter(obs, dim=1, index=indices.view(-1, 1), src=action.view(-1, 1))
        # update indices
        indices[action != self.tokenizer.eos_token_id] = indices[action != self.tokenizer.eos_token_id] + 1
        # update answer
        self.answer[:, self.a_idx] = action
        self.a_idx += 1

        # compute_reward
        reward, done = self.compute_reward(obs, already_done)

        return obs, indices, current_nodes, reward, done, None