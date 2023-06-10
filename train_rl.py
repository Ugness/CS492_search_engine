# import modules if needed.
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data.dataloader import get_tokenizer, SearchData
from model.model import GPT

if __name__ == '__main__':

    # define dataset
    tokenizer = get_tokenizer()
    dataset = SearchData('data/train.pkl', tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # define lookup model
    lookup_model = GPT(tokenizer, len(tokenizer), max_len=100)
    lookup_model = lookup_model.cuda()

    # define QA model
    qa_model = GPT(tokenizer, len(tokenizer), max_len=100)
    qa_model = qa_model.cuda()

    # define optimizer
    optimizer = Adam(list(lookup_model.parameters()) + list(qa_model.parameters()), lr=1e-3)

    # define environment and models for policy gradient
    def generate_key_string(lookup_model, data):
        lookup_model.eval()
        input_ids = tokenizer.encode(data, add_special_tokens=False, return_tensors='pt').cuda()
        output_ids = lookup_model.generate(input_ids, max_length=100, do_sample=True)
        key_string = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return key_string

    def compute_reward(qa_model, input_ids, gt_output_ids):
        qa_model.eval()
        output_ids = qa_model.generate(input_ids, max_length=100, do_sample=False)
        loss = torch.nn.functional.cross_entropy(output_ids.view(-1, output_ids.size(-1)), gt_output_ids.view(-1))
        return -loss.item()

    # define training loop
    for epoch in range(100):
        total_reward = 0
        for i, (data, gt_output) in enumerate(dataloader):
            data = data.cuda()
            gt_output = gt_output.cuda()
            optimizer.zero_grad()
            key_string = generate_key_string(lookup_model, data)
            lookup_result = dataset.lookup(key_string)
            input_ids = torch.cat((data, lookup_result), dim=1)
            reward = compute_reward(qa_model, input_ids, gt_output)
            total_reward += reward
            log_prob = qa_model.get_log_prob(input_ids)
            loss = -log_prob * reward
            loss.backward()
            optimizer.step()
        print(f'Epoch: {epoch}, Total Reward: {total_reward}')

    # define validation loop
    qa_model.eval()
    with torch.no_grad():
        for i, (data, gt_output) in enumerate(dataloader):
            data = data.cuda()
            gt_output = gt_output.cuda()
            key_string = generate_key_string(lookup_model, data)
            lookup_result = dataset.lookup(key_string)
            input_ids = torch.cat((data, lookup_result), dim=1)
            output_ids = qa_model.generate(input_ids, max_length=100, do_sample=False)
            loss = torch.nn.functional.cross_entropy(output_ids.view(-1, output_ids.size(-1)), gt_output.view(-1))
            print(f'Validation Loss: {loss.item()}')