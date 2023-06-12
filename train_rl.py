import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pickle

from tqdm import tqdm
import random
import numpy as np

from copy import deepcopy

from argparse import ArgumentParser
import os

from model.model import GPT
from data.dataloader import get_tokenizer, SearchData, SupervisionData2
from model.env import Env

def train_policy_gradient_iterative(
        act_model,
        act_opt,
        env_opt,
        train_loader,
        env,
        gamma=0.99,
        temperature=0.1,
        top_k=None,
        top_p=None,
        writer=None,
        reward_type='acc',
        eps=0.9,
        TAU=0.,
        n_epoch=0,):
    pbar = tqdm(train_loader)
    for i, (data, length, gt_y) in enumerate(train_loader):
        data, length, gt_y = data.cuda(), length.cuda(), gt_y.cuda()
        obs, indices, current_nodes = env.reset(data, length, gt_y)
        B = len(obs)
        done = torch.zeros(len(data), dtype=torch.bool, device=obs.device, requires_grad=False)
        episode_reward = torch.zeros(len(data), device=data.device, requires_grad=False)
        log_probs = []
        rewards = []
        terminates = []
        entropys = []
        while not done.all():
            action_logits, action_probs, log_prob, action = act_model.predict(obs, indices, current_nodes, env.trie,
                                                                              1.0, top_k, top_p)
            entropy = -(action_probs * torch.log(action_probs+1e-6))[action_probs>0].sum()
            entropys.append(entropy / len(obs))
            terminates.append(done)
            obs = obs.detach().clone()
            indices = indices.detach().clone()
            with torch.no_grad():
                obs, indices, current_nodes, reward, done, correct = env.step(obs, indices, current_nodes, action, done, reward_type)
            episode_reward += reward
            log_probs.append(log_prob)
            rewards.append(reward.detach())
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.stack(returns, -1) # (B, T)
        log_probs = torch.stack(log_probs, -1)
        terminates = torch.stack(terminates, -1).float()
        advantages = returns - returns.mean(-1, keepdim=True)
        # sum over timesteps, mean over episodes
        policy_loss = -(log_probs * advantages.detach() * (1.-terminates)).mean(0).sum(0)
        entropy_loss = -torch.stack(entropys, -1).mean(0).sum(0) # let the model to maximize the entropy (exploration)
        act_opt.zero_grad()
        loss = policy_loss + temperature * entropy_loss
        loss.backward()
        act_opt.step()

        # update env
        obs, indices, current_nodes, reward, done, correct = env.step(obs, indices, current_nodes, action, torch.zeros_like(done), 'ce')
        ce_loss = -reward.mean()
        acc = correct.float().mean()
        env_opt.zero_grad()
        ce_loss.backward()
        env_opt.step()

        if writer is not None:
            writer.add_scalar('Loss/policy_loss', policy_loss.item(), i + n_epoch * len(train_loader))
            writer.add_scalar('Loss/entropy_loss', entropy_loss.item(), i + n_epoch * len(train_loader))
            writer.add_scalar('Loss/ce_loss', ce_loss.item(), i + n_epoch * len(train_loader))
            writer.add_scalar('Reward/episode_reward', episode_reward.mean().item(), i + n_epoch * len(train_loader))
            writer.add_scalar('Accuracy/episode_accuracy', acc.item(), i + n_epoch * len(train_loader))
        pbar.set_description(f"Episode {i+1}: reward={episode_reward.mean().item():.4f}, policy_loss={policy_loss.item():.4f}, accuracy={acc.item():.4f}")
        pbar.update()

def train_policy_gradient(
        act_model,
        act_opt,
        train_loader,
        env,
        gamma=1.0,
        temperature=1.0,
        top_k=None,
        top_p=None,
        writer=None,
        reward_type='acc',):

    pbar = tqdm(train_loader)
    for i, (data, length, gt_y) in enumerate(train_loader):
        data, length, gt_y = data.cuda(), length.cuda(), gt_y.cuda()
        obs, indices, current_nodes = env.reset(data, length, gt_y)
        B = len(obs)
        done = torch.zeros(len(data), dtype=torch.bool, device=obs.device, requires_grad=False)
        episode_reward = torch.zeros(len(data), device=data.device, requires_grad=False)
        log_probs = []
        rewards = []
        terminates = []
        while not done.all():
            action_logits, action_probs, log_prob, action = act_model.predict(obs, indices, current_nodes, env.trie,
                                                                              temperature, top_k, top_p)
            terminates.append(done) # to compute loss properly.
            obs = obs.detach().clone()
            indices = indices.detach().clone()
            with torch.no_grad():
                obs, indices, current_nodes, reward, done, correct = env.step(obs, indices, current_nodes, action, done, reward_type)
            episode_reward += reward
            log_probs.append(log_prob)
            rewards.append(reward.detach())
        returns = []
        acc = correct.float().mean()
        terminates = torch.stack(terminates, -1).float()
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.stack(returns, -1) # (B, T)
        log_probs = torch.stack(log_probs, -1)
        advantages = returns - returns.mean(-1, keepdim=True)
        # sum over timesteps, mean over episodes
        policy_loss = -(log_probs * advantages.detach() * (1.-terminates)).mean(0).sum(0)
        act_opt.zero_grad()
        loss = policy_loss
        loss.backward()
        act_opt.step()
        if writer is not None:
            writer.add_scalar('Loss/policy_loss', policy_loss.item(), i)
            writer.add_scalar('Reward/episode_reward', episode_reward.mean().item(), i)
            writer.add_scalar('Accuracy/episode_accuracy', acc.item(), i)
        pbar.set_description(f"Episode {i+1}: reward={episode_reward.mean().item():.4f}, policy_loss={policy_loss.item():.4f}, accuracy={acc.item():.4f}")
        pbar.update()

def train_q_learning_iterative(
        model,
        optimizer,
        env_opt,
        train_loader,
        env,
        gamma=0.99,
        temperature=1.0,
        top_k=None,
        top_p=None,
        writer=None,
        reward_type='acc',
        eps=0.9,
        TAU=0.005,
        n_epoch=0):
    
    pbar = tqdm(train_loader)
    target_model = deepcopy(model)
    actor.train()
    env.train()
    for i, (data, length, gt_y) in enumerate(train_loader):
        data, length, gt_y = data.cuda(), length.cuda(), gt_y.cuda()
        obs, indices, current_nodes = env.reset(data, length, gt_y)
        B = len(obs)
        done = torch.zeros(len(data), dtype=torch.bool, device=obs.device, requires_grad=False)
        episode_reward = torch.zeros(len(data), device=data.device, requires_grad=False)
        total_loss = 0.
        while not done.all():
            terminates = done.clone()
            # uniform constrained_decoding
            sample = random.random() < eps
            if sample:
                action = model.random_act(current_nodes, env.trie)
            else:
                with torch.no_grad():
                    _, _, _, action = model.predict(obs, indices, current_nodes, env.trie, argmax=True)
            q_values = model.predict(obs.detach().clone(), indices.detach().clone(), current_nodes, env.trie)[0]
            with torch.no_grad():
                next_obs, indices, current_nodes, reward, done, correct = env.step(obs.detach().clone(), indices.detach().clone(), current_nodes, action, done, reward_type)
            episode_reward += reward
            with torch.no_grad():
                next_q_values = target_model.predict(next_obs.detach().clone(), indices.detach().clone(), current_nodes, env.trie)[0]

            q_values = torch.gather(q_values, -1, action.view(-1, 1, 1)).squeeze(-1)
            target_q_values = torch.zeros_like(q_values)

            target_q_values = reward.view(-1, 1) + gamma * next_q_values.max(-1)[0]
            loss = F.mse_loss(q_values, target_q_values, reduction='none') * (1.-terminates.float().view(-1, 1))
            total_loss += loss.mean()
            obs = next_obs.detach().clone()
            indices = indices.detach().clone()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        soft_target_update(model, target_model, TAU)
        
        # anothe episode with greedy_policy for training the reader.
        obs, indices, current_nodes = env.reset(data, length, gt_y)
        B = len(obs)
        done = torch.zeros(len(data), dtype=torch.bool, device=obs.device, requires_grad=False)
        with torch.no_grad():
            while not done.all():
                action_logits, action_probs, log_prob, action = model.predict(obs, indices, current_nodes, env.trie, argmax=True)
                obs = obs.detach().clone()
                indices = indices.detach().clone()
                obs, indices, current_nodes, reward, done, correct = env.step(obs, indices, current_nodes, action, done, reward_type)
        # update reader
        obs, indices, current_nodes, reward, done, correct = env.step(obs, indices, current_nodes, action, torch.zeros_like(done), 'ce')
        ce_loss = -reward.mean()
        acc = correct.float().mean()
        env_opt.zero_grad()
        ce_loss.backward()
        env_opt.step()

        if writer is not None:
            writer.add_scalar('Loss/q_loss', total_loss.item(), i + n_epoch * len(train_loader))
            writer.add_scalar('Loss/ce_loss', ce_loss.item(), i + n_epoch * len(train_loader))
            writer.add_scalar('Reward/episode_reward', episode_reward.mean().item(), i + n_epoch * len(train_loader))
            writer.add_scalar('Accuracy/episode_accuracy', acc.item(), i + n_epoch * len(train_loader))
        pbar.set_description(f"Epoch {n_epoch}, Episode {i+1}: reward={episode_reward.mean().item():.4f}, q_loss={total_loss.item():.4f}, accuracy={acc.item():.4f}")
        pbar.update()

def train_supervised_iterative(
        act_model,
        act_opt,
        env_opt,
        train_loader,
        env,
        gamma=0.99,
        temperature=1.0,
        top_k=None,
        top_p=None,
        writer=None,
        reward_type='acc',
        eps=0.9,
        TAU=0.,
        n_epoch=0,):
    pbar = tqdm(train_loader)
    # loader: []
    actor.train()
    env.train()
    for i, (data, target, mask, length, gt_y) in enumerate(train_loader):
        data, mask, target, length, gt_y = data.cuda(), mask.cuda(), target.cuda(), length.cuda(), gt_y.cuda()
        logits, loss = act_model(data, masks=mask, targets=target)
        loss = loss.mean()
        act_opt.zero_grad()
        loss.backward()
        act_opt.step()

        # anothe episode with greedy_policy for training the reader.
        obs, indices, current_nodes = env.reset(data, length, gt_y)
        B = len(obs)
        done = torch.zeros(len(data), dtype=torch.bool, device=obs.device, requires_grad=False)
        with torch.no_grad():
            while not done.all():
                action_logits, action_probs, log_prob, action = act_model.predict(obs, indices, current_nodes, env.trie, argmax=True)
                obs = obs.detach().clone()
                indices = indices.detach().clone()
                obs, indices, current_nodes, reward, done, correct = env.step(obs, indices, current_nodes, action, done, reward_type)
        # update reader
        obs, indices, current_nodes, reward, done, correct = env.step(obs, indices, current_nodes, action, torch.zeros_like(done), 'ce')
        ce_loss = -reward.mean()
        acc = correct.float().mean()
        env_opt.zero_grad()
        ce_loss.backward()
        env_opt.step()

        if writer is not None:
            writer.add_scalar('Loss/supervision_loss', loss.item(), i + n_epoch * len(train_loader))
            writer.add_scalar('Loss/ce_loss', ce_loss.item(), i + n_epoch * len(train_loader))
            writer.add_scalar('Accuracy/episode_accuracy', acc.item(), i + n_epoch * len(train_loader))
        pbar.set_description(f"Epoch {n_epoch}, Episode {i+1}: loss={loss.item():.4f}, accuracy={acc.item():.4f}")
        pbar.update()

def evaluate(actor, env, loader, reward_type='acc', num_epoch=0, writer=None):
    actor.eval()
    env.eval()
    corrects = 0
    total = 0
    with torch.no_grad():
        for i, (data, length, gt_y) in enumerate(loader):
            data, length, gt_y = data.cuda(), length.cuda(), gt_y.cuda()
            obs, indices, current_nodes = env.reset(data, length, gt_y)
            B = len(obs)
            done = torch.zeros(len(data), dtype=torch.bool, device=obs.device, requires_grad=False)
            while not done.all():
                action_logits, action_probs, log_prob, action = actor.predict(obs, indices, current_nodes, env.trie, argmax=True)
                obs = obs.detach().clone()
                indices = indices.detach().clone()
                obs, indices, current_nodes, reward, done, correct = env.step(obs, indices, current_nodes, action, done, reward_type)
            # update reader
            obs, indices, current_nodes, reward, done, correct = env.step(obs, indices, current_nodes, action, torch.zeros_like(done), 'ce')
            ce_loss = -reward.mean()
            corrects += correct.float().sum().item()
            total += B
        acc = corrects / total
    
    if writer is not None:
        writer.add_scalar('Accuracy/eval_accuracy', acc, num_epoch)
    print(f"Epoch {num_epoch}, Eval_Accuracy: {acc:.4f}")
    return acc



def soft_target_update(net, target_net, tau=0.005):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
      target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

func_dict = {
    'q_learning': train_q_learning_iterative,
    'policy_gradient': train_policy_gradient_iterative,
    # 'ppo': train_ppo_iterative,
    'supervision': train_supervised_iterative,
}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--algorithm', choices=['q_learning', 'policy_gradient', 'supervision', 'ppo'], default='q_learning')
    parser.add_argument('--reward_type', choices=['acc', 'ce'], default='acc')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--max_epochs', type=int, default=10000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr2', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--eps', type=float, default=0.9)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_dir', type=str, default='runs/rl')
    parser.add_argument('--model_dir', type=str, default='checkpoints/rl')
    parser.add_argument('--exp', default='')

    args = parser.parse_args()
    args.model_dir = os.path.join(args.model_dir, args.exp)
    args.log_dir = os.path.join(args.log_dir, args.exp)

    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    tokenizer = get_tokenizer()
    train_dataset = SearchData('data/train.pkl', tokenizer) if not args.algorithm == 'supervision' else SupervisionData2('data/train.pkl', tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataset = SearchData('data/val.pkl', tokenizer) if not args.algorithm == 'supervision' else SupervisionData2('data/val.pkl', tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    actor = GPT(tokenizer, len(tokenizer), block_size=100).cuda()
    reader = GPT(tokenizer, len(tokenizer), block_size=100).cuda()

    # load database
    with open('data/dataset.pkl', 'rb') as f:
        database = pickle.load(f)
        _, database = database
    # load trie
    with open('data/trie.pkl', 'rb') as f:
        trie = pickle.load(f)

    env = Env(reader, database, trie, tokenizer).cuda()
    act_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    env_opt = torch.optim.Adam(env.parameters(), lr=args.lr2)
    train_function = func_dict[args.algorithm]
    for i in range(args.max_epochs):
        train_function(actor, act_opt, env_opt, train_loader, env, args.gamma, writer=writer, reward_type=args.reward_type, TAU=args.tau, eps=args.eps, n_epoch=i, temperature=args.temperature)
        torch.save(actor.state_dict(), os.path.join(args.model_dir, f'actor_{i}.pt'))
        torch.save(env.state_dict(), os.path.join(args.model_dir, f'env_{i}.pt'))
        evaluate(actor, env, val_loader, args.reward_type, i, writer=SummaryWriter(args.log_dir))
