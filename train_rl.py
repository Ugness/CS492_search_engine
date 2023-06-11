import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
from copy import deepcopy

def train_policy_gradient_iterative(
        act_model,
        act_opt,
        env_opt,
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
        act_opt.zero_grad()
        loss = policy_loss
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
            writer.add_scalar('Loss/policy_loss', policy_loss.item(), i)
            writer.add_scalar('Loss/ce_loss', ce_loss.item(), i)
            writer.add_scalar('Reward/episode_reward', episode_reward.mean().item(), i)
            writer.add_scalar('Accuracy/episode_accuracy', acc.item(), i)
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
        eps=0.9,
        writer=None,
        reward_type='acc',
        TAU=0.005):
    
    pbar = tqdm(train_loader)
    target_model = deepcopy(model)
    for i, (data, length, gt_y) in enumerate(train_loader):
        data, length, gt_y = data.cuda(), length.cuda(), gt_y.cuda()
        obs, indices, current_nodes = env.reset(data, length, gt_y)
        B = len(obs)
        done = torch.zeros(len(data), dtype=torch.bool, device=obs.device, requires_grad=False)
        episode_reward = torch.zeros(len(data), device=data.device, requires_grad=False)
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

            target_q_values = reward.view(-1, 1, 1) + gamma * next_q_values.max(-1)[0].unsqueeze(-1)
            # target_q_values[action] = reward + gamma * next_q_values.max(-1)[0]
            loss = F.mse_loss(q_values, target_q_values, reduction='none') * (1.-terminates.float().view(-1, 1))
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            obs = next_obs.detach().clone()
            indices = indices.detach().clone()
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
            writer.add_scalar('Loss/q_loss', loss.item(), i)
            writer.add_scalar('Loss/ce_loss', ce_loss.item(), i)
            writer.add_scalar('Reward/episode_reward', episode_reward.mean().item(), i)
            writer.add_scalar('Accuracy/episode_accuracy', acc.item(), i)
        pbar.set_description(f"Episode {i+1}: reward={episode_reward.mean().item():.4f}, q_loss={loss.item():.4f}, accuracy={acc.item():.4f}")
        pbar.update()

def soft_target_update(net, target_net, tau=0.005):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
      target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)