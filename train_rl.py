import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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
    for i, (data, indices, gt_y) in enumerate(train_loader):
        data, indices, gt_y = data.cuda(), indices.cuda(), gt_y.cuda()
        obs, indices, current_nodes = env.reset(data, indices, gt_y)
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
    for i, (data, indices, gt_y) in enumerate(train_loader):
        data, indices, gt_y = data.cuda(), indices.cuda(), gt_y.cuda()
        obs, indices, current_nodes = env.reset(data, indices, gt_y)
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

def train_q_learning(model, optimizer, env, num_episodes, gamma=0.99):
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = model.act(obs)
            next_obs, reward, done, _ = env.step(action)
            episode_reward += reward
            q_values = model(obs)
            next_q_values = model(next_obs)
            target_q_values = q_values.clone()
            target_q_values[action] = reward + gamma * next_q_values.max()
            loss = F.mse_loss(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            obs = next_obs
        print(f"Episode {i+1}: reward={episode_reward}, epsilon={model.exploration_rate}")
        