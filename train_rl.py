import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train_policy_gradient(
        act_model,
        act_opt,
        train_loader,
        env,
        gamma=0.99,
        temperature=1.0,
        top_k=None,
        top_p=None,
        writer=None):

    pbar = tqdm(train_loader)
    for i, (data, indices, gt_y) in enumerate(train_loader):
        data, indices, gt_y = data.cuda(), indices.cuda(), gt_y.cuda()
        obs, indices, current_nodes = env.reset(data, indices, gt_y)
        B = len(obs)
        done = torch.zeros(len(data), dtype=torch.bool, device=obs.device, requires_grad=False)
        episode_reward = torch.zeros(len(data), device=data.device, requires_grad=False)
        log_probs = []
        rewards = []
        while not done.all():
            action_logits, action_probs, log_prob, action = act_model.predict(obs, indices, current_nodes, env.trie,
                                                                              temperature, top_k, top_p)
            obs, indices, current_nodes, reward, done, _ = env.step(obs, indices, current_nodes, action, done)
            episode_reward += reward.detach()
            log_probs.append(log_prob)
            rewards.append(reward.detach())
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.stack(returns, -1) # (B, T)
        log_probs = torch.stack(log_probs, -1)
        advantages = returns - returns.mean(-1, keepdim=True)
        policy_loss = -(log_probs * advantages.detach()).mean()
        act_opt.zero_grad()
        loss = policy_loss
        loss.backward()
        act_opt.step()
        if writer is not None:
            writer.add_scalar('Loss/policy_loss', policy_loss.item(), i)
            writer.add_scalar('Reward/episode_reward', episode_reward, i)

        pbar.update()
        pbar.set_description(f"Episode {i+1}: reward={episode_reward}, policy_loss={policy_loss.item()}")
        # print(f"Episode {i+1}: reward={episode_reward}, policy_loss={policy_loss.item()}")

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