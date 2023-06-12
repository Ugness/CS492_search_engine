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

from data.dataloader import get_tokenizer, SupervisionData
from model.model import GPT

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='gt')
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_only', default=None, type=str, help='path to checkpoint')
    args = parser.parse_args()

    args.save_dir = os.path.join(args.save_dir, 'baseline', f'{args.mode}_{args.lr}_{args.batch_size}')
    args.log_dir = os.path.join(args.log_dir, 'baseline', f'{args.mode}_{args.lr}_{args.batch_size}')

    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    tokenizer = get_tokenizer()
    train_dataset = SupervisionData('data/train.pkl', tokenizer, max_len=args.max_len, mode=args.mode)
    val_dataset = SupervisionData('data/val.pkl', tokenizer, max_len=args.max_len, mode=args.mode)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    writer = SummaryWriter(args.log_dir) if args.eval_only is None else None

    model = GPT(tokenizer, len(tokenizer), args.max_len)
    model = model.cuda()
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    os.makedirs(args.save_dir, exist_ok=True)
    if args.eval_only is not None:
        pbar = tqdm(total=len(val_dataloader))
        model.load_state_dict(torch.load(args.eval_only))
        with torch.no_grad():
            total_loss = 0
            total = 0
            correct = 0
            for i, (data, target, mask) in enumerate(val_dataloader):
                data = data.cuda().long()
                target = target.cuda().long()
                mask = mask.cuda()
                output, loss = model(data, targets=target, masks=mask)

                # compute accuracy
                total += len(data)
                correct += (output.argmax(dim=-1) == target)[mask].float().sum().item()
                acc = (output.argmax(dim=-1) == target)[mask].float().mean()
                # writer.add_scalar('Loss/val', loss.item())
                # writer.add_scalar('Accuracy/val', acc.item())

                total_loss += loss.item()
                pbar.update(1)
            print(f'Val Loss: {total_loss / len(val_dataloader)}')
            print(f'Val Acc: {correct / total}')
        exit(0)
    pbar = tqdm(total=len(train_dataloader))
    for epoch in range(args.epochs):
        for i, (data, target, mask) in enumerate(train_dataloader):
            data = data.cuda().long()
            target = target.cuda().long()
            mask = mask.cuda()
            optimizer.zero_grad()
            output, loss = model(data, targets=target, masks=mask)

            # compute accuracy
            acc = (output.argmax(dim=-1) == target)[mask].float().mean()

            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataloader) + i)
            writer.add_scalar('Acc/train', acc.item(), epoch * len(train_dataloader) + i)

            loss.backward()
            optimizer.step()
            pbar.update(1)
            pbar.set_description(f'Epoch: {epoch}, Iter: {i}, Loss: {loss.item()}, Acc: {acc.item()}')
        
        with torch.no_grad():
            total_loss = 0
            total = 0
            correct = 0
            for i, (data, target, mask) in enumerate(val_dataloader):
                data = data.cuda().long()
                target = target.cuda().long()
                mask = mask.cuda()
                output, loss = model(data, targets=target, masks=mask)

                # compute accuracy
                total += len(data)
                correct += (output.argmax(dim=-1) == target)[mask].float().sum().item()
                acc = (output.argmax(dim=-1) == target)[mask].float().mean()
                writer.add_scalar('Loss/val', loss.item(), epoch * len(val_dataloader) + i)
                writer.add_scalar('Acc/val', acc.item(), epoch * len(val_dataloader) + i)

                total_loss += loss.item()
            print(f'Epoch: {epoch}, Val Loss: {total_loss / len(val_dataloader)}')
            print(f'Epoch: {epoch}, Val Acc: {correct / total}')
        
        torch.save(model.state_dict(), f'{args.save_dir}/baseline_{epoch}.pt')