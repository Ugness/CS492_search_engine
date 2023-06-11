import math

import torch
import torch.nn as nn
from torch.nn import functional as F

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.
    resid_pdrop = 0.
    attn_pdrop = 0.

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class SEConfig(GPTConfig):
    """ Search Engine config (ref: https://arxiv.org/pdf/2201.02177.pdf)"""
    n_layer = 2
    n_head = 4
    n_embd = 128


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        mask = torch.tril(torch.ones(config.block_size,
                                     config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        present = torch.stack((k, v))
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if layer_past is None:
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, present   # TODO: check that this does not break anything


class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False):
        # if return_present: assert not self.training
        # layer past: tuple of length two with B, nh, T, hs
        attn, present = self.attn(self.ln1(x), layer_past=layer_past)

        x = x + attn
        x = x + self.mlp(self.ln2(x))
        if layer_past is not None or return_present:
            return x, present
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, tokenizer, vocab_size, block_size, n_layer=2, n_head=4, n_embd=128,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0.):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd)
        
        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        self.tokenizer = tokenizer

    @property
    def device(self):
        return self.head.weight.device

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, embeddings=None, targets=None, masks=None, reduce=True):
        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector

        if embeddings is not None: # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            _logits = logits[masks]
            _targets = targets[masks]
            loss = F.cross_entropy(_logits, _targets, reduction='mean' if reduce else 'none')
        return logits, loss

    # Copilot generated code
    @torch.no_grad()
    def get_log_prob(self, input_ids):
        output_logits = self.forward(input_ids)
        log_probs = torch.log_softmax(output_logits, dim=-1)
        log_prob = torch.sum(log_probs * torch.nn.functional.one_hot(input_ids[:, 1:], num_classes=log_probs.shape[-1]), dim=-1)
        return log_prob

    # def forward_with_past(self, idx, embeddings=None, targets=None, past=None, past_length=None):
    #     # TODO: use scatter.
    #     token_embeddings = self.tok_emb(idx)    # each index maps to a (learnable) vector
    #     if embeddings is not None:              # prepend explicit embeddings
    #         token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

    #     if past is not None:
    #         assert past_length is not None
    #         past = torch.cat(past, dim=-2)   # n_layer, 2, b, nh, len_past, dim_head
    #         past_shape = list(past.shape)
    #         expected_shape = [self.config.n_layer, 2, idx.shape[0], self.config.n_head, past_length, self.config.n_embd//self.config.n_head]
    #         assert past_shape == expected_shape, f"{past_shape} =/= {expected_shape}"
    #         position_embeddings = self.pos_emb[:, past_length, :]  # each position maps to a (learnable) vector
    #     else:
    #         position_embeddings = self.pos_emb[:, :token_embeddings.shape[1], :]

    #     x = self.drop(token_embeddings + position_embeddings)
    #     presents = []  # accumulate over layers
    #     for i, block in enumerate(self.blocks):
    #         x, present = block(x, layer_past=past[i, ...] if past is not None else None, return_present=True)
    #         presents.append(present)

    #     x = self.ln_f(x)
    #     logits = self.head(x)
    #     # if we are given some desired targets also calculate the loss
    #     loss = None
    #     if targets is not None:
    #         loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    #     return logits, loss, torch.stack(presents)  # _, _, n_layer, 2, b, nh, 1, dim_head

    # def sample_with_past(self, x, steps, trie, temperature=1., sample_logits=True,
    #                     top_k=None, top_p=None, callback=None):
    #     sample = x
    #     batch_size = x.shape[0]
    #     cond_len = x.shape[1]
    #     past = None
    #     current_nodes = [trie.root for _ in range(batch_size)]
    #     for n in range(steps):
    #         if callback is not None:
    #             callback(n)
    #         logits, _, present = self.forward_with_past(x, past=past, past_length=(n+cond_len-1))
    #         if past is None:
    #             past = [present]
    #         else:
    #             past.append(present)
            
    #         # TODO: shape check
    #         logits = logits[:, -1, :] / temperature
    #         logits = valid_tokens_masking(logits, trie, current_nodes, eos_id=self.tokenizer.eos_token_id)
    #         if top_k is not None:
    #             logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    #         probs = F.softmax(logits, dim=-1)
    #         if not sample_logits:
    #             _, x = torch.topk(probs, k=1, dim=-1)
    #         else:
    #             x = torch.multinomial(probs[:, -1, :], num_samples=1)
    #         # append to the sequence and continue
    #         current_nodes = [current_nodes[b].children[x[b].item()] for b in range(batch_size)]
    #         sample = torch.cat((sample, x), dim=1)
    #     del past
    #     sample = sample[:, cond_len:]  # cut conditioning off
    #     return sample

    def sample(self, query, indices, trie, temperature=1., sample_logits=True, top_k=None, top_p=None):
        # x: input text
        # indices: denotes the current position in the text
        # trie: prefix tree
        sample = query
        batch_size = query.shape[0]
        max_num_tokens = query.shape[1]
        current_nodes = [trie.root for _ in range(batch_size)]
        answers = torch.zeros_like(query)
        a_idx = 0
        # TODO: decode until all batch reach EOS.
        while True:
            logits = self(sample)[0]
            # TODO: use gather.
            logits = torch.gather(logits, dim=1, index=indices.view(-1, 1, 1).expand(-1, -1, logits.shape[-1]))
            logits = logits / temperature
            logits = valid_tokens_masking(logits, trie, current_nodes, eos_id=self.tokenizer.eos_token_id)
            if top_k is not None:
                logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            if not sample_logits:
                _, x = torch.topk(probs, k=1, dim=-1)
            else:
                x = torch.multinomial(probs[:, -1, :], num_samples=1)
            # append to the sequence and continue
            current_nodes = [current_nodes[b].children[x[b].item()] for b in range(batch_size)]

            # update x and indices
            sample.scatter_(dim=1, index=indices.view(-1, 1), src=x)
            answers[:, a_idx] = x[:, 0]
            indices = indices + 1
            indices = indices.clamp(min=0, max=max_num_tokens-1)

            a_idx += 1
            if indices.min() == (max_num_tokens-1):
                break
            if (x==self.tokenizer.eos_token_id).all():
                break
        return answers
    
    def predict(self, sample, indices, current_nodes, trie, temperature=1., top_k=None, top_p=None, argmax=False):
        # x: input text
        # indices: denotes the current position in the text
        # trie: prefix tree
        logits = self(sample)[0]
        logits = torch.gather(logits, dim=1, index=indices.view(-1, 1, 1).expand(-1, -1, logits.shape[-1]))
        logits = logits / temperature
        logits = valid_tokens_masking(logits, trie, current_nodes, eos_id=self.tokenizer.eos_token_id)
        if top_k is not None:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)
        if not argmax:
            action = torch.multinomial(probs[:, -1, :], num_samples=1).squeeze(-1)
        else:
            action = torch.argmax(probs[:, -1, :], dim=-1)

        log_prob = torch.log_softmax(logits, dim=-1)
        log_prob = torch.gather(log_prob, dim=-1, index=action.view(-1, 1, 1)).squeeze()

        return logits, probs, log_prob, action

    @torch.no_grad() 
    def random_act(self, current_nodes, trie):
        batch_size = len(current_nodes)
        logits = torch.zeros((batch_size, 1, len(self.tokenizer)), dtype=torch.float32, device=self.device)
        logits = valid_tokens_masking(logits, trie, current_nodes, eos_id=self.tokenizer.eos_token_id)
        probs = F.softmax(logits, dim=-1)
        actions = torch.multinomial(probs[:, -1, :], num_samples=1).squeeze(-1)
        return actions

def valid_tokens_masking(logits, trie, current_nodes, eos_id):
    batch_size = logits.shape[0]
    max_num_tokens = logits.shape[-1]
    mask = torch.zeros((batch_size, max_num_tokens), dtype=torch.bool, device=logits.device)
    for i in range(batch_size):
        valid_tokens = trie.get_valid_tokens(current_nodes[i])
        if len(valid_tokens) == 0:
            valid_tokens = [eos_id]
        mask[i, valid_tokens] = True
    mask = mask.unsqueeze(1).expand(-1, -1, max_num_tokens).bool()
    # simple trick to avoid inplace
    mask = torch.zeros_like(logits).masked_fill(~mask, -float('inf'))
    logits = logits + mask
    return logits

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits
