import pickle
import numpy as np

from tokenizers import Tokenizer
from tokenizers.models import Unigram
from transformers import PreTrainedTokenizerFast

from torch.utils.data import Dataset, DataLoader

def get_tokenizer():
    tokenizer = Tokenizer(Unigram(),)
    tokenizer.add_tokens(['+', '-', '*', '^', 'x', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    tokenizer.add_special_tokens(['<pad>', '<unk>', '<s>', '</s>'])

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='<unk>', pad_token='<pad>', bos_token='<s>', eos_token='</s>')
    return tokenizer

class SearchData(Dataset):
    '''
    Dataset for search (retrieval) engine.
    returns:
        - query: [full_expression, <pad>, x, <eos>]
        - gt_y: tokenized y.
    '''
    def __init__(self, dataset, tokenizer, max_len=100):
        super().__init__()
        with open(dataset, 'rb') as f:
            self.dataset = pickle.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __getitem__(self, index):
        full, abv, ys, x, y, full_tok, abv_tok, ys_tok, x_tok = self.dataset[index]
        data = np.ones(self.max_len, dtype=np.int64) * self.tokenizer.pad_token_id
        query = np.append(full_tok, self.tokenizer.pad_token_id)
        query = np.append(query, x_tok)
        query = np.append(query, self.tokenizer.eos_token_id)
        gt_y = ys_tok[x]
        data[:len(query)] = query
        return data, len(query), gt_y
    
    def __len__(self) -> int:
        return len(self.dataset)

class SupervisionData(Dataset):
    '''
    Dataset for baselines.
    supports 2 modes: <gt>, <none>
        - <gt> parse gt data from DB.
        - <none> No y0-y9.
    returns:
        - data: [full_expression, <pad>, x, <pad>, y0, y1, ..., y9, <eos>, <gt_y>]
        - target: shift query.
        - mask: point the target idx.
    '''
    def __init__(self, dataset, tokenizer, mode='gt', max_len=100):
        super().__init__()
        self.mode = mode
        with open(dataset, 'rb') as f:
            self.dataset = pickle.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __getitem__(self, index):
        full, abv, ys, x, y, full_tok, abv_tok, ys_tok, x_tok = self.dataset[index]
        data = np.ones(self.max_len, dtype=np.int32) * self.tokenizer.pad_token_id
        target = np.ones(self.max_len, dtype=np.int32) * self.tokenizer.pad_token_id
        mask = np.zeros(self.max_len, dtype=np.bool)

        query = np.append(full_tok, self.tokenizer.pad_token_id)
        query = np.append(query, x_tok)
        if self.mode == 'gt':
            query = np.append(query, self.tokenizer.pad_token_id)
            query = np.append(query, ys_tok)
        query = np.append(query, self.tokenizer.eos_token_id)
        gt_y = ys_tok[x]
        query = np.append(query, gt_y)

        data[:len(query)] = query
        target[:len(query)-1] = query[1:]
        mask[len(query)-2] = True # should point out <eos> for data, <gt_y> for target.

        return data, target, mask
    
    def __len__(self) -> int:
        return len(self.dataset)

class SupervisionData2(Dataset):
    '''
    Dataset for baselines.
    supports 2 modes: <gt>, <none>
        - <gt> parse gt data from DB.
        - <none> No y0-y9.
    returns:
        - data: [full_expression, <pad>, x, <pad>, y0, y1, ..., y9, <eos>, <gt_y>]
        - target: shift query.
        - mask: point the target idx.
    '''
    def __init__(self, dataset, tokenizer, mode='gt', max_len=100):
        super().__init__()
        self.mode = mode
        with open(dataset, 'rb') as f:
            self.dataset = pickle.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __getitem__(self, index):
        full, abv, ys, x, y, full_tok, abv_tok, ys_tok, x_tok = self.dataset[index]
        gt_y = ys_tok[x]

        data2 = np.ones(self.max_len, dtype=np.int64) * self.tokenizer.pad_token_id
        target2 = np.ones(self.max_len, dtype=np.int64) * self.tokenizer.pad_token_id
        mask2 = np.zeros(self.max_len, dtype=np.bool)

        query2 = np.append(full_tok, self.tokenizer.pad_token_id)
        query2 = np.append(query2, x_tok)
        start = len(query2)
        query2 = np.append(query2, self.tokenizer.eos_token_id)
        query2 = np.append(query2, abv_tok)
        query2 = np.append(query2, self.tokenizer.eos_token_id)
        end = len(query2)

        data2[:len(query2)] = query2
        target2[:len(query2)-1] = query2[1:]
        mask2[start:end] = True

        return data2, target2, mask2, start+1, gt_y
    
    def __len__(self) -> int:
        return len(self.dataset)