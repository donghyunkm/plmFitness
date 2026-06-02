import torch
from torch.utils.data import Dataset

class ProteinSequenceData(Dataset):
    def __init__(self, sequences, tokenizer, device=None):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.device = device
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]
    
    def collate(self, raw_batch):
        sequences = self.tokenizer(raw_batch, return_tensors='pt', padding=True, return_length=True)
        return sequences.to(self.device)

class MutantSequenceData(Dataset):
    def __init__(self, protein, tokenizer, mask=False, device=None):
        if mask:
            self.sequences = {}
            for positions in set(protein['df']['positions']):
                mutant = list(protein['wild_type'])
                for position in positions: # get masked mutant sequence
                    mutant[position] = '<mask>'
                self.sequences[positions] = ''.join(mutant)
        else:
            self.sequences = [protein['wild_type']]
        
        for key, value in protein['df'].items():
            setattr(self, key, value.to_list())
        self.tokenizer = tokenizer
        self.device = device
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        return self.wt_aas[idx], self.mt_aas[idx], self.positions[idx], self.DMS_score[idx], self.DMS_score_bin[idx]
    
    def collate(self, raw_batch):
        wt_aas, mt_aas, positions, scores, labels = zip(*raw_batch)
        
        if type(self.sequences) is dict: # identify duplicate positions, possibly multi-site
            unique_pos = {pos: i for i, pos in enumerate(set(positions))}
            inv_idx = torch.tensor([unique_pos[pos] for pos in positions], device=self.device)
            sequences = [self.sequences[pos] for pos in unique_pos.keys()]
        else:
            inv_idx = torch.zeros(len(positions), dtype=torch.long, device=self.device)
            sequences = self.sequences
        sequences = self.tokenizer(sequences, return_tensors='pt').to(self.device)
        
        positions = [torch.tensor(pos, device=self.device) + 1 for pos in positions]
        wt_aas = self.tokenizer(wt_aas, add_special_tokens=False)['input_ids']
        mt_aas = self.tokenizer(mt_aas, add_special_tokens=False)['input_ids']
        scores = torch.tensor(scores, device=self.device)
        labels = torch.tensor(labels, device=self.device)
        return dict(sequences=sequences,
                    inv_seq_idx=inv_idx,
                    wt_aas=wt_aas,
                    mt_aas=mt_aas,
                    positions=positions,
                    targets=scores,
                    labels=labels)

class RankingSequenceData(Dataset):
    def __init__(self, protein, tokenizer, mask=True,
                 constructor=MutantSequenceData, device=None):
        self.mutant_data = constructor(protein, tokenizer, mask, device)
    
    def __len__(self):
        return len(self.mutant_data)
    
    def __getitem__(self, idx):
        return self.mutant_data[idx]
    
    def collate(self, raw_batch):
        batch = self.mutant_data.collate(raw_batch)
        return batch
