import random
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from itertools import chain
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from peft import IA3Config, LoraConfig, get_peft_model, PeftModel
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import minmax_scale
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import EsmTokenizer, EsmForMaskedLM

from dataset import MutantSequenceData, RankingSequenceData
from utils.data import make_dir, split_data
from utils.score import (
    group_scores,
    listwise_ranking_loss,
    metrics,
    pairwise_ranking_loss,
    summarize_scores,
)
import json

with open("config.json", "r") as file:
    config = json.load(file)



def get_optimizer(optimizer, lr, params):
    params = filter(lambda p: p.requires_grad, params)
    if optimizer == 'sgd':
        return optim.SGD(params, lr=lr)
    elif optimizer == 'nag':
        return optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    elif optimizer == 'adagrad':
        return optim.Adagrad(params, lr=lr)
    elif optimizer == 'adadelta':
        return optim.Adadelta(params, lr=lr)
    elif optimizer == 'adam':
        return optim.Adam(params, lr=lr)
    else:
        raise ValueError('Unknown optimizer: ' + optimizer)


class RankingTrainer:
    def __init__(self, model, margin=1.0, pair_fn='hinge', score_fn=None,
                 optimizer='adam', lr=1e-4, epochs=100, max_grad_norm=5,
                 lr_decay=None, eval_metric='spearmanr',
                 log_metrics=('spearmanr',), save_dir=None, patience=5,
                 overwrite=True, list_size=2):
        self.model = model
        self.optimizer = get_optimizer(optimizer, lr, model.parameters())
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        if lr_decay:
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, lr_decay)
        self.eval_metric = eval_metric
        self.log_metrics = log_metrics
        self.save_dir = save_dir
        self.patience = patience
        self.overwrite = overwrite
        self.curr_epoch = self.curr_iter = self.best_epoch = 0
        self.best_score = float('-inf')
        self.logs = defaultdict(list)
        self.margin = margin
        self.pair_fn = pair_fn
        self.score_fn = score_fn
        self.list_size = list_size
    
    def save_states(self):
        print('Saving model states...')
        save_dir = self.save_dir if self.overwrite else f'{self.save_dir}/epoch_{self.curr_epoch}'
        self.model.save_pretrained(save_dir)
        torch.save(self.logs, self.save_dir + '/logs.pkl')
    
    def predict(self, batch):
        if self.score_fn is not None:
            return self.score_fn(self.model, batch)
        
        logits = self.model(**batch['sequences']).logits
        log_probs = torch.log_softmax(logits, dim=-1)
        
        predicts = []
        for inv_idx, positions, wt_aas, mt_aas in zip(
                batch['inv_seq_idx'], batch['positions'], batch['wt_aas'], batch['mt_aas']):
            log_prob = log_probs[inv_idx]
            predict = log_prob[positions, mt_aas] - log_prob[positions, wt_aas]
            predicts.append(predict.sum().unsqueeze(0))
        return torch.cat(predicts)

    def evaluate(self, eval_iter):
        self.model.eval()
        predicts, targets, labels = [], [], []
        pbar = tqdm(eval_iter, desc='Evaluating')
        
        with torch.no_grad():
            for batch in pbar:
                batch_preds = self.predict(batch)
                # compute metrics on full data
                predicts.append(batch_preds.to('cpu'))
                targets.append(batch['targets'].to('cpu'))
                labels.append(batch['labels'].to('cpu'))
        
        if not predicts:
            raise ValueError('Cannot evaluate an empty dataset.')
        predicts, targets, labels = torch.cat(predicts), torch.cat(targets), torch.cat(labels)
        logs = {}
        for metric in self.log_metrics:
            if metric == 'spearmanr':
                logs[metric] = spearmanr(predicts, targets).statistic
            elif metric == 'ndcg':
                std_tgts = minmax_scale(targets.unsqueeze(0), (0, 5), axis=1)
                logs[metric] = ndcg_score(std_tgts, predicts.unsqueeze(0))
            elif metric == 'topk_pr':
                k = min(len(predicts), 30)
                indices = predicts.topk(k).indices
                logs[metric] = torch.count_nonzero(labels[indices]).item() / k
            else:
                raise ValueError('Unknown metric: ' + metric)
        
        for key, value in logs.items():
            print('{}: {:.3f}'.format(key, value))
        return predicts, logs
    
    def fit(self, train_iter, eval_iter=None):
        for _ in range(self.epochs):
            self.model.train()
            train_loss = 0
            num_updates = 0
            pbar = tqdm(train_iter, desc=f'Training epoch {self.curr_epoch + 1}')

            for batch in pbar:
                self.optimizer.zero_grad()

                predicts = self.predict(batch)
                targets = batch['targets']
                if self.list_size == 1:
                    loss = F.mse_loss(predicts, targets)
                elif self.list_size == 2:
                    pair_count = len(predicts) // 2
                    if pair_count == 0:
                        continue
                    predicts = predicts[:pair_count * 2].reshape(-1, 2)
                    targets = targets[:pair_count * 2].reshape(-1, 2)
                    loss = pairwise_ranking_loss(
                        predicts[:, 0], predicts[:, 1],
                        targets[:, 0], targets[:, 1],
                        self.pair_fn, self.margin)
                else:
                    if len(predicts) < 2:
                        continue
                    loss = listwise_ranking_loss(
                        predicts.unsqueeze(0), targets.unsqueeze(0))

                loss.backward()
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                self.curr_iter += 1
                num_updates += 1
                train_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

            if hasattr(self, 'scheduler'):
                self.scheduler.step()

            self.curr_epoch += 1
            if num_updates == 0:
                raise ValueError('Training produced no optimizer updates. Ranking loss requires batches with at least two examples.')
            train_loss /= num_updates
            lr = self.optimizer.param_groups[0]['lr']
            self.logs['train_loss'].append(train_loss)
            self.logs['lr'].append(lr)
            print('train_loss: {:.3f}'.format(train_loss))
            print('lr: {:.1e}'.format(lr))

            if eval_iter is None:
                continue
            
            _, logs = self.evaluate(eval_iter)
            for key, value in logs.items():
                self.logs[key].append(value)
            
            score = logs[self.eval_metric]
            if score > self.best_score:
                self.best_epoch = self.curr_epoch
                self.best_score = score
                if self.save_dir:
                    self.save_states()
            elif self.curr_epoch - self.best_epoch >= self.patience:
                print(f'Early stopped at epoch {self.curr_epoch}')
                print(f'Best validating {self.eval_metric} reached at epoch {self.best_epoch}: {self.best_score:.3f}')
                break
            
        if self.save_dir and eval_iter is None:
            self.save_states()
        return self.logs

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)

def print_trainable_params(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f'Trainable params: {trainable_params} ({100 * trainable_params / all_param:.2f}%)')
    print(f'All params: {all_param}')


def uses_peft(args):
    return args.peft_type != 'none'


def get_device(device='auto', force_cpu=False):
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if force_cpu:
        return 'cpu'
    if device == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        if mps_available:
            return 'mps'
        return 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is not available.')
    if device == 'mps' and not mps_available:
        raise ValueError('MPS was requested but is not available.')
    return device


class Pipeline():
    def __init__(self, parsed_args, data_constructor=MutantSequenceData,
                 lora_modules=config['lora_modules'],
                 ia3_modules=config.get('ia3_modules', config['lora_modules']),
                 ia3_feedforward_modules=config.get('ia3_feedforward_modules', ['dense']),
                 score_fn=None):
        if parsed_args.n_sites == [0]:
            parsed_args.n_sites = None
        if parsed_args.train_size > 1:
            parsed_args.train_size = int(parsed_args.train_size)
        self.args = parsed_args
        self.device = get_device(parsed_args.device, parsed_args.force_cpu)
        self.data_constructor = data_constructor
        self.lora_modules = lora_modules
        self.ia3_modules = ia3_modules
        self.ia3_feedforward_modules = ia3_feedforward_modules
        self.score_fn = score_fn
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        set_seed(parsed_args.seed)
    
    def get_base_model(self, load_dir=None):
        args = self.args
        model_name = config['model_dir'][args.model]
        if load_dir is None:
            model = EsmForMaskedLM.from_pretrained(model_name)
            for name, param in model.named_parameters():
                if 'contact_head.regression' in name:
                    param.requires_grad = False
        else:
            model = EsmForMaskedLM.from_pretrained(load_dir)
        tokenizer = EsmTokenizer.from_pretrained(model_name)
        return model, tokenizer
        
    def get_save_dir(self, protein_name, prediction=False, timestamp=True):
        args = self.args
        if args.peft_type == 'lora':
            peft_name = f'r{args.lora_r}_alpha{args.lora_alpha}'
        else:
            peft_name = args.peft_type
        save_dir = '{}/{}/{}/{}{}{}{}{}{}'.format(
            config['pred_dir'] if prediction else config['ckpt_dir'],
            args.model,
            protein_name,
            peft_name,
            f'_ts{args.train_size}',
            '_' + '-'.join(args.augment) if args.augment else '',
            '_regr' if args.list_size == 1 else '',
            '_ms' if args.n_sites != [1] else '',
            args.save_postfix)
        if timestamp:
            save_dir += f'_{self.run_timestamp}'
        return save_dir

    def get_peft_config(self):
        args = self.args
        if not uses_peft(args):
            return None
        if args.peft_type == 'lora':
            if args.lora_r <= 0:
                raise ValueError('--lora_r must be greater than 0 when using --peft_type lora.')
            return LoraConfig(r=args.lora_r,
                              lora_alpha=args.lora_alpha,
                              target_modules=self.lora_modules,
                              lora_dropout=0.1,
                              bias='none')
        if args.peft_type == 'ia3':
            return IA3Config(target_modules=self.ia3_modules,
                             feedforward_modules=self.ia3_feedforward_modules)
        raise ValueError('Unknown PEFT type: ' + args.peft_type)

    def get_latest_checkpoint(self, protein_name):
        base_dir = self.get_save_dir(protein_name, timestamp=False)
        timestamped_dirs = sorted(
            path for path in Path(base_dir).parent.glob(Path(base_dir).name + '_*')
            if path.is_dir())
        if timestamped_dirs:
            return str(timestamped_dirs[-1])
        return base_dir

    def finetune(self, train, valid):
        args = self.args
        save_dir = self.get_save_dir(train['name'])
        if len(train['df']) < 2:
            raise ValueError(
                f'Cannot train from {len(train["df"])} example. '
                '--train_size values up to 1 are proportions; values greater than 1 '
                'are absolute mutant counts. Use a larger value.')
        if args.list_size > 1 and len(train['df']) < 2:
            raise ValueError('Ranking loss requires at least two training examples after the validation split.')
        if len(valid['df']) < 2:
            raise ValueError('Validation metrics require at least two validation examples. Use a smaller --train_size.')

        model, tokenizer = self.get_base_model()
        peft_config = self.get_peft_config()
        if peft_config is not None:
            model = get_peft_model(model, peft_config)
            print_trainable_params(model)
        
        train_data = RankingSequenceData(train, tokenizer,
                                         mask=args.mask in {'train', 'all'},
                                         constructor=self.data_constructor,
                                         device=self.device)
        train_iter = DataLoader(train_data,
                                batch_size=args.train_batch,
                                shuffle=True,
                                collate_fn=train_data.collate)
        trainer = RankingTrainer(model.to(self.device),
                                 optimizer=args.optimizer,
                                 lr=args.learning_rate,
                                 epochs=args.epochs,
                                 max_grad_norm=args.max_grad_norm,
                                 score_fn=self.score_fn,
                                 eval_metric=args.eval_metric,
                                 log_metrics=metrics,
                                 save_dir=save_dir,
                                 patience=args.patience,
                                 list_size=args.list_size)
        
        report = {}
        eval_data = self.data_constructor(valid, tokenizer,
                                            mask=args.mask in {'eval', 'all'},
                                            device=self.device)
        eval_iter = DataLoader(eval_data,
                                batch_size=args.eval_batch,
                                collate_fn=eval_data.collate)
        print('Computing zero-shot scores...')
        _, report['baseline'] = trainer.evaluate(eval_iter)
        
        logs = trainer.fit(train_iter, eval_iter)
        report.update(logs)
        report['best_epoch'] = trainer.best_epoch
        make_dir(save_dir + '/logs.pkl')
        torch.save(report, save_dir + '/logs.pkl')
        return report, save_dir
    
    
    def test(self, train, test, load_dir=None):
        args = self.args
        if args.epochs > 0:
            load_dir = load_dir or self.get_latest_checkpoint(test['name'])
            if not uses_peft(args):
                model, tokenizer = self.get_base_model(load_dir)
            else:
                model, tokenizer = self.get_base_model()
                model = PeftModel.from_pretrained(model, load_dir, is_trainable=True)
        else:
            model, tokenizer = self.get_base_model()
        
        test_data = self.data_constructor(test, tokenizer,
                                          mask=args.mask in {'eval', 'all'},
                                          device=self.device)
        test_iter = DataLoader(test_data,
                               batch_size=args.eval_batch,
                               collate_fn=test_data.collate)
        trainer = RankingTrainer(model.to(self.device), log_metrics=[], score_fn=self.score_fn)
        predicts, _ = trainer.evaluate(test_iter)
        predicts = predicts.tolist()
        
        predicts = pd.Series(predicts, index=test['df'].index, name='prediction')
        report, _ = group_scores(train['df'], predicts, test['df'])
        print('======================Breakdown results======================')
        print(report)
        
        print('Saving model predictions...')
        save_path = self.get_save_dir(test['name'], prediction=True)
        save_path += '_base.csv' if args.epochs == 0 else '.csv'
        make_dir(save_path)
        predicts.to_csv(save_path)
        return report
    
    def select_datasets(self, all_proteins):
        args = self.args
        print(all_proteins.keys())
        if args.protein in all_proteins.keys():
            return all_proteins[args.protein]
        
        proteins = chain(*all_proteins.values())
        if args.train_size > 1:
            proteins = filter(lambda x: len(x['df']) > args.train_size, proteins)
        
        if args.protein == 'all':
            return list(proteins)
        if args.protein == 'single-site':
            return list(filter(lambda x: x['n_sites'][-1] == 1, proteins))
        if args.protein == 'multi-site':
            return list(filter(lambda x: x['n_sites'][-1] > 1, proteins))
        if len(args.protein) == 2:
            proteins = list(proteins)
            N, i = int(args.protein[0]), int(args.protein[1])
            n = ceil(len(proteins) / N)
            j = (i - 1) * n
            return proteins[j:j + n]
    
    
    def augment_data(self, protein):
        args = self.args
        if args.augment == ['adaptive']:
            aug_models = pd.read_csv(f"{ config['retr_dir'] }/aug_models{args.save_postfix}.csv", index_col=0)
            aug_models = [aug_models.loc[protein['name'], args.train_size]]
        else:
            aug_models = args.augment
        
        raw_data = pd.read_csv(f"{config['raw_data_dir']}/{protein['name']}.csv", index_col='mutant',
                               usecols=aug_models + ['mutant'])
        aug_data = []
        for model_name in aug_models:
            new = deepcopy(protein)
            new['df']['DMS_score'] = raw_data[model_name]
            if new['n_sites'][-1] > 2:
                new, _ = split_data(new, len(new['df']), n_sites=[1, 2],
                                    random_state=args.seed, test_size=0)
            aug_data.append(new)
        return aug_data
    
    def run(self, all_proteins):
        args = self.args
        proteins = self.select_datasets(all_proteins)
        print(f'Using device: {self.device}')
        print(f'Using random seed: {args.seed}')
        
        print("RUN DATASET ", proteins[0].keys()) # dict_keys(['wild_type', 'df', 'offset', 'n_sites', 'name'])
        print(proteins[0]['df']) # [2497 rows x 5 columns]

        reports = {}
        for protein in proteins:
            print(f'**********************Current dataset: {protein["name"]}**********************')
            if protein['name'] == 'CCDB_ECOLI_Tripathi_2016':
                eval_metric = args.eval_metric
                args.eval_metric = 'ndcg' # in case of nan spearmanr
            
            train, valid, test = split_data(protein, args.train_size, n_sites=args.n_sites,
                                            neg_train=args.negative_train, scale=args.list_size == 1,
                                            random_state=args.seed, return_valid=True)
            print(f'Split sizes - train: {len(train["df"])}, validation: {len(valid["df"])}, test: {len(test["df"])}')
            if args.test:
                if test['df'].empty:
                    raise ValueError('No held-out test examples are available. Use --train_size below 1 when running --test.')
                report = self.test(train, test)
            else:
                if args.augment:
                    protein = self.augment_data(protein)[0]
                _, best_checkpoint_dir = self.finetune(train, valid)
                if test['df'].empty:
                    print('Skipping test evaluation: no held-out test examples are available.')
                    report = None
                else:
                    report = self.test(train, test, best_checkpoint_dir)
            if report is not None:
                reports[protein['name']] = report
            torch.cuda.empty_cache()
            
            if protein['name'] == 'CCDB_ECOLI_Tripathi_2016':
                args.eval_metric = eval_metric
        
        if reports and args.protein in {'single-site', 'multi-site', 'all'}:
            save_path = self.get_save_dir(args.protein, prediction=True)
            save_path += '_base.pkl' if args.epochs == 0 else '.pkl'
            make_dir(save_path)
            reports = summarize_scores(reports, save_path)
            print('**********************Score summary**********************')
            print(reports[args.eval_metric])
