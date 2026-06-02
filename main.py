import argparse
import torch
import json
from trainer import Pipeline
from utils.score import metrics

with open("config.json", "r") as file:
    config = json.load(file)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', '-t', action='store_true',
                        help='skip training and test the latest checkpoints')
    parser.add_argument('--model', '-md', type=str, choices=config['model_dir'].keys(), required=True,
                        help='name of the foundation model')
    parser.add_argument('--protein', '-p', type=str, default='all',
                        help='name of the target protein')
    parser.add_argument('--train_size', '-ts', type=float, required=True,
                        help='stratified training subset size after the fixed test holdout: values up to 1 are proportions; values greater than 1 are mutant counts')
    parser.add_argument('--train_batch', '-tb', type=int, default=16,
                        help='batch size for training')
    parser.add_argument('--eval_batch', '-eb', type=int, default=128,
                        help='batch size for evaluation')
    parser.add_argument('--lora_r', '-r', type=int, default=16,
                        help='rank r of the LoRA adapters')
    parser.add_argument('--lora_alpha', '-la', type=int, default=16,
                        help='alpha scaling factor of the LoRA adapters')
    parser.add_argument('--peft_type', '-peft', type=str, choices=['lora', 'ia3', 'none'], default='lora',
                        help='type of PEFT adapter to use')
    parser.add_argument('--optimizer', '-o', type=str, choices=['sgd', 'nag', 'adagrad', 'adadelta', 'adam'],
                        default='adam', help='optimizer for training')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                        help='maximum training epochs')
    parser.add_argument('--max_grad_norm', '-gn', type=float, default=3,
                        help='maximum gradient norm to clip to')
    parser.add_argument('--mask', '-mk', type=str, choices=['train', 'eval', 'all', 'none'], default='none',
                        help='whether to compute masked 0-shot scores')
    parser.add_argument('--list_size', '-ls', type=int, default=5,
                        help='ranking loss mode: 1 for regression, 2 for pairwise, or greater than 2 for listwise')
    parser.add_argument('--eval_metric', '-em', type=str, choices=metrics, default='spearmanr',
                        help='evaluation metric')
    parser.add_argument('--augment', '-a', nargs='*', type=str, default=[],
                        help='specify one or more models to use their zero-shot scores for data augmentation')
    parser.add_argument('--patience', '-pt', type=int, default=15,
                        help='number of epochs to wait until the validation score improves')
    parser.add_argument('--n_sites', '-ns', nargs='+', type=int, default=[1],
                        help='possible numbers of mutation sites in the training data. \
                              setting to 0 means no constraint')
    parser.add_argument('--negative_train', '-neg', action='store_true',
                        help='whether to constraint the training data to negative examples')
    parser.add_argument('--seed', '-s', type=int, default=0,
                        help='random seed for training')
    parser.add_argument('--save_postfix', '-sp', type=str, default='',
                        help='a custom string to append to all data paths (data, checkpoints and predictions)')
    parser.add_argument('--device', '-d', type=str, choices=['auto', 'cpu', 'cuda', 'mps'], default='auto',
                        help='device for training and evaluation; auto prefers CUDA, then MPS, then CPU')
    parser.add_argument('--force_cpu', '-cpu', action='store_true',
                        help='use CPU for training and evaluation; equivalent to --device cpu')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    path = config['data_path'].replace('.pkl', f'{args.save_postfix}.pkl')
    proteins = torch.load(path)
    
    pipeline = Pipeline(args)
    pipeline.run(proteins)
