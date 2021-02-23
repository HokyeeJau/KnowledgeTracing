import torch
import random
import os
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args(parser):
    params = parser.parse_known_args()[0]
    # Set down device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if params.gpu != 'none':
        os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu

    torch.manual_seed(params.random_seed)
    torch.cuda.manual_seed(params.random_seed)
    torch.cuda.manual_seed_all(params.random_seed)
    random.seed(params.random_seed)

    if torch.cuda.is_available():
        params.device = 'cuda'
        params.gpu = list(range(len(params.gpu.split(','))))
        if params.gpu is not None:
            torch.cuda.set_device(params.gpu[0])
    os.makedirs(params.weight_path, exist_ok=True)
    return params

# Set Down an Argument Parser for receiving parameters
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default='HokyeeJau')
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--gpu", type=str, default='none')
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--weight_path", type=str, default="./weights")
parser.add_argument("--dataset_name", type=str, default="assist2009_updated")
parser.add_argument("--base_path", type=str, default="./dataset")
parser.add_argument("--only_save_best", type=str2bool, default='0')
parser.add_argument("--model_type", type=str, default="DKT")
parser.add_argument("--valid_index", type=int, default=1)

parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--input_dim", type=int, default=256)
parser.add_argument("--dropout", type=float, default=0.2)

parser.add_argument("--random_seed", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.02)
parser.add_argument("--num_epochs", type=int, default=300)
parser.add_argument("--train_batch", type=int, default=512)
parser.add_argument("--test_batch", type=int, default=128)
parser.add_argument("--seq_size", type=int, default=128)
parser.add_argument("--warm_up_step_count", type=int, default=4000)
parser.add_argument("--eval_steps", type=int, default=5)
parser.add_argument("--cross_validation", type=str2bool, default='0')
ARGS = get_args(parser)

if __name__ == '__main__':
    ARGS = get_args(parser)
    print_args(ARGS)
