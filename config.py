import os
import argparse


def _parse_base_setting(parser):
    group = parser.add_argument_group('base', 'base settings')
    group.add_argument('--seed', default=6669, type=int)
    group.add_argument('--data_path', default='data', type=str)
    group.add_argument('--dataset', default='mimic3', type=str, choices=['mimic3', 'mimic4'])
    group.add_argument('--result_path', default='result', type=str)


def _parse_preprocess_setting(parser):
    group = parser.add_argument_group('preprocess', 'preprocess settings')
    group.add_argument('--from_saved', action='store_true')
    group.add_argument('--train_num', default=6000, type=int)
    group.add_argument('--sample_num', default=10000, type=int, help='for mimic4')


def _parse_model_structure_setting(parser):
    group = parser.add_argument_group('model', 'model structure setting')
    group.add_argument('--g_hidden_dim', default=256, type=int)
    group.add_argument('--g_attention_dim', default=64, type=int)
    group.add_argument('--c_hidden_dim', default=64, type=int)


def _parse_gan_training_setting(parser):
    group = parser.add_argument_group('gan_training', 'gan training setting')
    group.add_argument('--iteration', default=300000, type=int)
    group.add_argument('--batch_size', default=256, type=int)
    group.add_argument('--g_iter', default=1, type=int)
    group.add_argument('--g_lr', default=1e-4, type=float)
    group.add_argument('--c_iter', default=1, type=int)
    group.add_argument('--c_lr', default=1e-5, type=float)
    group.add_argument('--betas0', default=0.5, type=float)
    group.add_argument('--betas1', default=0.9, type=float)
    group.add_argument('--lambda_', default=10, type=float)
    group.add_argument('--decay_rate', default=0.1, type=float)
    group.add_argument('--decay_step', default=100000, type=int)


def _parse_base_transformer_setting(parser):  # ✅ rename
    group = parser.add_argument_group('base_transformer_training', 'base transformer training setting')
    group.add_argument('--base_transformer_epochs', default=200, type=int)
    group.add_argument('--base_transformer_lr', default=1e-3, type=float)


def _parse_log_setting(parser):
    group = parser.add_argument_group('log', 'log setting')
    group.add_argument('--test_freq', default=1000, type=int)
    group.add_argument('--save_freq', default=1000, type=int)
    group.add_argument('--save_batch_size', default=256, type=int)


def _parse_generate_setting(parser):
    group = parser.add_argument_group('generate', 'generate setting')
    group.add_argument('--batch_size', default=256, type=int)
    group.add_argument('--use_iteration', default=-1, type=int)
    group.add_argument('--number', default=1500, type=int)
    group.add_argument('--upper_bound', default=1e7, type=int)


def get_preprocess_args():
    parser = argparse.ArgumentParser(description='Parameters for Data Preprocess')
    _parse_base_setting(parser)
    _parse_preprocess_setting(parser)
    return parser.parse_args()


def get_training_args():
    parser = argparse.ArgumentParser(description='Parameters for training MTGAN')
    _parse_base_setting(parser)
    _parse_model_structure_setting(parser)
    _parse_base_transformer_setting(parser)  # ✅ update here
    _parse_gan_training_setting(parser)
    _parse_log_setting(parser)
    return parser.parse_args()


def get_generate_args():
    parser = argparse.ArgumentParser(description='Parameters for Generation')
    _parse_base_setting(parser)
    _parse_model_structure_setting(parser)
    _parse_generate_setting(parser)
    return parser.parse_args()


def get_paths(args):
    dataset_path = os.path.join(args.data_path, args.dataset)

    result_path = os.path.join(args.result_path, args.dataset)
    os.makedirs(result_path, exist_ok=True)

    records_path = os.path.join(result_path, 'records')
    os.makedirs(records_path, exist_ok=True)

    params_path = os.path.join(result_path, 'params')
    os.makedirs(params_path, exist_ok=True)

    return dataset_path, records_path, params_path
