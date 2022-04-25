import sys
import os
from os.path import dirname, abspath, join
import numpy as np
import torch
from tqdm import tqdm
from config import get_config
from agent import get_agent
from dataset import get_dataloader


def test():
    # create experiment config containing all hyperparameters
    config = get_config('test')

    # create dataloader
    test_loader = get_dataloader(config.dataset, 'test', config)

    # create network and eval agent
    agent = get_agent(config)
    agent.load_ckpt(config.ckpt)

    evaluate(config, test_loader, agent)
    evaluate(config, test_loader, agent, eval_ema=True)


def evaluate(config, test_loader, agent, eval_ema=False):
    ema_name = 'EMA_' if eval_ema else ''

    err_deg_lst = []
    testbar = tqdm(test_loader)
    for i, data in enumerate(testbar):
        fisher_dict, fisher_dict_unsuper, out_dict = agent.val_func(data, eval_ema=eval_ema)
        err_deg_lst.append(fisher_dict['err_deg'].detach().cpu().numpy())

    err_deg_lst = np.concatenate(err_deg_lst, 0)

    print(f'==== {ema_name}exp: {config.exp_name} ====')
    print(f'{ema_name}Fisher mean: {np.mean(err_deg_lst):.2f}')
    print(f'{ema_name}Fisher median: {np.median(err_deg_lst):.2f}')


if __name__ == '__main__':
    with torch.no_grad():
        test()

