import sys
from os.path import dirname, abspath, join
from tqdm import tqdm
import numpy as np
import torch
from config import get_config
from dataset import get_dataloader
from agent import get_agent
from utils import cycle, dict_get


def main():
    # create experiment config containing all hyperparameters
    config = get_config('train')

    # create dataloader
    train_loader = get_dataloader(config.dataset, 'train', config)
    test_loader = get_dataloader(config.dataset, 'test', config)
    ulb_train_loader = get_dataloader(config.dataset, 'ulb_train', config)
    iter_ulb_train_loader = cycle(ulb_train_loader)

    # create network and training agent
    agent = get_agent(config)

    if config.cont:
        # recover training
        agent.load_ckpt(config.ckpt)
        agent.clock.tock()

        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = config.lr


    # start training
    clock = agent.clock
    best_median_error = 360

    while True:
        # begin iteration
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            # train step

            # change lr for pascal3d stage2
            if config.dataset == 'pascal3d' and clock.iteration == config.stage1_iteration:
                stage1_clock = agent.clock.make_checkpoint()
                agent.load_ckpt('best')
                agent.clock.restore_checkpoint(stage1_clock)
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] *= 0.1

            if clock.iteration < config.stage1_iteration:
                # supervised
                s1 = True
                fisher_dict = agent.train_func_s1(data)
                loss = fisher_dict['loss']
            else:
                # ssl
                s1 = False
                ulb_data = next(iter_ulb_train_loader)
                fisher_dict, fisher_dict_unsuper, out_dict = agent.train_func(data, ulb_data)
                loss = out_dict['loss_all']

            if agent.clock.iteration % config.log_frequency == 0:
                agent.writer.add_scalar('train/lr', agent.optimizer.param_groups[0]['lr'], clock.iteration)
                agent.writer.add_scalar('train/loss', fisher_dict['loss'], clock.iteration)
                agent.writer.add_scalar('train/err_mean', fisher_dict['err_deg'].mean().item(), clock.iteration)
                if not s1:
                    agent.writer.add_scalar('train_SSL/unsuper_loss', dict_get(fisher_dict_unsuper, 'unsuper_loss', -1).item(), clock.iteration)
                    agent.writer.add_scalar('train_SSL/entropy', dict_get(fisher_dict_unsuper, 'entropy', -1).mean().item(), clock.iteration)
                    agent.writer.add_scalar('train_SSL/mask_ratio', dict_get(fisher_dict_unsuper, 'mask_ratio', -1).item(), clock.iteration)
                    agent.writer.add_scalar('train_SSL/err_weakAll_gt', dict_get(fisher_dict_unsuper, 'err_weakAll_gt', -1).mean().item(), clock.iteration)
                    agent.writer.add_scalar('train_SSL/err_weakPseudo_gt', dict_get(fisher_dict_unsuper, 'err_weakPseudo_gt', -1).mean().item(), clock.iteration)
                    agent.writer.add_scalar('train_SSL/err_strongSuper_pseudo', dict_get(fisher_dict_unsuper, 'err_strongSuper_pseudo', -1).mean().item(), clock.iteration)

            pbar.set_description("EPOCH[{}][{}]".format(clock.epoch, clock.minibatch))
            pbar.set_postfix({'loss': loss.item()})

            clock.tick()

            # evaluation
            if clock.iteration % config.val_frequency == 0:
                fisher_test_loss = []
                fisher_test_err_deg = []
                fisher_test_mask_ratio = []
                fisher_test_err_pseudo_gt = []

                testbar = tqdm(test_loader)
                for i, data in enumerate(testbar):
                    if s1:
                        fisher_dict = agent.val_func_s1(data)
                    else:
                        fisher_dict, fisher_dict_unsuper, out_dict = agent.val_func(data)

                        fisher_test_mask_ratio.append(out_dict['mask_ratio'])
                        if out_dict['err_pseudo_gt'] is not None:
                            fisher_test_err_pseudo_gt.append(out_dict['err_pseudo_gt'].detach().cpu().numpy())

                    fisher_test_loss.append(fisher_dict['loss'].item())
                    fisher_test_err_deg.append(fisher_dict['err_deg'].detach().cpu().numpy())

                fisher_test_err_deg = np.concatenate(fisher_test_err_deg, 0)
                agent.writer.add_scalar('test/loss', np.mean(fisher_test_loss), clock.iteration)
                agent.writer.add_scalar('test/err_median', np.median(fisher_test_err_deg), clock.iteration)
                agent.writer.add_scalar('test/err_mean', np.mean(fisher_test_err_deg), clock.iteration)
                if not s1:
                    fisher_test_err_pseudo_gt = [-1] if len(fisher_test_err_pseudo_gt) == 0 else \
                        np.concatenate(fisher_test_err_pseudo_gt, 0)
                    agent.writer.add_scalar('test/mask_ratio', np.mean(fisher_test_mask_ratio), clock.iteration)
                    agent.writer.add_scalar('test/err_pseudo_gt', np.mean(fisher_test_err_pseudo_gt), clock.iteration)
                
                # save the best checkpoint
                if np.median(fisher_test_err_deg) < best_median_error:
                    best_median_error = np.median(fisher_test_err_deg)
                    agent.save_ckpt('best')

                if not s1:
                    # For SSL, evaluate again by ema_model
                    fisher_test_loss = []
                    fisher_test_err_deg = []
                    fisher_test_mask_ratio = []
                    fisher_test_err_pseudo_gt = []

                    testbar = tqdm(test_loader)
                    for i, data in enumerate(testbar):
                        fisher_dict, fisher_dict_unsuper, out_dict = agent.val_func(data, eval_ema=True)

                        fisher_test_mask_ratio.append(out_dict['mask_ratio'])
                        if out_dict['err_pseudo_gt'] is not None:
                            fisher_test_err_pseudo_gt.append(out_dict['err_pseudo_gt'].detach().cpu().numpy())

                        fisher_test_loss.append(fisher_dict['loss'].item())
                        fisher_test_err_deg.append(fisher_dict['err_deg'].detach().cpu().numpy())

                    fisher_test_err_deg = np.concatenate(fisher_test_err_deg, 0)
                    agent.writer.add_scalar('test_ema/loss', np.mean(fisher_test_loss), clock.iteration)
                    agent.writer.add_scalar('test_ema/err_median', np.median(fisher_test_err_deg), clock.iteration)
                    agent.writer.add_scalar('test_ema/err_mean', np.mean(fisher_test_err_deg), clock.iteration)
                    fisher_test_err_pseudo_gt = [-1] if len(fisher_test_err_pseudo_gt) == 0 else \
                        np.concatenate(fisher_test_err_pseudo_gt, 0)
                    agent.writer.add_scalar('test_ema/mask_ratio', np.mean(fisher_test_mask_ratio), clock.iteration)
                    agent.writer.add_scalar('test_ema/err_pseudo_gt', np.mean(fisher_test_err_pseudo_gt), clock.iteration)

            # save checkpoint
            if clock.iteration % config.save_frequency == 0:
                agent.save_ckpt()


        clock.tock()

        if clock.iteration > config.max_iteration:
            break



if __name__ == '__main__':
    main()
