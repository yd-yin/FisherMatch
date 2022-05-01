import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pytorch3d import transforms as trans
import utils
from networks import get_network
from fisher.fisher_utils import vmf_loss as fisher_NLL, fisher_CE, batch_torch_A_to_R, fisher_entropy


def get_agent(config):
    return SSLAgent(config)


class SSLAgent:
    def __init__(self, config):
        self.config = config
        self.clock = utils.TrainClock()
        self.net = get_network(config)
        self.optimizer = optim.Adam(self.net.parameters(), config.lr)
        self.ema_net = get_network(config)
        # ema net is updated by ema, not training
        for param in self.ema_net.parameters():
            param.detach_()

        self.writer = SummaryWriter(log_dir=self.config.log_dir)

    def forward(self, data, ulb_data, eval_ema=False):
        ### 1. supervised loss
        img = data.get('img').cuda()
        gt = data.get('rot_mat').cuda()  # (b, 3, 3)

        # Teacher model or student model
        if eval_ema:
            net = self.ema_net
        else:
            net = self.net

        fisher_out = net(img)

        losses, pred_orth = fisher_NLL(fisher_out, gt, overreg=1.025)
        loss = losses.mean()

        err_deg = self.compute_err_deg_from_matrices(pred_orth, gt)

        fisher_dict = dict(
            loss=loss,
            pred=fisher_out,
            pred_orth=pred_orth,
            err_deg=err_deg
        )

        # usd for val_func
        if ulb_data is None:
            return fisher_dict, None

        ### 2. unsupervised loss
        ulb_img_weak = ulb_data.get('img').cuda()
        ulb_img_strong = ulb_data.get('img_strong').cuda()
        ulb_gt = ulb_data.get('rot_mat').cuda()  # ulb_gt is only used for evaluation

        # ema_net
        pred_weak = self.ema_net(ulb_img_weak)  # (b*nm, 9)

        utils.requires_grad(pred_weak, False)

        pred_strong = self.net(ulb_img_strong)


        entropy = fisher_entropy(pred_weak)
        mask_fisher = entropy < self.config.conf_thres  # (b, )

        mask_ratio_fisher = mask_fisher.sum() / len(mask_fisher)
        if mask_ratio_fisher > 0:
            pseudo_label_fisher = batch_torch_A_to_R(pred_weak[mask_fisher])
            if self.config.type_unsuper == 'ce':
                unsuper_loss = fisher_CE(pred_weak[mask_fisher], pred_strong[mask_fisher])
                unsuper_loss = unsuper_loss.mean()
            elif self.config.type_unsuper == 'nll':
                unsuper_losses, _ = fisher_NLL(pred_strong[mask_fisher], pseudo_label_fisher, overreg=1.025)
                unsuper_loss = unsuper_losses.mean()
        else:
            unsuper_loss = torch.tensor([0.], device='cuda').float()

        # We want unsupervised loss to be 1/(mu*B) Sum_{mu*B*mask} l, now unsuper_loss is 1/(mu*B*mask) Sum_{mu*B*mask} l, so multiply mask
        unsuper_loss = unsuper_loss * mask_ratio_fisher

        # errors
        err_weakAll_gt = self.compute_err_deg_from_matrices(batch_torch_A_to_R(pred_weak), ulb_gt)
        err_weakPseudo_gt = self.compute_err_deg_from_matrices(batch_torch_A_to_R(pred_weak[mask_fisher]), ulb_gt[mask_fisher])
        err_strongSuper_pseudo = self.compute_err_deg_from_matrices(
            batch_torch_A_to_R(pred_strong[mask_fisher]),
            batch_torch_A_to_R(pred_weak[mask_fisher])
        )

        fisher_dict_unsuper = dict(
            unsuper_loss=unsuper_loss,
            entropy=entropy,
            mask_ratio=mask_ratio_fisher,
            err_weakAll_gt=err_weakAll_gt,
            err_weakPseudo_gt=err_weakPseudo_gt,
            err_strongSuper_pseudo=err_strongSuper_pseudo,
        )

        return fisher_dict, fisher_dict_unsuper

    def train_func(self, data, ulb_data):
        """one step of training"""
        self.net.train()
        self.ema_net.train()

        stage2_iter = self.clock.iteration - self.config.stage1_iteration
        self.update_ema_variables(self.config.is_ema, self.config.ema_decay, stage2_iter)

        fisher_dict, fisher_dict_unsuper = self.forward(data, ulb_data)

        SSL_lambda = self.config.SSL_lambda

        loss_all = fisher_dict['loss'] + SSL_lambda * fisher_dict_unsuper['unsuper_loss']

        self.optimizer.zero_grad()
        loss_all.backward()
        self.optimizer.step()


        out_dict = dict(
            SSL_lambda=SSL_lambda,
            loss_all=loss_all
        )

        return fisher_dict, fisher_dict_unsuper, out_dict

    def val_func(self, data, eval_ema=False):
        """one step of validation"""
        self.net.eval()
        self.ema_net.eval()

        with torch.no_grad():
            fisher_dict, fisher_dict_unsuper = self.forward(data, None, eval_ema=eval_ema)

            # mask
            entropy = fisher_entropy(fisher_dict['pred'])

            fisher_mask = entropy < self.config.conf_thres  # (b, )
            fisher_mask_ratio = (fisher_mask.sum() / len(fisher_mask)).item()

            if fisher_mask_ratio > 0:
                # error for pseudo labels
                fisher_err_pseudo_gt = self.compute_err_deg_from_matrices(
                    fisher_dict['pred_orth'][fisher_mask], data.get('rot_mat').cuda()[fisher_mask])
            else:
                fisher_err_pseudo_gt = None

            out_dict = dict(
                mask_ratio=fisher_mask_ratio,
                err_pseudo_gt=fisher_err_pseudo_gt,
            )

            return fisher_dict, fisher_dict_unsuper, out_dict


    def train_func_s1(self, data):
        """supervised training"""
        self.net.train()

        fisher_dict, _ = self.forward(data, None)

        loss = fisher_dict['loss']

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return fisher_dict

    def val_func_s1(self, data):
        """supervised validation"""
        self.net.eval()

        with torch.no_grad():
            fisher_dict, _ = self.forward(data, None)
            return fisher_dict


    def update_ema_variables(self, is_ema, alpha, global_step):
        if is_ema:
            # Use the true average until the exponential average is more correct
            alpha = min(1 - 1 / (global_step + 1), alpha)
        else:
            # ema_param = param if is_ema is False
            alpha = 0
        for ema_param, param in zip(self.ema_net.parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha).add_(param.detach(), alpha=1 - alpha)

    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(self.config.model_dir, "ckpt_iteration{}.pth".format(self.clock.iteration))
            print("[{}/{}] Saving checkpoint iteration {}...".format(self.config.exp_name, self.config.date, self.clock.iteration))
        else:
            save_path = os.path.join(self.config.model_dir, "{}.pth".format(name))
            print("[{}/{}] Saving checkpoint {}...".format(self.config.exp_name, self.config.date, name))

        # self.net
        if isinstance(self.net, nn.DataParallel):
            model_state_dict = self.net.module.cpu().state_dict()
        else:
            model_state_dict = self.net.cpu().state_dict()

        # self.ema_net
        if isinstance(self.ema_net, nn.DataParallel):
            model_state_dict_ema = self.ema_net.module.cpu().state_dict()
        else:
            model_state_dict_ema = self.ema_net.cpu().state_dict()

        torch.save({
            'clock': self.clock.make_checkpoint(),
            'model_state_dict': model_state_dict,
            'model_state_dict_ema': model_state_dict_ema,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)

        self.net.cuda()
        self.ema_net.cuda()

    def load_ckpt(self, name=None):
        """load checkpoint from saved checkpoint"""
        if os.path.isabs(name):
            load_path = name
        else:
            load_path = os.path.join(self.config.model_dir, "{}.pth".format(name))
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
        print("Loading checkpoint from {} ...".format(load_path))
        if isinstance(self.net, nn.DataParallel):
            self.net.module.load_state_dict(checkpoint['model_state_dict'])
            if 'model_state_dict_ema' in checkpoint.keys():
                self.ema_net.module.load_state_dict(checkpoint['model_state_dict_ema'])
        else:
            self.net.load_state_dict(checkpoint['model_state_dict'])
            if 'model_state_dict_ema' in checkpoint.keys():
                self.ema_net.load_state_dict(checkpoint['model_state_dict_ema'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'], )
        self.clock.restore_checkpoint(checkpoint['clock'])

    @staticmethod
    def compute_err_deg_from_quats(pred, gt):
        err_rad = trans.so3_relative_angle(trans.quaternion_to_matrix(pred), trans.quaternion_to_matrix(gt))
        err_deg = torch.rad2deg(err_rad)
        return err_deg

    @staticmethod
    def compute_err_deg_from_matrices(pred, gt):
        err_rad = trans.so3_relative_angle(pred, gt)
        err_deg = torch.rad2deg(err_rad)
        return err_deg

