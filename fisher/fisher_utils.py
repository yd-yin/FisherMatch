from os.path import abspath, dirname, join
import sys
sys.path.append(dirname(abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch_norm_factor
import between_bingham_fisher as bbf
import bingham_utils


def vmf_loss(net_out, R, overreg=1.05):
    A = net_out.view(-1, 3, 3)
    loss_v = KL_Fisher(A, R, overreg=overreg)
    Rest = batch_torch_A_to_R(A)
    return loss_v, Rest


def KL_Fisher(A, R, overreg=1.05):
    """
    @param A: (b, 3, 3)
    @param R: (b, 3, 3)
    We find torch.svd() on cpu much faster than that on gpu in our case, so we apply svd operation on cpu.
    """
    A, R = A.cpu(), R.cpu()
    U, S, V = torch.svd(A)
    with torch.no_grad():  # sign can only change if the 3rd component of the svd is 0, then the sign does not matter
        s3sign = torch.det(torch.matmul(U, V.transpose(1, 2)))
    S_sign = torch.cat((S[:, :2], S[:, 2:] * s3sign[:, None]), -1)
    log_normalizer = torch_norm_factor.logC_F(S_sign)
    log_exponent = -torch.matmul(A.view(-1, 1, 9), R.view(-1, 9, 1)).view(-1)
    log_nll = log_exponent + overreg * log_normalizer
    log_nll = log_nll.cuda()
    return log_nll


def batch_torch_A_to_R(A):
    A = A.cpu()
    A = A.reshape(-1, 3, 3)
    U, S, V = torch.svd(A)
    with torch.no_grad():  # sign can only change if the 3rd component of the svd is 0, then the sign does not matter
        s3sign = torch.det(torch.matmul(U, V.transpose(1, 2)))
    U = torch.cat((U[:, :, :2], U[:, :, 2:] * s3sign[:, None][:, None]), -1)
    R = torch.matmul(U, V.transpose(1, 2))
    R = R.cuda()
    return R


def fisher_log_pdf(A, R):
    """
    @param A: (b, 3, 3)
    @param R: (b, 3, 3)
    """
    A, R = A.cpu(), R.cpu()
    U, S, V = torch.svd(A)
    with torch.no_grad():  # sign can only change if the 3rd component of the svd is 0, then the sign does not matter
        s3sign = torch.det(torch.matmul(U, V.transpose(1, 2)))
    S_sign = torch.cat((S[:, :2], S[:, 2:] * s3sign[:, None]), -1)
    log_normalizer = torch_norm_factor.logC_F(S_sign)
    log_exponent = torch.matmul(A.view(-1, 1, 9), R.view(-1, 9, 1)).view(-1)

    logp = -log_normalizer + log_exponent
    logp = logp.cuda()

    return logp


def fisher_entropy(A):
    """
    @param A: (b, 9) or (b, 3, 3)
    @return entropy: (b, )
    """
    A = A.reshape(-1, 3, 3)
    V, Lam = bbf.A_to_V_Lam(A)
    VB, LamB = bbf.convert_bingham_convention(V, Lam)
    # appr F
    entropy = bingham_utils.bingham_entropy(LamB)
    entropy = entropy - torch.tensor([np.log(2 * np.pi**2)]).float().to(entropy.device)
    return entropy


def fisher_CE(A1, A2):
    """
    A1 is gt
    A2 is prediction
    """
    A1 = A1.reshape(-1, 3, 3)
    A2 = A2.reshape(-1, 3, 3)
    V1, Lam1 = bbf.A_to_V_Lam(A1)
    V2, Lam2 = bbf.A_to_V_Lam(A2)
    VB1, LamB1 = bbf.convert_bingham_convention(V1, Lam1)
    VB2, LamB2 = bbf.convert_bingham_convention(V2, Lam2)
    CE = bingham_utils.bingham_CE(VB1, LamB1, VB2, LamB2)
    CE = CE - torch.tensor([np.log(2 * np.pi**2)]).float().to(CE.device)

    assert not torch.isnan(CE).any() and not torch.isinf(CE).any()
    return CE


