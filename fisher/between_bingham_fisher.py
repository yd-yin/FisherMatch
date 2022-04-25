import numpy as np
import torch
from pytorch3d import transforms as trans
import torch_norm_factor


def mu(x):
    X = trans.quaternion_to_matrix(x)
    return X


def mu_inv(X):
    x = trans.matrix_to_quaternion(X)
    return x


def Q(X):
    """
    @param X: (3, 3) or (3)
    """
    X = X.squeeze()
    if X.shape == (3,):
        X = torch.diag(X)
    x = mu_inv(X)
    Q4 = 4 * x[:, None] * x[None] - torch.eye(4)
    return Q4 / 4


def T(F):
    """
    @param F: (3, 3) or (3)
    """
    F = F.squeeze()
    if F.shape == (3,):
        F = torch.diag(F)

    u, S, v = proper_svd(F[None])
    s1, s2, s3 = S.squeeze().cpu()
    u, v = u.squeeze().cpu(), v.squeeze().cpu()

    l1 = s1 - s2 - s3
    l2 = s2 - s1 - s3
    l3 = s3 - s1 - s2
    l4 = -l1 - l2 - l3

    E1 = get_E(0).squeeze()
    E2 = get_E(1).squeeze()
    E3 = get_E(2).squeeze()
    E4 = torch.eye(3, dtype=torch.float32)[None].squeeze()

    m1 = u @ E1 @ v.transpose(0, 1)
    m2 = u @ E2 @ v.transpose(0, 1)
    m3 = u @ E3 @ v.transpose(0, 1)
    m4 = u @ E4 @ v.transpose(0, 1)

    T4 = l1 * Q(m1) + l2 * Q(m2) + l3 * Q(m3) + l4 * Q(m4)

    return T4 / 4


def proper_svd(F):
    """
    F = U * S * VT
    Proper svd guarantees U and V are rotation matrices, i.e., det(U) = det(V) = 1
    The last element of S may be negative if needed, i.e., s1 >= s2 >= |s3| >= 0
    """
    F = F.cpu()
    u1, s1, v1 = torch.svd(F)

    det = torch.det(u1).reshape(-1, 1, 1)
    u = torch.cat((u1[:, :, :-1], u1[:, :, -1:] * det), -1)

    det = torch.det(u1 @ v1).reshape(-1, 1)
    s = torch.cat((s1[:, :-1], s1[:, -1:] * det), -1)

    det = torch.det(v1).reshape(-1, 1, 1)
    v = torch.cat((v1[:, :, :-1], v1[:, :, -1:] * det), -1)

    u, s, v = u.cuda(), s.cuda(), v.cuda()
    return u, s, v


def S_to_Lam(S):
    """
    Convert `S` in matrix Fisher to `Lambda` in Bingham with `Fisher convention`
    """
    s1 = S[:, 0]
    s2 = S[:, 1]
    s3 = S[:, 2]
    l1 = s1 - s2 - s3
    l2 = s2 - s1 - s3
    l3 = s3 - s1 - s2
    l4 = -l1 - l2 - l3
    Lam = torch.stack((l1, l2, l3, l4), 1)
    return Lam


def get_E(t):
    et = torch.zeros((3, 1), dtype=torch.float32)[None]
    et[:, t] += 1
    Et = 2 * torch.bmm(et, et.transpose(1, 2)) - torch.eye(3)[None]
    return Et


def A_to_V_Lam(A):
    """
    Convert `A` in matrix Fisher to `V` and `Lambda` in Bingham with `Fisher convention`
    @param A: (b, 3, 3)
    @return V: (b, 4, 4)
    @return Lam: (b, 3)
    """
    # svd to F
    u, s, v = proper_svd(A)
    Lam = S_to_Lam(s)

    E1 = get_E(0).to(A.device)
    E2 = get_E(1).to(A.device)
    E3 = get_E(2).to(A.device)
    E4 = torch.eye(3, dtype=torch.float32)[None].to(A.device)

    m1 = u @ E1 @ v.transpose(1, 2)
    m2 = u @ E2 @ v.transpose(1, 2)
    m3 = u @ E3 @ v.transpose(1, 2)
    m4 = u @ E4 @ v.transpose(1, 2)

    a1 = mu_inv(m1)
    a2 = mu_inv(m2)
    a3 = mu_inv(m3)
    a4 = mu_inv(m4)

    V = torch.stack((a1, a2, a3, a4), 2)

    return V, Lam


def convert_bingham_convention(V, Lam):
    """
    Convert to `Bingham convention`
    """
    c = -Lam.max(1)[0]

    Lam = Lam + c.unsqueeze(-1)
    Lam, order = Lam.sort(descending=True)

    if V is None:
        return Lam

    V = torch.gather(V, -1, order[:, None, :].repeat(1, V.shape[1], 1))

    return V, Lam


def ensure_bingham_convention(LamB):
    assert LamB.shape[1] == 4 or LamB.shape[1] == 3, 'LamB.shape[1] should be 3 or 4.'
    if LamB.shape[1] == 4:
        assert torch.allclose(LamB[:, 0], torch.zeros_like(LamB[:, 0]))
    else:
        LamB = torch.cat((torch.zeros_like(LamB[:, [0]]), LamB), -1)
    return LamB


def Lam_to_S(Lam):
    """
    `Lambda` in Bingham with `Fisher convention` to `S` in matrix Fisher
    """
    l1 = Lam[:, 0]
    l2 = Lam[:, 1]
    l3 = Lam[:, 2]
    l4 = Lam[:, 3]

    s1 = 1/4 * (l1 - l2 - l3 + l4)
    s2 = 1/4 * (- l1 + l2 - l3 + l4)
    s3 = 1/4 * (- l1 - l2 + l3 + l4)

    S = torch.stack((s1, s2, s3), -1)
    return S


def LamB_to_S(LamB):
    """
    `Lambda` in Bingham with `Bingham convention` to `S` in matrix Fisher
    """
    S1 = Lam_to_S(LamB)
    S = torch.abs(S1)
    S = S.sort(descending=True)[0]

    sign = torch.sign(S1[:, 0] * S1[:, 1] * S1[:, 2])
    S[:, -1] *= sign
    return S


def constant_bingham_appr_fromS(S, c):
    """
    Given the corresponding `S`, compute the normalization constant of Bingham
    """
    yFy = torch.exp(torch_norm_factor.logC_F(S) + c)
    const = 2 * np.pi ** 2 * yFy
    return const
