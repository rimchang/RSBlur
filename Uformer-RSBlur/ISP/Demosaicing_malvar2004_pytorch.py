import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from ISP.masks import masks_CFA_Bayer

class Demosaic(nn.Module):
    # based on https://github.com/GuoShi28/CBDNet/blob/master/SomeISP_operator_python/Demosaicing_malvar2004.py

    def __init__(self):
        super(Demosaic, self).__init__()

        GR_GB = np.asarray(
            [[0, 0, -1, 0, 0],
             [0, 0, 2, 0, 0],
             [-1, 2, 4, 2, -1],
             [0, 0, 2, 0, 0],
             [0, 0, -1, 0, 0]]) / 8  # yapf: disable

        # [5,5] => rot90 => [1, 1, 5, 5]
        self.GR_GB_pt = torch.tensor(np.rot90(GR_GB, k=2).copy(), dtype=torch.float32)

        Rg_RB_Bg_BR = np.asarray(
            [[0, 0, 0.5, 0, 0],
             [0, -1, 0, -1, 0],
             [-1, 4, 5, 4, - 1],
             [0, -1, 0, -1, 0],
             [0, 0, 0.5, 0, 0]]) / 8  # yapf: disable
        self.Rg_RB_Bg_BR_pt = torch.tensor(np.rot90(Rg_RB_Bg_BR, k=2).copy(), dtype=torch.float32)

        Rg_BR_Bg_RB = np.transpose(Rg_RB_Bg_BR)
        self.Rg_BR_Bg_RB_pt = torch.tensor(np.rot90(Rg_BR_Bg_RB, k=2).copy(), dtype=torch.float32)

        Rb_BB_Br_RR = np.asarray(
            [[0, 0, -1.5, 0, 0],
             [0, 2, 0, 2, 0],
             [-1.5, 0, 6, 0, -1.5],
             [0, 2, 0, 2, 0],
             [0, 0, -1.5, 0, 0]]) / 8  # yapf: disable

        self.Rb_BB_Br_RR_pt = torch.tensor(np.rot90(Rb_BB_Br_RR, k=2).copy(), dtype=torch.float32)


    def cuda(self, device=None):
        self.GR_GB_pt = self.GR_GB_pt.cuda(device)
        self.Rg_RB_Bg_BR_pt = self.Rg_RB_Bg_BR_pt.cuda(device)
        self.Rg_BR_Bg_RB_pt = self.Rg_BR_Bg_RB_pt.cuda(device)
        self.Rb_BB_Br_RR_pt = self.Rb_BB_Br_RR_pt.cuda(device)


    def forward(self, CFA_inputs, pattern='RGGB'):
        batch_size, c, h, w = CFA_inputs.shape

        R_m, G_m, B_m = masks_CFA_Bayer([h, w], pattern)

        # CFA mask
        R_m_pt = torch.from_numpy(R_m[np.newaxis, np.newaxis, :, :]).to(CFA_inputs.device)
        G_m_pt = torch.from_numpy(G_m[np.newaxis, np.newaxis, :, :]).to(CFA_inputs.device)
        B_m_pt = torch.from_numpy(B_m[np.newaxis, np.newaxis, :, :]).to(CFA_inputs.device)

        R = CFA_inputs * R_m_pt
        G = CFA_inputs * G_m_pt
        B = CFA_inputs * B_m_pt

        # True : GR_GB, False : G
        GR_GB_result = F.conv2d(CFA_inputs, weight=self.GR_GB_pt.expand(1, 1, -1, -1), padding=2, groups=1)
        Rm_Bm = np.logical_or(R_m, B_m)[np.newaxis, np.newaxis, :, :]
        Rm_Bm = np.tile(Rm_Bm, [batch_size, 1, 1, 1])
        Rm_Bm_pt = torch.tensor(Rm_Bm.copy(), dtype=torch.bool).to(CFA_inputs.device)
        G = GR_GB_result * Rm_Bm_pt + G * torch.logical_not(Rm_Bm_pt)

        RBg_RBBR = F.conv2d(CFA_inputs, weight=self.Rg_RB_Bg_BR_pt.expand(1, 1, -1, -1), padding=2, groups=1)
        RBg_BRRB = F.conv2d(CFA_inputs, weight=self.Rg_BR_Bg_RB_pt.expand(1, 1, -1, -1), padding=2, groups=1)
        RBgr_BBRR = F.conv2d(CFA_inputs, weight=self.Rb_BB_Br_RR_pt.expand(1, 1, -1, -1), padding=2, groups=1)

        # Red rows.
        R_r = np.transpose(np.any(R_m == 1, axis=1)[np.newaxis]) * np.ones(R_m.shape)
        # Red columns.
        R_c = np.any(R_m == 1, axis=0)[np.newaxis] * np.ones(R_m.shape)
        # Blue rows.
        B_r = np.transpose(np.any(B_m == 1, axis=1)[np.newaxis]) * np.ones(B_m.shape)
        # Blue columns
        B_c = np.any(B_m == 1, axis=0)[np.newaxis] * np.ones(B_m.shape)

        # rg1g2b
        Rr_Bc = R_r * B_c
        Br_Rc = B_r * R_c

        Rr_Bc = np.tile(Rr_Bc[np.newaxis, np.newaxis, :, :], [batch_size, 1, 1, 1])
        Br_Rc = np.tile(Br_Rc[np.newaxis, np.newaxis, :, :], [batch_size, 1, 1, 1])
        Rr_Bc_pt = torch.tensor(Rr_Bc.copy(), dtype=torch.bool).to(CFA_inputs.device)
        Br_Rc_pt = torch.tensor(Br_Rc.copy(), dtype=torch.bool).to(CFA_inputs.device)

        R = RBg_RBBR * Rr_Bc_pt + R * torch.logical_not(Rr_Bc_pt)
        R = RBg_BRRB * Br_Rc_pt + R * torch.logical_not(Br_Rc_pt)

        Br_Rc = B_r * R_c
        Rr_Bc = R_r * B_c

        Br_Rc = np.tile(Br_Rc[np.newaxis, np.newaxis, :, :], [batch_size, 1, 1, 1])
        Rr_Bc = np.tile(Rr_Bc[np.newaxis, np.newaxis, :, :], [batch_size, 1, 1, 1])
        Br_Rc_pt = torch.tensor(Br_Rc.copy(), dtype=torch.bool).to(CFA_inputs.device)
        Rr_Bc_pt = torch.tensor(Rr_Bc.copy(), dtype=torch.bool).to(CFA_inputs.device)

        B = RBg_RBBR * Br_Rc_pt + B * torch.logical_not(Br_Rc_pt)
        B = RBg_BRRB * Rr_Bc_pt + B * torch.logical_not(Rr_Bc_pt)

        Br_Bc = B_r * B_c
        Rr_Rc = R_r * R_c

        Br_Bc = np.tile(Br_Bc[np.newaxis, np.newaxis, :, :], [batch_size, 1, 1, 1])
        Rr_Rc = np.tile(Rr_Rc[np.newaxis, np.newaxis, :, :], [batch_size, 1, 1, 1])
        Br_Bc_pt = torch.tensor(Br_Bc.copy(), dtype=torch.bool).to(CFA_inputs.device)
        Rr_Rc_pt = torch.tensor(Rr_Rc.copy(), dtype=torch.bool).to(CFA_inputs.device)

        R = RBgr_BBRR * Br_Bc_pt + R * torch.logical_not(Br_Bc_pt)
        B = RBgr_BBRR * Rr_Rc_pt + B * torch.logical_not(Rr_Rc_pt)

        new_out = torch.cat([R, G, B], dim=1)

        return new_out