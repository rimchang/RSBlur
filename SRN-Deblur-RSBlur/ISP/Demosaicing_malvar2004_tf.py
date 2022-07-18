import tensorflow as tf
import numpy as np
from ISP.masks import masks_CFA_Bayer

"""
Malvar (2004) Bayer CFA Demosaicing
===================================
*Bayer* CFA (Colour Filter Array) *Malvar (2004)* demosaicing.
References
----------
-   :cite:`Malvar2004a` : Malvar, H. S., He, L.-W., Cutler, R., & Way, O. M.
    (2004). High-Quality Linear Interpolation for Demosaicing of
    Bayer-Patterned Color Images. In International Conference of Acoustic,
    Speech and Signal Processing (pp. 5-8). Institute of Electrical and
    Electronics Engineers, Inc. Retrieved from
    http://research.microsoft.com/apps/pubs/default.aspx?id=102068
    https://colour-demosaicing.readthedocs.io/en/develop/_modules/colour_demosaicing/bayer/demosaicing/malvar2004.html
"""


def demosaicing_CFA_Bayer_Malvar2004_tf(CFA_inputs, pattern='RGGB', batch_size=4):
    # based on Demosaic of CBDNet (https://github.com/GuoShi28/CBDNet/blob/master/SomeISP_operator_python/Demosaicing_malvar2004.py)

    b, h, w, c = CFA_inputs.get_shape().as_list()

    R_m, G_m, B_m = masks_CFA_Bayer([h, w], pattern)

    R_m_tf = R_m[np.newaxis, :, :, np.newaxis]
    G_m_tf = G_m[np.newaxis, :, :, np.newaxis]
    B_m_tf = B_m[np.newaxis, :, :, np.newaxis]

    GR_GB = np.asarray(
        [[0, 0, -1, 0, 0],
         [0, 0, 2, 0, 0],
         [-1, 2, 4, 2, -1],
         [0, 0, 2, 0, 0],
         [0, 0, -1, 0, 0]]) / 8  # yapf: disable

    GR_GB_tf = tf.constant(np.rot90(GR_GB, k=2), dtype=tf.float32)[:, :, tf.newaxis, tf.newaxis]

    Rg_RB_Bg_BR = np.asarray(
        [[0, 0, 0.5, 0, 0],
         [0, -1, 0, -1, 0],
         [-1, 4, 5, 4, - 1],
         [0, -1, 0, -1, 0],
         [0, 0, 0.5, 0, 0]]) / 8  # yapf: disable
    Rg_RB_Bg_BR_tf = tf.constant(np.rot90(Rg_RB_Bg_BR, k=2), dtype=tf.float32)[:, :, tf.newaxis, tf.newaxis]

    Rg_BR_Bg_RB = np.transpose(Rg_RB_Bg_BR)
    Rg_BR_Bg_RB_tf = tf.constant(np.rot90(Rg_BR_Bg_RB, k=2), dtype=tf.float32)[:, :, tf.newaxis, tf.newaxis]

    Rb_BB_Br_RR = np.asarray(
        [[0, 0, -1.5, 0, 0],
         [0, 2, 0, 2, 0],
         [-1.5, 0, 6, 0, -1.5],
         [0, 2, 0, 2, 0],
         [0, 0, -1.5, 0, 0]]) / 8  # yapf: disable

    Rb_BB_Br_RR_tf = tf.constant(np.rot90(Rb_BB_Br_RR, k=2), dtype=tf.float32)[:, :, tf.newaxis, tf.newaxis]

    R = CFA_inputs * R_m_tf
    G = CFA_inputs * G_m_tf
    B = CFA_inputs * B_m_tf

    GR_GB_result = tf.nn.depthwise_conv2d(
        input=CFA_inputs,
        filter=GR_GB_tf,
        strides=(1, 1, 1, 1),
        padding="SAME",
    )

    Rm_Bm = np.logical_or(R_m, B_m)[np.newaxis, :, :, np.newaxis]
    Rm_Bm = np.tile(Rm_Bm, [batch_size, 1, 1, 1])
    G = tf.where(Rm_Bm, GR_GB_result, G)  # g_r, g_b

    RBg_RBBR = tf.nn.depthwise_conv2d(
        input=CFA_inputs,
        filter=Rg_RB_Bg_BR_tf,
        strides=(1, 1, 1, 1),
        padding="SAME",
    )

    RBg_BRRB = tf.nn.depthwise_conv2d(
        input=CFA_inputs,
        filter=Rg_BR_Bg_RB_tf,
        strides=(1, 1, 1, 1),
        padding="SAME",
    )

    RBgr_BBRR = tf.nn.depthwise_conv2d(
        input=CFA_inputs,
        filter=Rb_BB_Br_RR_tf,
        strides=(1, 1, 1, 1),
        padding="SAME",
    )

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

    Rr_Bc = np.tile(Rr_Bc[np.newaxis, :, :, np.newaxis], [batch_size, 1, 1, 1])
    Br_Rc = np.tile(Br_Rc[np.newaxis, :, :, np.newaxis], [batch_size, 1, 1, 1])

    R = tf.where(Rr_Bc, RBg_RBBR, R)  # r_g1
    R = tf.where(Br_Rc, RBg_BRRB, R)  # r_g2

    Br_Rc = B_r * R_c
    Rr_Bc = R_r * B_c

    Br_Rc = np.tile(Br_Rc[np.newaxis, :, :, np.newaxis], [batch_size, 1, 1, 1])
    Rr_Bc = np.tile(Rr_Bc[np.newaxis, :, :, np.newaxis], [batch_size, 1, 1, 1])

    B = tf.where(Br_Rc, RBg_RBBR, B)  # b_g2
    B = tf.where(Rr_Bc, RBg_BRRB, B)  # b_g1

    Br_Bc = B_r * B_c
    Rr_Rc = R_r * R_c

    Br_Bc = np.tile(Br_Bc[np.newaxis, :, :, np.newaxis], [batch_size, 1, 1, 1])
    Rr_Rc = np.tile(Rr_Rc[np.newaxis, :, :, np.newaxis], [batch_size, 1, 1, 1])

    R = tf.where(Br_Bc, RBgr_BBRR, R)  # r_b
    B = tf.where(Rr_Rc, RBgr_BBRR, B)  # b_r

    new_out = tf.concat([R, G, B], axis=3)

    return new_out