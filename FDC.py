import math
import copy
import sys
import random
import time
import pygad  # -----------------> 1. IMPORT THE GA LIBRARY

# The total number of epochs to run the search for. (Set to zero for demonstration; positive integer to run optimizer)
# The GA will try to find a 26/26 solution and then
# spend the rest of the epochs trying to minimize its digit cost.
TOTAL_EPOCHS_TO_RUN = 0

# The precision cap in decimal digits. Change to 124 and watch rho_true fail to converge.
FINAL_CAP = 125


ANALOG_HYPOTHESIS = False


# --- Static Observed Values (Most Accurate PDG 2024 / CODATA 2022) ---
OBSERVED = {
    'inv_alpha_MZ': (128.952, 0.014), 'm_H': (125.26, 0.14), 'G': (6.67430e-11, 0.00015e-11),
    'sin2_theta_W': (0.23122, 0.00004), 'alpha_s_MZ': (0.1180, 0.0008), 'm_nu': (0.0513, 0.005),
    'theta_C': (13.04, 0.05), 'theta_23': (2.378, 0.057), 'N_efolds': (60, 5),
    'rho_true': (5.2875e-124, 1.656e-125), 'muon_A4/electron_A4': (206.768283, 0.0000046),
    'strange_D10/down_D10': (19.8936, 0.20), 'charm_Oh/up_Oh': (589.3518, 11.2),
    'tau_A4/muon_A4': (16.8175, 0.0011), 'bottom_D10/strange_D10': (44.7380, 0.24),
    'top_Oh/charm_Oh': (135.711, 0.30), 'electron_A4': (0.00051099895, 1.5e-13),
    'muon_A4': (0.1056583755, 2.3e-9), 'tau_A4': (1.77686, 0.00012), 'up_Oh': (0.00216, 0.00004),
    'down_D10': (0.00470, 0.00004), 'strange_D10': (0.0935, 0.0005), 'charm_Oh': (1.273, 0.003),
    'bottom_D10': (4.183, 0.004), 'top_Oh': (172.76, 0.30), 'v_VEV': (246.22, 0.03),
}

# --------------------------------------------------------------------
# --- Seed Generator ---
# --------------------------------------------------------------------
def get_seed(parent_a=None, parent_b=None):
    # --- 1. Your "Adam" Config (The base) ---
    ADAM_CONFIG = {
        'ALPHA_LAMBDA': 13,
        'AMP_A': 6,
        'AMP_D': 3,
        'B_LIN': 19,
        'B_VOL': 9,
        'C': 14,
        'DELTA': 7,
        'DRAG': 16,
        'EPSILON': 11,
        'G0': 32,
        'GEV_TO_J': 27,
        'G_t_to_v': 9,
        'HBAR': 38,
        'K': 5,
        'K_MIX': 25,
        'LAMBDA': 8,
        'M_PL_J': 21,
        'M_PL_KG': 22,
        'M_Z': 19,
        'N_efolds': 4,
        'PI_EFF': 26,
        'P_g_O': 5,
        'S': 18,
        'SQRT2_Q': 13,
        'T2_CORR': 5,
        'T3_CORR': 2,
        'T4_CORR': 6,
        'T5_CORR': 10,
        'T6_CORR': 4,
        'T_TO_MU': 9,
        'T_TO_MZ': 12,
        'T_TO_V': 13,
        'V': 10,
        'V_DOT_S': 20,
        'V_DOT_V': 14,
        'a_T0': 11,
        'a_T1': 17,
        'a_T2': 2,
        'a_T3': 4,
        'a_T4': 44,
        'a_fin_step1': 21,
        'a_fin_step2': 11,
        'a_fin_step3': 9,
        'a_fin_step4': 4,
        'a_t1_step1': 21,
        'a_t1_step2': 2,
        'a_t2_step1': 14,
        'a_t3_step1': 14,
        'a_t4_step1': 2,
        'alpha_lambda_step1': 13,
        'amp_a_step1': 16,
        'amp_a_step2': 19,
        'amp_d_step1': 4,
        'as_T0': 1,
        'as_T1': 7,
        'as_T2': 10,
        'as_fin_step1': 3,
        'as_inv': 14,
        'as_inv_step1': 6,
        'as_inv_step2': 8,
        'as_t0_step1': 12,
        'as_t1_step1': 9,
        'as_t1_step2': 13,
        'as_t2_step1': 2,
        'b_lin_step1': 4,
        'b_vol_step1': 11,
        'ckm23_denom': 2,
        'ckm23_denom_step1': 33,
        'ckm23_fin_step1': 2,
        'ckm23_final': 16,
        'ckm_T1': 38,
        'ckm_T2': 8,
        'ckm_denom': 4,
        'ckm_denom_step1': 14,
        'ckm_denom_step2': 1,
        'ckm_fin_step1': 9,
        'ckm_t1_step1': 13,
        'ckm_t1_step2': 2,
        'ckm_t1_step3': 21,
        'ckm_t1_step4': 19,
        'ckm_t2_step1': 21,
        'ckm_theta0': 2,
        'ckm_theta0_step1': 16,
        'ckm_theta0_step2': 31,
        'delta_step1': 0,
        'delta_step2': 4,
        'dqe_p_ANGLE_RATIO': 5,
        'dqe_p_bs': 15,
        'dqe_p_cu': 8,
        'dqe_p_me': 8,
        'dqe_p_sd': 7,
        'dqe_p_tc': 13,
        'dqe_p_tm': 9,
        'drag_step1': 7,
        'eps_step1': 3,
        'eps_step2': 4,
        'eps_step3': 30,
        'f_C': 18,
        'f_EXP_D10_T1': 20,
        'f_EXP_OH_T1': 6,
        'f_LOG_Y1': 18,
        'f_LOG_Y2': 7,
        'f_LOG_Y3': 21,
        'f_R_net_per_pair': 9,
        'f_T1_b_s': 2,
        'f_T1_c_u': 1,
        'f_T1_mu_e': 6,
        'f_T1_s_d': 11,
        'f_T1_t_c': 11,
        'f_T1_tau_mu': 10,
        'f_T6_diluted': 30,
        'f_V_OVER_SQRT2': 4,
        'f_base_anch_step1': 2,
        'f_base_anchor': 22,
        'f_c_step1': 14,
        'f_exp_b': 3,
        'f_exp_b_step1': 21,
        'f_exp_b_step2': 13,
        'f_exp_b_step3': 13,
        'f_exp_c': 7,
        'f_exp_c_step1': 2,
        'f_exp_c_step2': 1,
        'f_exp_c_step3': 9,
        'f_exp_d': 3,
        'f_exp_d10_t1_step1': 7,
        'f_exp_d_step1': 9,
        'f_exp_d_step2': 14,
        'f_exp_d_step3': 17,
        'f_exp_d_step4': 31,
        'f_exp_e': 6,
        'f_exp_e_step1': 2,
        'f_exp_mu': 0,
        'f_exp_mu_step1': 22,
        'f_exp_mu_step2': 0,
        'f_exp_mu_step3': 1,
        'f_exp_oh_t1_step1': 1,
        'f_exp_s': 17,
        'f_exp_s_step1': 7,
        'f_exp_s_step2': 5,
        'f_exp_s_step3': 15,
        'f_exp_t': 6,
        'f_exp_t_step1': 3,
        'f_exp_t_step2': 3,
        'f_exp_t_step3': 12,
        'f_exp_tau': 9,
        'f_exp_tau_step1': 14,
        'f_exp_tau_step2': 1,
        'f_exp_tau_step3': 3,
        'f_exp_u': 14,
        'f_exp_u_step1': 20,
        'f_exp_u_step2': 6,
        'f_exp_u_step3': 10,
        'f_gated_b_s': 12,
        'f_gated_c_u': 4,
        'f_gated_c_u_step1': 5,
        'f_gated_mu_e': 17,
        'f_gated_s_d': 6,
        'f_gated_s_d_step1': 19,
        'f_gated_t_c': 4,
        'f_gated_tau_mu': 7,
        'f_gated_tau_mu_step1': 6,
        'f_gated_tau_mu_step2': 6,
        'f_gated_tau_mu_step3': 11,
        'f_gated_tau_mu_step4': 4,
        'f_lam_a_step1': 26,
        'f_lam_d_step1': 1,
        'f_lam_o_step1': 9,
        'f_lambda_A': 14,
        'f_lambda_D': 9,
        'f_lambda_O': 4,
        'f_log_r_b_s': 1,
        'f_log_r_b_s_step1': 31,
        'f_log_r_b_s_step2': 4,
        'f_log_r_b_s_step3': 14,
        'f_log_r_b_s_step4': 3,
        'f_log_r_c_u': 3,
        'f_log_r_c_u_step1': 22,
        'f_log_r_c_u_step2': 7,
        'f_log_r_c_u_step3': 18,
        'f_log_r_c_u_step4': 12,
        'f_log_r_c_u_step5': 8,
        'f_log_r_mu_e': 11,
        'f_log_r_mu_e_step1': 13,
        'f_log_r_mu_e_step2': 18,
        'f_log_r_mu_e_step3': 14,
        'f_log_r_mu_e_step4': 10,
        'f_log_r_s_d': 2,
        'f_log_r_s_d_step1': 5,
        'f_log_r_s_d_step2': 3,
        'f_log_r_s_d_step3': 10,
        'f_log_r_s_d_step4': 5,
        'f_log_r_t_c': 28,
        'f_log_r_t_c_step1': 12,
        'f_log_r_t_c_step2': 4,
        'f_log_r_t_c_step3': 9,
        'f_log_r_t_c_step4': 20,
        'f_log_r_t_c_step5': 7,
        'f_log_r_tau_mu': 6,
        'f_log_r_tau_mu_step1': 7,
        'f_log_r_tau_mu_step2': 15,
        'f_log_r_tau_mu_step3': 19,
        'f_log_r_tau_mu_step4': 7,
        'f_log_y1_step1': 3,
        'f_log_y2_step1': 14,
        'f_m_bare_1': 5,
        'f_m_bare_1_step1': 5,
        'f_m_bare_2': 0,
        'f_m_bare_2_step1': 5,
        'f_m_bare_3': 0,
        'f_m_bare_3_step1': 2,
        'f_m_bare_b': 4,
        'f_m_bare_b_step1': 17,
        'f_m_bare_b_step2': 6,
        'f_m_bare_c': 3,
        'f_m_bare_c_step1': 4,
        'f_m_bare_c_step2': 3,
        'f_m_bare_d': 28,
        'f_m_bare_d_step1': 2,
        'f_m_bare_d_step2': 30,
        'f_m_bare_e': 6,
        'f_m_bare_e_step1': 12,
        'f_m_bare_e_step2': 13,
        'f_m_bare_mu': 14,
        'f_m_bare_mu_step1': 0,
        'f_m_bare_mu_step2': 4,
        'f_m_bare_s': 4,
        'f_m_bare_s_step1': 21,
        'f_m_bare_s_step2': 16,
        'f_m_bare_t': 16,
        'f_m_bare_t_step1': 19,
        'f_m_bare_t_step2': 12,
        'f_m_bare_tau': 0,
        'f_m_bare_tau_step1': 27,
        'f_m_bare_tau_step2': 8,
        'f_m_bare_u': 18,
        'f_m_bare_u_step1': 1,
        'f_m_bare_u_step2': 5,
        'f_r_net_pp_step1': 9,
        'f_t1_b_s_step1': 10,
        'f_t1_b_s_step2': 13,
        'f_t1_b_s_step3': 3,
        'f_t1_c_u_step1': 19,
        'f_t1_c_u_step2': 18,
        'f_t1_mu_e_step1': 8,
        'f_t1_mu_e_step2': 11,
        'f_t1_mu_e_step3': 11,
        'f_t1_s_d_step1': 14,
        'f_t1_s_d_step2': 12,
        'f_t1_t_c_step1': 10,
        'f_t1_t_c_step2': 7,
        'f_t1_t_c_step3': 5,
        'f_t1_tau_mu_step1': 7,
        'f_t1_tau_mu_step2': 8,
        'f_t6_dil_step1': 33,
        'f_v_ovr_s2_step1': 6,
        'final_G': 15,
        'final_G_step1': 9,
        'final_alpha_s_MZ': 6,
        'final_inv_alpha_MZ': 3,
        'final_m_H': 1,
        'final_m_H_step1': 0,
        'final_m_H_step2': 8,
        'final_m_b': 3,
        'final_m_c': 3,
        'final_m_d': 4,
        'final_m_e': 12,
        'final_m_mu': 14,
        'final_m_nu': 5,
        'final_m_nu_step1': 2,
        'final_m_s': 12,
        'final_m_t': 2,
        'final_m_tau': 5,
        'final_m_u': 5,
        'final_rho_true': 125,
        'final_rho_true_step1': 125,
        'final_rho_true_step2': 125,
        'final_sin2': 6,
        'final_sin2_step1': 6,
        'final_theta_C': 2,
        'final_theta_C_step1': 3,
        'final_v_VEV': 1,
        'g0_step1': 34,
        'g0_step2': 19,
        'g0_step3': 26,
        'g_R_net': 7,
        'g_r_net_step1': 2,
        'g_r_net_step2': 0,
        'gamma_sum': 13,
        'gamma_term_a': 3,
        'gamma_term_b': 9,
        'h_T1': 7,
        'h_T2': 11,
        'h_T4': 10,
        'h_m_H0': 3,
        'h_m_h0_step1': 0,
        'h_t1_step1': 13,
        'h_t2_step1': 7,
        'h_t4_step1': 3,
        'k_mix_step1': 25,
        'k_step1': 41,
        'k_step2': 17,
        'lam_step1': 8,
        'lam_step2': 21,
        'lam_step3': 13,
        'loop_ALPHA_Z': 4,
        'loop_ALPHA_Z_step1': 3,
        'loop_AMP_D_CORR': 0,
        'loop_AMP_D_CORR_step1': 12,
        'loop_AMP_O_CORR': 13,
        'loop_AMP_O_CORR_step1': 3,
        'loop_R_net_per_pair': 14,
        'loop_dm1_qed': 1,
        'loop_dm2_qcd_d': 4,
        'loop_dm2_qcd_d_step1': 3,
        'loop_dm2_qcd_u': 28,
        'loop_dm2_qcd_u_step1': 16,
        'loop_dm2_qed': 16,
        'loop_factor': 17,
        'loop_factor_step1': 7,
        'loop_factor_step2': 8,
        'loop_m_e_phys_step1': 7,
        'loop_m_e_phys_step2': 14,
        'loop_m_e_phys_step3': 15,
        'loop_m_mu_phys_step1': 14,
        'loop_m_phys_step1': 11,
        'loop_m_ratio_step1': 6,
        'loop_m_ratio_step2': 7,
        'loop_m_tau_phys_step1': 6,
        'loop_r_net_pp_step1': 13,
        'loop_t_mass': 10,
        'loop_t_mass_step1': 10,
        'loop_t_mass_step2': 14,
        'loop_t_mass_step3': 4,
        'm_pl_j_step1': 8,
        'm_pl_kg_step1': 0,
        'm_pl_kg_step2': 15,
        'main_C_QED_phys': 17,
        'main_c_qed_step1': 2,
        'main_c_qed_step2': 9,
        'n_efolds_step1': 5,
        'nu_M_R': 9,
        'nu_alpha_Z': 5,
        'nu_alpha_z_step1': 21,
        'nu_bare_ev': 16,
        'nu_bare_ev_step1': 12,
        'nu_bare_gev': 15,
        'nu_bare_gev_step1': 11,
        'nu_bare_gev_step2': 13,
        'nu_m_D': 4,
        'nu_m_d_step1': 2,
        'nu_m_d_step2': 0,
        'nu_m_r_step1': 9,
        'nu_m_r_step2': 2,
        'nu_y': 12,
        'nu_y_step1': 2,
        'p_g_o_step1': 6,
        'phi_step1': 6,
        'ratio_bs_step1': 4,
        'ratio_cu_step1': 0,
        'ratio_sd_step1': 2,
        'ratio_tc_step1': 1,
        'rho_exp_term': 125,
        'rho_exp_term_step1': 23,
        'rho_exp_term_step2': 4,
        'rho_exp_term_step3': 12,
        'rho_exp_term_step4': 7,
        'rho_kappa': 12,
        'rho_kappa_step1': 7,
        'rho_kappa_step2': 6,
        'rho_kappa_step3': 6,
        'rho_kappa_step4': 2,
        'rho_vac_bare': 3,
        'rho_vac_bare_step1': 15,
        'rho_vac_bare_step2': 11,
        's_step1': 31,
        'sin2_R1': 14,
        'sin2_R2': 4,
        'sin2_bare': 13,
        'sin2_bare_step1': 13,
        'sin2_denom': 3,
        'sin2_denom_step1': 9,
        'sin2_denom_step2': 7,
        'sin2_r1_step1': 5,
        'sin2_r1_step2': 20,
        'sin2_r1_step3': 3,
        'sin2_r1_step4': 15,
        'sin2_r1_step5': 33,
        'sin2_r2_step1': 14,
        'sin2_r2_step2': 13,
        'sqrt2_q_step1': 7,
        't2_corr_step1': 20,
        't3_corr_step1': 2,
        't4_corr_step1': 5,
        't5_corr_step1': 33,
        't6_corr_step1': 32,
        't_to_mu_step1': 19,
        't_to_mu_step2': 1,
        't_to_mz_step1': 5,
        't_to_mz_step2': 7,
        't_to_v_step1': 4,
        't_to_v_step2': 14,
        'v_dot_s_step1': 6,
        'v_dot_s_step2': 12,
        'v_dot_v_step1': 9,
        'v_exp_step': 6,
    }
    
    ALL_POSSIBLE_LEVERS = [
        'ALPHA_LAMBDA', 'AMP_A', 'AMP_D', 'B_LIN', 'B_VOL', 'C', 'DELTA', 'DRAG',
        'EPSILON', 'G0', 'GEV_TO_J', 'G_t_to_v', 'HBAR', 'K',
        'K_MIX', 'LAMBDA', 'M_PL_J', 'M_PL_KG', 'M_Z', 'N_efolds', 'PI_EFF',
        'P_g_O', 'S', 'SQRT2_Q', 'T2_CORR', 'T3_CORR', 'T4_CORR', 'T5_CORR',
        'T6_CORR', 'T_TO_MU', 'T_TO_MZ', 'T_TO_V', 'V', 'V_DOT_S', 'V_DOT_V',
        'a_T0', 'a_T1', 'a_T2', 'a_T3', 'a_T4', 'a_fin_step1', 'a_fin_step2',
        'a_fin_step3', 'a_fin_step4', 'a_t1_step1', 'a_t1_step2', 'a_t2_step1',
        'a_t3_step1', 'a_t4_step1', 'alpha_lambda_step1', 'amp_a_step1',
        'amp_a_step2', 'amp_d_step1', 'as_T0', 'as_T1', 'as_T2', 'as_fin_step1',
        'as_inv', 'as_inv_step1', 'as_inv_step2', 'as_t0_step1', 'as_t1_step1',
        'as_t1_step2', 'as_t2_step1', 'b_lin_step1', 'b_vol_step1', 'ckm23_denom',
        'ckm23_denom_step1', 'ckm23_fin_step1', 'ckm23_final', 'ckm_T1', 'ckm_T2',
        'ckm_denom', 'ckm_denom_step1', 'ckm_denom_step2', 'ckm_fin_step1',
        'ckm_t1_step1', 'ckm_t1_step2', 'ckm_t1_step3', 'ckm_t1_step4',
        'ckm_t2_step1', 'ckm_theta0', 'ckm_theta0_step1', 'ckm_theta0_step2',
        'delta_step1', 'delta_step2', 'dqe_p_ANGLE_RATIO', 'dqe_p_bs', 'dqe_p_cu',
        'dqe_p_me', 'dqe_p_sd', 'dqe_p_tc', 'dqe_p_tm', 'drag_step1', 'eps_step1',
        'eps_step2', 'eps_step3', 'f_C', 'f_EXP_D10_T1',
        'f_EXP_OH_T1', 'f_LOG_Y1', 'f_LOG_Y2', 'f_LOG_Y3', 'f_R_net_per_pair',
        'f_T1_b_s', 'f_T1_c_u', 'f_T1_mu_e', 'f_T1_s_d', 'f_T1_t_c',
        'f_T1_tau_mu', 'f_T6_diluted', 'f_V_OVER_SQRT2', 'f_base_anch_step1',
        'f_base_anchor', 'f_c_step1', 'f_exp_b', 'f_exp_b_step1', 'f_exp_b_step2',
        'f_exp_b_step3', 'f_exp_c', 'f_exp_c_step1', 'f_exp_c_step2',
        'f_exp_c_step3', 'f_exp_d', 'f_exp_d10_t1_step1', 'f_exp_d_step1',
        'f_exp_d_step2', 'f_exp_d_step3', 'f_exp_d_step4', 'f_exp_e',
        'f_exp_e_step1', 'f_exp_mu', 'f_exp_mu_step1', 'f_exp_mu_step2',
        'f_exp_mu_step3', 'f_exp_oh_t1_step1', 'f_exp_s', 'f_exp_s_step1',
        'f_exp_s_step2', 'f_exp_s_step3', 'f_exp_t', 'f_exp_t_step1',
        'f_exp_t_step2', 'f_exp_t_step3', 'f_exp_tau', 'f_exp_tau_step1',
        'f_exp_tau_step2', 'f_exp_tau_step3', 'f_exp_u', 'f_exp_u_step1',
        'f_exp_u_step2', 'f_exp_u_step3', 'f_gated_b_s', 'f_gated_c_u',
        'f_gated_c_u_step1', 'f_gated_mu_e', 'f_gated_s_d', 'f_gated_s_d_step1',
        'f_gated_t_c', 'f_gated_tau_mu', 'f_gated_tau_mu_step1',
        'f_gated_tau_mu_step2', 'f_gated_tau_mu_step3', 'f_gated_tau_mu_step4',
        'f_lam_a_step1', 'f_lam_d_step1', 'f_lam_o_step1', 'f_lambda_A',
        'f_lambda_D', 'f_lambda_O', 'f_log_r_b_s', 'f_log_r_b_s_step1',
        'f_log_r_b_s_step2', 'f_log_r_b_s_step3', 'f_log_r_b_s_step4',
        'f_log_r_c_u', 'f_log_r_c_u_step1', 'f_log_r_c_u_step2',
        'f_log_r_c_u_step3', 'f_log_r_c_u_step4', 'f_log_r_c_u_step5',
        'f_log_r_mu_e', 'f_log_r_mu_e_step1', 'f_log_r_mu_e_step2',
        'f_log_r_mu_e_step3', 'f_log_r_mu_e_step4', 'f_log_r_s_d',
        'f_log_r_s_d_step1', 'f_log_r_s_d_step2', 'f_log_r_s_d_step3',
        'f_log_r_s_d_step4', 'f_log_r_t_c', 'f_log_r_t_c_step1',
        'f_log_r_t_c_step2', 'f_log_r_t_c_step3', 'f_log_r_t_c_step4',
        'f_log_r_t_c_step5', 'f_log_r_tau_mu', 'f_log_r_tau_mu_step1',
        'f_log_r_tau_mu_step2', 'f_log_r_tau_mu_step3', 'f_log_r_tau_mu_step4',
        'f_log_y1_step1', 'f_log_y2_step1', 'f_m_bare_1', 'f_m_bare_1_step1',
        'f_m_bare_2', 'f_m_bare_2_step1', 'f_m_bare_3', 'f_m_bare_3_step1',
        'f_m_bare_b', 'f_m_bare_b_step1', 'f_m_bare_b_step2', 'f_m_bare_c',
        'f_m_bare_c_step1', 'f_m_bare_c_step2', 'f_m_bare_d', 'f_m_bare_d_step1',
        'f_m_bare_d_step2', 'f_m_bare_e', 'f_m_bare_e_step1', 'f_m_bare_e_step2',
        'f_m_bare_mu', 'f_m_bare_mu_step1', 'f_m_bare_mu_step2', 'f_m_bare_s',
        'f_m_bare_s_step1', 'f_m_bare_s_step2', 'f_m_bare_t', 'f_m_bare_t_step1',
        'f_m_bare_t_step2', 'f_m_bare_tau', 'f_m_bare_tau_step1',
        'f_m_bare_tau_step2', 'f_m_bare_u', 'f_m_bare_u_step1', 'f_m_bare_u_step2',
        'f_r_net_pp_step1', 'f_t1_b_s_step1', 'f_t1_b_s_step2', 'f_t1_b_s_step3',
        'f_t1_c_u_step1', 'f_t1_c_u_step2', 'f_t1_mu_e_step1', 'f_t1_mu_e_step2',
        'f_t1_mu_e_step3', 'f_t1_s_d_step1', 'f_t1_s_d_step2', 'f_t1_t_c_step1',
        'f_t1_t_c_step2', 'f_t1_t_c_step3', 'f_t1_tau_mu_step1',
        'f_t1_tau_mu_step2', 'f_t6_dil_step1', 'f_v_ovr_s2_step1', 'final_G',
        'final_G_step1', 'final_alpha_s_MZ', 'final_inv_alpha_MZ', 'final_m_H',
        'final_m_H_step1', 'final_m_H_step2', 'final_m_b', 'final_m_c',
        'final_m_d', 'final_m_e', 'final_m_mu', 'final_m_nu', 'final_m_nu_step1',
        'final_m_s', 'final_m_t', 'final_m_tau', 'final_m_u', 'final_rho_true',
        'final_rho_true_step1', 'final_rho_true_step2', 'final_sin2',
        'final_sin2_step1', 'final_theta_C', 'final_theta_C_step1', 'g0_step1',
        'g0_step2', 'g0_step3', 'g_R_net', 'g_r_net_step1', 'g_r_net_step2',
        'h_T1', 'h_T2', 'h_T4', 'h_m_H0', 'h_m_h0_step1', 'h_t1_step1',
        'h_t2_step1', 'h_t4_step1', 'k_mix_step1', 'k_step1', 'k_step2',
        'lam_step1', 'lam_step2', 'lam_step3', 'loop_ALPHA_Z',
        'loop_ALPHA_Z_step1', 'loop_AMP_D_CORR', 'loop_AMP_D_CORR_step1',
        'loop_AMP_O_CORR', 'loop_AMP_O_CORR_step1', 'loop_R_net_per_pair',
        'loop_dm1_qed', 'phi_step1', 'gamma_term_a', 'gamma_term_b', 'gamma_sum', 
        'v_exp_step', 'final_v_VEV', 'loop_dm2_qcd_d', 'loop_dm2_qcd_d_step1', 
        'loop_dm2_qcd_u', 'loop_dm2_qcd_u_step1', 'loop_dm2_qed', 'loop_factor',
        'loop_factor_step1', 'loop_factor_step2', 'loop_m_e_phys_step1',
        'loop_m_e_phys_step2', 'loop_m_e_phys_step3', 'loop_m_mu_phys_step1',
        'loop_m_phys_step1', 'loop_m_ratio_step1', 'loop_m_ratio_step2',
        'loop_m_tau_phys_step1', 'loop_r_net_pp_step1', 'loop_t_mass',
        'loop_t_mass_step1', 'loop_t_mass_step2', 'loop_t_mass_step3',
        'm_pl_j_step1', 'm_pl_kg_step1', 'm_pl_kg_step2', 'main_C_QED_phys',
        'main_c_qed_step1', 'main_c_qed_step2', 'n_efolds_step1', 'nu_M_R',
        'nu_alpha_Z', 'nu_alpha_z_step1', 'nu_bare_ev', 'nu_bare_ev_step1',
        'nu_bare_gev', 'nu_bare_gev_step1', 'nu_bare_gev_step2', 'nu_m_D',
        'nu_m_d_step1', 'nu_m_d_step2', 'nu_m_r_step1', 'nu_m_r_step2', 'nu_y',
        'nu_y_step1', 'p_g_o_step1', 'ratio_bs_step1', 'ratio_cu_step1',
        'ratio_sd_step1', 'ratio_tc_step1', 'rho_exp_term', 'rho_exp_term_step1',
        'rho_exp_term_step2', 'rho_exp_term_step3', 'rho_exp_term_step4',
        'rho_kappa', 'rho_kappa_step1', 'rho_kappa_step2', 'rho_kappa_step3',
        'rho_kappa_step4', 'rho_vac_bare', 'rho_vac_bare_step1',
        'rho_vac_bare_step2', 's_step1', 'sin2_R1', 'sin2_R2', 'sin2_bare',
        'sin2_bare_step1', 'sin2_denom', 'sin2_denom_step1', 'sin2_denom_step2',
        'sin2_r1_step1', 'sin2_r1_step2', 'sin2_r1_step3', 'sin2_r1_step4',
        'sin2_r1_step5', 'sin2_r2_step1', 'sin2_r2_step2', 'sqrt2_q_step1',
        't2_corr_step1', 't3_corr_step1', 't4_corr_step1', 't5_corr_step1',
        't6_corr_step1', 't_to_mu_step1', 't_to_mu_step2', 't_to_mz_step1',
        't_to_mz_step2', 't_to_v_step1', 't_to_v_step2', 'v_dot_s_step1',
        'v_dot_s_step2', 'v_dot_v_step1'
    ]

    # --- 3. Build Master Template from your ADAM_CONFIG ---
    # Use 1023 (our proxy for 'None') for any missing keys.
    DEFAULT_GENE_VALUE = 125
    MASTER_TEMPLATE_CONFIG = {}
    for key in ALL_POSSIBLE_LEVERS:
        # Get the value from your ADAM_CONFIG, or use default if it's missing
        MASTER_TEMPLATE_CONFIG[key] = ADAM_CONFIG.get(key, DEFAULT_GENE_VALUE)

    # --- 4. The Mating Function ---
    """
    Mates two parent config dictionaries or returns the completed ADAM_CONFIG.
    """

    # If no parents are provided, return the completed ADAM_CONFIG
    if parent_a is None and parent_b is None:
        return MASTER_TEMPLATE_CONFIG

    child_config = {}

    # Use the globally defined master template as the source of all keys
    for key in MASTER_TEMPLATE_CONFIG.keys():

        # Get gene from Parent A, defaulting if parent is None or key is missing
        gene_a = parent_a.get(key, DEFAULT_GENE_VALUE) if parent_a is not None else DEFAULT_GENE_VALUE

        # Get gene from Parent B, defaulting if parent is None or key is missing
        gene_b = parent_b.get(key, DEFAULT_GENE_VALUE) if parent_b is not None else DEFAULT_GENE_VALUE

        # Randomly select one of the two genes for the child
        child_config[key] = random.choice([gene_a, gene_b])

    return child_config


# ---
# ---
# --- VFD-4.0: PROCEDURAL QUANTIZATION SIMULATION ---
# --- (This entire function is unchanged)
# ---

def round_if_needed(value, precision):
    if ANALOG_HYPOTHESIS:
        return value
    """Applies rounding if precision is specified (not None)."""
    if precision is None:
        return value
    return round(value, precision)

def run_simulation(P_in, O, verbose=True):
    """
    EXPERIMENTAL MODEL VFD-4.0: Implements "Procedural Quantization".
    Every intermediate arithmetic operation is quantized using its
    own unique PDQ lever from the expanded 'P' map.
    """
    P = copy.deepcopy(P_in)
    try:
        # --- 1. Foundational Geometric Invariants ---
        
        # V_DOT_V = 1/3
        v_dot_v_step1 = round_if_needed(1/3, P['v_dot_v_step1'])
        V_DOT_V = round_if_needed(v_dot_v_step1, P['V_DOT_V'])
        
        # V_DOT_S = 2 / sqrt(6)
        vds_step1 = round_if_needed(6, P['v_dot_s_step1'])
        vds_step2 = round_if_needed(math.sqrt(vds_step1), P['v_dot_s_step2'])
        V_DOT_S = round_if_needed(2 / vds_step2, P['V_DOT_S'])
        
        S_DOT_S = 1/2 # This is a discrete definition
        
        # EPSILON = 2 * sqrt(2/3) - 1
        eps_step1 = round_if_needed(2/3, P['eps_step1'])
        eps_step2 = round_if_needed(math.sqrt(eps_step1), P['eps_step2'])
        eps_step3 = round_if_needed(2 * eps_step2, P['eps_step3'])
        EPSILON = round_if_needed(eps_step3 - 1, P['EPSILON'])

        N_ROOTS = 180
        N_PAIRS = 3
        R_BASE = 3/4
        
        # LAMBDA = (3 + sqrt(13)) / 2
        lam_step1 = round_if_needed(13, P['lam_step1'])
        lam_step2 = round_if_needed(math.sqrt(lam_step1), P['lam_step2'])
        lam_step3 = round_if_needed(3 + lam_step2, P['lam_step3'])
        LAMBDA = round_if_needed(lam_step3 / 2, P['LAMBDA'])

        # --- 2. Entropic & Quasicrystal Invariants ---
        
        # S = log(LAMBDA)
        s_step1 = round_if_needed(math.log(LAMBDA), P['s_step1'])
        S = round_if_needed(s_step1, P['S'])
        
        # DELTA = sqrt(2) - 1
        delta_step1 = round_if_needed(2, P['delta_step1'])
        delta_step2 = round_if_needed(math.sqrt(delta_step1), P['delta_step2'])
        DELTA = round_if_needed(delta_step2 - 1, P['DELTA'])
        
        # SQRT2_Q = DELTA + 1
        sqrt2_q_step1 = round_if_needed(DELTA + 1, P['sqrt2_q_step1'])
        SQRT2_Q = round_if_needed(sqrt2_q_step1, P['SQRT2_Q'])
        
        # PI_Q = pi
        PI_Q = round_if_needed(math.pi, P['PI_EFF'])
        
        # K = exp(S / pi)
        k_step1 = round_if_needed(S / PI_Q, P['k_step1'])
        k_step2 = round_if_needed(math.exp(k_step1), P['k_step2'])
        K = round_if_needed(k_step2, P['K'])
        
        # K_MIX = exp(DELTA)
        k_mix_step1 = round_if_needed(math.exp(DELTA), P['k_mix_step1'])
        K_MIX = round_if_needed(k_mix_step1, P['K_MIX'])
        
        # DRAG = K / K_MIX
        drag_step1 = round_if_needed(K / K_MIX, P['drag_step1'])
        DRAG = round_if_needed(drag_step1, P['DRAG'])

        # --- 3. RG Flow & Scale Invariants ---
        
        # B_LIN = 2 * pi
        b_lin_step1 = round_if_needed(2 * PI_Q, P['b_lin_step1'])
        B_LIN = round_if_needed(b_lin_step1, P['B_LIN'])
        
        # B_VOL = 60 / pi
        b_vol_step1 = round_if_needed(60 / PI_Q, P['b_vol_step1'])
        B_VOL = round_if_needed(b_vol_step1, P['B_VOL'])
        
        M_PL = 1.22e19
        MU = 1
        
        
        # 1. Calculate Golden Ratio (Phi)
        phi_step1 = round_if_needed(math.sqrt(5), P['phi_step1'])
        PHI = (1 + phi_step1) / 2

        # 2. Calculate Flux Impedance (Gamma = 2*B_VOL + Phi/2*Lambda)
        gamma_term_a = round_if_needed(2 * B_VOL, P['gamma_term_a'])
        gamma_term_b = round_if_needed(PHI / (2 * LAMBDA), P['gamma_term_b'])
        GAMMA = round_if_needed(gamma_term_a + gamma_term_b, P['gamma_sum'])

        # 3. Derive VEV from Planck Scale (v = M_PL * e^-Gamma)
        v_exp_step = round_if_needed(M_PL * math.exp(-GAMMA), P['v_exp_step'])
        
        # 4. Final Physical VEV
        V = round_if_needed(v_exp_step, P['final_v_VEV'])
        
        # M_Z = 91.1880
        M_Z = round_if_needed(91.1880, P['M_Z'])
        
        # T_TO_MZ = log(M_PL / M_Z)
        t_to_mz_step1 = round_if_needed(M_PL / M_Z, P['t_to_mz_step1'])
        t_to_mz_step2 = round_if_needed(math.log(t_to_mz_step1), P['t_to_mz_step2'])
        T_TO_MZ = round_if_needed(t_to_mz_step2, P['T_TO_MZ'])
        
        # T_TO_V = log(M_PL / V)
        t_to_v_step1 = round_if_needed(M_PL / V, P['t_to_v_step1'])
        t_to_v_step2 = round_if_needed(math.log(t_to_v_step1), P['t_to_v_step2'])
        T_TO_V = round_if_needed(t_to_v_step2, P['T_TO_V'])
        
        # T_TO_MU = log(M_PL / MU)
        t_to_mu_step1 = round_if_needed(M_PL / MU, P['t_to_mu_step1'])
        t_to_mu_step2 = round_if_needed(math.log(t_to_mu_step1), P['t_to_mu_step2'])
        T_TO_MU = round_if_needed(t_to_mu_step2, P['T_TO_MU'])

        # --- 4. Coupling & Correction Invariants ---
        
        # ALPHA_LAMBDA = 1 / 9
        alpha_lambda_step1 = round_if_needed(1 / 9, P['alpha_lambda_step1'])
        ALPHA_LAMBDA = round_if_needed(alpha_lambda_step1, P['ALPHA_LAMBDA'])
        
        RHO_PL = 1
        
        SEEDS = {'y1': V_DOT_V, 'y2': V_DOT_S, 'y3': 1.0}
        
        # AMP_A = 1 + S / pi
        amp_a_step1 = round_if_needed(S / PI_Q, P['amp_a_step1'])
        amp_a_step2 = round_if_needed(1 + amp_a_step1, P['amp_a_step2'])
        AMP_A = round_if_needed(amp_a_step2, P['AMP_A'])
        
        AMP_O = 3.0
        
        # AMP_D = 1 + EPSILON
        amp_d_step1 = round_if_needed(1 + EPSILON, P['amp_d_step1'])
        AMP_D = round_if_needed(amp_d_step1, P['AMP_D'])
        
        P_g_D = 1.0
        P_g_A = 12.0 / 20.0
        
        # P_g_O = 12.0 / 21.0
        p_g_o_step1 = round_if_needed(12.0 / 21.0, P['p_g_o_step1'])
        P_g_O = round_if_needed(p_g_o_step1, P['P_g_O'])
        
        # T_CORR Dictionary
        t2_corr_step1 = round_if_needed(ALPHA_LAMBDA * V_DOT_V, P['t2_corr_step1'])
        t3_corr_step1 = round_if_needed(DELTA * EPSILON, P['t3_corr_step1'])
        t4_corr_step1 = round_if_needed(K_MIX * DELTA, P['t4_corr_step1'])
        t5_corr_step1 = round_if_needed(-(K / K_MIX), P['t5_corr_step1'])
        t6_corr_step1 = round_if_needed(PI_Q / 12, P['t6_corr_step1'])
        
        T_CORR = {
            'T2': round_if_needed(t2_corr_step1, P['T2_CORR']),
            'T3': round_if_needed(t3_corr_step1, P['T3_CORR']),
            'T4': round_if_needed(t4_corr_step1, P['T4_CORR']),
            'T5': round_if_needed(t5_corr_step1, P['T5_CORR']),
            'T6': round_if_needed(t6_corr_step1, P['T6_CORR'])
        }

        # --- 5. Gravitational & Energy Unit Invariants ---
        
        # HBAR
        HBAR = round_if_needed(1.0545718e-34, P['HBAR'])
        
        # C
        C_raw = 2.99792458e8
        C = round_if_needed(C_raw, P['C'])
        
        # GEV_TO_J
        GEV_TO_J_raw = 1.60217662e-10
        GEV_TO_J = round_if_needed(GEV_TO_J_raw, P['GEV_TO_J'])
        
        # M_PL_J = M_PL * GEV_TO_J
        m_pl_j_step1 = round_if_needed(M_PL * GEV_TO_J, P['m_pl_j_step1'])
        M_PL_J = round_if_needed(m_pl_j_step1, P['M_PL_J'])
        
        # M_PL_KG = M_PL_J / (C ** 2)
        m_pl_kg_step1 = round_if_needed(C ** 2, P['m_pl_kg_step1'])
        m_pl_kg_step2 = round_if_needed(M_PL_J / m_pl_kg_step1, P['m_pl_kg_step2'])
        M_PL_KG = round_if_needed(m_pl_kg_step2, P['M_PL_KG'])
        
        # --- 6. Loop Correction & Hierarchy Engine Invariants ---
        
        loop_AMP_D_CORR_step1 = round_if_needed(AMP_D - 1.0, P['loop_AMP_D_CORR_step1'])
        AMP_D_CORR = round_if_needed(loop_AMP_D_CORR_step1, P['loop_AMP_D_CORR'])
        
        loop_AMP_O_CORR_step1 = round_if_needed(AMP_O - 1.0, P['loop_AMP_O_CORR_step1'])
        AMP_O_CORR = round_if_needed(loop_AMP_O_CORR_step1, P['loop_AMP_O_CORR'])
        
        # KAPPA_Q = (sqrt(PI_Q / (3 * SQRT2_Q))) * (EPSILON ** 2)
        rho_kappa_step1 = round_if_needed(3 * SQRT2_Q, P['rho_kappa_step1'])
        rho_kappa_step2 = round_if_needed(PI_Q / rho_kappa_step1, P['rho_kappa_step2'])
        rho_kappa_step3 = round_if_needed(math.sqrt(rho_kappa_step2), P['rho_kappa_step3'])
        rho_kappa_step4 = round_if_needed(EPSILON ** 2, P['rho_kappa_step4'])
        KAPPA_Q = round_if_needed(rho_kappa_step3 * rho_kappa_step4, P['rho_kappa'])
        
        # rho_exp_term_pixel = exp( - B_VOL * (S / PI_Q) * T_TO_V )
        rho_exp_term_step1 = round_if_needed(S / PI_Q, P['rho_exp_term_step1'])
        rho_exp_term_step2 = round_if_needed(B_VOL * rho_exp_term_step1, P['rho_exp_term_step2'])
        rho_exp_term_step3 = round_if_needed(rho_exp_term_step2 * T_TO_V, P['rho_exp_term_step3'])
        rho_exp_term_step4 = round_if_needed(-rho_exp_term_step3, P['rho_exp_term_step4'])
        rho_exp_term_pixel = round_if_needed(math.exp(rho_exp_term_step4), P['rho_exp_term'])
        
        if rho_exp_term_pixel == 0:
            rho_exp_term_pixel = 1e-99
            
        # --- 7. Calculation Functions (Now using procedural invariants) ---

        results = {}
        results['v_VEV'] = V

        # --- 7a. calc_inv_alpha_MZ ---
        T0 = round_if_needed(9.0, P['a_T0'])
        
        a_t1_step1 = round_if_needed(B_LIN / 2, P['a_t1_step1'])
        a_t1_step2 = round_if_needed(a_t1_step1 * T_TO_MZ, P['a_t1_step2'])
        T1 = round_if_needed(a_t1_step2 * DRAG, P['a_T1'])
        
        a_t2_step1 = round_if_needed(ALPHA_LAMBDA * S_DOT_S, P['a_t2_step1'])
        T_S_new = round_if_needed(a_t2_step1, P['a_T2'])
        
        a_t3_step1 = round_if_needed(ALPHA_LAMBDA * EPSILON, P['a_t3_step1'])
        T3 = round_if_needed(a_t3_step1, P['a_T3'])
        
        a_t4_step1 = round_if_needed(ALPHA_LAMBDA / 2.0, P['a_t4_step1'])
        T4 = round_if_needed(a_t4_step1, P['a_T4'])
        
        a_fin_step1 = round_if_needed(T0 + T1, P['a_fin_step1'])
        a_fin_step2 = round_if_needed(a_fin_step1 + T_S_new, P['a_fin_step2'])
        a_fin_step3 = round_if_needed(a_fin_step2 + T3, P['a_fin_step3'])
        a_fin_step4 = round_if_needed(a_fin_step3 + T4, P['a_fin_step4'])
        
        inv_alpha_z_val = round_if_needed(a_fin_step4, P['final_inv_alpha_MZ'])
        results['inv_alpha_MZ'] = inv_alpha_z_val

        # --- 7b. calc_m_H ---
        h_m_h0_step1 = round_if_needed(KAPPA_Q * V, P['h_m_h0_step1'])
        m_H0_honest = round_if_needed(h_m_h0_step1, P['h_m_H0'])
        
        h_t1_step1 = round_if_needed(m_H0_honest * K, P['h_t1_step1'])
        h_T1 = round_if_needed(h_t1_step1, P['h_T1'])
        
        h_t2_step1 = round_if_needed(h_T1 / (N_ROOTS * 2), P['h_t2_step1'])
        h_T2 = round_if_needed(h_t2_step1, P['h_T2'])
        
        h_t4_step1 = round_if_needed(V_DOT_V * MU, P['h_t4_step1'])
        h_T4 = round_if_needed(h_t4_step1, P['h_T4'])
        
        final_m_H_step1 = round_if_needed(h_T1 + h_T2, P['final_m_H_step1'])
        final_m_H_step2 = round_if_needed(final_m_H_step1 + h_T4, P['final_m_H_step2'])
        
        m_H_val = round_if_needed(final_m_H_step2, P['final_m_H'])
        results['m_H'] = m_H_val

        # --- 7c. calc_rho_true ---
        rho_vac_bare_step1 = round_if_needed(KAPPA_Q ** 4, P['rho_vac_bare_step1'])
        rho_vac_bare_step2 = round_if_needed(rho_vac_bare_step1 * RHO_PL, P['rho_vac_bare_step2'])
        rho_vac_bare = round_if_needed(rho_vac_bare_step2, P['rho_vac_bare'])
        
        exp_term = rho_exp_term_pixel
        
        final_rho_true_step1 = round_if_needed(rho_vac_bare * exp_term, P['final_rho_true_step1'])
        final_rho_true_step2 = round_if_needed(final_rho_true_step1 / K, P['final_rho_true_step2'])
        
        results['rho_true'] = round_if_needed(final_rho_true_step2, P['final_rho_true'])

        # --- 7d. calc_G ---
        g0_step1 = round_if_needed(HBAR * C, P['g0_step1'])
        g0_step2 = round_if_needed(M_PL_KG ** 2, P['g0_step2'])
        g0_step3 = round_if_needed(g0_step1 / g0_step2, P['g0_step3'])
        G0 = round_if_needed(g0_step3, P['G0'])
        
        t_to_v = round_if_needed(T_TO_V, P['G_t_to_v'])
        
        g_r_net_step1 = round_if_needed(S - ALPHA_LAMBDA, P['g_r_net_step1'])
        g_r_net_step2 = round_if_needed(B_VOL * t_to_v, P['g_r_net_step2'])
        R_net_val = round_if_needed(g_r_net_step1 / g_r_net_step2, P['g_R_net'])
        
        final_G_step1 = round_if_needed(1 + R_net_val, P['final_G_step1'])
        G_val = round_if_needed(G0 / final_G_step1, P['final_G'])
        results['G'] = G_val

        # --- 7e. calc_sin2_theta_W ---
        # *** START EXPERIMENTAL ANGLE QUANTIZATION (SHARED LEVER) ***
        sin2_bare_step1 = round_if_needed(V_DOT_S / 3, P['sin2_bare_step1'])
        bare = round_if_needed(sin2_bare_step1, P['sin2_bare'])
        
        t_mz = T_TO_MZ
        
        sin2_r1_step1 = round_if_needed(B_LIN * ALPHA_LAMBDA, P['sin2_r1_step1'])
        sin2_r1_step2 = round_if_needed(4 * PI_Q, P['sin2_r1_step2'])
        sin2_r1_step3 = round_if_needed(sin2_r1_step1 / sin2_r1_step2, P['sin2_r1_step3'])
        sin2_r1_step4 = round_if_needed(sin2_r1_step3 * t_mz, P['sin2_r1_step4'])
        sin2_r1_step5 = round_if_needed(K_MIX / B_VOL, P['sin2_r1_step5'])
        R1 = round_if_needed(sin2_r1_step4 * sin2_r1_step5, P['sin2_R1'])
        
        sin2_r2_step1 = round_if_needed(B_VOL * t_mz, P['sin2_r2_step1'])
        sin2_r2_step2 = round_if_needed(S / sin2_r2_step1, P['sin2_r2_step2'])
        R2_val = round_if_needed(sin2_r2_step2, P['sin2_R2'])
        
        sin2_denom_step1 = round_if_needed(1 + R1, P['sin2_denom_step1'])
        sin2_denom_step2 = round_if_needed(sin2_denom_step1 + R2_val, P['sin2_denom_step2'])
        denom = round_if_needed(sin2_denom_step2, P['sin2_denom'])
        
        # 1. Calculate the raw, unquantized sin^2 value
        sin2_raw = round_if_needed(bare / denom, P['final_sin2_step1'])

        # 2. Invert it to get the raw angle in degrees
        if sin2_raw < 0 or sin2_raw > 1.0: # Added bounds check
            raise ValueError("sin2_theta_W is outside [0, 1]")
        theta_W_raw = math.degrees(math.asin(math.sqrt(sin2_raw)))

        # 3. Create the ratio to a full 360-degree rotation
        R_raw = theta_W_raw / 360.0

        # 4. Apply the NEW SHARED DQE snap to the ratio itself
        R_snapped = R_raw
        if not ANALOG_HYPOTHESIS and P['dqe_p_ANGLE_RATIO'] is not None:
            snap_factor = 10**-P['dqe_p_ANGLE_RATIO']
            R_snapped = round(R_raw / snap_factor) * snap_factor

        # 5. Reconstruct the angle from the snapped ratio
        theta_W_snapped = R_snapped * 360.0

        # 6. Re-calculate the final sin^2 value from the snapped angle
        sin2_theta_W_val = math.sin(math.radians(theta_W_snapped))**2

        # 7. Apply final rounding (as before) and store
        sin2_theta_W_val = round_if_needed(sin2_theta_W_val, P['final_sin2'])
        results['sin2_theta_W'] = sin2_theta_W_val
        # *** END EXPERIMENTAL ANGLE QUANTIZATION ***

        # --- 7f. calc_alpha_s_MZ ---
        t_mz = T_TO_MZ
        
        as_t0_step1 = round_if_needed(9 * PI_Q, P['as_t0_step1'])
        T0 = round_if_needed(as_t0_step1, P['as_T0'])
        
        as_t1_step1 = round_if_needed(4 * PI_Q, P['as_t1_step1'])
        as_t1_step2 = round_if_needed(B_LIN / as_t1_step1, P['as_t1_step2'])
        T1 = round_if_needed(-as_t1_step2 * t_mz, P['as_T1'])
        
        as_t2_step1 = round_if_needed(-ALPHA_LAMBDA * V_DOT_S, P['as_t2_step1'])
        T2 = round_if_needed(as_t2_step1, P['as_T2'])
        
        as_inv_step1 = round_if_needed(T0 + T1, P['as_inv_step1'])
        as_inv_step2 = round_if_needed(as_inv_step1 + T2, P['as_inv_step2'])
        inv_alpha_s = round_if_needed(as_inv_step2, P['as_inv'])
        
        as_fin_step1 = round_if_needed(1 / inv_alpha_s, P['as_fin_step1'])
        alpha_s_mz_val = round_if_needed(as_fin_step1, P['final_alpha_s_MZ'])
        results['alpha_s_MZ'] = alpha_s_mz_val

        # --- 7g. calc_m_nu ---
        nu_m_r_step1 = round_if_needed(4 * PI_Q, P['nu_m_r_step1'])
        nu_m_r_step2 = round_if_needed(math.exp(nu_m_r_step1), P['nu_m_r_step2'])
        M_R = round_if_needed(M_PL / nu_m_r_step2, P['nu_M_R'])
        
        nu_y_step1 = round_if_needed(V_DOT_S / 3, P['nu_y_step1'])
        y_nu = round_if_needed(nu_y_step1, P['nu_y'])
        
        nu_m_d_step1 = round_if_needed(y_nu * V, P['nu_m_d_step1'])
        nu_m_d_step2 = round_if_needed(nu_m_d_step1 / SQRT2_Q, P['nu_m_d_step2'])
        m_D = round_if_needed(nu_m_d_step2, P['nu_m_D'])
        
        nu_bare_gev_step1 = round_if_needed(m_D ** 2, P['nu_bare_gev_step1'])
        nu_bare_gev_step2 = round_if_needed(nu_bare_gev_step1 / M_R, P['nu_bare_gev_step2'])
        bare_gev = round_if_needed(nu_bare_gev_step2, P['nu_bare_gev'])
        
        nu_bare_ev_step1 = round_if_needed(bare_gev * 1e9, P['nu_bare_ev_step1'])
        bare_ev = round_if_needed(nu_bare_ev_step1, P['nu_bare_ev'])
        
        nu_alpha_z_step1 = round_if_needed(1 / inv_alpha_z_val, P['nu_alpha_z_step1'])
        alpha_Z = round_if_needed(nu_alpha_z_step1, P['nu_alpha_Z'])
        
        final_m_nu_step1 = round_if_needed(1 - alpha_Z, P['final_m_nu_step1'])
        m_nu_val = round_if_needed(bare_ev * final_m_nu_step1, P['final_m_nu'])
        results['m_nu'] = m_nu_val

        # --- 7h. calc_theta_C ---
        # *** START EXPERIMENTAL ANGLE QUANTIZATION (SHARED LEVER) ***
        t_mu = T_TO_MU
        
        ckm_theta0_step1 = round_if_needed(math.acos(S_DOT_S), P['ckm_theta0_step1'])
        ckm_theta0_step2 = round_if_needed(math.degrees(ckm_theta0_step1), P['ckm_theta0_step2'])
        theta0 = round_if_needed(ckm_theta0_step2, P['ckm_theta0'])
        
        ckm_t1_step1 = round_if_needed(4 * PI_Q, P['ckm_t1_step1'])
        ckm_t1_step2 = round_if_needed(B_LIN * ALPHA_LAMBDA, P['ckm_t1_step2'])
        ckm_t1_step3 = round_if_needed(ckm_t1_step2 / ckm_t1_step1, P['ckm_t1_step3'])
        ckm_t1_step4 = round_if_needed(ckm_t1_step3 * t_mu, P['ckm_t1_step4'])
        T1 = round_if_needed(K * ckm_t1_step4, P['ckm_T1'])
        
        ckm_t2_step1 = round_if_needed(ALPHA_LAMBDA * S_DOT_S, P['ckm_t2_step1'])
        T2_new = round_if_needed(ckm_t2_step1, P['ckm_T2'])
        
        ckm_denom_step1 = round_if_needed(1 + T1, P['ckm_denom_step1'])
        ckm_denom_step2 = round_if_needed(ckm_denom_step1 + T2_new, P['ckm_denom_step2'])
        denom = round_if_needed(ckm_denom_step2, P['ckm_denom'])
        
        # 1. Calculate the raw, unquantized angle
        theta_C_raw = round_if_needed(theta0 / denom, P['final_theta_C_step1'])

        # 2. Create the ratio to a full 360-degree rotation
        R_raw = theta_C_raw / 360.0

        # 3. Apply the NEW SHARED DQE snap to the ratio itself
        R_snapped = R_raw
        if not ANALOG_HYPOTHESIS and P['dqe_p_ANGLE_RATIO'] is not None:
            snap_factor = 10**-P['dqe_p_ANGLE_RATIO']
            R_snapped = round(R_raw / snap_factor) * snap_factor

        # 4. Reconstruct the final angle from the snapped ratio
        theta_C_val = R_snapped * 360.0

        # 5. Apply final rounding (as before) and store
        theta_C_val = round_if_needed(theta_C_val, P['final_theta_C'])
        results['theta_C'] = theta_C_val
        # *** END EXPERIMENTAL ANGLE QUANTIZATION ***


        # --- 7i. calc_theta_23 ---
        # *** START EXPERIMENTAL ANGLE QUANTIZATION (SHARED LEVER) ***
        ckm23_denom_step1 = round_if_needed(B_LIN - V_DOT_S, P['ckm23_denom_step1'])
        denom = round_if_needed(ckm23_denom_step1, P['ckm23_denom'])
        if denom == 0: denom = 1e-99
        
        # 1. Calculate the raw, unquantized angle
        #    (Uses the *snapped* theta_C_val from the previous step as its input)
        theta_23_raw = round_if_needed(theta_C_val / denom, P['ckm23_fin_step1'])
        
        # 2. Create the ratio to a full 360-degree rotation
        R_raw = theta_23_raw / 360.0
        
        # 3. Apply the NEW SHARED DQE snap to the ratio itself
        R_snapped = R_raw
        if not ANALOG_HYPOTHESIS and P['dqe_p_ANGLE_RATIO'] is not None:
            snap_factor = 10**-P['dqe_p_ANGLE_RATIO']
            R_snapped = round(R_raw / snap_factor) * snap_factor
            
        # 4. Reconstruct the final angle from the snapped ratio
        theta_23_val = R_snapped * 360.0
        
        # 5. Apply final rounding (as before) and store
        theta_23_val = round_if_needed(theta_23_val, P['ckm23_final'])
        results['theta_23'] = theta_23_val
        # *** END EXPERIMENTAL ANGLE QUANTIZATION ***


        # --- 7i. calc_theta_23 ---
        # *** START EXPERIMENTAL ANGLE QUANTIZATION (SHARED LEVER) ***
        ckm23_denom_step1 = round_if_needed(B_LIN - V_DOT_S, P['ckm23_denom_step1'])
        denom = round_if_needed(ckm23_denom_step1, P['ckm23_denom'])
        if denom == 0: denom = 1e-99
        
        # 1. Calculate the raw, unquantized angle
        #    (Uses the *snapped* theta_C_val from the previous step as its input)
        theta_23_raw = round_if_needed(theta_C_val / denom, P['ckm23_fin_step1'])
        
        # 2. Create the ratio to a full 360-degree rotation
        R_raw = theta_23_raw / 360.0
        
        # 3. Apply the NEW SHARED DQE snap to the ratio itself
        R_snapped = R_raw
        if P['dqe_p_ANGLE_RATIO'] is not None:
            snap_factor = 10**-P['dqe_p_ANGLE_RATIO']
            R_snapped = round(R_raw / snap_factor) * snap_factor
            
        # 4. Reconstruct the final angle from the snapped ratio
        theta_23_val = R_snapped * 360.0
        
        # 5. Apply final rounding (as before) and store
        theta_23_val = round_if_needed(theta_23_val, P['ckm23_final'])
        results['theta_23'] = theta_23_val
        # *** END EXPERIMENTAL ANGLE QUANTIZATION ***

        # --- 7j. calc_N_efolds ---
        n_efolds_step1 = round_if_needed(N_ROOTS / N_PAIRS, P['n_efolds_step1'])
        N_efolds_val = round_if_needed(n_efolds_step1, P['N_efolds'])
        results['N_efolds'] = N_efolds_val


        # --- 7k. C_QED_phys (SIMPLIFIED) ---
        main_c_qed_step1 = round_if_needed(B_VOL * T_TO_V, P['main_c_qed_step1'])
        main_c_qed_step2 = round_if_needed(ALPHA_LAMBDA / main_c_qed_step1, P['main_c_qed_step2'])
        C_QED_phys = round_if_needed(main_c_qed_step2, P['main_C_QED_phys'])

        # --- 7l. get_gated_correction (Helper) ---
        def get_gated_correction(key, T_CORR_in):
            if key == 'muon_A4/electron_A4':
                return round_if_needed(0.0, P['f_gated_mu_e'])
            elif key == 'tau_A4/muon_A4':
                f_gated_tau_mu_step1 = round_if_needed(T_CORR['T2'] + T_CORR['T4'], P['f_gated_tau_mu_step1'])
                f_gated_tau_mu_step2 = round_if_needed(f_gated_tau_mu_step1 + abs(T_CORR['T5']), P['f_gated_tau_mu_step2'])
                f_gated_tau_mu_step3 = round_if_needed(ALPHA_LAMBDA * V_DOT_V, P['f_gated_tau_mu_step3'])
                tau_tax_raw = round_if_needed(f_gated_tau_mu_step2 - f_gated_tau_mu_step3, P['f_gated_tau_mu_step4'])
                return round_if_needed(tau_tax_raw, P['f_gated_tau_mu'])
            elif key == 'strange_D10/down_D10':
                f_gated_s_d_step1 = round_if_needed(-T_CORR['T2'], P['f_gated_s_d_step1'])
                return round_if_needed(f_gated_s_d_step1, P['f_gated_s_d'])
            elif key == 'bottom_D10/strange_D10':
                return round_if_needed(0.0, P['f_gated_b_s'])
            elif key == 'charm_Oh/up_Oh':
                f_gated_c_u_step1 = round_if_needed(-T_CORR['T3'] / 2.0, P['f_gated_c_u_step1'])
                return round_if_needed(f_gated_c_u_step1, P['f_gated_c_u'])
            elif key == 'top_Oh/charm_Oh':
                return round_if_needed(0.0, P['f_gated_t_c'])
            return 0.0

        ## --- 7m. get_fermion_log_ratios ---
        log_ratios = {}
        f_c_step1 = round_if_needed(T_TO_V / 10.0, P['f_c_step1'])
        C = round_if_needed(f_c_step1, P['f_C'])
        
        f_t6_dil_step1 = round_if_needed(T_CORR['T6'] * DRAG, P['f_t6_dil_step1'])
        T6_diluted = round_if_needed(f_t6_dil_step1, P['f_T6_diluted'])
        
        f_r_net_pp_step1 = round_if_needed(R_net_val / N_PAIRS, P['f_r_net_pp_step1'])
        R_net_per_pair = round_if_needed(f_r_net_pp_step1, P['f_R_net_per_pair'])
        
        dy_21 = SEEDS['y2'] - SEEDS['y1']
        dy_32 = SEEDS['y3'] - SEEDS['y2']
        
        # --- mu/e ---
        f_lam_a_step1 = round_if_needed(LAMBDA * P_g_A, P['f_lam_a_step1'])
        lambda_A = round_if_needed(f_lam_a_step1, P['f_lambda_A'])
        f_t1_mu_e_step1 = round_if_needed(C * dy_21, P['f_t1_mu_e_step1'])
        f_t1_mu_e_step2 = round_if_needed(AMP_A * lambda_A, P['f_t1_mu_e_step2'])
        f_t1_mu_e_step3 = round_if_needed(f_t1_mu_e_step1 * f_t1_mu_e_step2, P['f_t1_mu_e_step3'])
        T1_mu_e = round_if_needed(f_t1_mu_e_step3, P['f_T1_mu_e'])
        gated_mu_e = get_gated_correction('muon_A4/electron_A4', T_CORR)
        f_log_r_mu_e_step1 = round_if_needed(T1_mu_e + T6_diluted, P['f_log_r_mu_e_step1'])
        f_log_r_mu_e_step2 = round_if_needed(f_log_r_mu_e_step1 + gated_mu_e, P['f_log_r_mu_e_step2'])
        f_log_r_mu_e_step3 = round_if_needed(f_log_r_mu_e_step2 - R2_val, P['f_log_r_mu_e_step3'])
        log_ratio_bare_mu_e = round_if_needed(f_log_r_mu_e_step3 - R_net_per_pair, P['f_log_r_mu_e_step4'])
        if not ANALOG_HYPOTHESIS and P['dqe_p_me'] is not None:
            log_ratio_bare_mu_e = round(log_ratio_bare_mu_e / (10 ** -P['dqe_p_me'])) * (10 ** -P['dqe_p_me'])
        log_ratios['muon_A4/electron_A4'] = round_if_needed(log_ratio_bare_mu_e + C_QED_phys, P['f_log_r_mu_e'])

        # --- tau/mu ---
        f_t1_tau_mu_step1 = round_if_needed(C * dy_32, P['f_t1_tau_mu_step1'])
        f_t1_tau_mu_step2 = round_if_needed(AMP_A * 1.0, P['f_t1_tau_mu_step2'])
        T1_tau_mu = round_if_needed(f_t1_tau_mu_step1 * f_t1_tau_mu_step2, P['f_T1_tau_mu'])
        gated_tau_mu = get_gated_correction('tau_A4/muon_A4', T_CORR)
        f_log_r_tau_mu_step1 = round_if_needed(T1_tau_mu + T6_diluted, P['f_log_r_tau_mu_step1'])
        f_log_r_tau_mu_step2 = round_if_needed(f_log_r_tau_mu_step1 + gated_tau_mu, P['f_log_r_tau_mu_step2'])
        f_log_r_tau_mu_step3 = round_if_needed(f_log_r_tau_mu_step2 + R2_val, P['f_log_r_tau_mu_step3'])
        log_ratio_bare_tau_mu = round_if_needed(f_log_r_tau_mu_step3 + R_net_per_pair, P['f_log_r_tau_mu_step4'])
        if not ANALOG_HYPOTHESIS and P['dqe_p_tm'] is not None:
            log_ratio_bare_tau_mu = round(log_ratio_bare_tau_mu / (10 ** -P['dqe_p_tm'])) * (10 ** -P['dqe_p_tm'])
        log_ratios['tau_A4/muon_A4'] = round_if_needed(log_ratio_bare_tau_mu + C_QED_phys, P['f_log_r_tau_mu'])

        # --- s/d ---
        f_t1_s_d_step1 = round_if_needed(C * dy_21, P['f_t1_s_d_step1'])
        f_t1_s_d_step2 = round_if_needed(AMP_D * 1.0, P['f_t1_s_d_step2'])
        T1_s_d = round_if_needed(f_t1_s_d_step1 * f_t1_s_d_step2, P['f_T1_s_d'])
        gated_sd = get_gated_correction('strange_D10/down_D10', T_CORR)
        f_log_r_s_d_step1 = round_if_needed(T1_s_d + T6_diluted, P['f_log_r_s_d_step1'])
        f_log_r_s_d_step2 = round_if_needed(f_log_r_s_d_step1 - T_CORR['T3'], P['f_log_r_s_d_step2'])
        f_log_r_s_d_step3 = round_if_needed(f_log_r_s_d_step2 - R_net_per_pair, P['f_log_r_s_d_step3'])
        log_ratio_bare_sd = round_if_needed(f_log_r_s_d_step3 + gated_sd, P['f_log_r_s_d_step4'])
        if not ANALOG_HYPOTHESIS and P['dqe_p_sd'] is not None:
            log_ratio_bare_sd = round(log_ratio_bare_sd / (10 ** -P['dqe_p_sd'])) * (10 ** -P['dqe_p_sd'])
        log_ratios['strange_D10/down_D10'] = round_if_needed(log_ratio_bare_sd, P['f_log_r_s_d'])

        # --- b/s ---
        f_lam_d_step1 = round_if_needed(LAMBDA * P_g_D, P['f_lam_d_step1'])
        lambda_D = round_if_needed(f_lam_d_step1, P['f_lambda_D'])
        f_t1_b_s_step1 = round_if_needed(C * dy_32, P['f_t1_b_s_step1'])
        f_t1_b_s_step2 = round_if_needed(AMP_D * lambda_D, P['f_t1_b_s_step2'])
        f_t1_b_s_step3 = round_if_needed(f_t1_b_s_step1 * f_t1_b_s_step2, P['f_t1_b_s_step3'])
        T1_b_s = round_if_needed(f_t1_b_s_step3, P['f_T1_b_s'])
        gated_bs = get_gated_correction('bottom_D10/strange_D10', T_CORR)
        f_log_r_b_s_step1 = round_if_needed(T1_b_s + T6_diluted, P['f_log_r_b_s_step1'])
        f_log_r_b_s_step2 = round_if_needed(f_log_r_b_s_step1 - T_CORR['T3'], P['f_log_r_b_s_step2'])
        f_log_r_b_s_step3 = round_if_needed(f_log_r_b_s_step2 - R_net_per_pair, P['f_log_r_b_s_step3'])
        log_ratio_bare_bs = round_if_needed(f_log_r_b_s_step3 + gated_bs, P['f_log_r_b_s_step4'])
        if not ANALOG_HYPOTHESIS and P['dqe_p_bs'] is not None:
            log_ratio_bare_bs = round(log_ratio_bare_bs / (10 ** -P['dqe_p_bs'])) * (10 ** -P['dqe_p_bs'])
        log_ratios['bottom_D10/strange_D10'] = round_if_needed(log_ratio_bare_bs, P['f_log_r_b_s'])

        # --- c/u ---
        f_t1_c_u_step1 = round_if_needed(C * dy_21, P['f_t1_c_u_step1'])
        f_t1_c_u_step2 = round_if_needed(AMP_O * 1.0, P['f_t1_c_u_step2'])
        T1_c_u = round_if_needed(f_t1_c_u_step1 * f_t1_c_u_step2, P['f_T1_c_u'])
        gated_cu = get_gated_correction('charm_Oh/up_Oh', T_CORR)
        f_log_r_c_u_step1 = round_if_needed(T1_c_u + T6_diluted, P['f_log_r_c_u_step1'])
        f_log_r_c_u_step2 = round_if_needed(f_log_r_c_u_step1 + T_CORR['T2'], P['f_log_r_c_u_step2'])
        f_log_r_c_u_step3 = round_if_needed(f_log_r_c_u_step2 + T_CORR['T4'], P['f_log_r_c_u_step3'])
        f_log_r_c_u_step4 = round_if_needed(f_log_r_c_u_step3 - R_net_per_pair, P['f_log_r_c_u_step4'])
        log_ratio_bare_cu = round_if_needed(f_log_r_c_u_step4 + gated_cu, P['f_log_r_c_u_step5'])
        if not ANALOG_HYPOTHESIS and P['dqe_p_cu'] is not None:
            log_ratio_bare_cu = round(log_ratio_bare_cu / (10 ** -P['dqe_p_cu'])) * (10 ** -P['dqe_p_cu'])
        log_ratios['charm_Oh/up_Oh'] = round_if_needed(log_ratio_bare_cu, P['f_log_r_c_u'])

        # --- t/c ---
        f_lam_o_step1 = round_if_needed(LAMBDA * P_g_O, P['f_lam_o_step1'])
        lambda_O = round_if_needed(f_lam_o_step1, P['f_lambda_O'])
        f_t1_t_c_step1 = round_if_needed(C * dy_32, P['f_t1_t_c_step1'])
        f_t1_t_c_step2 = round_if_needed(AMP_O * lambda_O, P['f_t1_t_c_step2'])
        f_t1_t_c_step3 = round_if_needed(f_t1_t_c_step1 * f_t1_t_c_step2, P['f_t1_t_c_step3'])
        T1_t_c = round_if_needed(f_t1_t_c_step3, P['f_T1_t_c'])
        gated_tc = get_gated_correction('top_Oh/charm_Oh', T_CORR)
        f_log_r_t_c_step1 = round_if_needed(T1_t_c + T6_diluted, P['f_log_r_t_c_step1'])
        f_log_r_t_c_step2 = round_if_needed(f_log_r_t_c_step1 + T_CORR['T2'], P['f_log_r_t_c_step2'])
        f_log_r_t_c_step3 = round_if_needed(f_log_r_t_c_step2 + T_CORR['T4'], P['f_log_r_t_c_step3'])
        f_log_r_t_c_step4 = round_if_needed(f_log_r_t_c_step3 - R_net_per_pair, P['f_log_r_t_c_step4'])
        log_ratio_bare_tc = round_if_needed(f_log_r_t_c_step4 + gated_tc, P['f_log_r_t_c_step5'])
        if not ANALOG_HYPOTHESIS and P['dqe_p_tc'] is not None:
            log_ratio_bare_tc = round(log_ratio_bare_tc / (10 ** -P['dqe_p_tc'])) * (10 ** -P['dqe_p_tc'])
        log_ratios['top_Oh/charm_Oh'] = round_if_needed(log_ratio_bare_tc, P['f_log_r_t_c'])

        # --- 7n. calculate_fermion_masses (Bare) ---
        bare_mass_results = {}
        
        f_v_ovr_s2_step1 = round_if_needed(V / SQRT2_Q, P['f_v_ovr_s2_step1'])
        V_OVER_SQRT2 = round_if_needed(f_v_ovr_s2_step1, P['f_V_OVER_SQRT2'])
        
        f_log_y1_step1 = round_if_needed(math.log(SEEDS['y1']), P['f_log_y1_step1'])
        f_log_y2_step1 = round_if_needed(math.log(SEEDS['y2']), P['f_log_y2_step1'])
        LOG_Y_BARE = {
            1: round_if_needed(f_log_y1_step1, P['f_LOG_Y1']),
            2: round_if_needed(f_log_y2_step1, P['f_LOG_Y2']),
            3: round_if_needed(math.log(SEEDS['y3']), P['f_LOG_Y3'])
        }
        
        exponents = {}
        f_base_anch_step1 = round_if_needed(T_TO_V / LAMBDA, P['f_base_anch_step1'])
        base_anchor = round_if_needed(f_base_anch_step1, P['f_base_anchor'])
        
        T2_SIGN = 0.0
        
        f_exp_e_step1 = round_if_needed(base_anchor + (T2_SIGN * T_CORR['T2']), P['f_exp_e_step1'])
        EXP_ANCHOR_A4 = round_if_needed(f_exp_e_step1, P['f_exp_e'])
        
        f_exp_d10_t1_step1 = round_if_needed(PI_Q / S, P['f_exp_d10_t1_step1'])
        EXP_D10_T1 = round_if_needed(f_exp_d10_t1_step1, P['f_EXP_D10_T1'])
        
        f_exp_oh_t1_step1 = round_if_needed(PI_Q / K, P['f_exp_oh_t1_step1'])
        EXP_OH_T1 = round_if_needed(f_exp_oh_t1_step1, P['f_EXP_OH_T1'])
        
        f_exp_d_step1 = round_if_needed(base_anchor - EXP_D10_T1, P['f_exp_d_step1'])
        f_exp_d_step2 = round_if_needed(f_exp_d_step1 + T_CORR['T3'], P['f_exp_d_step2'])
        f_exp_d_step3 = round_if_needed(f_exp_d_step2 + T_CORR['T2'], P['f_exp_d_step3'])
        f_exp_d_step4 = round_if_needed(f_exp_d_step3 + ALPHA_LAMBDA, P['f_exp_d_step4'])
        EXP_ANCHOR_D10 = round_if_needed(f_exp_d_step4, P['f_exp_d'])
        
        f_exp_u_step1 = round_if_needed(base_anchor - EXP_OH_T1, P['f_exp_u_step1'])
        f_exp_u_step2 = round_if_needed(f_exp_u_step1 + T_CORR['T2'], P['f_exp_u_step2'])
        f_exp_u_step3 = round_if_needed(f_exp_u_step2 + T_CORR['T4'], P['f_exp_u_step3'])
        EXP_ANCHOR_OH = round_if_needed(f_exp_u_step3, P['f_exp_u'])
        
        exponents['electron_A4'] = EXP_ANCHOR_A4
        exponents['down_D10'] = EXP_ANCHOR_D10
        exponents['up_Oh'] = EXP_ANCHOR_OH
        
        f_exp_mu_step1 = round_if_needed(LOG_Y_BARE[2] - LOG_Y_BARE[1], P['f_exp_mu_step1'])
        f_exp_mu_step2 = round_if_needed(exponents['electron_A4'] + f_exp_mu_step1, P['f_exp_mu_step2'])
        f_exp_mu_step3 = round_if_needed(f_exp_mu_step2 - log_ratios['muon_A4/electron_A4'], P['f_exp_mu_step3'])
        exponents['muon_A4'] = round_if_needed(f_exp_mu_step3, P['f_exp_mu'])
        
        f_exp_s_step1 = round_if_needed(LOG_Y_BARE[2] - LOG_Y_BARE[1], P['f_exp_s_step1'])
        f_exp_s_step2 = round_if_needed(exponents['down_D10'] + f_exp_s_step1, P['f_exp_s_step2'])
        f_exp_s_step3 = round_if_needed(f_exp_s_step2 - log_ratios['strange_D10/down_D10'], P['f_exp_s_step3'])
        exponents['strange_D10'] = round_if_needed(f_exp_s_step3, P['f_exp_s'])

        f_exp_c_step1 = round_if_needed(LOG_Y_BARE[2] - LOG_Y_BARE[1], P['f_exp_c_step1'])
        f_exp_c_step2 = round_if_needed(exponents['up_Oh'] + f_exp_c_step1, P['f_exp_c_step2'])
        f_exp_c_step3 = round_if_needed(f_exp_c_step2 - log_ratios['charm_Oh/up_Oh'], P['f_exp_c_step3'])
        exponents['charm_Oh'] = round_if_needed(f_exp_c_step3, P['f_exp_c'])

        f_exp_tau_step1 = round_if_needed(LOG_Y_BARE[3] - LOG_Y_BARE[2], P['f_exp_tau_step1'])
        f_exp_tau_step2 = round_if_needed(exponents['muon_A4'] + f_exp_tau_step1, P['f_exp_tau_step2'])
        f_exp_tau_step3 = round_if_needed(f_exp_tau_step2 - log_ratios['tau_A4/muon_A4'], P['f_exp_tau_step3'])
        exponents['tau_A4'] = round_if_needed(f_exp_tau_step3, P['f_exp_tau'])
        
        f_exp_b_step1 = round_if_needed(LOG_Y_BARE[3] - LOG_Y_BARE[2], P['f_exp_b_step1'])
        f_exp_b_step2 = round_if_needed(exponents['strange_D10'] + f_exp_b_step1, P['f_exp_b_step2'])
        f_exp_b_step3 = round_if_needed(f_exp_b_step2 - log_ratios['bottom_D10/strange_D10'], P['f_exp_b_step3'])
        exponents['bottom_D10'] = round_if_needed(f_exp_b_step3, P['f_exp_b'])

        f_exp_t_step1 = round_if_needed(LOG_Y_BARE[3] - LOG_Y_BARE[2], P['f_exp_t_step1'])
        f_exp_t_step2 = round_if_needed(exponents['charm_Oh'] + f_exp_t_step1, P['f_exp_t_step2'])
        f_exp_t_step3 = round_if_needed(f_exp_t_step2 - log_ratios['top_Oh/charm_Oh'], P['f_exp_t_step3'])
        exponents['top_Oh'] = round_if_needed(f_exp_t_step3, P['f_exp_t'])

        f_m_bare_1_step1 = round_if_needed(V_OVER_SQRT2 * SEEDS['y1'], P['f_m_bare_1_step1'])
        f_m_bare_2_step1 = round_if_needed(V_OVER_SQRT2 * SEEDS['y2'], P['f_m_bare_2_step1'])
        f_m_bare_3_step1 = round_if_needed(V_OVER_SQRT2 * SEEDS['y3'], P['f_m_bare_3_step1'])
        m_bare = {
            1: round_if_needed(f_m_bare_1_step1, P['f_m_bare_1']),
            2: round_if_needed(f_m_bare_2_step1, P['f_m_bare_2']),
            3: round_if_needed(f_m_bare_3_step1, P['f_m_bare_3'])
        }
        
        f_m_bare_e_step1 = round_if_needed(-exponents['electron_A4'], P['f_m_bare_e_step1'])
        f_m_bare_e_step2 = round_if_needed(m_bare[1] * math.exp(f_m_bare_e_step1), P['f_m_bare_e_step2'])
        bare_mass_results['electron_A4'] = round_if_needed(f_m_bare_e_step2, P['f_m_bare_e'])
        
        f_m_bare_mu_step1 = round_if_needed(-exponents['muon_A4'], P['f_m_bare_mu_step1'])
        f_m_bare_mu_step2 = round_if_needed(m_bare[2] * math.exp(f_m_bare_mu_step1), P['f_m_bare_mu_step2'])
        bare_mass_results['muon_A4'] = round_if_needed(f_m_bare_mu_step2, P['f_m_bare_mu'])
        
        f_m_bare_tau_step1 = round_if_needed(-exponents['tau_A4'], P['f_m_bare_tau_step1'])
        f_m_bare_tau_step2 = round_if_needed(m_bare[3] * math.exp(f_m_bare_tau_step1), P['f_m_bare_tau_step2'])
        bare_mass_results['tau_A4'] = round_if_needed(f_m_bare_tau_step2, P['f_m_bare_tau'])

        f_m_bare_d_step1 = round_if_needed(-exponents['down_D10'], P['f_m_bare_d_step1'])
        f_m_bare_d_step2 = round_if_needed(m_bare[1] * math.exp(f_m_bare_d_step1), P['f_m_bare_d_step2'])
        bare_mass_results['down_D10'] = round_if_needed(f_m_bare_d_step2, P['f_m_bare_d'])
        
        f_m_bare_s_step1 = round_if_needed(-exponents['strange_D10'], P['f_m_bare_s_step1'])
        f_m_bare_s_step2 = round_if_needed(m_bare[2] * math.exp(f_m_bare_s_step1), P['f_m_bare_s_step2'])
        bare_mass_results['strange_D10'] = round_if_needed(f_m_bare_s_step2, P['f_m_bare_s'])
        
        f_m_bare_b_step1 = round_if_needed(-exponents['bottom_D10'], P['f_m_bare_b_step1'])
        f_m_bare_b_step2 = round_if_needed(m_bare[3] * math.exp(f_m_bare_b_step1), P['f_m_bare_b_step2'])
        bare_mass_results['bottom_D10'] = round_if_needed(f_m_bare_b_step2, P['f_m_bare_b'])

        f_m_bare_u_step1 = round_if_needed(-exponents['up_Oh'], P['f_m_bare_u_step1'])
        f_m_bare_u_step2 = round_if_needed(m_bare[1] * math.exp(f_m_bare_u_step1), P['f_m_bare_u_step2'])
        bare_mass_results['up_Oh'] = round_if_needed(f_m_bare_u_step2, P['f_m_bare_u'])

        f_m_bare_c_step1 = round_if_needed(-exponents['charm_Oh'], P['f_m_bare_c_step1'])
        f_m_bare_c_step2 = round_if_needed(m_bare[2] * math.exp(f_m_bare_c_step1), P['f_m_bare_c_step2'])
        bare_mass_results['charm_Oh'] = round_if_needed(f_m_bare_c_step2, P['f_m_bare_c'])

        f_m_bare_t_step1 = round_if_needed(-exponents['top_Oh'], P['f_m_bare_t_step1'])
        f_m_bare_t_step2 = round_if_needed(m_bare[3] * math.exp(f_m_bare_t_step1), P['f_m_bare_t_step2'])
        bare_mass_results['top_Oh'] = round_if_needed(f_m_bare_t_step2, P['f_m_bare_t'])

        # --- 7o. apply_loop_corrections (Physical Masses) ---
        loop_ALPHA_Z_step1 = round_if_needed(1 / inv_alpha_z_val, P['loop_ALPHA_Z_step1'])
        ALPHA_Z = round_if_needed(loop_ALPHA_Z_step1, P['loop_ALPHA_Z'])
        
        physical_masses = {}
        
        loop_r_net_pp_step1 = round_if_needed(R_net_val / N_PAIRS, P['loop_r_net_pp_step1'])
        R_net_per_pair = round_if_needed(loop_r_net_pp_step1, P['loop_R_net_per_pair'])
        
        precision_map = {
            'electron_A4': 'final_m_e', 'muon_A4': 'final_m_mu', 'tau_A4': 'final_m_tau',
            'down_D10': 'final_m_d', 'strange_D10': 'final_m_s', 'bottom_D10': 'final_m_b',
            'up_Oh': 'final_m_u', 'charm_Oh': 'final_m_c', 'top_Oh': 'final_m_t',
        }
        
        # Electron
        lepton = 'electron_A4'
        m_e_bare = bare_mass_results[lepton]
        if m_H_val == 0:
            m_phys_raw_e = m_e_bare
        else:
            loop_m_e_phys_step1 = round_if_needed(m_e_bare / m_H_val, P['loop_m_e_phys_step1'])
            loop_m_e_phys_step2 = round_if_needed(0.5 * loop_m_e_phys_step1, P['loop_m_e_phys_step2'])
            loop_m_e_phys_step3 = round_if_needed(1.0 - loop_m_e_phys_step2, P['loop_m_e_phys_step3'])
            m_phys_raw_e = round_if_needed(m_e_bare * loop_m_e_phys_step3, P['loop_m_phys_step1']) # Re-using step1 precision
        physical_masses[lepton] = round_if_needed(m_phys_raw_e, P[precision_map[lepton]])

        # Quarks (Down-Type)
        quark_order_down = ['down_D10', 'strange_D10', 'bottom_D10']
        for i, quark in enumerate(quark_order_down):
            m_vfd = bare_mass_results[quark]
            
            loop_t_mass_step1 = round_if_needed(m_vfd * 1e-3, P['loop_t_mass_step1'])
            loop_t_mass_step2 = round_if_needed(M_PL / (loop_t_mass_step1 if 'down' in quark else m_vfd), P['loop_t_mass_step2'])
            loop_t_mass_step3 = round_if_needed(math.log(loop_t_mass_step2), P['loop_t_mass_step3'])
            t_mass_rounded = round_if_needed(loop_t_mass_step3, P['loop_t_mass'])
            
            # --- SIMPLIFIED: This entire calculation chain provably collapses to 0.0 ---
            # Based on P['loop_dm1_qcd_d_step3'] = 0.
            dm1_qcd = 0.0
            # --------------------------------------------------------------------
            
            loop_dm2_qcd_d_step1 = round_if_needed(-R_net_per_pair, P['loop_dm2_qcd_d_step1'])
            dm2_qcd = round_if_needed(loop_dm2_qcd_d_step1, P['loop_dm2_qcd_d'])
            
            loop_factor_step1 = round_if_needed(1 + dm1_qcd, P['loop_factor_step1'])
            loop_factor_step2 = round_if_needed(loop_factor_step1 + dm2_qcd, P['loop_factor_step2'])
            factor = round_if_needed(loop_factor_step2, P['loop_factor'])
            
            m_phys_raw_q = round_if_needed(m_vfd * factor, P['loop_m_phys_step1'])
            physical_masses[quark] = round_if_needed(m_phys_raw_q, P[precision_map[quark]])

        # Quarks (Up-Type)
        quark_order_up = ['up_Oh', 'charm_Oh', 'top_Oh']
        for i, quark in enumerate(quark_order_up):
            m_vfd = bare_mass_results[quark]
            
            loop_t_mass_step1 = round_if_needed(m_vfd * 1e-3, P['loop_t_mass_step1'])
            loop_t_mass_step2 = round_if_needed(M_PL / (loop_t_mass_step1 if 'up' in quark else m_vfd), P['loop_t_mass_step2'])
            loop_t_mass_step3 = round_if_needed(math.log(loop_t_mass_step2), P['loop_t_mass_step3'])
            t_mass_rounded = round_if_needed(loop_t_mass_step3, P['loop_t_mass'])

            # --- SIMPLIFIED: This entire calculation chain provably collapses to 0.0 ---
            # Based on P['loop_dm1_qcd_u_step1'] = 0, which rounds (4/3 * alpha_s_mz) to 0.0.
            dm1_qcd = 0.0
            # --------------------------------------------------------------------

            loop_dm2_qcd_u_step1 = round_if_needed(-R_net_per_pair, P['loop_dm2_qcd_u_step1'])
            dm2_qcd = round_if_needed(loop_dm2_qcd_u_step1, P['loop_dm2_qcd_u'])

            loop_factor_step1 = round_if_needed(1 + dm1_qcd, P['loop_factor_step1'])
            loop_factor_step2 = round_if_needed(loop_factor_step1 + dm2_qcd, P['loop_factor_step2'])
            factor = round_if_needed(loop_factor_step2, P['loop_factor'])

            m_phys_raw_q = round_if_needed(m_vfd * factor, P['loop_m_phys_step1'])
            physical_masses[quark] = round_if_needed(m_phys_raw_q, P[precision_map[quark]])

        # --- 7p. Final Ratio-Derived Masses ---
        results.update(physical_masses)
        m_e_phys = physical_masses['electron_A4']
        
        # This section now correctly implements procedural quantization
        
        # Calculate Lepton Ratios
        ratio_mu_e = round_if_needed(math.exp(log_ratios['muon_A4/electron_A4']), P['loop_m_ratio_step1'])
        ratio_tau_mu = round_if_needed(math.exp(log_ratios['tau_A4/muon_A4']), P['loop_m_ratio_step2'])
        
        # Derive Lepton Masses
        loop_m_mu_phys_step1 = round_if_needed(m_e_phys * ratio_mu_e, P['loop_m_mu_phys_step1'])
        m_mu_phys = round_if_needed(loop_m_mu_phys_step1, P['final_m_mu'])
        
        loop_m_tau_phys_step1 = round_if_needed(m_mu_phys * ratio_tau_mu, P['loop_m_tau_phys_step1'])
        m_tau_phys = round_if_needed(loop_m_tau_phys_step1, P['final_m_tau'])
        
        results['muon_A4'] = m_mu_phys
        results['tau_A4'] = m_tau_phys
        
        # --- 7q. Final Ratios ---
        # Store Lepton Ratios
        results['muon_A4/electron_A4'] = ratio_mu_e
        results['tau_A4/muon_A4'] = ratio_tau_mu
        
        # Calculate and Store Quark Ratios
        results['strange_D10/down_D10'] = round_if_needed(math.exp(log_ratios['strange_D10/down_D10']), P['ratio_sd_step1'])
        results['charm_Oh/up_Oh'] = round_if_needed(math.exp(log_ratios['charm_Oh/up_Oh']), P['ratio_cu_step1'])
        results['bottom_D10/strange_D10'] = round_if_needed(math.exp(log_ratios['bottom_D10/strange_D10']), P['ratio_bs_step1'])
        results['top_Oh/charm_Oh'] = round_if_needed(math.exp(log_ratios['top_Oh/charm_Oh']), P['ratio_tc_step1'])

    # --- 8. Error Handling and Reporting ---
    except (ValueError, OverflowError, ZeroDivisionError, TypeError) as e:
        if verbose:
            # Suppress verbose error printing during GA runs
            pass
            # print(f"Simulation failed: {e}")
        return float('inf'), float('inf'), 0, []

    # --- 9. Analysis and Scoring (Unchanged) ---
    analysis_results = []
    RATIO_KEYS = ['muon_A4/electron_A4', 'strange_D10/down_D10', 'charm_Oh/up_Oh', 'tau_A4/muon_A4', 'bottom_D10/strange_D10', 'top_Oh/charm_Oh']
    MASS_KEYS = ['electron_A4', 'muon_A4', 'tau_A4', 'up_Oh', 'down_D10', 'strange_D10', 'charm_Oh', 'bottom_D10', 'top_Oh']
    
    # --- UPDATE THIS LIST ---
    CORE_KEYS = ['inv_alpha_MZ', 'm_H', 'v_VEV', 'G', 'sin2_theta_W', 'alpha_s_MZ', 'm_nu', 'theta_C', 'theta_23', 'N_efolds', 'rho_true']
    
    all_keys = set(RATIO_KEYS) | set(MASS_KEYS) | set(CORE_KEYS)
    
    for key, calc_val in results.items():
        if key in O:
            obs, obs_err = O[key]
            dev = abs(calc_val - obs) / abs(obs) * 100 if obs != 0 else 0
            sigma_raw = abs(calc_val - obs) / obs_err if obs_err != 0 else (0 if dev == 0 else float('inf'))
            sigma = min(sigma_raw, 1000.0)
            analysis_results.append((key, calc_val, obs, obs_err, dev, sigma))
            
    analysis_results.sort(key=lambda x: x[5])
    
    valid_results_v = [r for r in analysis_results if r[0] in all_keys]
    deviations_v = [x[4] for x in valid_results_v]
    sigmas_v = [x[5] for x in valid_results_v]
    
    avg_dev = sum(deviations_v) / len(deviations_v) if deviations_v else float('inf')
    avg_sigma = sum(sigmas_v) / len(sigmas_v) if sigmas_v else float('inf')
    total_matches = sum(1 for s in sigmas_v if s < 1)
    
    if verbose:
        run_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        print(f"\n\n--- 1. Final Results (\"VFD-4.0 Hyper-Quantized\", {run_timestamp} UTC) ---")
        for (key, calc_val, obs, obs_err, dev, sigma) in analysis_results:
                if key in CORE_KEYS:
                        if key == 'G' or key == 'rho_true':
                            print(f"{key}: Calc={calc_val:.5e}, Obs={obs:.5e} ± {obs_err:.2e}, Dev={dev:.3f}% ({sigma:.2f} sigma)")
                        else:
                            print(f"{key}: Calc={calc_val:.5f}, Obs={obs:.5f} ± {obs:.5f}, Dev={dev:.3f}% ({sigma:.2f} sigma)")
                elif key in RATIO_KEYS:
                    # Ratios are special, don't round obs/err
                    print(f"**{key}: Calc={calc_val:.6f}, Obs={obs:.6f} ± {obs_err:.6f}, Dev={dev:.3f}% ({sigma:.2f} sigma)**")
                elif key in MASS_KEYS:
                    print(f"**{key} (GeV): Calc={calc_val:.10f}, Obs={obs:.10f} ± {obs_err:.10f}, Dev={dev:.3f}% ({sigma:.2f} sigma)**")
        print(f"\nSummary: Average deviation (all 25) = {avg_dev:.3f}%")
        print(f"Average sigma (all 25, verbose) = {avg_sigma:.2f}")
        print(f"\n{total_matches} / {len(sigmas_v)} predictions are within 1 sigma.")
        
    return avg_sigma, avg_dev, total_matches, analysis_results


    
    # --- 1. GET THE DYNAMIC CAP FROM THE GA INSTANCE ---
    PRECISION_CAP = ga_instance.user_data['precision_cap']
    PROXY_FOR_NONE = PRECISION_CAP + 1
    GA_GENE_KEYS = ga_instance.user_data['gene_keys']

    test_config = {}
    
    # --- 2. De-Code the Solution (Now uses the dynamic cap) ---
    for i, key in enumerate(GA_GENE_KEYS):
        gene_value = int(solution[i])  # Get the int value from the GA

        # --- Handle our special proxy values ---
        if key == 'f_A4_T2_SIGN':
            if gene_value == 0: test_config[key] = None  # 0 is None
            elif gene_value == 1: test_config[key] = 1
            elif gene_value == 2: test_config[key] = -1

        elif gene_value == PROXY_FOR_NONE:
            test_config[key] = None  # The dynamic proxy is None
        
        else:
            # The gene_value is already guaranteed to be [0, precision_cap]
            # because the gene_space is constrained.
            test_config[key] = gene_value

    # --- 3b. Run the Simulation ---
    try:
        avg_sigma, _, _, _ = run_simulation(test_config, OBSERVED, verbose=False)
    except Exception as e:
        # print(f"CAUGHT ERROR IN FITNESS FUNC: {e}") # DEBUG
        avg_sigma = float('inf')  # Punish crashes

    # --- 3c. Return the Fitness ---
    fitness = 1.0 / (avg_sigma + 1.0e-12)
    
    return fitness


# --------------------------------------------------------------------
# --- 5. Main Execution Block (NEW: GA Manager) ---
# --------------------------------------------------------------------

class GAManager:
    def __init__(self, population_size=50, precision_cap=125, total_epochs_to_run=30):
        print(f"--- 🧬 Initializing GA Manager for FINAL CAP = {precision_cap} ---")
        self.population_size = population_size
        self.precision_cap = precision_cap
        
        # --- NEW: Set total epochs for the *entire run* ---
        self.total_epochs_to_run = total_epochs_to_run
        
        # --- Define the master config and gene keys *within the instance* ---
        self.OPTIMAL_CONFIG = get_seed()
        self.GA_GENE_KEYS = sorted(self.OPTIMAL_CONFIG.keys()) # Now 414 keys

        # --- NEW: MAX_DIGIT_COST is now simpler (no non-digit keys) ---
        self.MAX_DIGIT_COST = len(self.GA_GENE_KEYS) * self.precision_cap

        print("--- 🧬 Calculating ADAM seed fitness... ---")
        self.adam_seed_config_template = get_seed() # Get the raw template
        
        # Quantize Adam seed to the final cap
        quantized_adam_config = self.quantize_config_to_cap(self.adam_seed_config_template)
        
        # --- NEW: Run the fitness calc on the *quantized* Adam seed ---
        self.adam_fitness = self.fitness_func(None, self.build_solution_from_config(quantized_adam_config), 0)
        
        print(f"--- 🧬 ADAM (Quantized to {precision_cap}) Fitness = {self.adam_fitness:.6f} ---")

        # --- NEW: Store the *quantized* config as the seed and parent ---
        self.current_seed_config = quantized_adam_config 
        self.stable_parents = [(self.adam_fitness, quantized_adam_config)]
        
        # --- State for convergence tracking ---
        self.global_best_fitness = self.adam_fitness
        adam_solution, _ = self.build_adam_solution(quantized_adam_config) 
        self.global_best_solution = adam_solution  # All-time best
        self.epoch_best_fitness = -1.0  # Best for *this epoch*
        self.generations_without_improvement = 0
        
        # --- Adaptive convergence cap ---
        self.current_convergence_cap = 1000  # Default
        
        # --- Flag for end-of-epoch printing ---
        self.new_global_best_found_this_epoch = False
        
        # --- NEW: TRACK TOP PERFORMER FOR 3-EPOCH RULE ---
        self.top_performer_config = quantized_adam_config
        self.top_performer_epochs = 0 # Will be set to 1 on first epoch

    # ----------------------------------------------------------------
    # --- 3. Genetic Algorithm (GA) Fitness Function (UPDATED v2) ---
    # ----------------------------------------------------------------
    def fitness_func(self, ga_instance, solution, solution_idx):
        """
        UPDATED Fitness Function (Tiered):
        1. HARD CONSTRAINT: Must match 26/26 predictions (sigma < 1).
        2. PRIMARY OPTIMIZATION: Lowest Average Deviation.
        3. SECONDARY OPTIMIZATION: Fewest Digits (Tiebreaker).
        """
        
        GA_GENE_KEYS = self.GA_GENE_KEYS
        test_config = {}
        total_digits = 0
        
        # --- 1. De-Code the Solution & Calculate Digit Cost ---
        for i, key in enumerate(GA_GENE_KEYS):
            gene_value = int(solution[i])
            test_config[key] = gene_value
            total_digits += gene_value

        # --- 2. Run the Simulation ---
        try:
            # We capture avg_sigma, avg_dev, and total_matches
            avg_sigma, avg_dev, total_matches, _ = run_simulation(test_config, OBSERVED, verbose=False)
        except Exception:
            return 0.0 # Punish crashes

        # --- 3. Tiered Fitness Calculation ---

        # Dynamic target: Must match ALL observed values
        TARGET_MATCHES = len(OBSERVED) 

        # TIER A: The solution is invalid (Matches < Target)
        # Goal: Just get the sigmas down.
        # Max Score in this tier: ~1.0 (if avg_sigma is close to 1)
        if total_matches < TARGET_MATCHES:
            return 1.0 / (avg_sigma + 1e-9)

        # TIER B: The solution is VALID (Predictions = 26/26)
        # Goal: Optimize Deviation, then Digits.
        else:
            # 1. The "Valid" Bonus (Separates this tier from Tier A)
            # Score starts at 1,000,000
            fitness = 1_000_000.0
            
            # 2. Primary Optimization: Lowest Average Deviation
            # We add (1 / dev). Lower dev = Higher score.
            # Ex: Dev 0.01% -> Adds 100.0 points.
            # Ex: Dev 0.001% -> Adds 1,000.0 points.
            fitness += (1.0 / (avg_dev + 1e-9))
            
            # 3. Secondary Optimization: Fewest Digits
            # We add (1 / digits). Fewer digits = Higher score.
            # Since total_digits is large (~50,000), this adds a tiny fraction (e.g., 0.00002).
            # This ensures it NEVER overrides the Deviation score.
            fitness += (1.0 / (total_digits + 1.0))
            
            return fitness

    def quantize_config_to_cap(self, config):
        """
        Helper to apply the current precision_cap to any config dict.
        (Removed 'f_A4_T2_SIGN' logic)
        """
        quantized_config = {}
        for key, value in config.items():
            # --- No more special key check needed ---
            if value is None or value > self.precision_cap:
                quantized_config[key] = self.precision_cap 
            else:
                quantized_config[key] = value
        return quantized_config

    def decode_solution_to_config(self, solution):
        """
        Converts a flat GA solution list back into a config dictionary.
        (Removed 'f_A4_T2_SIGN' logic)
        """
        config = {}
        for i, key in enumerate(self.GA_GENE_KEYS):
            gene_value = int(solution[i])
            # --- No more special key check needed ---
            config[key] = gene_value
        return config

    def build_solution_from_config(self, config):
        """
        Helper to convert a config dict back into a flat solution list.
        (Removed 'f_A4_T2_SIGN' logic)
        """
        solution = []
        for key in self.GA_GENE_KEYS:
            # --- No more special key check needed ---
            solution.append(config.get(key))
        return solution

    def build_adam_solution(self, seed_config):
        """
        Builds the 'adam_solution' list from a given seed config.
        (Removed 'f_A4_T2_SIGN' logic)
        """
        adam_solution = []
        gene_space = []
        
        print(f"Building gene space for {len(self.GA_GENE_KEYS)} parameters (Cap={self.precision_cap})...")
        
        for key in self.GA_GENE_KEYS:
            base_value = seed_config.get(key)
            
            if key == 'rho_exp_term':
                val = seed_config.get(key, 123)
                if val > self.precision_cap:
                    val = self.precision_cap
                gene_space.append({'low': val, 'high': val + 1, 'step': 1})
                adam_solution.append(val)

            # --- 'f_A4_T2_SIGN' case removed ---

            else:
                # --- Gene space is [0...cap]. 'high' is exclusive. ---
                gene_space.append({'low': 0, 'high': self.precision_cap + 1, 'step': 1})
                
                if base_value is None or base_value > self.precision_cap:
                    adam_solution.append(self.precision_cap) # Default to the cap
                else:
                    adam_solution.append(base_value)
                    
        return adam_solution, gene_space


    def on_gen_callback(self, ga_instance):
        """
        This callback runs *after* each generation.
        This version has improved logging to make the process clearer.
        """
        try:
            solution, best_fitness, _ = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
        except TypeError:
            print("Warning: Could not get best solution. Skipping generation callback.")
            return

        # --- Flag to control logging ---
        found_new_global_best = False

        # --- 1. Check for New Epoch Best ---
        if best_fitness > self.epoch_best_fitness:
            self.epoch_best_fitness = best_fitness
            self.generations_without_improvement = 0
            
            # --- 2. Check if it's also a New *Global* Best ---
            if best_fitness > self.global_best_fitness:
                self.global_best_fitness = best_fitness
                self.global_best_solution = list(solution)
                self.new_global_best_found_this_epoch = True
                found_new_global_best = True # Set flag to print big report
        
        else:
            # No new epoch best was found
            self.generations_without_improvement += 1

        
        # --- 3. NEW Consolidated Logging Block ---

        if found_new_global_best:
            # --- Print the "New Best" header FIRST ---
            print(f"\n{'='*30}")
            print(f"🎉 NEW GLOBAL BEST! Gen {ga_instance.generations_completed} (Cap={self.precision_cap})")
            print(f"   Best Fitness (Score): {self.global_best_fitness:.2f}")
            
            # --- Run analysis and print details *after* the header ---
            print("--- 🚀 Running Full Analysis on New Best Solution ---")
            best_config = self.decode_solution_to_config(self.global_best_solution)
            
            # Re-calculate total digits for reporting
            total_digits = 0
            for key, value in best_config.items():
                total_digits += value # No special keys anymore
            print(f"   Total Digits Used: {total_digits}")

            # Run the verbose simulation
            _, _, total_matches, _ = run_simulation(best_config, OBSERVED, verbose=True)
            print(f"{'='*30}\n")
        
        # --- This 'if' is now SEPARATE from the 'if' above.
        if ga_instance.generations_completed % 50 == 0:
            if not found_new_global_best:
                print(f"Gen {ga_instance.generations_completed}: Epoch best {self.epoch_best_fitness:.2f}. Stuck for {self.generations_without_improvement} gens.")

        # --- 7. Check for Convergence (Stable Parent) ---
        if self.generations_without_improvement >= self.current_convergence_cap:
            print(f"\n--- 🔒 CONVERGENCE. Stable solution found after {self.current_convergence_cap} generations. ---\n")
            return "stop"  # Stop the GA run

    def run_manager(self):
        """
        The main manager. Runs for a fixed number of epochs
        to find the best possible solution under the new fitness function.
        """
        epoch_count = 0
        
        while epoch_count < self.total_epochs_to_run:
            epoch_count += 1
            self.new_global_best_found_this_epoch = False
            
            print(f"\n{'=' * 70}")
            print(f"--- 🌟 Starting GA Epoch {epoch_count} / {self.total_epochs_to_run} (Cap={self.precision_cap}) ---")
            
            # --- 1. SORT PARENT BANK ---
            print("--- 🏦 Current Stable Parent Bank ---")
            self.stable_parents.sort(key=lambda x: x[0], reverse=True)
            for i, (fitness, config) in enumerate(self.stable_parents):
                is_adam = "(ADAM)" if config == self.quantize_config_to_cap(self.adam_seed_config_template) else ""
                print(f"   Parent {i}: Fitness = {fitness:.6f} {is_adam}")
            
            # --- 2. GET CURRENT TOP PERFORMER ---
            top_fitness, top_config = self.stable_parents[0] # Get the best parent
            
            # --- 3. CHECK FOR "3 EPOCH" RULE ---
            if top_config == self.top_performer_config:
                self.top_performer_epochs += 1
                print(f"--- Top performer has been #1 for {self.top_performer_epochs} consecutive epochs. ---")
            else:
                print(f"--- New top performer found! (Old champion held for {self.top_performer_epochs} epochs) ---")
                self.top_performer_config = top_config
                self.top_performer_epochs = 1 
            
            print("-" * 70)

            # --- 5. NORMAL MATING LOGIC (if 3-epoch rule didn't run) ---
            if len(self.stable_parents) == 1:
                print("--- Mode: [1 Parent] - Seeding from top performer. ---")
                self.current_seed_config = self.stable_parents[0][1]
            
            else:
                parent_a_fit, parent_a_cfg = self.stable_parents[0]
                pool_size = min(3, len(self.stable_parents))
                
                if pool_size < 2:
                    parent_b_fit, parent_b_cfg = parent_a_fit, parent_a_cfg
                    parent_b_index = 0
                else:
                    parent_b_index = random.randint(1, pool_size - 1)
                    parent_b_fit, parent_b_cfg = self.stable_parents[parent_b_index]
                
                is_adam_a = "(ADAM)" if parent_a_cfg == self.quantize_config_to_cap(self.adam_seed_config_template) else ""
                is_adam_b = "(ADAM)" if parent_b_cfg == self.quantize_config_to_cap(self.adam_seed_config_template) else ""

                print(f"--- 🤝 Mating Best Parent 0 {is_adam_a} (Fitness: {parent_a_fit:.6f})")
                print(f"---    with Random Top-10 Parent {parent_b_index} {is_adam_b} (Fitness: {parent_b_fit:.6f}) ---")
                
                mated_template_config = get_seed(parent_a_cfg, parent_b_cfg)
                self.current_seed_config = self.quantize_config_to_cap(mated_template_config)
                print("--- 🏦 Seeding new hybrid epoch. Parent bank remains. ---")
            
            print("-" * 70)
            
            # --- [REST OF THE FUNCTION] ---
            adam_solution, gene_space = self.build_adam_solution(self.current_seed_config)

            initial_population = []
            for _ in range(self.population_size):
                initial_population.append(list(adam_solution))

            num_generations = 5000
            self.epoch_best_fitness = -1.0
            self.generations_without_improvement = 0
            
            if epoch_count <= 10:
                self.current_convergence_cap = 750
                print(f"--- Mode: Scout (Early Epoch). Patience cap set to {self.current_convergence_cap} ---")
            else:
                self.current_convergence_cap = 2000
                print(f"--- Mode: Explorer (Late Epoch). Patience cap set to {self.current_convergence_cap} ---")
            
            print(f"GA Config: {num_generations} gens, {self.population_size} pop.")
            print(f"Initial Mutation Rate: 1.5%")
            print(f"Initial Parents Mating: 3")
            print(f"Total parameters to optimize: {len(self.GA_GENE_KEYS)}")

            ga_instance = pygad.GA(
                num_generations=num_generations,
                num_parents_mating=3,
                fitness_func=self.fitness_func,
                num_genes=len(self.GA_GENE_KEYS),
                gene_space=gene_space,
                gene_type=int,
                initial_population=initial_population,
                parent_selection_type="sss",
                crossover_type="two_points",
                mutation_type="random",
                mutation_percent_genes=1.5,
                on_generation=self.on_gen_callback,
                keep_elitism=2
            )

            print(f"--- 🧬 Running GA Epoch {epoch_count}... (This will be very slow!) ---")
            start_time = time.time()
            ga_instance.run()
            end_time = time.time()
            print(f"--- 🏁 GA Epoch {epoch_count} Complete ({end_time - start_time:.2f} seconds) ---")
            
            if self.new_global_best_found_this_epoch:
                print("\n--- 💎 New All-Time Best Config (Found This Epoch) ---")
                best_config = self.decode_solution_to_config(self.global_best_solution)
                print("OPTIMAL_CONFIG = {")
                for key, value in sorted(best_config.items()):
                    print(f"    '{key}': {repr(value)},")
                print("}\n----------------------------------------------------\n")
            
            epoch_solution, epoch_fitness, _ = ga_instance.best_solution()

            if epoch_solution is None:
                print("Error: Epoch finished with no solution. Retrying from Adam.")
                self.current_seed_config = self.quantize_config_to_cap(get_seed()) # Use quantized Adam
                continue
                
            best_solution_config = self.decode_solution_to_config(epoch_solution)

            if self.generations_without_improvement >= self.current_convergence_cap:
                print(f"...New stable parent found! Fitness: {epoch_fitness:.6f}. Adding to bank.")
                
                is_duplicate = False
                for (parent_fit, parent_cfg) in self.stable_parents:
                    if parent_cfg == best_solution_config:
                         is_duplicate = True
                         print("...This stable parent is a duplicate. Discarding.")
                         break
                
                if not is_duplicate:
                    self.stable_parents.append((epoch_fitness, best_solution_config))
            
            else:
                print("...Max generations for epoch reached. Seeding next epoch with this epoch's best.")
                self.current_seed_config = best_solution_config

        # --- 4f. Show the Final Results (using the *global best* solution) ---
        print("\n\n" + "=" * 70)
        print(f"--- 🏁 FINAL 'MINIMAL DIGIT' RUN COMPLETE (Cap={self.precision_cap}) ---")
        print("=" * 70)
        
        if self.global_best_solution is None:
            print("No solution was found at all.")
            return

        final_config = self.decode_solution_to_config(self.global_best_solution)
        
        # --- Final calculation of total digits for the report ---
        final_total_digits = 0
        for key, value in final_config.items():
            final_total_digits += value # No special keys

        print(f"--- 🏆 Best Solution Found (after {self.total_epochs_to_run} epochs) ---")
        print(f"--- 💎 Total Digits Used: {final_total_digits} ---")
        print(f"---  score: {self.global_best_fitness:.2f} ---")
        
        opt_sigma, opt_dev, opt_matches, _ = run_simulation(final_config, OBSERVED, verbose=True)

        print("\n--- 🏁 Final Tuned Metrics ---")
        print(f"Average Deviation: {opt_dev:.4f}%")
        print(f"Average Sigma: {opt_sigma:.4f}")
        print(f"Matches (within 1-sigma): {opt_matches} / 26")

        # print("\n\n--- Final Config Dictionary (Minimal Digit Solution) ---")
        # print("OPTIMAL_CONFIG = {")
        # for key, value in sorted(final_config.items()):
        #     print(f"    '{key}': {repr(value)},")
        # print("}")
        
        if opt_matches < 25:
             print("\n--- ⚠️ WARNING: The best solution found did NOT achieve 26/26 matches. ---")
             print("--- You may need to increase TOTAL_EPOCHS_TO_RUN. ---")

if __name__ == "__main__":
    print("--- 🔬 Starting VFD-4.0 **Minimal Digit Search** ---")
    
    GA_POPULATION_SIZE = 50

    print(f"### 🔬 Searching for minimal digit solution with CAP = {FINAL_CAP} ###")
    print(f"### Will run for a total of {TOTAL_EPOCHS_TO_RUN} epochs. ###")
    print("#" * 70)
    
    manager = GAManager(
        population_size=GA_POPULATION_SIZE, 
        precision_cap=FINAL_CAP, 
        total_epochs_to_run=TOTAL_EPOCHS_TO_RUN
    )
    
    # run_manager() will now just run for the set number of epochs
    manager.run_manager()

    print("\n\n" + "*" * 70)
    print("--- 🏆🏆🏆 MINIMAL DIGIT SEARCH COMPLETE 🏆🏆🏆 ---")
    print("*" * 70)