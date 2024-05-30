#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:02:51 2024

@author: mjt-zdz
"""

import numpy as np
import fug
import HD_sEMG
import os
import scipy.io


if __name__=="__main__":
    # 定义参数
    config = {
        # MU数量
        't1': 102, # 1~400 
        't2a': 16, # 1~100
        't2b': 2, # 1~100
        # 采样频率 Hz
        'sampling': 2048,
        # 模拟时长 [ms]
        'duration': 10e3, 
        # 招募范围
        'rr': 30, # 10~200
        # 峰值放电率差值 [Hz]
        'pfrd' : 20, # -30~30 
        # 第一个峰值放电率 [Hz]
        'firstPFR': 35, # 10~55
        # 最小放电率 [Hz]
        'mfr': 3, # 1~20
        # 兴奋驱动和放电率之间的增益
        'gain_cte': False, # 设置为True表示所MU的增益一样
        'gain_factor': 2, # 增益因子 1~10
        # 激活运动神经元的总数
        'LR': 1, 
        # 招募最后一个运动神经元所需的相对兴奋驱动 
        'rrc': 67, # 5~99.1 %
        # 最小峰间间隔 [ms]
        'ISI_limit': 15, # 根据肌肉类型进行设置 5~30 ms
                
        ### 兴奋驱动曲线参数
        'mode': 'Sinusoidal', # 兴奋驱动模式，可选Trapezoidal Sinusoidal 
        'intensity': 30, # 兴奋驱动强度 0~100 %
        # 梯形
        't00': 500, # 兴奋驱动开始时间
        't01': 2500, # 兴奋驱动平台开始时间
        't02': 7500, # 兴奋驱动平台结束时间
        't03': 9500, # 兴奋驱动结束时间
        # 正弦
        'freq_sin': 1, # 正弦波频率 0~2
        
        ### 运动神经元同步参数
        'synch_level': 25, # 同步比例 0~30 %
        'sigma': 2, # 正态分布标准差 0~5 ms
        'CoV': 20, # isi变异系数 1~50 %
        
        ### 体积导体模型参数
        # 肌肉形态
        'morpho': 'Circle', # 肌肉截面形态，可选Circle Ring Pizza Ellipse
        'csa': 200, # 肌肉截面积 100~1000 mm^2
        'fat': 1, # 脂肪层厚度 0~5 mm
        'skin': 0.3, # 皮肤层厚度 0~3 mm
        'theta': 0.9, # opening弧度 0.05~np.pi/2
        'prop': 0.4, # 内外径或短轴长轴比例 0.1~0.8
        'length': 300, # 肌肉长度 200~400 mm
        'nmj_pos': 150, # 神经肌肉接点为位置（z坐标） mm
        'iz_width': 10, # 神经支配区的宽度 5~20 mm
        # 电极阵列
        'elect_z': 14, # 阵列中沿肌纤维方向的电极数量
        'elect_x': 9, # 阵列中横跨肌纤维的电极数量
        'elect_d': 4, # 电极间距离 mm
        'elect_loc': 130, # 电极位置 < 肌肉长度 mm
        # 运动神经元及其支配的肌纤维参数
        'first': 21, # 最小运动神经元支配的肌纤维数量
        'ratio': 84, # 最小和最大运动神经元之间的神经支配（肌纤维数量）比例 10~200
        't1m': 0.7, # I型MUs的距离分布平均（相对于肌肉半径） 0~1
        't2m':1, # # II型MUs的距离分布平均（相对于肌肉半径） 0~1 
        't1dp':0.5, # I型MUs的距离分布标准差（相对于肌肉半径），用于计算生成随机极坐标半径的标准差 0.2~1
        't2dp':0.25, # II型MUs的距离分布标准差（相对于肌肉半径），用于计算生成随机极坐标半径的标准差 0.2~1
        # 运动单元动作电位参数
        'v1': 1, # 第一个招募的运动单元动作电位振幅因子 0.005~100 mV
        'v2': 130, # 最后一个招募的运动单元动作电位振幅因子 2~200 mV
        'd1': 3, # 第一个招募的运动单元动作电位持续时间因子 0.1~10 ms
        'd2': 1, # 最后一个招募的运动单元动作电位持续时间因子 0.1~10 ms
         # 'add_hr': '2st order', # HR函数的阶数
         # 'cv_mean': 4.0, # 运动单元传导速度均值 1~10 m/s（慢肌）10~120 m/s（快肌） 
         # 'cv_std': 0.35, # 运动单元标准差 m/s
         # 'rc': 0.02, # 径向电导率 0.02~0.05 S/m
         # 'arc_ratio': 10, # 轴向和径向电导率的比值 4~25
         # 体积导体衰减常数
         'ampk': 5, # 振幅衰减常数 1~20 mm
         'durak': 0.1, # 持续时间衰减常数 0.1~10 mm^(-1)
        ### 肌电信号处理参数
        'add_noise': False, # 是否添加噪声
        'noise_level':0, # 噪声水平
        'add_filter': False, # 是否使用滤波器
        'order': 4, # 滤波器阶数
        'lowcut': 0, # 低端截止频率
        'highcut': 0, # 高端截止频率
        'order': 4, # 滤波器阶数
        }
    # 模拟步长 [s]
    dt = 1/config['sampling']
    # 时间数组
    t = np.arange(0, config['duration']*1e3, dt*1e3)
    
    config['dt'] = dt
    config['t'] = t
    config['t_size'] = len(t)
    config['save'] = '/media/root/data/mjt/doctoral_project/neuromuscular_notebook-master/results_3d_Sinusoidal'
    if not os.path.exists(config['save']):
        os.makedirs(config['save'])
            
    mn_pool = fug.Phemo() # 定义运动神经元池实例
    
    # view_organization(self, rr, mfr, firstPFR, PFRD, RRC, t1, t2a, t2b, gain_factor, gain_CTE, save)
    mn_pool.view_organization(config['rr'], config['mfr'], config['firstPFR'], config['pfrd'], config['rrc'], 
                              config['t1'], config['t2a'], config['t2b'], config['gain_factor'], config['gain_cte'], config['save'])
    
    # view_excitatory(self, t0, t1, t2, t3, freq_sin, mode, intensity, sample_time, sim_time, save)
    mn_pool.view_excitatory(config['t00'], config['t01'], config['t02'], config['t03'], config['freq_sin'], config['mode'], config['intensity'],
                            config['sampling'], config['duration'], config['save'])
    
    # view_neural_command(self, CoV, synch_level, sigma)
    mn_pool.view_neural_command(config['CoV'], config['synch_level'], config['sigma'])
    
    conductor = HD_sEMG.Emg_mod(mn_pool) # 定义体积导体实例
    
    # view_morpho(self, csa, fat, skin, theta, prop, length, morpho, elect_z, elect_x, elect_d, elect_loc, save)
    conductor.view_morpho(config['csa'], config['fat'], config['skin'], config['theta'], config['prop'], config['length'],
                          config['morpho'], config['elect_z'], config['elect_x'], 
                          config['elect_d'], config['elect_loc'], config['save'])

    # view_distribution(self, ratio, t1m, t1dp, t2m, t2dp)
    conductor.view_distribution(config['ratio'], config['t1m'], config['t1dp'], config['t2m'], config['t2dp'])

    # view_muap(self, v1, v2, d1, d2)
    conductor.view_muap(config['v1'], config['v2'], config['d1'], config['d2'])
        
    # view_semg(self, nmj_pos, iz_width, ampk, durak, add_noise, noise_level, add_filter, order, bplc, bphc)
    conductor.view_semg(config['nmj_pos'], config['iz_width'], config['ampk'], config['durak'], config['add_noise'], config['noise_level'], 
                        config['add_filter'], config['order'], config['lowcut'], config['highcut'])
    
    spike_trains = mn_pool.neural_input
    t_array = mn_pool.t
    sEMG = conductor.emg
    muaps = conductor.mu_emg
    
    save_data = {'spike_trains': spike_trains, 't': t_array, 'sEMG': sEMG, 'muaps': muaps, 'fs': config['sampling']}
    
    scipy.io.savemat('sEMG_sim.mat', save_data)
    
    
    
    
    

