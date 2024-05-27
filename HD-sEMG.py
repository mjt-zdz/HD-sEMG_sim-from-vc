#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 09:39:32 2024

@author: root
"""

import sys
import numpy as np
from scipy.integrate import cumtrapz
from scipy.signal import butter, filtfilt, welch, get_window, spectrogram
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from IPython.display import clear_output
import os


class Emg_mod():
    def __init__(self, mnpool):
        self.morpho     = 'Ring' # 肌肉截面形态 可选Circle Ring Pizza Ellipse                
        self.csa        = 150 # 肌肉截面积 mm^2
        self.fat        = 0.2 # 脂肪厚度 mm
        self.skin       = 0.1 # 皮肤厚度 mm
        self.theta      = 0.9 # 定义所选几何形态的“opening”，即经过原点的垂线和经过原点的肌肉横截面的边界线之间形成的夹角 rad
        self.prop       = 0.4 # 比例 Ring环形——定义内外肌肉半径的比例；Ellipse椭圆形——定义短轴和长轴的比例
        self.first      = 21 # 最小运动神经元支配的肌纤维数量
        self.ratio      = 84 # 最大和最小运动神经元之间的神经支配比例
        self.t1m        = 0.7 # I型MUs的平均极坐标半径占肌肉截面半径的比例
        self.t1m_save   = 0 # 用于保存t1m
        self.t2m_save   = 0 # 用于保存t2m
        self.t1dp       = 0.5 # I型MUs的径向分布比例，用于计算生成随机极坐标半径的标准差
        self.t2m        = 1 # II型MUs的平均极坐标半径占肌肉截面半径的比例
        self.t2dp       = 0.25 # II型MUs的径向分布比例，用于计算生成随机极坐标半径的标准差
        self.emg        = [] # 肌电信号
        self.n          = mnpool.n # 运动单元总数
        self.t1         = mnpool.t1 # I型运动单元数量
        self.t2a        = mnpool.t2a # IIa型运动单元数量
        self.t2b        = mnpool.t2b # IIb型运动单元数量
        self.MUFN       = np.zeros(self.n) # 运动单元支配的肌纤维数量 
        self.MUradius   = np.zeros(self.n) # 运动单元半径
        self.LR         = mnpool.LR # 运动单元放电率
        self.t          = mnpool.t # 模拟时长
        self.rr         = mnpool.rr # 招募范围
        self.v1         = 1 # 第一个招募的运动单元动作电位振幅因子
        self.v2         = 130 # 最后一个招募的运动单元动作电位振幅因子
        self.d1         = 3 # 第一个招募的运动单元动作电位持续时间因子
        self.d2         = 1 # 最后一个招募的运动单元动作电位持续时间因子
        self.expn_interpol()
        self.exp_interpol()
        self.neural_input = mnpool.neural_input # 运动单元脉冲序列
        self.sampling     = mnpool.sampling # 采样频率
        self.mnpool       = mnpool # 运动神经元池类的实例
        self.config       = {}
        self.mu_distance  = []
        self.add_noise    = False # 是否添加噪声
        self.noise_level  = 0 # 噪声水平
        self.add_filter   = False # 是否使用滤波器
        self.lowcut       = 0 # 低端截止频率 
        self.highcut      = 0 # 高端截止频率

        
    def defineMorpho(self):
        """ 
        Defines the muscle morphology and electrode position
        定义肌肉形态和电极位置
        """
        if self.morpho == 'Circle':
            self.circle_tissues()
        elif self.morpho == 'Ring':
            self.ring_tissues()
        elif self.morpho == 'Pizza':
            self.pizza_tissues()
        elif self.morpho == 'Ellipse':
            self.prop = 1/self.prop
            self.ellipse_tissues()


    #### 需要增加肌肉的长度（肌肉截面改成三维柱状），神经支配区的位置，电极大小（z，x），电极间距，电极中心的位置 ——> 每个电极的x，y和z坐标
    def circle_tissues(self):
        """ 
        Draw Circle Tissue limits by creating arrays coordinates of the tissue boundaries.
        通过创建数组坐标来绘制圆形组织的边界
        """
        self.r = r =  np.sqrt(self.csa/np.pi) # 肌肉截面半径
        circle = np.arange(0,2*np.pi,0.01) # 极坐标肌肉层边界点数组
        self.ma = r * np.cos(circle) # 肌肉层边界点的x坐标
        self.mb = r * np.sin(circle) # 肌肉边界点的y坐标
        self.fa = (r+self.fat)* np.cos(circle) # 脂肪层边界点的x坐标
        self.fb = (r+self.fat)* np.sin(circle) # 脂肪层边界点的y坐标
        self.sa = (r+self.fat+self.skin)* np.cos(circle) # 皮肤层边界点的x坐标
        self.sb = (r+self.fat+self.skin)* np.sin(circle) # 皮肤层边界点的y坐标
        self.elec = r+self.fat+self.skin # 电极位置
        
        
    def ring_tissues(self):
        """ 
        Draw ring Tissue limits by creating arrays coordinates of the tissue boundaries. 
        通过创建组织边界的数组坐标绘制环形组织的边界
        """
        self.re = np.sqrt((self.csa)/(self.theta*(1-self.prop**2))) # 肌肉组织的外径
        self.ri = self.re * self.prop # 肌肉组织的内径
        angle = np.arange(np.pi/2-self.theta,np.pi/2+self.theta,0.01) # 肌肉层环形的弧度范围 
        t1 = [self.ri*np.cos(np.pi/2-self.theta)] # 肌肉内层边界点的x坐标
        t2 = self.re*np.cos(angle) # 肌肉外层边界点的x坐标
        self.ma = np.concatenate((t1,t2,np.flip(self.ri*np.cos(angle),0))) # 肌肉层边界点的x坐标
        t1 = [self.ri*np.sin(np.pi/2-self.theta)]  # 肌肉内层边界点的y坐标
        t2 = self.re*np.sin(angle) # 肌肉外层边界点的y坐标
        self.mb = np.concatenate((t1,t2,np.flip(self.ri*np.sin(angle),0))) # 肌肉层边界点的y坐标  
        self.fa = (self.re + self.fat) * np.cos(angle) # 脂肪层边界点的x坐标
        self.fb = (self.re + self.fat) * np.sin(angle) # 脂肪层边界点的y坐标
        self.sa = (self.re + self.fat + self.skin) * np.cos(angle) # 皮肤层边界点的x坐标
        self.sb = (self.re + self.fat + self.skin) * np.sin(angle) # 皮肤层边界点的y坐标
        self.elec = self.re + self.fat + self.skin # 电极位置
        
        
    def pizza_tissues(self):
        """ 
        Draw pizza like muscle tissue limits by creating arrays coordinates of the tissue boundaries.
        通过创建组织边界的数组坐标绘制批萨形组织的边界
        """
        self.r = np.sqrt(self.csa/self.theta) # 肌肉截面半径
        angle = np.arange(np.pi/2-self.theta, np.pi/2+self.theta, 0.01) # 批萨形的弧度范围
        self.ma = self.r * np.cos(angle) # 肌肉层边界点的x坐标
        self.mb = self.r * np.sin(angle) # 肌肉层边界点的y坐标
        self.ma = np.concatenate(([0],self.ma,[0])) 
        self.mb = np.concatenate(([0],self.mb,[0]))
        self.fa = (self.r+self.fat)* np.cos(angle) # 脂肪层边界点的x坐标
        self.fb = (self.r+self.fat)* np.sin(angle) # 脂肪层边界点的y坐标
        self.sa = (self.r+self.fat+self.skin)* np.cos(angle) # 皮肤层边界点的x坐标
        self.sb = (self.r+self.fat+self.skin)* np.sin(angle) # 皮肤层边界点的y坐标
        self.elec = self.r+self.fat+self.skin # 电极位置
        
        
    def ellipse_tissues(self):
        """ 
        Draw ellipse like muscle tissue limits by creating arrays of coordinates of the tissue boundaries. 
        通过创建组织边界的数组坐标绘制椭圆形组织的边界
        """
        self.b = np.sqrt(self.csa/(self.prop*np.pi)) # 椭圆肌肉组织的长轴
        self.a = self.prop*self.b # 椭圆肌肉组织的短轴
        circle = np.arange(0,2*np.pi,0.01) # 椭圆形的弧度范围
        self.ma = self.a * np.cos(circle) # 肌肉层边界点的x坐标
        self.mb = self.b * np.sin(circle) # 肌肉层边界点的y坐标
        self.fa = (self.a + self.fat) * np.cos(circle) # 脂肪层边界点的x坐标
        self.fb = (self.b + self.fat) * np.sin(circle) # 脂肪层边界点的y坐标
        self.sa = (self.a + self.fat + self.skin) * np.cos(circle) # 皮肤层边界点的x坐标
        self.sb = (self.b + self.fat + self.skin) * np.sin(circle) # 皮肤层边界点的y坐标
        self.elec = self.b + self.fat + self.skin # 电极位置


    def view_morpho(self, CSA, prop, theta, sk, fa, morpho, save):
        """
        Plot graphic with muscle cross-sectional area morphology.
        绘制肌肉横截面

        Args:
            CSA (float): Muscle cross sectional area [mm²]. 肌肉横截面积
            prop (float): Muscle radius proportion. 内外径或短长轴比例
            theta (float): Theta angle. Parameter to define muscle morphology. 'opening'角度
            sk (float): Skin tickness [mm]. 皮肤层厚度
            fa (float): Adipose tickness [mm]. 脂肪层厚度
            morpho (string): Muscle morphology. 肌肉截面形态
        """
        self.csa, self.skin, self.fat = CSA*10**-6, sk*10**-3, fa*10**-3
        self.theta, self.prop, self.morpho= theta, prop , morpho
        self.save = save
        self.defineMorpho()
        plt.figure(figsize=(5,4))
        plt.plot(self.ma*1e3, self.mb*1e3, ls='-.', label='Muscle boundaries')
        plt.plot(self.fa*1e3, self.fb*1e3, ls='--', label='Fat tissue')
        plt.plot(self.sa*1e3, self.sb*1e3, label='Skin boundaries')
        plt.plot(self.elec*1e3, marker=7, ms='15', label='Electrode')
        plt.legend(loc=10)
        plt.axis('equal')
        plt.xlabel('[mm]')
        plt.ylabel('[mm]')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save, 'muscle-morphology_'+self.morpho+'.tif'), dpi=600)
        
    def innervateRatio(self):
        """ 
        Calculates the number of innervated muscle fibers for each motorneuron in the pool and the motor unit territory radius. 
        Based on the work of Enoka e Fuglevand, 2001.
        计算池中每个运动神经元支配的肌纤维数量以及运动单元半径
        """
        n_fibers = 0
        for i in range(self.n):
            self.MUFN[i] = self.first*np.exp(np.log(self.ratio)*(i)/self.n) # 第i个MU中的肌纤维数量
            n_fibers = n_fibers + self.MUFN[i] # 计算肌纤维总数
        fiber_area = self.csa/n_fibers # 单个肌纤维的面积
        MUarea = self.MUFN*fiber_area # 每个MU的截面积
        self.MUradius = np.sqrt(MUarea/np.pi) # 每个MU的半径


    def gen_distribution(self):
        """
        Defines the motor units  x and y coordinates 
        定义MUs的x和y坐标
        """
        self.t1m_save = self.t1m
        self.t2m_save = self.t2m
        if self.morpho == 'Circle':
            self.t1m = self.t1m*self.r # I型MUs的平均极坐标半径
            self.t2m = self.t2m*self.r # II型MUs的平均极坐标半径
            self.circle_normal_distribution_otimize()
        elif self.morpho == 'Ring':
            self.t1m = self.ri + (self.re-self.ri) * self.t1m
            self.t2m = self.ri + (self.re-self.ri) * self.t2m
            flag = 1 
            while(flag == 1):
                flag = self.ring_normal_distribution_otimize()
        elif self.morpho == 'Pizza':
            self.t1m = self.t1m * self.r
            self.t2m = self.t2m * self.r
            self.pizza_normal_distribution_otimize()
        elif self.morpho == 'Ellipse':
            self.ellipse_normal_distribution_otimize()


    def circle_normal_distribution_otimize(self):
        """ 
        Generate Motor unit Territory (MUT) center coordinates for circle cross sectional area (CSA) muscle morpholigies. 
        Verifies MUT placed before to otimize by reducing the distribution variability across CSA. 
        生成圆形肌肉截面里的运动单元中心坐标
        验证已分配的MUs中心，通过减少分布变异性和重叠程度，优化MUs的分布
        """
        self.x= np.zeros(self.n) # MUs中心的x坐标
        self.y= np.zeros(self.n) # MUs中心的y坐标
        i = self.n - 1 
        while (i > self.t1): # i大于I型MUs的数量时，生成II型MUs的中心坐标
            temp = (self.r-self.MUradius[i])*self.t2dp # 极坐标半径的标准差
            r_temp = np.random.normal(self.t2m, temp) # 极坐标半径的正态分布随机数
            t_temp = np.random.uniform(0, 2*np.pi) # 极坐标角度的均匀分布随机数
            x_temp = r_temp*np.cos(t_temp) # x坐标
            y_temp = r_temp*np.sin(t_temp) # y坐标
            # 验证和分配II型MU区域
            if r_temp <= (self.r - self.MUradius[i]) and r_temp>= 0: # 如果r_temp在允许的半径范围内
                if i == 0: # i为0就直接分配坐标
                    self.x[i] = x_temp
                    self.y[i] = y_temp
                    i = i-1
                else: 
                    ant_d = self.r
                    for j in range(i, self.n): 
                        temp = (x_temp-self.x[j])**2+(y_temp-self.y[j])**2
                        d = np.sqrt(temp) # 两个中心之间的距离
                        min_d = min(d, ant_d) # 生成的第i个中心坐标与已分配的其他MUs中心的最小距离
                        ant_d = min_d
                        if min_d == d:
                            mur_min = self.MUradius[j] # 最相邻MU的半径
                    if min_d >= self.MUradius[i]+mur_min/2: # 如果满足最小重叠条件，则分配坐标
                        self.x[i] = x_temp
                        self.y[i] = y_temp
                        i = i - 1
        while (i >= 0): # 生成I型MUs的中心坐标
            temp = (self.r-self.MUradius[i])*self.t1dp # 极坐标半径的标准差
            r_temp = np.random.normal(self.t1m, temp) # 极坐标半径的正态分布随机数
            t_temp = np.random.uniform(0,2*np.pi) # 极坐标角度的均匀分布随机数
            x_temp = r_temp*np.cos(t_temp) # x坐标
            y_temp = r_temp*np.sin(t_temp) # y坐标
            if r_temp <= (self.r-self.MUradius[i]) and r_temp>= 0:
                if i == 0: # i为0直接分配坐标
                    self.x[i] = x_temp
                    self.y[i] = y_temp
                    i = i-1
                else:
                    ant_d = self.r
                    for j in range(i, self.n):
                        temp = (x_temp-self.x[j])**2+(y_temp-self.y[j])**2
                        d = np.sqrt(temp) # 两个中心之间的距离
                        min_d = min(d,ant_d) # 生成的第i个中心坐标与已分配的其他MUs中心的最小距离
                        ant_d = min_d
                        if min_d == d:
                            mur_min = self.MUradius[j] # 最相邻MU的半径
                    if min_d >= self.MUradius[i]+mur_min/2: # 如果满足最小重叠条件，则分配坐标
                        self.x[i] = x_temp
                        self.y[i] = y_temp
                        i = i - 1


    def motorUnitTerritory(self):
        """ 
        Based on motor unit territory centers and muscle radius, creates the motor unit territory boundaries.
        基于运动单元中心坐标和半径，生成运动单元边界
        """
        theta = np.arange(0, 2*np.pi, 0.01)
        self.MUT = np.zeros((self.n, 2, len(theta)))
        for i in range(self.n):
            self.MUT[i][0] = self.x[i] + self.MUradius[i]*np.cos(theta)
            self.MUT[i][1] = self.y[i] + self.MUradius[i]*np.sin(theta)


    def quantification_of_mu_regionalization(self):
        """ 
        Calculates the motor unit type II territory radial eccentricity. 
        计算II型运动单元的径向偏心度
        分析不同类型MU的分布和偏心程度
        """
        n_f     = [self.n, self.n]
        n_i     = [0, self.t1]
        peso    = [0, 0] # 总权重 第一个元素MU 0~n，第二个元素MU t1~n 支配的肌纤维总数
        r_cg    = [0, 0] # 加权距离
        for j in range(2):
            for i in range (n_i[j], n_f[j]):
                peso[j] = peso[j] + self.MUFN[i] 
                temp    = self.x[i]**2 + self.y[i]**2 
                r_cg[j] = r_cg[j] + np.sqrt(temp)*self.MUFN[i]
            r_cg[j] = r_cg[j]/peso[j] # 质心半径
        self.eccentricity = r_cg[1] - r_cg[0] # 计算径向偏心度


    def generate_density_grid(self):
        """ 
        Generates the muscle cross sectional area density of motor unit territories (to use with 2d histogram) 
        生成肌肉截面上不同运动单元的密度网格，用于绘制2D直方图，分析和可视化运动单元的分布情况
        """
        self.grid_step = 5e-5 
        self.gx = np.zeros(1)
        self.gy = np.zeros(1)
        for i in range(self.n):
            murad = self.MUradius[i] # 半径
            x_temp = np.arange(-murad, murad, self.grid_step)
            for j in range(len(x_temp)):
                Y = np.sqrt(self.MUradius[i]**2-x_temp[j]**2) # 圆的方程
                y_temp  = np.arange(-Y,Y,self.grid_step) # 在圆范围内生成y坐标点
                x_temp2 = (x_temp[j]*np.ones(len(y_temp))+self.x[i]) # 生成平移后的x坐标
                self.gx = np.append(self.gx,x_temp2) # 添加新的x坐标
                self.gy = np.append(self.gy,(y_temp+self.y[i])) # 添加新的y坐标
    
    
    def get_distribution(self, ratio, t1m, t1dp, t2m, t2dp):
        self.ratio, self.t1m, self.t1dp = ratio, t1m, t1dp
        self.t2m, self.t2dp = t2m, t2dp
        self.innervateRatio() # 计算MUs半径和肌纤维数量
        self.gen_distribution() # 定义MUs的中心坐标
        self.motorUnitTerritory() # 定义MUs的边界点
        self.quantification_of_mu_regionalization() # 计算II型MUs的径向偏心度
        self.generate_density_grid() # 生成密度网格
        self.LR = self.mnpool.LR # 激活的MUs数量


    def view_distribution(self, ratio, t1m, t1dp, t2m, t2dp):
        """
        Generates graphic with motor unit distribution within muscle csa and 2d histogram.
        在肌肉截面和2D直方图中绘制运动单元分布情况
        
        Args:
            ratio (float): ratio between first and last motor unit muscle fiber quantity. 第一和最后一个MU的肌纤维数量比例
            t1m (float): Type I MU distance distribution mean (% relative to the muscle radius). I型MU距离分布平均（相对于肌肉半径）
            t1dp (float): Type I MU distance distribution standard deviation (% relative to the muscle radius). I型MU距离分布标准差（相对于肌肉半径）
            t2m (float): Type II MU distance distribution mean (% relative to the muscle radius). II型MU距离分布平均（相对于肌肉半径）
            t2dp (float): Type II MU distance distribution standard deviation (% relative to the muscle radius). II型MU距离分布标准差（相对于肌肉半径）
        """
        self.get_distribution(ratio, t1m, t1dp, t2m, t2dp)
        hst_step  = 0.1
        hist_bins = [np.arange(min(self.ma)*1e3, max(self.ma)*1e3, hst_step), 
                     np.arange(min(self.mb)*1e3, max(self.mb)*1e3, hst_step)]
        hist,xedges,yedges = np.histogram2d(self.gx*1e3, self.gy*1e3, 
                                            bins=hist_bins)
        f, axes = plt.subplots(1, 2, figsize=(8, 4), sharex = 'all')
        plt.sca(axes[0])
        plt.ylabel('[mm]')
        plt.xlabel('[mm]')
        ecc2 = self.eccentricity*1e3
        print("MU type II radial eccentricity:{:.2f} [mm]".format(ecc2))
        fill_blue = mpl.patches.Patch(label='Recruited type I MUs', 
                                      fc=(0,0,1,0.4))
        fill_red = mpl.patches.Patch(label='Recruited type II MUs', 
                                      fc=(1,0,0,0.4))
        blue_line = mpl.lines.Line2D([], [], color='b', label='Type I MU')
        red_line = mpl.lines.Line2D([], [], color='r', ls='--', 
                                     label='Type II MU')
        plt.legend(handles=[fill_blue, fill_red, blue_line, red_line],
                   loc=3, fontsize=8)
        for i in range(self.t1):
            if (i <= self.LR):
                plt.fill(self.MUT[i,0]*1e3, self.MUT[i,1]*1e3, 
                         fc=(0,0,1,0.4), lw=0.5)
            plt.plot(self.MUT[i,0]*1e3, self.MUT[i,1]*1e3, color='b')
        for i in range(self.t1, self.n):
            if (i <= self.LR):
                plt.fill(self.MUT[i,0]*1e3, self.MUT[i,1]*1e3, 
                         fc=(1,0,0, 0.4))
            plt.plot(self.MUT[i,0]*1e3, self.MUT[i,1]*1e3, color='r', ls='--')
        plt.plot(self.ma*1e3, self.mb*1e3, self.fa*1e3, self.fb*1e3, 
                 self.sa*1e3, self.sb*1e3)
        plt.plot(self.elec*1e3, marker=7, ms='15')
        plt.axis('equal')
        plt.sca(axes[1])
        axes1  = plt.gca()
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        with plt.style.context('ggplot'):
            im1 = plt.imshow(hist.T, extent=extent, cmap=plt.cm.jet,
                                 interpolation='nearest', origin='lower')
            axins1 = inset_axes(axes1, width="5%", height="100%", loc=3, 
                                bbox_to_anchor=(1.01, 0., 1, 1),
                                bbox_transform=axes1.transAxes, borderpad=0)
            cbar = plt.colorbar(im1, cax=axins1)
            axins1.xaxis.set_ticks_position("bottom")
            cbar.ax.set_ylabel('[a.u.]',rotation=0, va='bottom')
            cbar.ax.yaxis.set_label_coords(0.5,1.05)
            axes1.axis('equal')
            axes1.set_xlabel('[mm]')
        plt.subplots_adjust(wspace = 0.2, hspace = 0.05)
        plt.savefig(os.path.join(self.save, 'MUs-distribution_'+self.morpho+'.tif'), dpi=600)

        
    def ring_normal_distribution_otimize(self):
        """ 
        Generate Motor unit Territory (MUT) center coordinates for ring cross sectional area (CSA) muscle morpholigies. 
        Verifies MUT placed before to otimize by reducing the distribution variability across CSA. 
        生成环形肌肉截面里的运动单元中心坐标
        验证已分配的MUs中心，通过减少分布变异性和重叠程度，优化MUs的分布    
        """
        counter = 0
        counter_limit = 10e3 # 限制尝试生成有效坐标的此书，避免无限循环
        flag = 0 # 标志是否完成优化，1——未完成，0——已完成
        self.x = np.zeros(self.n) # MUs中心的x坐标
        self.y = np.zeros(self.n) # MUs中心的y坐标
        i = self.n - 1
        while (i > self.t1): # 分配II型MUs的中心坐标
            std_temp = (self.re-self.MUradius[i])*self.t2dp # 极坐标半径的标准差
            r_temp = np.random.normal(self.t2m, std_temp) # 极坐标半径的正态分布随机数
            t_temp = np.random.uniform(-self.theta, self.theta) # 极坐标角度的均匀分布随机数
            x_temp = r_temp*np.sin(t_temp) # x坐标
            y_temp = r_temp*np.cos(t_temp) # y坐标
            ### 判断角度是否在允许范围内
            if self.MUradius[i]/r_temp > 1: theta_c = np.arcsin(1)
            elif self.MUradius[i]/r_temp < -1:  theta_c = np.arcsin(-1)
            else: theta_c = np.arcsin(self.MUradius[i]/r_temp)
            phi_c = np.arcsin(x_temp/r_temp)
            c1 = (r_temp <= self.re-self.MUradius[i])
            c2 = (r_temp >= self.MUradius[i] + self.ri)
            c3 = (phi_c <= self.theta - theta_c)
            c4 = (phi_c >= -(self.theta - theta_c))
            if  c1 and c2 and c3 and c4:
                if i == 0: # i为0直接分配坐标
                    self.x[i] = x_temp
                    self.y[i] = y_temp
                    i = i-1
                else: 
                    ant_d = self.re
                    for j in range(i,self.n):
                        dsquare = (x_temp-self.x[j])**2+(y_temp-self.y[j])**2
                        d = np.sqrt(dsquare)
                        min_d = min(d,ant_d) # 生成的第i个中心坐标与已分配的其他MUs中心的最小距离
                        ant_d = min_d
                        if min_d == d:
                            mur_min = self.MUradius[j] # 最相邻的MU半径
                    if min_d >= self.MUradius[i]+mur_min: # 如果满足最小重叠条件，则分配坐标
                        self.x[i] = x_temp
                        self.y[i] = y_temp
                        i = i - 1
                    else:
                        counter += 1
                        if counter > counter_limit:
                            flag = 1
                            break
        while (i>=0): # 分配I型MUs的中心坐标
            std_temp = (self.re-self.MUradius[i])*self.t1dp
            r_temp = np.random.normal(self.t1m,std_temp)
            t_temp = np.random.uniform(-self.theta,self.theta)
            x_temp = r_temp*np.sin(t_temp)
            y_temp = r_temp*np.cos(t_temp)
            if self.MUradius[i]/r_temp > 1: theta_c = np.arcsin(1)
            elif self.MUradius[i]/r_temp < -1:  theta_c = np.arcsin(-1)
            else: theta_c = np.arcsin(self.MUradius[i]/r_temp)
            phi_c = np.arcsin(x_temp/r_temp)
            c1 = (r_temp <= self.re-self.MUradius[i])
            c2 = (r_temp>= self.MUradius[i]+self.ri)
            c3 = (phi_c <= self.theta - theta_c)
            c4 = (phi_c >= -(self.theta - theta_c))
            if c1 and c2 and c3 and c4:
                if i == 0:
                    self.x[i] = x_temp
                    self.y[i] = y_temp
                    i = i-1
                else:
                    ant_d = self.re
                    for j in range(i,self.n):
                        dsquare = (x_temp-self.x[j])**2+(y_temp-self.y[j])**2
                        d = np.sqrt(dsquare)
                        min_d = min(d,ant_d)
                        ant_d = min_d
                        if min_d == d:
                            mur_min = self.MUradius[j]
                    if min_d >= self.MUradius[i]+mur_min/2:
                        self.x[i] = x_temp
                        self.y[i] = y_temp
                        i = i - 1
                    else:
                        counter += 1
                        if counter > counter_limit:
                            flag = 1
                            break
        return flag
                        

    def pizza_normal_distribution_otimize(self):
        """
        Generate Motor unit Territory (MUT) center coordinates for pizza like cross sectional area (CSA) muscle morpholigies. 
        Verifies MUT placed before to otimize by reducing the distribution variability across CSA.
        生成批萨形形肌肉截面里的运动单元中心坐标
        验证已分配的MUs中心，通过减少分布变异性和重叠程度，优化MUs的分布 
        """
        self.x= np.zeros(self.n)
        self.y= np.zeros(self.n)
        i= self.n - 1
        while (i > self.t1):
            r_temp = np.random.normal(self.t2m,(self.r-self.MUradius[i])*self.t2dp)
            t_temp = np.random.uniform(-self.theta,self.theta)
            x_temp = r_temp*np.sin(t_temp)
            y_temp = r_temp*np.cos(t_temp)
            if self.MUradius[i]/r_temp > 1: theta_c = np.arcsin(1)
            elif self.MUradius[i]/r_temp < -1:  theta_c = np.arcsin(-1)
            else: theta_c = np.arcsin(self.MUradius[i]/r_temp)
            phi_c = np.arcsin(x_temp/r_temp)
            c1 = (r_temp <= self.r-self.MUradius[i])
            c2 = (r_temp >= self.MUradius[i])
            c3 = (phi_c <= self.theta - theta_c)
            c4 = (phi_c >= -(self.theta - theta_c))
            if c1 and c2 and c3 and c4:
                if i == 0:
                    self.x[i] = x_temp
                    self.y[i] = y_temp
                    i = i-1
                else:
                    ant_d = self.r
                    for j in range(i,self.n):
                        d = np.sqrt((x_temp-self.x[j])**2+(y_temp-self.y[j])**2)
                        min_d = min(d,ant_d)
                        ant_d = min_d
                        if min_d == d:
                            mur_min = self.MUradius[j]
                    if min_d >= self.MUradius[i]+mur_min:
                        self.x[i] = x_temp
                        self.y[i] = y_temp
                        i = i - 1
        while (i>=0):
            #r_temp = r*np.random.uniform()
            r_temp = np.random.normal(self.t1m,(self.r-self.MUradius[i])*self.t1dp)
            t_temp = np.random.uniform(-self.theta,self.theta)
            x_temp = r_temp*np.sin(t_temp)
            y_temp = r_temp*np.cos(t_temp)
            if self.MUradius[i]/r_temp > 1: theta_c = np.arcsin(1)
            elif self.MUradius[i]/r_temp < -1:  theta_c = np.arcsin(-1)
            else: theta_c = np.arcsin(self.MUradius[i]/r_temp)
            phi_c = np.arcsin(x_temp/r_temp)
            c1 = (r_temp <= self.r-self.MUradius[i])
            c2 = (r_temp >= self.MUradius[i])
            c3 = (phi_c <= self.theta - theta_c)
            c4 = (phi_c >= -(self.theta - theta_c))
            if c1 and c2 and c3 and c4:
                if i == 0:
                    self.x[i] = x_temp
                    self.y[i] = y_temp
                    i = i-1
                else:
                    ant_d = self.r
                    for j in range(i,self.n):
                        d = np.sqrt((x_temp-self.x[j])**2+(y_temp-self.y[j])**2)
                        min_d = min(d,ant_d)
                        ant_d = min_d
                        if min_d == d:
                            mur_min = self.MUradius[j]
                    if min_d >= self.MUradius[i]+mur_min/2:
                        self.x[i] = x_temp
                        self.y[i] = y_temp
                        i = i - 1
                        

    def ellipse_normal_distribution_otimize(self):
        """
        Generate Motor unit Territory (MUT) center coordinates for pizza like cross sectional area (CSA) muscle morpholigies. 
        Verifies MUT placed before to otimize by reducing the distribution variability across CSA.
        生成批萨形形肌肉截面里的运动单元中心坐标
        验证已分配的MUs中心，通过减少分布变异性和重叠程度，优化MUs的分布 
        """
        self.x= np.zeros(self.n)
        self.y= np.zeros(self.n)
        i= self.n - 1
        while (i > self.t1):
            t_temp = np.random.uniform(0, 2*np.pi)
            raio = self.a * self.b/np.sqrt((self.b*np.cos(t_temp))**2+(self.a*np.sin(t_temp))**2)
            r_temp = np.random.normal(raio*self.t2m, (raio-self.MUradius[i])*self.t2dp)
            x_temp = r_temp*np.sin(t_temp)
            y_temp = r_temp*np.cos(t_temp)
            if r_temp <= raio-self.MUradius[i] and r_temp >= 0:
                if i == 0:
                    self.x[i] = x_temp
                    self.y[i] = y_temp
                    i = i-1
                else:
                    ant_d = raio
                    for j in range(i,self.n):
                        d = np.sqrt((x_temp-self.x[j])**2+(y_temp-self.y[j])**2)
                        min_d = min(d,ant_d)
                        ant_d = min_d
                        if min_d == d:
                            mur_min = self.MUradius[j]
                    if min_d >= self.MUradius[i]+mur_min:
                        self.x[i] = x_temp
                        self.y[i] = y_temp
                        i = i - 1
        while (i >= 0):
            t_temp = np.random.uniform(0,2*np.pi)
            raio = self.a*self.b/np.sqrt((self.b*np.cos(t_temp))**2+(self.a*np.sin(t_temp))**2)
            r_temp = np.random.normal(raio*self.t1m,(raio-self.MUradius[i])*self.t1dp)
            x_temp = r_temp*np.sin(t_temp)
            y_temp = r_temp*np.cos(t_temp)
            if r_temp <= raio-self.MUradius[i] and r_temp >= 0:
                if i == 0:
                    self.x[i] = x_temp
                    self.y[i] = y_temp
                    i = i-1
                else:
                    ant_d = raio
                    for j in range(i,self.n):
                        d = np.sqrt((x_temp-self.x[j])**2+(y_temp-self.y[j])**2)
                        min_d = min(d,ant_d)
                        ant_d = min_d
                        if min_d == d:
                            mur_min =self.MUradius[j]
                    if min_d >= self.MUradius[i]+mur_min/2:
                        self.x[i] = x_temp
                        self.y[i] = y_temp
                        i = i - 1
        temp = self.x
        self.x = self.y
        self.y = temp


    def exp_interpol(self):
        """
        Creates an growing exponential interpolation with n points between two values (v1 and v2) adjusted by the rr factor.
        创建在两个值（v1和v2）之间的递增指数插值（n个），并根据招募范围因子进行调整
        """
        a=np.log(self.rr)/self.n
        rte = np.zeros(self.n) # 招募阈值兴奋
        self.amp_array = np.zeros(self.n) # MUs中心的MUAP振幅
        for i in range(self.n):
            rte[i]=np.exp(a*(i+1))
            self.amp_array[i] = rte[i]*(self.v2-self.v1)/self.rr + self.v1
        
        # rte = self.mnpool.rte
        # self.amp_array = rte * (self.v2-self.v1)/self.rr +self.v1

    
    def expn_interpol(self):  
        """
        Creates an descending exponential interpolation with n points between two values (d1 and d2) adjusted by the rr factor.
        创建在两个值（d1和d2）之间的递减指数插值（n个），并根据招募范围因子进行调整
        """
        a = np.log(self.rr)/self.n
        rte = np.zeros(self.n)
        self.lm_array = np.zeros(self.n) # MUs中心的MUAP持续时间
        for i in range(self.n):
            rte[i]= np.exp(-a*(i+1))
            self.lm_array[i] = rte[i]*(self.d1-self.d2)+self.d2 
        
        # rte = 1/self.mnpool.rte
        # self.lm_array = rte*(self.d1-self.d2)+self.d2


    #### 当有多个通道的时候，mu_distance应该是一个数组
    def vc_filter(self):
        """
        Apply the filtering effect of the muscle tissue (volume conductor) on the duration and amplitude factors of the hermite rodriguez functions.
        应用肌肉组织（体积导体）的过滤效应对Hermite-Rodriguez函数的持续时间和幅度因子进行调整，即计算电极记录到的运动单元动作电位
        """
        self.mu_distance = np.sqrt((self.elec-self.y)**2+self.x**2) # 计算每个运动单元与电极之间的距离
        self.ampvar = self.amp_array*np.exp(-self.mu_distance/self.ampk) # 计算衰减后的运动单元动作电位振幅（即表面电极记录得到的MUAP振幅）
        self.lmvar = self.lm_array*(1+self.durak*self.mu_distance) # 计算衰减后的运动单元动作电位持续时间（即表面电极记录得到的MUAP持续时间）

    ### 多通道时，z[i]应该是一个二维矩阵
    def get_MUAPs(self):
        """
        计算表面电极上记录的运动单元动作电位

        Returns:
            x (np.array): 时间数组.
            z (np.array): 运动单元动作电位数组.

        """
        comp = max(self.lmvar) / (1.8)
        x = np.arange(0, 12 * comp, 0.1)
        z = np.zeros((self.LR, len(x))) 
        for i in range(self.LR):
            if (self.hr_order[i] == 1): # 根据运动单元类型选择合适的辅助函数的阶数
                z[i] = self.hr1_f(x, self.ampvar[i], self.lmvar[i], [max(self.lmvar) * 3]) # 一阶HR函数，表示双相MUAP
            else:
                z[i] = self.hr2_f(x, self.ampvar[i], self.lmvar[i], [max(self.lmvar) * 3]) # 二阶HR函数，表示三相MUAP
        return x, z

    def get_ptp(self):
        """
        计算每个运动单元的峰-峰值，即波形最大值和最小值之间的差值和时间间隔

        Returns:
            ptp_amps (float): 运动单元动作电位最大值和最小值之间的差值.
            ptp_durs (float): 运动单元动作电位最大值和最小值之间的时间间隔.
        """
        x, muaps = self.get_MUAPs() # 获取每个MU的MUAPs
        ptp_durs = np.zeros(self.LR) # 时间间隔
        ptp_amps = np.zeros(self.LR) # 峰-峰值
        for i in range(self.LR):
            min_val = min(muaps[i]) # 最小值
            index_min = np.where(muaps[i] == min_val)[0][0] # 最小值索引
            max_val = max(muaps[i]) # 最大值
            index_max = np.where(muaps[i] == max_val)[0][0] # 最大值索引
            ptp_durs[i] = abs(x[index_min]-x[index_max]) # 时间间隔
            ptp_amps[i] =  max_val - min_val # 峰-峰值
        return ptp_amps, ptp_durs


    def view_muap(self, v1, v2, d1, d2, add_hr):
        """
        plot graphic with motor unit action potential
        绘制运动单元动作电位

        Args:
            v1 (float): First recruited muap amplitude factor. 第一个招募的运动单元动作电位振幅因子
            v2 (float): Last recruited muap amplitude factor. 最后一个招募的运动单元动作电位振幅因子
            d1 (float): First recruited muap duration factor. 第一个招募的运动单元动作电位持续时间因子
            d2 (float): Last recruited muap duration factor. 最后一个招募的运动单元动作电位持续时间因子
            add_hr (float): Visualize 2nd order hermite rodrigues (MUAP) function. HR函数的阶数
        """
        self.v1, self.v2, self.d1, self.d2 = v1, v2, d1, d2
        self.v2 = v2
        self.d1 = d1
        self.expn_interpol() # 计算MUs中心的MUAPs的持续时间
        self.exp_interpol() # 计算MUs中心的MUAPs的振幅
        
        comp = max(self.lm_array) / (1.8)
        x = np.arange(0, 12 * comp, 0.1) # 持续时间
        y = np.arange(self.n) 
        z = np.zeros((y.shape[0], x.shape[0])) # 振幅
        
        fig,ax = plt.subplots(3,1,figsize=(5,9))
        plt.sca(ax[0])
        for i in range(self.n):
            if (add_hr == '1st order'):
                z[i] = self.hr1_f(x,self.amp_array[i],self.lm_array[i],[max(self.lm_array) * 3])
            else:
                z[i] = self.hr2_f(x,self.amp_array[i],self.lm_array[i],[max(self.lm_array) * 3])
            if i % int(self.n/5) == 0:
                plt.plot(x, z[i], label = 'MU${}_{%d}$'%(i+1)) 
        plt.plot(x,z[-1], label = 'MU${}_{%d}$'%(i+1)) 
                
        plt.xlabel('Time [ms]')
        plt.ylabel('Amplitude [mV]')
        plt.legend(loc=1)
        plt.xlim(left=0)

        plt.sca(ax[1])
        plt.plot(np.linspace(1,self.n,self.n), self.amp_array)
        plt.xlabel('MU $i$')
        plt.ylabel('$A_M$ [ms]')
        plt.xlim(1,self.n)
        plt.ylim(bottom =0)

        plt.sca(ax[2])
        plt.plot(np.linspace(1,self.n,self.n), self.lm_array)
        plt.xlabel('MU $i$')
        plt.ylabel('$\\lambda_M$ [mV]')
        plt.xlim(1,self.n)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save, 'MUAPs_'+self.morpho+'.tif'), dpi=600)
    
    def hr1_f(self, t, amp, lm, tspk):
        """
        Hermite-rodriguez 1nd order function (Cisi e Kohn, 2008)
        

        Args:
            t (np.array): Time simulation array (ms). 时间数组
            amp (float): MUAP amplitude. 运动单元动作电位振幅
            lm (float): MUAP duration. 运动单元动作电位持续时间
            tspk (list of floats): Motorneuron discharge times (ms). 运动神经元放电时间

        Returns:
            hr1 (np.array): Biphasic motor unit action potential train over time (mV). 双相MUAP的时间序列
        """
        n = len(t)
        hr1= np.zeros(n)
        j=0
        sbase = lm*3
        for w in range(n):
            if (t[w] > tspk[j] + sbase) and (j < len(tspk)-1): # and j < len(tspk):
                j = j+1
            # hr1(i) = first order Hermite-Rodriguez in instant 't'
            hr1[w] = amp*((t[w]-tspk[j])/lm)*np.exp(-1*(((t[w]-tspk[j])/lm)**2))
        return hr1


    def hr2_f(self, t, amp, lm, tspk):
        """
        Hermite-rodriguez 2nd order function (Cisi e Kohn, 2008)
        
        
        Args:
            t (np.array): Time simulation array (ms). 时间数组
            amp (float): MUAP amplitude. 运动单元动作电位振幅
            lm (float): MUAP duration. 运动单元动作电位持续时间
            tspk (list of floats): Motorneuron discharge times (ms). 运动神经元放电时间

        Returns:
            hr2 (np.array): triphasic motor unit action potential train over time (mV). 三相MUAP随时间变化的序列
        """
        n = len(t)
        hr2= np.zeros(n)
        j=0
        sbase = lm*3
        for w in range(n):
            if t[w] > tspk[j]+ sbase and j<len(tspk)-1 and j < len(tspk):
                j = j+1
            # hr2(i) =  Second order Hermite-Rodriguez in instant 't'
            hr2[w] = amp*(1-2*((t[w]-tspk[j])/lm)**2) * np.exp(-1*(((t[w]-tspk[j])/lm)**2))
        return hr2
        

    def view_attenuation(self, ampk, durak):
        """
        Generates graphic plot of the volume conduction attenuation.
        绘制体积传导的振幅衰减和持续时间扩展图

        Args:
            ampk (float): Volume conductor amplitude attenuation constant. 体积导体振幅衰减常数，描述电信号通过介质时的振幅衰减程度
            durak (float): Volume conductor duration widening constant. 体积导体持续时间扩展常数，描述电信号通过介质时持续时间的扩展程度
        """
        self.ampk = ampk*1e-3 
        self.durak = durak*1e3 
        step = 1e-4
        ga = np.arange(min(self.ma), max(self.ma), step)
        gb = np.arange(min(self.mb), max(self.mb), step)
        Ga, Gb = np.meshgrid(ga, gb)
        mu_distance2d = np.sqrt((self.elec-Gb) ** 2 + Ga ** 2)
        apvar2d = np.exp(-mu_distance2d / self.ampk)
        lmvar2d = 1 + self.durak * mu_distance2d
        f = plt.figure(figsize = (9,4))
        axes1 = plt.subplot(121)
        axes1.axis('equal')
        plt.xlabel('[mm]')
        plt.ylabel('[mm]')
        
        plt.plot(self.ma * 1e3, self.mb * 1e3, self.fa * 1e3, self.fb * 1e3, self.sa * 1e3, self.sb *1e3)
        plt.plot(self.elec * 1e3, marker = 7, ms = '15')
        CS = plt.contour(Ga * 1e3, Gb * 1e3, apvar2d, 10, cmap=plt.cm.jet_r)
        ax1= plt.gca()
        cbar1 = plt.colorbar(CS, ax = ax1)
        cbar1.ax.set_ylabel('Attenuation',rotation=0,va='bottom')
        cbar1.ax.yaxis.set_label_coords(0.5,1.05)
        axes1.clabel(CS, inline = 1, fontsize = 12)
        ax2 = plt.subplot(122)
        plt.plot(self.ma * 1e3, self.mb * 1e3, self.fa * 1e3, self.fb * 1e3, self.sa * 1e3, self.sb * 1e3)
        plt.plot(self.elec * 1e3, marker = 7, ms = '15')
        CS2 = plt.contour(Ga * 1e3, Gb * 1e3, lmvar2d, 8, cmap=plt.cm.jet)
        ax2 = plt.gca()
        cbar2 = plt.colorbar(CS2, ax = ax2)
        cbar2.ax.set_ylabel('Widening',rotation=0,va='bottom')
        cbar2.ax.yaxis.set_label_coords(0.5,1.05)
        ax2.clabel(CS2, inline = 1, fontsize = 12)
        ax2.axis('equal')
        plt.xlabel('[mm]')
        f.subplots_adjust(wspace=0.2)
        plt.savefig(os.path.join(self.save, 'attenuation_'+self.morpho+'.tif'), dpi=600)
        
    
    ### 多通道信号
    def semg(self): 
        """
        Generates the surface EMG signal.
        生成sEMG信号
        """
        self.raw_emg = np.zeros(len(self.t)) # 表面肌电信号
        self.mu_emg = np.zeros((self.LR, len(self.t))) # 每个激活MU对应的表面肌电信号
        self.hr_order = np.zeros(self.LR) # 每个激活MU的动作电位函数的阶数（随机分配）
        for i in range (self.LR):
            if np.random.randint(2) == 1: 
                temp = self.hr2_f(self.t,self.ampvar[i],self.lmvar[i],self.neural_input[i])
                self.hr_order[i] = 2
            else:
                temp = self.hr1_f(self.t,self.ampvar[i],self.lmvar[i],self.neural_input[i])
                self.hr_order[i] = 1
            self.mu_emg[i] = temp
            self.raw_emg = self.raw_emg + temp
        
        
    def get_semg(self, add_noise, noise_level, add_filter, bplc, bphc):
        """
        获取添加噪声和滤波之后的表面肌电信号        

        Args:
            add_noise (boolean): Add noise to the surface EMG. 是否添加噪声
            noise_level (int): Standard deviation of the noise level. 噪声水平的标准差
            add_filter (boolean): If true, filters the EMG with a butterworth 4-order filter. 是否使用滤波器
            bplc (float): Butterworth bandpass low cut frequency (Hz). 低端截止频率
            bphc (float): Butterworth bandpass high cut frequency (Hz). 高端截止频率
        """
        self.add_noise, self.noise_level, self.add_filter = add_noise, noise_level, add_filter
        self.lowcut, self.highcut   = bplc, bphc
        self.neural_input = self.mnpool.neural_input
        self.LR = self.mnpool.LR
        self.t = self.mnpool.t
        if self.neural_input == []:
            print("Neuronal drive to the muscle not found.")
            print("Please click on the button 'run interact' in the motor unit spike trains section.")
        else:
            print("Processing...")
            self.vc_filter() # 计算表面电极记录到的MUAPs振幅和持续时间
            self.semg() # 计算肌电信号
            if add_noise:
                self.emg = self.raw_emg  + np.random.normal(0, noise_level, len(self.raw_emg))
            else:
                self.emg = self.raw_emg
            if add_filter:
                self.butter_bandpass_filter()


    def view_semg(self,add_noise, noise_level, add_filter, bplc, bphc):
        """
        Generates and plot surface EMG.
        生成和绘制表面肌电信号
        
        Args:
            add_noise (boolean): Add noise to the surface EMG.
            noise_level (int): Standard deviation of the noise level.
            add_filter (boolean): If true, filters the EMG with a butterworth 4-order filter.
            bplc (float): Butterworth bandpass low cut frequency (Hz).
            bphc (float): Butterworth bandpass high cut frequency (Hz).
        """
        bins = 50
        self.get_semg(add_noise, noise_level, add_filter, bplc, bphc)
        amps, durs = self.get_ptp()
        clear_output()
        fig,ax = plt.subplots(3,1,figsize = (9, 9))
        ax[0].set_ylabel("Amplitude [mV]")
        ax[0].set_xlabel('Time [ms]')
        ax[0].plot(self.t, self.emg, lw = 0.5)
        ax[0].set_xlim(0, self.t[-1])
        
        ax[1].hist(amps,bins=bins)
        ax[1].set_xlabel('MUAP amplitude [mV]')
        ax[1].set_ylabel('Count')
        
        ax[2].hist(durs,bins=bins)
        ax[2].set_xlabel('MUAP duration [ms]')
        ax[2].set_ylabel('Count')
        plt.tight_layout()
        fig.align_ylabels()
        plt.savefig(os.path.join(self.save, 'sEMG.tif'), dpi=600)


    def butter_bandpass_filter(self, order=4):
        """
         Apply Butterworth bandpass filter on data.
         Butterworth带通滤波器

        Args:
            order (int, optional): Order of the filter. Defaults to 4.
        """
        nyq = 0.5 * self.sampling
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        self.emg = filtfilt(b, a, self.emg)
    

    def MedFreq(self, freq, psd):
        """
        Calculates the median frequency and the power of the median frequency of a Power spectral density data
        计算功率谱密度的中值频率及其对应的功率

        Args:
            freq (TYPE): Frequency vector generated with the PSD. 频率向量
            psd (TYPE): Power spectrum density. 功率谱密度

        Returns:
            f (TYPE): DESCRIPTION.
            mfpsdvalue (TYPE): DESCRIPTION.

        """
        cum = cumtrapz(psd,freq,initial = 0)
        f = np.interp(cum[-1]/2,cum,freq)
        mfpsdvalue = cum[-1]/2
        return f, mfpsdvalue
        
    
    def analysis(self, a_interval, add_rms, add_spec, add_welch, rms_length, spec_w, spec_w_size, 
                 spec_ol, welch_w, welch_w_size, welch_ol, add_mu_cont, mu_c_index):
        """
        
        

        Args:
            a_interval (TYPE): DESCRIPTION.
            add_rms (TYPE): DESCRIPTION.
            add_spec (TYPE): DESCRIPTION.
            add_welch (TYPE): DESCRIPTION.
            rms_length (TYPE): DESCRIPTION.
            spec_w (TYPE): DESCRIPTION.
            spec_w_size (TYPE): DESCRIPTION.
            spec_ol (TYPE): DESCRIPTION.
            welch_w (TYPE): DESCRIPTION.
            welch_w_size (TYPE): DESCRIPTION.
            welch_ol (TYPE): DESCRIPTION.
            add_mu_cont (TYPE): DESCRIPTION.
            mu_c_index (TYPE): DESCRIPTION.
        """
        self.t = self.mnpool.t
        self.sampling = self.mnpool.sampling
        self.dt = self.mnpool.sampling
        if self.emg == []:
            print("Surface EMG not found.")
            print("Please click the 'Run interact' button in Surface EMG generation section and run this cell again.")
        else:
            a_init = int(a_interval[0] * self.sampling / 1e3)
            a_end =int(a_interval[1] * self.sampling / 1e3)
            aemg = self.emg[a_init:a_end]
            at = self.t[a_init:a_end]
            # g_count = 0
            if add_rms:
                rms_length = int(rms_length * self.sampling/1e3)
                moving_average = np.sqrt(np.convolve(np.power(aemg,2), np.ones((rms_length,)) / rms_length, mode = 'same'))
                plt.figure(figsize = (9, 4))
                self.plot_rms(at, aemg, moving_average, 'Surface EMG Moving Average',
                              "Amplitude [mV]")
            if add_spec:
                if (spec_w_size <= spec_ol):
                    spec_w_size = spec_ol + 1
                spec_w_size = int(spec_w_size*self.sampling / 1e3)
                spec_ol = int(spec_ol*self.sampling / 1e3)
                spec_window = get_window(spec_w, spec_w_size)
                f,tf,Sxx = spectrogram(aemg, self.sampling, window = spec_window, 
                                       nperseg = spec_w_size, noverlap = spec_ol)
                plt.figure(figsize = (9, 4))
                ax1 = plt.subplot(111)
                self.plot_spec(tf * 1e3 + a_interval[0], f, Sxx, ax1, 300)
            if add_welch:
                if (welch_w_size <= welch_ol):
                    welch_w_size = welch_ol + 1
                welch_w_size = int(welch_w_size * self.sampling / 1e3)
                welch_ol = int(welch_ol * self.sampling / 1e3)
                fwelch, PSDwelch = welch(aemg * 1e-3, self.sampling, window = welch_w, 
                                         nperseg = welch_w_size, noverlap = welch_ol)
                emgfm, psdfm = self.MedFreq(fwelch, PSDwelch)
                plt.figure(figsize = (9, 4))
                self.plot_welch(fwelch, PSDwelch, emgfm, psdfm, 500, "Power [mV\u00b2/Hz]")
            if add_mu_cont:
                plt.figure(figsize = (9, 4))
                self.plot_mu_cont(at, self.mu_emg[mu_c_index - 1][a_init:a_end], mu_c_index)
                
        
    def plot_rms(self, at, aemg, moving_average, title, ylabel):
        """
        
        

        Args:
            at (TYPE): DESCRIPTION.
            aemg (TYPE): DESCRIPTION.
            moving_average (TYPE): DESCRIPTION.
            title (TYPE): DESCRIPTION.
            ylabel (TYPE): DESCRIPTION.
        """
        plt.ylabel(ylabel)
        plt.xlabel('Time [ms]')
        # plt.title(title)
        plt.plot(at,aemg,lw=0.5,label='Raw sEMG')
        plt.plot(at,moving_average,label='Moving RMS',lw=2,color='red')
        # plt.annotate("EMG RMS = %.3f mV" %(np.sqrt(np.mean(np.square(aemg)))), xy=(0.1,0.90), xycoords = ("axes fraction"))
        print("sEMG RMS = %.3f mV" %(np.sqrt(np.mean(np.square(aemg)))))
        plt.legend(loc=1)
        plt.xlim(at[0],at[-1])
        
        
    def plot_spec(self, tf, f, Sxx, spec_axis, ylim):
        """
        

        Args:
            tf (TYPE): DESCRIPTION.
            f (TYPE): DESCRIPTION.
            Sxx (TYPE): DESCRIPTION.
            spec_axis (TYPE): DESCRIPTION.
            ylim (TYPE): DESCRIPTION.
        """
        cf=plt.contourf(tf,f,Sxx, levels = 20, cmap=plt.cm.jet)
        # plt.title('Spectrogram')
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [ms]")
        plt.ylim(0,ylim)
        ax_in = inset_axes(spec_axis, width="5%",height="100%", loc=3, bbox_to_anchor=(1.01, 0., 1, 1),
                            bbox_transform=spec_axis.transAxes, borderpad=0)
        cbar = plt.colorbar(cf,cax = ax_in)
        ax_in.xaxis.set_ticks_position("bottom")
        cbar.ax.set_ylabel('[$mV^2$]',rotation=0, va='bottom')
        cbar.ax.yaxis.set_label_coords(0.5,1.05)
        
        
    def plot_welch(self, fwelch, PSDwelch, emgfm, psdfm, xlim, ylabel):
        """
        


        Args:
            fwelch (TYPE): DESCRIPTION.
            PSDwelch (TYPE): DESCRIPTION.
            emgfm (TYPE): DESCRIPTION.
            psdfm (TYPE): DESCRIPTION.
            xlim (TYPE): DESCRIPTION.
            ylabel (TYPE): DESCRIPTION.
        """
        # plt.title("sEMG Power Spectrum Density")
        plt.plot(fwelch,PSDwelch*10**6)
        plt.axvline(x=emgfm,ls = '--',lw=0.5, c = 'k')
        # plt.annotate("Median Freq. = %.2f Hz" %emgfm, xy=(0.5,0.9), xycoords = ("axes fraction"))
        print("PSD median Frequency = %.2f Hz" %emgfm)
        plt.ylabel(ylabel)
        plt.xlabel("Frequency [Hz]")
        plt.xlim(0,xlim)
        plt.ylim(bottom=0)


    def plot_mu_cont(self, at, mu_emg, mu_c_index):
        """
        
        

        Args:
            at (TYPE): DESCRIPTION.
            mu_emg (TYPE): DESCRIPTION.
            mu_c_index (TYPE): DESCRIPTION.
        """
        plt.plot(at,mu_emg)
        plt.xlabel('Time [ms]')
        plt.ylabel('Amplitude [mV]')
        plt.title('MU # {}'.format(mu_c_index))
        
        
    def save_config(self):
        """
        
        

        Returns:
            TYPE: DESCRIPTION.

        """
        try:
            self.config.update({'CSA Morphology': self.morpho, 
                                'CSA[mm^2]':  self.csa*1e6,
                                'Skin Layer [mm]': round(self.skin*1e3,6), # 小数点后六位
                                'Fat Layer [mm]':  self.fat*1e3,
                                'Proportion':  self.prop,
                                'Theta [rad]':  self.theta,
                                'Type I MU mu':  self.t1m_save,
                                'Type I MU sigma': self.t1dp,
                                'Type II MU mu':  self.t2m_save,
                                'Type II MU sigma':  self.t2dp,
                                'R_in':  self.ratio,
                                'MU_1 A_M [mV]': self.v1,
                                'MU_n A_M [mV]':  self.v2,
                                'MU_1 lambda_M [ms]': self.d1,
                                'MU_n lambda_M [ms]': self.d2,
                                'tau_at': self.ampk*1e-3,
                                'C': self.durak*1e3,
                                'Add noise':self.add_noise, 
                                'Noise std [mV]':self.noise_level,
                                'Add filter':self.add_filter,
                                'High-pass cutoff frequency [Hz]':self.lowcut,
                                'Low-pass cutoff frequency [Hz]':self.highcut})
        except AttributeError as error:
            print("Oops!", sys.exc_info()[0], "occurred: ", error)
            print(  'Could not save EMG parameters in configuration file. \
                    Try to click on \'run interact\' button on EMG \
                    generation cell.')
        return self.config
