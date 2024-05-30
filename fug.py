import numpy as np
from math import log
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os

class Phemo():

    def __init__(self):
        """ 
        Phenonomenologycal approach to model recruitment and rate coding 
        organization of a motorneuron pool
        模拟运动神经元池的招募和速率编码组织的现象学方法 参数初始化
        招募：激活不同数量的运动神经元来控制肌肉力量的过程，随着力量需求的增加，通常有更多的运动神经元被招募参与肌肉收缩。
        速率编码：通过改变已被招募的运动神经元的放电频率来调节肌肉力量的过程，当需要更大的力量输出时，运动神经元的放电频率会增加。
        """
        self.t1         = 101 # Number of type I motor units I型MUs的数量
        self.t2a        = 17 # Number of type IIa motor units IIa型MUs的数量
        self.t2b        = 2 # Number of type IIb motor units IIb型MUs的数量
        self.n          = self.t1 + self.t2a + self.t2b # Size of MU pool 池的大小（总MUs数量）
        self.sampling   = 2e4 # Sampling Frequency [Hz] 采样频率
        self.dt         = 1/self.sampling # simulation step time [s] 模拟步长
        self.t          = np.arange(0,5e3,self.dt*1e3) # time array 时间数组
        self.t_size     = len(self.t)
        self.rr         = 30 # range of recruitment 招募范围 （运动神经元在不同激活阈值下的招募范围）
        self.pfrd       = 20 # peak firing rate difference [Hz] 运动神经元池中第一个和最后一个招募的MU的峰值放电率之间的差值
        self.mfr        = 3 # Minimum firing rate [Hz] 最小放电率
        self.firstPFR   = 35 # First recruited Peak firing rate [Hz] 第一个招募的运动神经元的峰值放电率
        self.gain_cte   = False # If true, all motor unit gain (exicatory drive x firing rate) are equal  兴奋驱动和放电率之间的增益
        self.gain_factor= 2 # Gain factor 增益因子 pps/10% MVC
        self.LR         = 1 # Last recruited 最后一个招募的运动神经元的编号，即激活运动神经元的总数
        self.rrc        = 67 # Recruitment range condition 招募范围条件，招募最后一个MU的兴奋驱动的归一化值
        self.ISI_limit  = 15 # Minimum interspike interval  [ms] 最小峰间间隔
        self.recruitThreshold() # Defines Recruitment threshold for all motor units 定义所有MUs的招募阈值兴奋
        self.peakFireRate() # Defines peak firing rate for all motor units 定义所有MUs的峰值放电率
        self.recruitmentRangeCondition() # Defines Maximum excitatory drive and gain for all motor units 定义最大兴奋驱动和所有MUs的放电率增益
        self.neural_input   = [] # Spike train for all motor units 所有MUs的脉冲序列
        self.intensity      = 0 # excitatory drive intensity 兴奋驱动强度
        self.config         = {} # Configuration 
        self.CV             = 0 # 峰间间隔的变异系数

    
    def recruitThreshold(self): 
        """
        Defines the recruitment threshold for each motorneuron of the pool.
        Equal to the original proposed by Fuglevand et Al, 1993.
        定义池中每个运动神经元的招募阈值兴奋
        """
        # a=np.log(self.rr)/self.n
        # fuglevand
        self.rte = np.exp(np.arange(1, self.n + 1) * np.log(self.rr) / self.n)
        # self.rte = np.zeros(self.n) # Recruitment threshold excitation 招募阈值兴奋
        # for i in range(self.n):
        #     self.rte[i]=np.exp(a*(i+1))
        
    
    def peakFireRate(self):
        """
        Defines the Peak firing rate for each motorneuron in the pool.
        Equal to the original proposed by Fuglevand et Al, 1993.
        定义池中每个运动神经元的峰值放电率
        """
        self.pfr = np.zeros(self.n)
        for i in range(self.n):
            self.pfr[i] = self.firstPFR - self.pfrd*self.rte[i]/self.rte[-1]
        

    def recruitmentRangeCondition(self):
        """
        Defines the Firing rate x excitatory drive gain for each motorneuron and the maximum exitatory drive.
        定义每个运动神经元的放电率随兴奋驱动变化的增益和最大兴奋驱动
        """
        self.Emax = self.rte[-1]/(self.rrc/100) # 计算最大兴奋驱动
        var_gain = np.linspace(self.gain_factor,1,self.n) # 等距递减的增益（均匀分布）
        self.gain = var_gain*(self.pfr - self.mfr)/(self.Emax-self.rte) # 计算每个运动神经元的增益
        if self.gain_cte: # 每个运动神经元的增益一样的话，增益因子设置为gain_factor
            # last_gain = self.gain[-1]
            self.gain = np.ones(self.n) * self.gain[0] # linear 
            # for i in range(self.n):
            #     self.gain[i] = last_gain
        ### 还可以实现不同的增益算法     
           
                    
    def fireRate(self, E):
        """
        Calculates the firing rate of the motorneuron pool over time
        计算运动神经元池随时间变化的放电率（每个时间单位内运动神经元发放动作电位的频率）
        Args:
            E (np.array): Excitatory drive over simulation time (a.u.). 随时间变化的兴奋驱动
        Returns:
            fr (2D np.array): Firing rate for each motorneuron over time as function of E (Hz). 
            E的函数，每个运动神经元随时间变化的放电率
        """
        t_size = len(E)
        lastrec = np.zeros(t_size)
        for i in range(len(E)):
            for j in range(self.n):
                if E[i] >= self.rte[j]:
                    lastrec[i] = j + 1
        self.LR = int(max(lastrec))
        fr = np.zeros((self.LR, t_size))
        for i in range(self.n):
            for j in range(t_size):
                if (E[j] > self.rte[i]):
                    fr[i][j] = self.gain[i]*(E[j]-self.rte[i]) + self.mfr
                    if (self.pfr[i] < fr[i][j]):
                        fr[i][j] = self.pfr[i]
        ### 循环比较费时间，可以改简单一点，最后再算出self.LR就行了
        # if E.ndim == 1:
        #     E = E.reshape(1, E.shape[-1])
        # fr = np.minimum(self.maxfr, self.minfr + (E - self.rte) * self.slope_fr)
        # fr[E < self.rte] = 0
        # self.LR = np.sum(np.any(fr != 0, axis=1))
        return fr
    

    def graph_FRxExcitation(self):
        """ 
        plot recruitment and firing rate organization of the motorneuron pool
        绘制运动神经元池中放电率和兴奋驱动的关系曲线
        """
        e = np.linspace(0, self.Emax, 200)
        fr = self.fireRate(e)
        
        plt.figure(figsize = (5,4))
        for i in range(self.t1-1):
            plt.plot(100*e/self.Emax, fr[i], c = '#4C72B0')
        plt.plot(100*e/self.Emax, fr[self.t1-1], c = '#4C72B0', label = "MN I")   
        for i in range(self.t1, self.t1 + self.t2a-1):
            plt.plot(100*e/self.Emax, fr[i], c = '#55A868', ls = "--")
        plt.plot(100*e/self.Emax, fr[self.t1 + self.t2a - 1], c = '#55A868', ls = "--", label = "MN IIa")  
        for i in range(self.t1 + self.t2a, self.n - 1):
            plt.plot(100*e/self.Emax, fr[i], c = '#C44E52', ls = "-.")
        plt.plot(100*e/self.Emax, fr[self.n - 1], c = '#C44E52', ls = "-.", label = "MN IIb")  
        plt.xlabel('Excitatory drive [%]')
        plt.ylim(self.mfr)  
        plt.xlim(0, 100)
        plt.ylabel('Firing Rate [imp/s]')
        plt.legend(loc=2)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save,'firing-rate_excitatory-drive.tif'), dpi=600)

        
    def view_organization(self, rr, mfr, firstPFR, PFRD, RRC, t1, t2a, t2b, gain_factor, gain_CTE, save):
        """
        update and plot recruitment and firing rate organization of the motorneuron pool
        更新和绘制运动神经元池中放电率和兴奋驱动的关系曲线
        
        Args:
            rr (int): Range of recruitment excitations of the motorneuron pool(x fold). 招募范围
            mfr (int): Minimal firing rate in the motorneuron pool (Hz). 最小放电率 
            firstPFR (int): Peak firing rate of the first recruited motorneuron (Hz). 第一个招募的运动神经元的峰值放电率
            PFRD (int): Desired peak firing rate difference between first and last motorneuron (Hz). 峰值放电率差值
            RRC (float): Defines the relative excitatory drive necessary to recruit the last motorneuron (%). 
            招募最后一个运动神经元所需的相对兴奋驱动
            即为了使运动神经元池中的最后一个运动神经元（通常阈值最高）被招募，所需要的相对兴奋驱动强度
            t1 (int): Quantity of type I motorneurons in the pool. I型MUs的数量
            t2a (int): Quantity of type IIa motorneurons in the pool. IIa型MUs的数量
            t2b (int): Quantity of type IIb motorneurons in the pool. IIb型MUs的数量
            gain_factor (float): Gain factor for the first recruited motorneuron. 第一个招募的运动神经元的兴奋驱动-放电率的增益因子
            gain_CTE (boolean): Flag to define all motorneuron gains equal to the first recruited (flag). 是否所有运动神经元使用统一增益
        """
        self.save = save
        self.get_mu_pool_org(rr, mfr, firstPFR, PFRD, RRC, t1, t2a, t2b, gain_factor,gain_CTE)
        self.graph_FRxExcitation()
        

    def get_mu_pool_org(self,rr, mfr, firstPFR, PFRD, RRC, t1, t2a, t2b, gain_factor,gain_CTE):
        self.t1 = t1
        self.t2a = t2a
        self.t2b = t2b
        self.n = t1 + t2a + t2b
        self.rr = rr
        self.pfrd = PFRD
        self.mfr = mfr
        self.firstPFR = firstPFR
        self.gain_cte = gain_CTE
        self.gain_factor = gain_factor
        self.rrc = RRC/100
        self.recruitThreshold()
        self.peakFireRate()
        self.recruitmentRangeCondition()
 
    
    def excitation_curve(self, t0, t1, t2, t3, freq_sin, mode, intensity):
        """
        Caclulates the excitatory drive over the simulation time
        计算随模拟时间变化的兴奋驱动E

        Args:
            t0 (float): Excitatory drive onset time (ms). 兴奋驱动开始时间
            t1 (float): Excitatory drive plateau on time (ms). 兴奋驱动平台开始时间
            t2 (float): Excitatory drive plateau off time (ms). 兴奋驱动平台结束时间
            t3 (float): Excitatory drive offset time (ms). 兴奋驱动结束时间
            f (float): Sinusoidal frequency excitatory drive time variation (Hz). 正弦波频率
            mode (string): Excitatory drive mode, 'trap' for trapezoidal or 'sin' for sinusoidal excitation curve.兴奋驱动模式
            intensity (float): plateou relative excitatory drive for 'trap' mode and peak excitatory drive for 'sin' mode (%). 兴奋驱动强度
        """
        dt =  self.dt
        Emax = self.Emax
        self.intensity = intensity
        self.t00 = t0
        self.t01 = t1
        self.t02 = t2
        self.t03 = t3
        self.mode = mode
        self.e_freq = freq_sin
        
        ramp_init = int(t0/dt)
        plateau_init = int(t1/dt)
        dramp_init = int(t2/dt)
        dramp_end = int(t3/dt)
        
        ramp = (t1-t0)/dt
        if ramp == 0:
            stepup = 0
        else:
            stepup = intensity*Emax/ramp
            
        dramp = (t3-t2)/dt
        if dramp == 0:
            stepdown = 0
        else:
            stepdown = intensity*Emax/dramp
            
        if (mode == "Trapezoidal"):
            self.E[0:ramp_init] = 0
            self.E[ramp_init:plateau_init] = stepup * np.array([i-ramp_init for i in range(ramp_init, plateau_init)])
            self.E[plateau_init:dramp_init] = intensity*Emax
            self.E[dramp_init:dramp_end] = intensity*Emax - stepdown * np.array([i+1-dramp_init for i in range(dramp_init, dramp_end)])
            self.E[dramp_end:self.t_size] = 0
            # for i in range(0, ramp_init):
            #     self.E[i] = 0
            # for i in range(ramp_init, plateau_init):
            #     self.E[i] = stepup*(i-ramp_init)
            # for i in range(plateau_init, dramp_init):
            #     self.E[i] = intensity*Emax
            # for i in range(dramp_init, dramp_end):
            #     self.E[i] = intensity*Emax - stepdown*(i+1-dramp_init)
            # for i in range(dramp_end, self.t_size):
            #     self.E[i] = 0
        if (mode == "Sinusoidal"):
            self.E = intensity*Emax/2 + intensity*Emax/2*np.sin(2*np.pi*freq_sin*self.t)
            # for i in range(self.t_size):
            #     self.E[i] = intensity*Emax/2 + intensity*Emax/2*np.sin(2*np.pi*freq_sin*self.t[i])
    
    
    def set_excitatory(self, t0, t1, t2, t3, freq_sin, mode, intensity, sample_time, sim_time):
        self.sampling = sample_time
        self.sim_time = sim_time
        self.dt = 1e3/sample_time # ms
        self.t = np.arange(0, self.sim_time, self.dt) # Time Array in [ms]
        self.t_size = len(self.t)
        self.E = np.zeros(self.t_size)
        self.excitation_curve(t0,t1,t2,t3,freq_sin*1e-3,mode,intensity/100)


    def view_excitatory(self, t0, t1, t2, t3, freq_sin, mode, intensity, sample_time, sim_time, save):
        """
        Caculates and plot the excitatory drive over the simulation time
        计算并绘制随模拟时间变化的兴奋驱动
        
        Args:
            t0 (float): Excitatory drive onset time (ms). 兴奋驱动开始时间
            t1 (float): Excitatory drive plateau on time (ms). 兴奋驱动平台开始时间
            t2 (float): Excitatory drive plateau off time (ms). 兴奋驱动平台结束时间
            t3 (float): Excitatory drive offset time (ms). 兴奋驱动结束时间
            freq_sin (float): Sinusoidal frequency excitatory drive time variation (Hz). 正弦波频率
            mode (string): Excitatory drive mode, 'trap' for trapezoidal or 'sin' for sinusoidal excitation curve.兴奋驱动模式
            intensity (float): Plateou relative excitatory drive for 'trap' mode and peak excitatory drive for 'sin' mode (%). 兴奋驱动强度
            sample_time: Simulation sampling time (Hz). 采样频率
            sim_time (float): Total simulation time (ms). 模拟时长
        """
        self.save = save
        self.set_excitatory(t0, t1, t2, t3, freq_sin, mode, intensity, sample_time, sim_time)
        plt.figure(figsize=(4,4))
        plt.plot(self.t, self.E/self.Emax * 100)
        plt.xlabel('Time [ms]')
        plt.ylabel('Excitation [%]')      
        plt.tight_layout()
        plt.savefig(os.path.join(self.save, 'excitatory-drive_'+mode+'.tif'), dpi=600)

    def neuralInput(self):
        """
        Calculates the motorneuron pool discharge times over the simulation time
        计算随时间变化的运动神经元池放电时间
        """
        x,y = self.fr.shape # x=MUs数量，y=时间点数量
        self.neural_input = []
        for i in range (x): # for each MU 
            spike_train = []
            next_spike = 0
            flag = 0
            for j in range(y): # for each instant
                if (self.fr[i][j]>0 and self.t[j]>next_spike):
                    sigma = self.CV*1e3/self.fr[i][j] # 1e3转换为ms fr=1/mean(ISI) std=cov✖mean(ISI)=cov/fr
                    if not spike_train:
                        next_spike = self.t[j] + self.add_spike(sigma,self.fr[i][j])
                        spike_train.append(next_spike)
                        k = 0
                    else:
                        if (flag == 1 and (self.t[j] - spike_train[k] > 2)): # 放电时间间隔大于2ms
                            flag = 0
                            next_spike = self.t[j] + self.add_spike(sigma,self.fr[i][j])
                        else:
                            next_spike = self.add_spike(sigma,self.fr[i][j]) + spike_train[k]
                        if (next_spike > self.t[-1]):
                            break
                        if (next_spike -  spike_train[k] > 2):
                            spike_train.append(next_spike)
                            k = k + 1
                if (self.fr[i][j] == 0):
                    flag = 1 # MU stopped
            self.neural_input.append(np.asarray(spike_train)) # self.n ✖ self.t_size 初始脉冲序列


    def add_spike(self, sigma, FR):
        """
        Calculates the inter spike interval (ISI) for the next discharge time of a motorneuron
        为一个运动神经元的下一个放电时间计算峰间间隔
        
        Args:
            sigma (float): Standard deviation of the ISI (ms). 峰间间隔的标准差
            FR (float): Mean Firing rate of a motorneuron at a given time (Hz). 一个运动神经元在给定时间点的平均放电率

        Returns:
            float: New inter spike interval (ms). 新的峰间间隔
        """
        sig = np.random.normal(0,sigma) # 生成一个均值为0，标准差为sigma的正态分布随机数
        while abs(sig) > sigma*3.9: # sig应该在平均值正负3.9标准差范围内，避免出现异常值
            sig = np.random.normal(0, sigma)
        return  sig + 1e3/FR # 新的峰间间隔

    # FUNCTION NAME: synchSpikes
    # FUNCTION DESCRIPTION: Apply the synchronization algorithm [yao et Al, 2001]
    #                       to the discharge times of a MN pool. 
    # INPUT PARAMS:  1) synch_level: desired level of synchrony (%) [float]
    #                2) sigma: standard deviation of the normal distribution add to the 
    #                   synchronized discharge (ms) [float]
    def synchSpikes(self, synch_level, sigma):
        """
        Apply the synchronization algorithm [yao et Al, 2001] to the discharge times of a MN pool.
        将同步算法应用到运动神经元池的放电时间中
        
        Args:
            synch_level (float): Desired level of synchrony (%). 同步水平，在同一时间段内同步放电的运动神经元比例
            sigma (float): Standard deviation of the normal distribution add to the synchronized discharge (ms).
            添加到同步放电中的正态分布的标准差，模拟同步放电事件在时间上的微小随机变动，使模型更加逼近真实的生物现象
        """
        pool = np.arange(0,self.LR, dtype=int) # Create the array of all recruited motor units indexes 创建数组表示所有被招募的MUs的索引
        np.random.shuffle(pool) # Shuffle the Pool 随机打乱运动神经元池
        synch_pool_limit = int(synch_level*self.LR) # set the limit number of synched MUs 设置同步MUs的数量
        for i in pool: # for all active MUs in the pool do: 对于池中所有激活的MUs
            i_index= np.argwhere(pool == i) # localize the index of [i] 定位第i个MU
            synchin_pool = np.delete(pool,i_index) # remove the reference MU from the pool indexes 从池索引中删除参考MU
            # Define the reference spikes to synchronization of the mu[i] 定义参考脉冲以同步MU[i]
            ref_spikes = np.random.choice(self.neural_input[i], int(synch_level*len(self.neural_input[i])), replace = False)
            # 从MU[i]的神经输入数组随机选择元素，得到参考的放电时间数组（不放回抽取）
            for j in ref_spikes: # for all the reference spikes of MU[i] do: 对于MU[i]的所有参考脉冲
                np.random.shuffle(synchin_pool) # shuffle the order of the pool to be synchronized 随机打乱用于同步的运动神经元池
                synched_MUs = 0 # Synchronized motor units 同步MUs的数量
                w = 0 # synchronizing pool index 同步的运动神经元索引
                while (synched_MUs < synch_pool_limit and w < self.LR-1):
                    k = synchin_pool[w] # 获取运动神经元索引
                    w += 1 
                    # Vector of differences between discharge ref and candidate MN. 计算参考放电时间和候选MN放电时间之间的差异
                    difs = abs(self.neural_input[k] - j) 
                    minimum = min(difs) # Minimum diference between then 最小差异
                    min_index= np.argwhere(difs == minimum)[0][0] # Index of the minimum 最小差异的索引
                    # If k motorneuron(MN) candidate is recruited and we did not reach the total quantity of synched MNs.
                    # 
                    if self.ISI_limit == 0: # 最小峰间间隔为0
                        adjusted_spike_time = j + np.random.normal(0,sigma) # New spike position 
                        self.neural_input[k][min_index] = adjusted_spike_time
                        synched_MUs += 1
                    else:
                        if (minimum < self.ISI_limit): # 如果最小差异小于最小峰间间隔  
                            adjusted_spike_time = j + np.random.normal(0,sigma) # New spike position
                            self.neural_input[k][min_index] = adjusted_spike_time
                            synched_MUs += 1
    
    
    def get_neural_command(self, CoV, synch_level, sigma):
        print("Processing...")
        self.CV = CoV/100
        self.synch_level = synch_level
        self.synch_sigma = sigma
        self.fireRate(self.E) # Defines the mean firing rate
        self.fr = self.fireRate(self.E)
        self.neuralInput() # Generates the neural input to the muscles 生成肌肉的神经输入
        self.synchSpikes(synch_level/100, sigma) # Promotes synchronism between MU 


    def view_neural_command(self, CoV, synch_level, sigma):
        """
        Plot neural command and other performance indicators
        绘制神经指令和其他性能指标

        Args:
            CoV (float): Initial cv to be used in the interpolation, if cv_factor = 0, this value will be used for all excitatory drives (%).
            插值中使用的初始变异系数，如果cv_factor设置为0，这个初始值将用于所有兴奋驱动
            synch_level (float): desired level of synchrony (%). 同步水平
            sigma (float): standard deviation of the normal distribution add to the synchronized discharge (ms). 
            添加到同步放电中的正态分布的标准差
        """
        print("Processing...")
        self.get_neural_command(CoV, synch_level, sigma)
        # Inter Spike interval Analysis
        ISI = [np.diff(mu_isi) for mu_isi in self.neural_input if mu_isi != []]
        isi_hist = [item for mu_isi in ISI for item in mu_isi]
        isi_mean = [np.mean(mu_isi) for mu_isi in ISI]
        isi_std = [np.std(mu_isi) for mu_isi in ISI]
        isi_cv = [mu_isi_std / mu_isi_mean for mu_isi_std, mu_isi_mean in zip(isi_std, isi_mean)]
        clear_output()
        f, axes = plt.subplots(4, 1, figsize=(8,8))
        axes[0] = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
        axes[1] = plt.subplot2grid((4, 1), (2, 0))
        axes[2] = plt.subplot2grid((4, 1), (3, 0))
        axes[0].eventplot(self.neural_input)
        plt.sca(axes[0])
        plt.ylabel("MU #")
        plt.xlabel('Time (ms)')
        plt.xlim(0, self.t[-1])
        plt.ylim(0, self.LR+1)
        plt.sca(axes[1])
        plt.hist(isi_hist, bins = np.arange(0, 500, 10), edgecolor = 'k')
        print("ISI mean: {:.2f}".format(np.mean(isi_mean)))
        print("ISI Std. Dev.: {:.2f}".format(np.mean(isi_std)))
        print("ISI Coef. Var.: {:.2f}".format(np.mean(isi_cv)))    
        plt.ylabel('Count')
        plt.xlabel('Interspike Interval (ms)')
        plt.sca(axes[2])
        plt.plot(np.asarray(isi_cv)*100, ls='',marker = 'o')
        plt.ylabel('ISI CoV [%]')
        plt.xlabel('MN index')
        plt.tight_layout()
        f.align_ylabels()
        plt.savefig(os.path.join(self.save, 'neural-input_'+self.mode+'.tif'), dpi=600)

    def save_config(self):
        """
        Generate dictionary with motorneuron pool model organization
        保存运动单元池模型的参数配置

        Returns:
            self.config (dict): Dictionary with motorneuron pool model parameters. 运动神经元模型的字典
        """
        try:
            self.config.update({'# Type I MU': self.t1,
                     '# Type IIa MU': self.t2a,
                     '# Type IIb MU': self.t2b,
                     '# Number of MUs': self.n,
                     'RR': self.rr,
                     'PFRD [Hz]': self.pfrd,
                     'MFR [Hz]': self.mfr,
                     'PFR_1 [Hz]': self.firstPFR,
                     'Same gain for all MNs?':self.gain_cte,
                     'g_1 [a.u.]':self.gain_factor,
                     'e_LR [%]': self.rrc,
                     'Plateau/peak intensity [%]': self.intensity,
                     'Onset [ms]': self.t00,
                     'Plateau on [ms]':self.t01,
                     'Plateau off [ms]':self.t02,
                     'Offset [ms]':self.t03,
                     'Modulation':self.mode,
                     'Frequency [Hz]':self.e_freq*1e3,
                     'Sampling [Hz]': self.sampling,
                     'Duration [ms]': self.sim_time,
                     'ISI COV [%]':self.CV,
                     'synch. level [%]': self.synch_level,
                     'Synch. sigma [ms]': self.synch_sigma})
        except:
            print('Couldn\'t save motorneuron pool parameters, try to click \'run interact\' on neural command generation cell.')
        return self.config