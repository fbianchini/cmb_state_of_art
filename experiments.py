import numpy as np

cmbs = ['TT','EE','BB','TE','EB','TB','KK']

# Stolen from Dominic Beck
def binit(cl,bins):
    nb = len(bins)-1
    clb = np.zeros(nb)
    for i in range(nb):
        clb[i] = np.mean(cl[bins[i]:bins[i+1]])

    return clb

class Experiment(object):
    def __init__(self, telescope, exp_type, color=None):
        self.telescope = telescope
        self.exp_type = exp_type
        self.color = color

        self.l = {}
        self.l_lo = {}
        self.l_hi = {}
        self.dl = {}
        self.dl_err = {}
        for cmb in cmbs:
            self.l[cmb] = None
            self.l_lo[cmb] = None
            self.l_hi[cmb] = None
            self.dl_err[cmb] = None
            self.dl[cmb] = None

class SPTSZ(Experiment):
    def __init__(self, telescope='SPT', exp_type='ground', color='#e26813'):
        super().__init__(telescope, exp_type, color)

        self.l['TT'], _, self.dl_err['TT'] = np.loadtxt('data/bandpowers_spt2500deg2_lps12.txt', unpack=1)
        self.dl['TT'] = np.loadtxt('data/SPT_cosmomc/data/sptsz_2500d_tt/spt_s13_margfg_cl_hat.dat',unpack=1, usecols=1)

class SPTpol(Experiment):
    def __init__(self, telescope='SPT', exp_type='ground', color='#e26813'):
        super().__init__(telescope, exp_type, color)

        self.l['BB'], self.l_lo['BB'], self.l_hi['BB'], self.dl['BB'], self.dl_err['BB'] = np.loadtxt('data/bb_plotting.txt', unpack=1,)

class SPT3G(Experiment):
    def __init__(self, telescope='SPT', exp_type='ground', color='#e26813'):
        super().__init__(telescope, exp_type, color)

        self.l_lo['TE'], self.l_hi['TE'], self.l['TE'], self.dl['TE'], self.dl_err['TE'] = np.loadtxt('data/bp_for_plotting_v3.txt', unpack=1, usecols=[0,1,2,3,4])
        self.l_lo['EE'], self.l_hi['EE'], self.l['EE'], self.dl['EE'], self.dl_err['EE'] = np.loadtxt('data/bp_for_plotting_v3.txt', unpack=1, usecols=[0,1,5,6,7])
        
class ACTpol_DR4(Experiment):
    def __init__(self, telescope='ACT', exp_type='ground', color='#b3074f'):
        super().__init__(telescope, exp_type, color)

        for cmb in ['TT','EE','TE']:
            self.l[cmb], self.dl[cmb], self.dl_err[cmb] = np.loadtxt('data/cmbonly_spectra_dr4.01/act_dr4.01_D_ell_%s_cmbonly.txt'%cmb,unpack=1)
        #from table 9 of 2007.07289
        self.l_lo['BB'] = np.array([300, 651, 1001, 1401, 2801])
        self.l_hi['BB'] = np.array([650, 1000, 1400, 2800, 4000])
        self.l['BB']  = np.array([475, 825.5, 1200.5, 2100.5, 3400.5])
        self.dl['BB'] = np.array([0.090, 0.029, 0.094, -0.113, -0.30])
        self.dl_err['BB'] = np.array([0.043, 0.057, 0.073, 0.092, 0.24])



class POLARBEAR17(Experiment):
    def __init__(self, telescope='POLARBEAR', exp_type='ground', color='#33673B'):
        super().__init__(telescope, exp_type, color)

        self.l_lo['BB'], self.l_hi['BB'], self.l['BB'], self.dl['BB'], self.dl_err['BB'] = np.loadtxt('data/Polarbear_BB_2017.txt', unpack=1, usecols=[1,2,3,4,5])

class POLARBEAR22(Experiment):
    def __init__(self, telescope='POLARBEAR', exp_type='ground', color='#33673B'):
        super().__init__(telescope, exp_type, color)

        self.l_lo['BB'] = np.arange(50,600,50)
        self.l_hi['BB'] = np.arange(100,650,50)
        self.l['BB'] = 0.5*(self.l_lo['BB']+self.l_hi['BB'])
        self.dl['BB'] = np.array([0.0249,0.0029,0.0218 ,0.0207 ,-0.0521,0.0481 ,0.0259 ,0.1016 ,-0.0376,-0.0772,0.0664 ,])
        self.dl_err['BB'] = np.array([0.0126 ,0.0135 ,0.0207 ,0.0287 , 0.0403,0.0528 ,0.0650 ,0.0835 , 0.0912, 0.1114,0.1323 ,])

class BK18(Experiment):
    def __init__(self, telescope='BK', exp_type='ground', color='#9B5DE5'):
        super().__init__(telescope, exp_type, color)

        self.l_lo['BB'], self.l['BB'], self.l_hi['BB'], self.dl['BB'], dl_lo_bb, dl_hi_bb = np.loadtxt('data/BK18_components_20210607.txt', unpack=1, usecols=[0,1,2,3,4,5])
        self.dl_err['BB'] = [self.dl['BB']-dl_lo_bb, dl_hi_bb-self.dl['BB']]

class Planck(Experiment):
    def __init__(self, telescope='Planck', exp_type='space', bins=np.array([2, 15, 32]), color='#137fbf'):
        super().__init__(telescope, exp_type, color)

        # For binning the Planck l-by-l low-l spectra
        self.bins = bins

        self.l['TT'], self.dl['TT'], self.dl_err['TT'] = np.loadtxt('data/COM_PowerSpect_CMB-TT-binned_R3.01.txt', unpack=1, usecols=(0,1,2))
        self.l['EE'], self.dl['EE'], self.dl_err['EE'] = np.loadtxt('data/COM_PowerSpect_CMB-EE-binned_R3.02.txt', unpack=1, usecols=(0, 1, 2))
        self.l['TE'], self.dl['TE'], self.dl_err['TE'] = np.loadtxt('data/COM_PowerSpect_CMB-TE-binned_R3.02.txt', unpack=1, usecols=(0, 1, 2))
        self.l['BB'], self.l_lo['BB'], self.l_hi['BB'], self.dl['BB'] = np.loadtxt('data/cl_planck_lolEB_NPIPE_BB.dat.txt', unpack=1, usecols=(1, 0, 2, 7))
        planck_l_tt_lo, planck_dl_tt_lo, planck_dl_err_tt_lo = np.loadtxt('data/COM_PowerSpect_CMB-TT-full_R3.01.txt', unpack=1, usecols=(0,1,2))
        planck_l_ee_lo, planck_dl_ee_lo, planck_dl_err_ee_lo = np.loadtxt('data/COM_PowerSpect_CMB-EE-full_R3.01.txt', unpack=1, usecols=(0,1,2))

        self.l['TT'] = np.insert(self.l['TT'], 0, planck_l_tt_lo[:25])
        self.dl['TT'] = np.insert(self.dl['TT'], 0, planck_dl_tt_lo[:25])
        self.dl_err['TT'] = np.insert(self.dl_err['TT'], 0, planck_dl_err_tt_lo[:25])
        self.l['EE'] = np.insert(self.l['EE'], 0, binit(planck_l_ee_lo, self.bins))
        self.dl['EE'] = np.insert(self.dl['EE'], 0, binit(planck_dl_ee_lo, self.bins))
        self.dl_err['EE'] = np.insert(self.dl_err['EE'], 0, binit(planck_dl_err_ee_lo, self.bins))
