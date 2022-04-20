import numpy as np

cmbs = ['TT','EE','BB','TE','EB','TB','KK']

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

        self.l['TT'], _, self.dl_err['TT'] = np.loadtxt('data/bandpowers_spt2500deg2_lps12.txt', unpack=1)
        self.l_lo['TE'], self.l_hi['TE'], self.l['TE'], self.dl['TE'], self.dl_err['TE'] = np.loadtxt('data/bp_for_plotting_v3.txt', unpack=1, usecols=[0,1,2,3,4])
        self.l_lo['EE'], self.l_hi['EE'], self.l['EE'], self.dl['EE'], self.dl_err['EE'] = np.loadtxt('data/bp_for_plotting_v3.txt', unpack=1, usecols=[0,1,5,6,7])
        
class ACTpol(Experiment):
    def __init__(self, telescope='ACT', exp_type='ground', color='#b3074f'):
        super().__init__(telescope, exp_type, color)

        self.l['TT'], self.dl['TT'], self.dl_err['TT'] = np.loadtxt('data/cmbonly_spectra_dr4.01/act_dr4.01_D_ell_TT_cmbonly.txt',unpack=1)
        self.l['EE'], self.dl['EE'], self.dl_err['EE'] = np.loadtxt('data/cmbonly_spectra_dr4.01/act_dr4.01_D_ell_EE_cmbonly.txt',unpack=1)
        self.l['TE'], self.dl['TE'], self.dl_err['TE'] = np.loadtxt('data/cmbonly_spectra_dr4.01/act_dr4.01_D_ell_TE_cmbonly.txt',unpack=1)

class POLARBEAR(Experiment):
    def __init__(self, telescope, exp_type, color='#33673B'):
        super().__init__(telescope, exp_type, color)

        self.l_lo['BB'], self.l_hi['BB'], self.l['BB'], self.dl['BB'], self.dl_err['BB'] = np.loadtxt('data/Polarbear_BB_2017.txt', unpack=1, usecols=[1,2,3,4,5])

class BICEPKeck(Experiment):
    def __init__(self, telescope, exp_type, color='#9B5DE5'):
        super().__init__(telescope, exp_type, color)

        self.l_, self.l_bb, self.lmax, self.dl_bb, self.dl_lo_bb, self.dl_hi_bb = np.loadtxt('data/BK18_components_20210607.txt', unpack=1, usecols=[0,1,2,3,4,5])
        self.xerr_bb = (bk_lmax - bk_lmin)/2
        bk_yerr_bb = [self.dl_max_bb-bk_dl_lo_bb,self.dl_hi_bb-self.dl_max_bb]

class Planck(Experiment):
    def __init__(self, telescope='Planck', exp_type='space', color='#137fbf'):
        super().__init__(telescope, exp_type)

        # self.l['TT'], self.dl['TT'], self.dl_err['TT'] = np.loadtxt('data/COM_PowerSpect_CMB-TT-binned_R3.01.txt', unpack=1, usecols=(0,1,2))
        # self.l['EE'], self.dl['EE'], self.dl_err['EE'] = np.loadtxt('data/COM_PowerSpect_CMB-EE-binned_R3.02.txt', unpack=1, usecols=(0,1,2))
        self.l['BB'], self.l_lo['BB'], self.l_hi['BB'], self.dl['BB']  = np.loadtxt('data/cl_planck_lolEB_NPIPE_BB.dat.txt', unpack=1, usecols=(1,0,2,7))
        self.l['TT']_lo, self.dl['TT']_lo, self.dl_err['TT']_lo = np.loadtxt('data/COM_PowerSpect_CMB-TT-full_R3.01.txt', unpack=1, usecols=(0,1,2))
        self.l_ee_lo, self.dl_ee_lo, self.dl_err_ee_lo = np.loadtxt('data/COM_PowerSpect_CMB-EE-full_R3.01.txt', unpack=1, usecols=(0,1,2))
