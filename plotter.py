import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc, rcParams
from matplotlib.ticker import FormatStrFormatter, NullFormatter, LogLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import experiments

# rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})

def SetPlotStyle():
    rc('text',usetex=True)
    rc('font',**{'family':'serif','serif':['Computer Modern']})
    plt.rcParams['axes.linewidth']  = 1.
    plt.rcParams['axes.labelsize']  = 15
    plt.rcParams['axes.titlesize']  = 15
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['xtick.major.size'] = 7
    plt.rcParams['ytick.major.size'] = 7
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['legend.fontsize']  = 15
    plt.rcParams['legend.frameon']  = False

    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1

def make_plot(exps, cmbs=['TT','EE','BB'],
              scalex_l='log', scalex_r='linear',
              nsigma=3, sigma_det=1, theory='camb',
              lsep=200, datams=4, upperlimitms=4, error_lw=0.5, txtsize=12,
              theorycolor='grey', theorylw=0.7,  cutsom_style=True, labsize=18,
              ylim=[3e-5, 1e4],
              **kwargs):

    lmin = kwargs.get('lmin', 2)
    lmax = kwargs.get('lmax', 3000)

    assert(lmin<lsep)
    
    # Get a list of all the experiments in experiments.py
    all_exps = {}
    for objname in dir(experiments):
        obj = getattr(experiments, objname)
        if isinstance(obj, (type,)):
            all_exps[obj.__name__] = obj
    all_exps.pop('Experiment', None)

    exps_to_plot = []  # Name of experiments to plot

    for exp in np.atleast_1d(exps):
        # Let's plot them all?
        if exp.lower() == 'all':
            for exp_tmp in all_exps.keys():
                exps_to_plot.append(exp_tmp)
        
        # Check whether a specific dataset is requested
        if exp in all_exps.keys():
            exps_to_plot.append(exp)

        # Check whether all ground/balloon/space datasets are requested
        if exp.lower() == 'ground' or exp.lower() == 'balloon' or exp.lower() == 'space':
            for exp_tmp in all_exps.keys():
                if all_exps[exp_tmp]().exp_type == exp:
                    exps_to_plot.append(exp_tmp)
                    
        # Finally check whether all datasets from a given telescope are requested
        else:
            for exp_tmp in all_exps.keys():
                if all_exps[exp_tmp]().telescope == exp.upper():
                    exps_to_plot.append(exp_tmp)

    exps_to_plot = np.unique(np.asarray(exps_to_plot))
    print('Plotting experiments:', exps_to_plot)

    if cutsom_style:
        SetPlotStyle()

    dl_lens = {}
    dl_tens = {}
    
    if theory == 'camb':
        import camb
        pars = camb.CAMBparams()

        # Some Planck2018 best-fit cosmology
        pars.set_cosmology(H0=kwargs.get('H0', 67.32117), 
                        ombh2=kwargs.get('ombh2',0.02238280), 
                        omch2=kwargs.get('omch2',0.1201075), 
                        mnu=kwargs.get('mnu',0.0543), 
                        omk=kwargs.get('omk', 0),
                        tau=kwargs.get('tau',0.06)
                        )

        pars.WantTensors = True
        pars.InitPower.set_params(As=kwargs.get('As',2.100549e-9), 
                                  ns=kwargs.get('ns',0.9660499), 
                                  r=kwargs.get('r',0.01)
                                  )
        pars.set_for_lmax(kwargs.get('lmax',5000)+500, lens_potential_accuracy=kwargs.get('lens_potential_accuracy',1))
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')

        dl_lens['TT'] = powers['lensed_scalar'][:,0]
        dl_lens['EE'] = powers['lensed_scalar'][:,1]
        dl_lens['BB'] = powers['lensed_scalar'][:,2]
        dl_tens['BB'] = powers['tensor'][:,2] 
        l_th = np.arange(dl_lens['TT'].size)

    else:
        l_th, dl_lens['TT'], dl_lens['EE'], dl_lens['BB'] = np.loadtxt('data/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt', unpack = 1, usecols = (0, 1, 3, 4))
        dl_tens['BB'] = np.zeros_like(dl_lens['TT'])
 
    plt.figure(figsize=kwargs.get('figsize'))
    
    axR = plt.subplot(111)
    
    axR.set_xscale(scalex_r)
    axR.set_yscale('log')
    axR.set_xlim((lsep, lmax))
    axR.set_ylim(ylim)
    axR.yaxis.set_ticks_position('right')
    axR.yaxis.set_visible(False)
    axR.spines['left'].set_visible(False)
    # axR.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    
    divider = make_axes_locatable(axR)
    axL = divider.append_axes("left", size=2.0, pad=0, sharey=axR)
    axL.set_xscale(scalex_l)
    axL.set_xlim((lmin, lsep))
    axL.spines['right'].set_visible(False)
    # y_minor = LogLocator(base=10.0, subs=np.arange(1.0, 10.0) *10, numticks=10)
    # axR.yaxis.set_minor_locator(y_minor)
    axL.yaxis.set_minor_formatter(NullFormatter())
    axL.yaxis.set_ticks_position('left')
    axL.set_ylabel(r'Power $\frac{\ell(\ell+1)}{2\pi}C_{\ell}$ [$\mu K^2$]', size=15)


    plt.annotate(r'Multipole $\ell$',
                xy=(1.2, -0.01), xytext=(0, 4),
                xycoords=('axes fraction', 'figure fraction'),
                textcoords='offset points',
                size=labsize, ha='center', va='bottom')
    
    for cmb in np.atleast_1d(cmbs):
        
        axR.plot(l_th, dl_lens[cmb], color=theorycolor, linewidth=theorylw)
        axL.plot(l_th, dl_lens[cmb], color=theorycolor, linewidth=theorylw)
        
        if cmb.upper() == 'BB':
            axR.plot(l_th, dl_tens[cmb], color=theorycolor, linewidth=theorylw, ls='--')        
            axL.plot(l_th, dl_tens[cmb], color=theorycolor, linewidth=theorylw, ls='--')
        
        for exp in exps_to_plot:
            exp_tmp = all_exps[exp]()
            if exp_tmp.dl[cmb] is not None:
                # ugh...
                if np.atleast_2d(np.asarray(exp_tmp.dl_err[cmb])).shape[0] == 1:
                    detbins = exp_tmp.dl[cmb]/exp_tmp.dl_err[cmb] > sigma_det
                else:
                    detbins = exp_tmp.dl[cmb]/exp_tmp.dl_err[cmb][0] > sigma_det 

                # detections
                if np.atleast_2d(np.asarray(exp_tmp.dl_err[cmb])).shape[0] == 1:
                    axR.errorbar(exp_tmp.l[cmb][detbins], exp_tmp.dl[cmb][detbins], yerr=exp_tmp.dl_err[cmb][detbins], fmt='.', elinewidth=error_lw, color=exp_tmp.color, ms=datams)
                    axL.errorbar(exp_tmp.l[cmb][detbins], exp_tmp.dl[cmb][detbins], yerr=exp_tmp.dl_err[cmb][detbins], fmt='.', elinewidth=error_lw, color=exp_tmp.color, ms=datams)
                else:
                    axR.errorbar(exp_tmp.l[cmb][detbins], exp_tmp.dl[cmb][detbins], yerr=[exp_tmp.dl_err[cmb][0][detbins],exp_tmp.dl_err[cmb][1][detbins]], fmt='.', elinewidth=error_lw, color=exp_tmp.color, ms=datams)
                    axL.errorbar(exp_tmp.l[cmb][detbins], exp_tmp.dl[cmb][detbins], yerr=[exp_tmp.dl_err[cmb][0][detbins],exp_tmp.dl_err[cmb][1][detbins]], fmt='.', elinewidth=error_lw, color=exp_tmp.color, ms=datams)
                    
                # non-detections
                axR.errorbar(exp_tmp.l[cmb][~detbins], exp_tmp.dl[cmb][~detbins]*nsigma, yerr=0, fmt='v', ms=upperlimitms, elinewidth=error_lw, color=exp_tmp.color)
                axL.errorbar(exp_tmp.l[cmb][~detbins], exp_tmp.dl[cmb][~detbins]*nsigma, yerr=0, fmt='v', ms=upperlimitms, elinewidth=error_lw, color=exp_tmp.color)

