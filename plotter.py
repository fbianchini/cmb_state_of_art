import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc, rcParams
from matplotlib.ticker import FormatStrFormatter, NullFormatter, LogLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import experiments

# rc('text',usetex=True)

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
              xscaleL='log', xscaleR='linear', yscale='log', 
              nsigma={'TT':2,'EE':2,'BB':3,'TE':0}, sigma_det={'%s'%cmb:1 for cmb in experiments.cmbs}, 
              theory='camb',
              lsep=200, datams=4, upperlimitms=4, error_lw=0.5, txtsize=12,
              plot_xerr=False,
              theorycolor={'%s'%cmb:'grey' for cmb in experiments.cmbs}, 
              theorylw={'%s'%cmb:0.7 for cmb in experiments.cmbs}, 
              theoryls={'%s'%cmb:'-' for cmb in experiments.cmbs}, 
              custom_style=True, labsize=18,
              ylim=(3e-5,1e4),
              **kwargs):

    lmin = kwargs.get('lmin', 2)
    lmax = kwargs.get('lmax', 3000)

    if lsep is not None: assert(lmin<lsep)
    
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

    if custom_style:
        SetPlotStyle()

    dl_lens = {}
    dl_tens = {}
    
    if theory == 'camb':
        try:
            import camb
            pars = camb.CAMBparams(max_l_tensor=kwargs.get('max_l_tensor',10000), 
                                   max_eta_k_tensor=kwargs.get('max_eta_k_tensor',25000))

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
                                    r=kwargs.get('r',1),
                                    nt=kwargs.get('nt',0)
                                    )
            pars.set_for_lmax(kwargs.get('lmax',10000)+500,
                    lens_potential_accuracy=kwargs.get('lens_potential_accuracy',2))
            results = camb.get_results(pars)
            powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')

            dl_lens['TT'] = powers['lensed_scalar'][:,0]
            dl_lens['EE'] = powers['lensed_scalar'][:,1]
            dl_lens['BB'] = powers['lensed_scalar'][:,2]
            dl_lens['TE'] = powers['lensed_scalar'][:,3]
            dl_tens['BB'] = powers['tensor'][:,2] 
            l_th = np.arange(dl_lens['TT'].size)
        except:
            l_th, dl_lens['TT'], dl_lens['EE'], dl_lens['BB'] = np.loadtxt('data/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt', unpack = 1, usecols = (0, 1, 3, 4))
            dl_tens['BB'] = np.zeros_like(dl_lens['TT'])
    elif theory == 'fromfile':
        l_th, dl_lens['TT'], dl_lens['EE'], dl_lens['BB'] = np.loadtxt('data/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt', unpack = 1, usecols = (0, 1, 3, 4))
        dl_tens['BB'] = np.zeros_like(dl_lens['TT'])
 
    plt.figure() #figsize=kwargs.get('figsize')); this wasn't working for unkonwn reasons
    if kwargs.get('figsize') is not None:
        fs = kwargs.get('figsize')
        plt.gcf().set_size_inches(fs[0],fs[1])

    axR = plt.subplot(111)
    
    axR.set_xscale(xscaleR)
    axR.set_yscale(yscale)
    axR.set_xlim([lmin, lmax])
    axR.set_ylim(ylim)
    ylab = kwargs.get('ylabel',r'Power $\frac{\ell(\ell+1)}{2\pi}C_{\ell}$ [$\mu K^2$]')
    axR.set_ylabel(ylab, size=kwargs.get('ylabsize',15))
    xlab_pos = (0.5, -0.01)
    # axR.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    
    axlist = [axR]

    def add_left_axes(axR):
        axR.set_xlim((lsep, lmax))
        axR.yaxis.set_ticks_position('right')
        axR.yaxis.set_visible(False)
        axR.spines['left'].set_visible(False)

        divider = make_axes_locatable(axR)
        axL = divider.append_axes("left", size=2.0, pad=0, sharey=axR)
        axL.set_xscale(xscaleL)
        axL.set_yscale(yscale)
        axL.set_xlim((lmin, lsep))
        axL.spines['right'].set_visible(False)
        # y_minor = LogLocator(base=10.0, subs=np.arange(1.0, 10.0) *10, numticks=10)
        # axR.yaxis.set_minor_locator(y_minor)
        axL.yaxis.set_minor_formatter(NullFormatter())
        axL.yaxis.set_ticks_position('left')
        axR.set_ylabel(r'')
        axL.set_ylabel(ylab, size=15)

        return axL 

    if lsep is not None:
        axL = add_left_axes(axR)
        axlist.append(axL)
        xlab_pos = (1.2, -0.01)

    plt.annotate(kwargs.get('xlabel',r'Multipole $\ell$'), 
                xy=xlab_pos, xytext=(0, 4),
                xycoords=('axes fraction', 'figure fraction'),
                textcoords='offset points',
                size=labsize, ha='center', va='bottom')

    def plot_on_axes(axes, x_in, y_in, **kwargs):
        # axes is a list
        for axis in axes:
            axis.plot(x_in, y_in, **kwargs)
    
    def errorbars_on_axes(axes, x_in, y_in, **kwargs):
        for axis in axes:
            axis.errorbar(x_in, y_in, **kwargs)

    
        
    # Plotting the theory
    if 'plot_all_theory' in kwargs:
        cmbth = ['TT','EE','BB']
    else:
        cmbth = np.atleast_1d(cmbs)

    for cmb in cmbth:
        plot_on_axes(axlist, l_th, dl_lens[cmb], color=theorycolor[cmb],
                    lw=theorylw[cmb], ls=theoryls[cmb])

        if cmb.upper() == 'BB':
            rscale = kwargs.get('rscale', 0.01)
            tcolor = kwargs.get('tensor_color',theorycolor[cmb])
            tls    = kwargs.get('tensor_ls', '--')
            plot_on_axes(axlist, l_th, dl_tens[cmb]*rscale,
                         color=tcolor,
                         linewidth=theorylw[cmb], ls=tls)
            if 'rscale2' in kwargs:
                rscale2 = kwargs.get('rscale2') 
                plot_on_axes(axlist, l_th, dl_tens[cmb]*rscale2,
                         color=tcolor,
                         linewidth=theorylw[cmb], ls=tls)

    for cmb in np.atleast_1d(cmbs):

        # Plotting the experiments
        for exp in exps_to_plot:
            exp_tmp = all_exps[exp](color=kwargs.get('expt_color')[exp]) if 'expt_color' in kwargs else all_exps[exp]()
            if exp_tmp.dl_err[cmb] is not None:
                # ugh...
                if np.atleast_2d(np.asarray(exp_tmp.dl_err[cmb])).shape[0] == 1:
                    detbins = exp_tmp.dl[cmb]/exp_tmp.dl_err[cmb] > sigma_det[cmb]
                else:
                    detbins = exp_tmp.dl[cmb]/exp_tmp.dl_err[cmb][0] > sigma_det[cmb]
                if cmb == 'TE': detbins = np.ones_like(detbins, dtype=bool)

                # detections
                xerr = (exp_tmp.l_hi[cmb]-exp_tmp.l_lo[cmb])[detbins]/2.0 if (plot_xerr and exp_tmp.l_lo[cmb] is not None) else None
                if np.atleast_2d(np.asarray(exp_tmp.dl_err[cmb])).shape[0] == 1:
                    errorbars_on_axes(axlist, exp_tmp.l[cmb][detbins],
                        exp_tmp.dl[cmb][detbins],
                        yerr=exp_tmp.dl_err[cmb][detbins], 
                        xerr=xerr, 
                        fmt='.',
                        elinewidth=error_lw, color=exp_tmp.color, ms=datams)
                else:
                    errorbars_on_axes(axlist, exp_tmp.l[cmb][detbins], exp_tmp.dl[cmb][detbins], 
                                     yerr=[exp_tmp.dl_err[cmb][0][detbins],exp_tmp.dl_err[cmb][1][detbins]], 
                                     xerr=xerr,
                                     fmt='.', elinewidth=error_lw, color=exp_tmp.color, ms=datams)


                # non-detections
                xerr = (exp_tmp.l_hi[cmb]-exp_tmp.l_lo[cmb])[~detbins]/2.0 if (plot_xerr and exp_tmp.l_lo[cmb] is not None) else None
                errorbars_on_axes(axlist, exp_tmp.l[cmb][~detbins], exp_tmp.dl[cmb][~detbins]*nsigma[cmb], 
                                yerr=0, xerr=xerr, fmt='v', ms=upperlimitms, elinewidth=error_lw, color=exp_tmp.color)
    return axR


def make_PBDR_BB_plot():
    def SetPlotStyle2():
        rcParams_dict = {'figure.dpi': 360,
                 'font.family': 'serif',
                 'font.serif': 'Charter', #'Computer Modern Roman',
                 'font.size': 10.5*12/11.,
                 'text.usetex': True,
                 'axes.axisbelow': False,
                 'axes.labelsize': 10.5*12/11.,
                 'axes.linewidth': 0.550,
                 'lines.linewidth': 1.3,
                 'legend.fontsize': 7.5*12/11.,
                 'legend.labelspacing': 0.2,
                 'legend.borderaxespad': 0.3,
                 'legend.frameon': False,
                 'xtick.top': True,
                 'xtick.direction': 'in',
                 'xtick.labelsize': 8.5*12/11.,
                 'xtick.major.size': 2.8,
                 'xtick.minor.size': 1.556,
                 'xtick.major.width': 0.549,
                 'xtick.minor.width': 0.366,
                 'xtick.major.pad': 3.,
                 'xtick.minor.pad': 3.,
                 'xtick.minor.visible': True,
                 'ytick.right': True,
                 'ytick.direction': 'in',
                 'ytick.labelsize': 8.5*12/11.,
                 'ytick.major.size': 2.8,
                 'ytick.minor.size': 1.556,
                 'ytick.major.width': 0.549,
                 'ytick.minor.width': 0.366,
                 'ytick.major.pad': 3.,
                 'ytick.minor.pad': 3.,
                 'ytick.minor.visible': True,
                 'savefig.dpi': 360,
                 'savefig.format': 'eps',
                 'savefig.bbox': 'tight',
                 'savefig.pad_inches': 0.05,
                 'savefig.transparent': True
                }
        plt.rcParams.update(rcParams_dict)
        latex_line = r'\usepackage[scale=0.97]{XCharter}'
        latex_line += r'\usepackage[xcharter, bigdelims, vvarbb, scaled=1.05]{newtxmath}'
        rcParams['text.latex.preamble'] = latex_line

    SetPlotStyle2()

    ratio = (1+np.sqrt(5.))/2.

    act_color    = 'gold'
    bk_color     = 'orangered'
    spt_color    = 'lightsalmon'
    pb_color     = '#feb308'

    axMain = make_plot(['BK18','SPTpol','ACTpol_DR4','POLARBEAR17'], lsep=None,
               cmbs='BB', xscaleR='log', 
               plot_xerr=True,
               ylim=[1e-6, 1e4],
               lmax = 5000,
               theorycolor={'TT':'k', 'EE':'firebrick', 'BB':'forestgreen'},
               tensor_color='navy',
               tensor_ls = '-',
               rscale=0.001,
               rscale2=0.03,
               theorylw={'%s'%cmb:1.3 for cmb in experiments.cmbs},
               xlabel=r'Multipole moment\ $\ell$',
               ylabel=r'$\ell (\ell+1)\hskip1.5pt C_{\ell} / (2 \pi)\ [\mu\mathrm{K}^2]$',
               figsize=[6, 6./ratio], 
               custom_style = False, 
               labsize=plt.rcParams.get('axes.labelsize'),
               ylabsize=plt.rcParams.get('axes.labelsize'),
               plot_all_theory=True,
               expt_color={'BK18':bk_color, 'SPTpol':spt_color,
                   'ACTpol_DR4':act_color, 'POLARBEAR17':pb_color},
               datams=5, upperlimitms=3,
               )

    #angular scale, top x ticks (consider moving to make_plot())
    ax2 = axMain.twiny()
    ax2.plot([0,0],[1e-6,20000])
    ax2.set_xlim(axMain.get_xlim())
    ax2.set_xscale('log')
    ax2.set_xticks([180/10**ii for ii in range(-1,2)])
    ax2.set_xticklabels(['$0.1$', '$1$', '$10$'], minor=False)
    ax2.set_xticks([180/(n * 10**ii) for ii in range(-2,2) for n in range(2, 10)], minor=True)
    ax2.set_xlabel('Angular scale\ $[^\circ]$', labelpad=6)

    axMain.set_xticks([10, 100, 1000])
    axMain.set_xticklabels([10, 100, 1000])
    axMain.set_yticks([10**ii for ii in range(-6, 5, 2)])
    axMain.set_yticks([10**ii for ii in range(-5, 4, 2)], minor=True)
    axMain.set_yticklabels([], minor=True)

    txtsize = plt.rcParams.get('legend.fontsize')
    axMain.text(9, 1150, 'Temperature', color='k', rotation=2, size=txtsize) #-2)
    axMain.text(11, 0.0045, 'E-modes', color='firebrick', rotation=17, size=txtsize) #-2,)
    axMain.text(2.2, 6.6e-4, 'r=0.03', color='navy', size=txtsize) #-2)
    axMain.text(2.2, 2.2e-5, 'r=0.001', color='navy', size=txtsize) #-2)
    axMain.text(10.2, 1.4e-4, 'GW B-modes', color='navy',rotation=20, size=txtsize) #-2)
    axMain.text(10, 1.3e-5, 'Lensing B-modes', color='forestgreen', rotation=22, size=txtsize) 

    axMain.text(1000, 0.008, 'BICEP/Keck', color=bk_color, size=txtsize, horizontalalignment='center')
    axMain.text(1000, 0.003,'SPTpol', color=spt_color, size=txtsize, horizontalalignment='center')
    axMain.text(1000, 0.0011,  'POLARBEAR', color=pb_color, size=txtsize, horizontalalignment='center')
    axMain.text(1000, 0.00045,  'ACT', color=act_color, size=txtsize, horizontalalignment='center')

    return

