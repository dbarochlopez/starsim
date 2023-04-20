import starsim
import numpy as np
import random
import matplotlib.pyplot as plt

ss=starsim.StarSim(conf_file_path='starsim.conf')
ss.compute_forward(t=np.linspace(0,10,100),observables=['rv'])


#x_ph,y_ph,dy_ph, _ =np.loadtxt('/home/baroch/Desktop/PhD/Articles/starsim/solarfit/TSI_sun_SORCE_Sep2017.txt',unpack=1)
#x_sp, y_rv, dy_rv, y_fwhm, dy_fwhm, y_bis, dy_bis, y_cont, dy_cont=np.loadtxt('/home/baroch/Desktop/PhD/Articles/starsim/solarfit/sun_spectroscopic_time_series_Sep2017.txt',unpack=1)#

##Llegeix les dades que vols utilitzar  per fer la inversio.
#ss.load_data(t=x_ph,y=y_ph,yerr=dy_ph,instrument='SORCE',observable='lc',wvmin=2000,wvmax=15000,offset=1,jitter=0.00,fix_jitter=False,fix_offset=False,filter_name='flat_TSI.dat')
#ss.load_data(t=x_sp,y=y_rv,yerr=dy_rv,instrument='HARPS-N',observable='rv',wvmin=3740,wvmax=6910,offset=0.0,jitter=0.00,fix_jitter=False,fix_offset=False)
#ss.load_data(t=x_sp,y=y_fwhm,yerr=dy_fwhm,instrument='HARPS-N',observable='fwhm',wvmin=3740,wvmax=6910,offset=1,jitter=0.00,fix_jitter=False,fix_offset=False)
#ss.load_data(t=x_sp,y=y_bis,yerr=dy_bis,instrument='HARPS-N',observable='bis',wvmin=3740,wvmax=6910,offset=1,jitter=0.00,fix_jitter=False,fix_offset=False)
##Contrast no quadra
##ss.load_data(t=x_sp,y=y_cont,yerr=dy_cont,instrument='HARPS-N',observable='contrast',wvmin=3740,wvmax=6910,offset=0,jitter=0.00,fix_jitter=False,fix_offset=False)#

##Fes la inversio, seleccionant el numero dinversions. Usara els CPUs que li diguis al starsim.conf
#best_maps, lnLs =ss.compute_inverseSA(N_inversions=18) #retorna el spotmap de cada inversio i el lnL corresponent. Tot aixo esta a un fitxer a results.#

#ss.plot_inversion_results(best_maps,lnLs,Npoints=200)
#ss.plot_spot_map(best_maps,tref=np.linspace(2457980,2458030,50))
#ss.plot_active_longitudes(best_maps,tini=2457980,tfin=2458030,N_obs=200)