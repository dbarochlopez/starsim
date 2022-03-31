import sys
from scipy import optimize
import numpy as np
import math as m
import random
from . import nbspectra
from . import spectra


########################################################################################
########################################################################################
#                              INVERSION    FUNCTIONS                                  #
########################################################################################
########################################################################################
def inversion_parallel_MCMC(self,P,pbound,logpriors,typ,i):
    p_used,best_map,best_lnL=lnposteriorMCMC(self,P[i],pbound,logpriors,typ)
    # percent = ("{0:." + str(0) + "f}").format(100 * (i / float(self.steps)))
    # filledLength = int(100 * i // self.steps)
    # bar = '\u2588' * filledLength + '-' * (100 - filledLength)
    # sys.stdout.write("\r |{}| {}% {}/{}".format(bar,percent,i,self.steps))
    sys.stderr.flush()
    np.set_printoptions(precision=3,suppress=True)
    print(P[i],'{:.2f}'.format(best_lnL))
    return p_used,best_map, best_lnL

def lnposteriorMCMC(self,P,pbound,logprior,typ):
    """
    The natural logarithm of the joint posterior.

    Args:
        P (array): contains the individual parameter values
        pbound (2D array): contains the upper and lower bounds of the individual parameters
        logprior (2D array): contains information abount the priors used. Flag, mean and std.
        vparams (array): values of all parameters, including fixed parameters.
        fit (array): flag indicating is the parameter is to be fitted
        typ (array): indicates if its lc, rv or crx.
    """

    lp = lnpriorMCMC(P,pbound,logprior) #get the prior

    # if the prior is not finite return a probability of zero (log probability of -inf), to avoid computing the likelihood and save time
    if not np.isfinite(lp):
        return -np.inf

    p_used, best_map, lnL=lnlikeMCMC(self,P,typ)

    # return the likeihood times the prior (log likelihood plus the log prior)
    return p_used, best_map, lp + lnL



def lnpriorMCMC(P,pbound,logprior):
    """
    The natural logarithm of the prior probability.

    Args:
        P (array): contains the individual parameter values
        pbound (2D array): contains the upper and lower bounds of the individual parameters
        logprior (2D array): contains information abount the priors used. Flag, mean and std.

    Note:
        We can ignore the normalisations of the prior here.
    """

    lp = 0.
    if np.any((pbound[:,1]<P)+(P<pbound[:,0])): #check if the parameters are outside bounds
        return -np.inf

    for i in range(len(P)):
        
        if logprior[i,0]==0:
            #uniform prior
            lp+=0.

        if logprior[i,0]==1:
            #Gaussian prior
            lp-= 0.5 * ((P[i]-logprior[i,1])/logprior[i,2])**2

        if logprior[i,0]==2:
            #log-Gaussian prior
            lp-= 0.5 * ((np.log(P[i])-logprior[i,1])/logprior[i,2])**2

    return lp


def lnlikeMCMC(self,P,typ):
    """
    The natural logarithm of the joint Gaussian likelihood.

    Args:
        P (array): contains the individual parameter values
        vparams (array): values of all parameters, including fixed parameters.
        fit (array): flag indicating is the parameter is to be fitted
        typ (array): indicates if its lc, rv or crx.

    """

    #Variable p contains all the parameters available, fixed and optimized. P are the optimized parameters,vparam are the fixed params.
    p=np.zeros(len(self.vparam))
    # print(P)
    ii=0
    for i in range(len(self.fit)):
      if self.fit[i]==0:
        p[i]=self.vparam[i]
      elif self.fit[i]==1:
        p[i]=P[ii]
        ii=ii+1


    #Assign the new variables to the parameters, in the order they are defined.
    self.temperature_photosphere = p[0]
    self.spot_T_contrast = p[1]
    self.facula_T_contrast = p[2]
    self.facular_area_ratio = p[3]
    self.convective_shift = p[4]
    self.rotation_period = p[5]
    self.inclination = np.deg2rad(90-p[6]) #axis inclinations in rad (inc=0 has the axis pointing up). The input was in deg defined as usual.
    self.radius = p[7] #in Rsun
    self.limb_darkening_q1 = p[8]
    self.limb_darkening_q2 = p[9]
    self.planet_period = p[10]
    self.planet_transit_t0 = p[11]
    self.planet_semi_amplitude = p[12]
    self.planet_esinw = p[13]
    self.planet_ecosw = p[14]
    if (self.planet_esinw**2 + self.planet_ecosw**2)>=1: return p, self.spot_map, -np.inf #check if eccentricity is valid
    self.planet_radius = p[15]
    self.planet_impact_param = p[16]
    self.planet_spin_orbit_angle = p[17]*np.pi/180 #deg2rad


    gbest,gbest_lnL,DeltalnL,counter=inversion_PSO(self,typ,only_inversion=False)
  
    return p, gbest, gbest_lnL








def lnposteriorPSO(self,P,pbound,logprior,typ,mode_lc_params,mode_rv_params):

    lp = lnpriorPSO(self,P,pbound,logprior) #get the prior. Check if spots are overlapping
    # if the prior is not finite return a probability of zero (log probability of -inf), to avoid computing the likelihood and save time
    if not np.isfinite(lp):
        return -np.inf

    lnL=lnlikePSO(self,typ,mode_lc_params,mode_rv_params)
    # np.set_printoptions(precision=3,suppress=True)
    # print(P,lp,lnL,lp+lnL)
    # return the likeihood times the prior (log likelihood plus the log prior)
    return lp + lnL

def lnpriorPSO(self,P,pbound,logprior):
    """
    The natural logarithm of the prior probability.

    Args:
        P (array): contains the individual parameter values
        pbound (2D array): contains the upper and lower bounds of the individual parameters
        logprior (2D array): contains information abount the priors used. Flag, mean and std.

    Note:
        We can ignore the normalisations of the prior here.
    """

    lp = 0.
    if np.any((pbound[:,1]<P)+(P<pbound[:,0])): #check if the parameters are outside bounds
        return -np.inf

    if nbspectra.check_spot_overlap(self.spot_map,self.facular_area_ratio):
        return -np.inf


    for i in range(len(P)):
        
        if logprior[i,0]==0:
            #uniform prior
            lp+=0.

        if logprior[i,0]==1:
            #Gaussian prior
            lp-= 0.5 * ((P[i]-logprior[i,1])/logprior[i,2])**2

        if logprior[i,0]==2:
            #log-Gaussian prior
            lp-= 0.5 * ((np.log(P[i])-logprior[i,1])/logprior[i,2])**2
    return lp






def lnlikePSO(self,typ,mode_lc_params,mode_rv_params):
    #Compute the model for each instrument and observable, and the corresponding lnL
    lnL=0.0 
    l=0
    r=0
    for i in range(len(self.instruments)): #for each instrument
        for j in np.unique(typ[i]): #for each observable of the instrument
            if j==0: #photometric case
                t=self.data[self.instruments[i]]['lc']['t']
                _,FLUX,_,_,_,_=nbspectra.generate_rotating_photosphere_fast_lc(t,mode_lc_params[l][0],mode_lc_params[l][1],mode_lc_params[l][2],mode_lc_params[l][3],mode_lc_params[l][4],mode_lc_params[l][5],mode_lc_params[l][6],5,self.use_phoenix_limb_darkening,self.limb_darkening_law,self.limb_darkening_q1,self.limb_darkening_q2,self.spot_map,self.reference_time,self.rotation_period,self.differential_rotation,self.spots_evo_law,self.facular_area_ratio,self.inclination,self.temperature_photosphere,self.temperature_facula,self.simulate_planet,self.planet_esinw,self.planet_ecosw,self.planet_transit_t0,self.planet_period,self.planet_radius,self.planet_impact_param,self.planet_semi_major_axis,self.planet_spin_orbit_angle)

                if (self.data[self.instruments[i]]['lc']['fix_jitter'] and self.data[self.instruments[i]]['lc']['fix_offset']):
                    offset=1.0 #offsets and jitters already set
                    jitter=0.0
                    data=self.data[self.instruments[i]]['lc']['y']
                    error=self.data[self.instruments[i]]['lc']['yerr']
                    newerror=np.sqrt((error/offset)**2+jitter**2)
                    lnL=lnL-0.5*np.sum(((data/offset-FLUX)/(newerror))**2.0+np.log(2.0*np.pi)+np.log(newerror**2))

                elif self.data[self.instruments[i]]['lc']['fix_offset']:
                    offset=1.0
                    data=self.data[self.instruments[i]]['lc']['y']/offset
                    error=self.data[self.instruments[i]]['lc']['yerr']/offset
                    res=optimize.minimize(nbspectra.fit_only_jitter,2*np.mean(error), args=(FLUX,data,error), method='Nelder-Mead')
                    lnL=lnL-res.fun
                
                elif self.data[self.instruments[i]]['lc']['fix_jitter']:
                    jitter=0.0
                    newerror=np.sqrt(self.data[self.instruments[i]]['lc']['yerr']**2+jitter**2)
                    res=optimize.minimize(nbspectra.fit_only_multiplicative_offset,np.mean(self.data[self.instruments[i]]['lc']['y'])/(np.mean(FLUX)+0.0001), args=(FLUX,self.data[self.instruments[i]]['lc']['y'],newerror), method='Nelder-Mead')
                    lnL=lnL-res.fun
                
                else:
                    res=optimize.minimize(nbspectra.fit_multiplicative_offset_jitter,[np.mean(self.data[self.instruments[i]]['lc']['y'])/(np.mean(FLUX)+0.0001),2*np.mean(self.data[self.instruments[i]]['lc']['yerr'])], args=(FLUX,self.data[self.instruments[i]]['lc']['y'],self.data[self.instruments[i]]['lc']['yerr']), method='Nelder-Mead')
                    lnL=lnL-res.fun

                l+=1

            if j==1: #spectroscopic case
                idx_rv=np.where(np.array(typ[i])==1)[0] #indexs of observables that are rv bis or fwhm, contrast. Ideally only one

                t=self.data[self.instruments[i]][self.observables[i][idx_rv[0]]]['t']
                _,CCF,_,_,_,_=nbspectra.generate_rotating_photosphere_fast_rv(t,mode_rv_params[r][0],mode_rv_params[r][1],mode_rv_params[r][2],mode_rv_params[r][3],mode_rv_params[r][4],mode_rv_params[r][5],mode_rv_params[r][6],mode_rv_params[r][7],mode_rv_params[r][8],mode_rv_params[r][9],mode_rv_params[r][10],mode_rv_params[r][11],mode_rv_params[r][12],mode_rv_params[r][13],mode_rv_params[r][14],5,self.use_phoenix_limb_darkening,self.limb_darkening_law,self.limb_darkening_q1,self.limb_darkening_q2,self.spot_map,self.reference_time,self.rotation_period,self.differential_rotation,self.spots_evo_law,self.facular_area_ratio,self.inclination,self.vsini,self.convective_shift,self.temperature_photosphere,self.temperature_facula,self.simulate_planet,self.planet_esinw,self.planet_ecosw,self.planet_transit_t0,self.planet_period,self.planet_radius,self.planet_impact_param,self.planet_semi_major_axis,self.planet_spin_orbit_angle)
                if self.simulate_planet:
                    rvkepler = spectra.keplerian_orbit(t,[self.planet_period,self.planet_semi_amplitude,self.planet_esinw,self.planet_ecosw,self.planet_transit_t0])
                else:
                    rvkepler = 0.0

                ccf_params=spectra.compute_ccf_params(self,mode_rv_params[r][4],CCF,plot_test=False)
                self.results['rv']=ccf_params[0] + rvkepler - mode_rv_params[r][-4] #subtract rv of immaculate photosphere
                self.results['contrast']=ccf_params[1] #- mode_rv_params[r][-3] #offsets
                self.results['fwhm']=ccf_params[2] #- mode_rv_params[r][-2]
                self.results['bis']=ccf_params[3] #- mode_rv_params[r][-1] 
                for k in idx_rv:

                    if (self.data[self.instruments[i]][self.observables[i][k]]['fix_jitter'] and self.data[self.instruments[i]][self.observables[i][k]]['fix_offset']):
                        offset=0.0
                        jitter=0.0
                        newerror=np.sqrt(self.data[self.instruments[i]][self.observables[i][k]]['yerr']**2+jitter**2)
                        lnL=lnL-0.5*np.sum(((self.data[self.instruments[i]][self.observables[i][k]]['y']-offset-self.results[self.observables[i][k]])/(newerror))**2.0+np.log(2.0*np.pi)+np.log(newerror**2))

                    elif self.data[self.instruments[i]][self.observables[i][k]]['fix_offset']:
                        offset=0.0
                        res=optimize.minimize(nbspectra.fit_only_jitter,2*np.mean(self.data[self.instruments[i]][self.observables[i][k]]['yerr']), args=(self.results[self.observables[i][k]],self.data[self.instruments[i]][self.observables[i][k]]['y']-offset,self.data[self.instruments[i]][self.observables[i][k]]['yerr']), method='Nelder-Mead')
                        lnL=lnL-res.fun
                    
                    elif self.data[self.instruments[i]][self.observables[i][k]]['fix_jitter']:
                        jitter=0.0
                        newerror=np.sqrt(self.data[self.instruments[i]][self.observables[i][k]]['yerr']**2+jitter**2)
                        res=optimize.minimize(nbspectra.fit_only_linear_offset,np.mean(self.data[self.instruments[i]][self.observables[i][k]]['y'])-(np.mean(self.results[self.observables[i][k]])), args=(self.results[self.observables[i][k]],self.data[self.instruments[i]][self.observables[i][k]]['y'],newerror), method='Nelder-Mead')
                        lnL=lnL-res.fun

                    else:
                        res=optimize.minimize(nbspectra.fit_linear_offset_jitter,[np.mean(self.data[self.instruments[i]][self.observables[i][k]]['y'])-(np.mean(self.results[self.observables[i][k]])),2*np.mean(self.data[self.instruments[i]][self.observables[i][k]]['yerr'])], args=(self.results[self.observables[i][k]],self.data[self.instruments[i]][self.observables[i][k]]['y'],self.data[self.instruments[i]][self.observables[i][k]]['yerr']), method='Nelder-Mead')
                        lnL=lnL-res.fun

                    r+=1

            if j==2: #chromatic-spectroscopic case
                idx_crx=np.where(np.array(typ[i])==2)[0] #indexs of observables that are crx. Ideally only one
                self.wavelength_lower_limit=self.data[self.instruments[i]]['wvmin']
                self.wavelength_upper_limit=self.data[self.instruments[i]]['wvmax']
                self.compute_forward(observables=['crx'],t=self.data[self.instruments[i]][self.observables[i][idx_crx[0]]]['t'],inversion=True)
                for k in idx_crx:
                    
                    if (self.data[self.instruments[i]][self.observables[i][k]]['fix_jitter'] and self.data[self.instruments[i]][self.observables[i][k]]['fix_offset']):
                        offset=0.0
                        jitter=0.0
                        newerror=np.sqrt(self.data[self.instruments[i]][self.observables[i][k]]['yerr']**2+jitter**2)
                        lnL=lnL-0.5*np.sum(((self.data[self.instruments[i]][self.observables[i][k]]['y']-offset-self.results[self.observables[i][k]])/(newerror))**2.0+np.log(2.0*np.pi)+np.log(newerror**2))

                    elif self.data[self.instruments[i]][self.observables[i][k]]['fix_offset']:
                        offset=0.0
                        res=optimize.minimize(nbspectra.fit_only_jitter,2*np.mean(self.data[self.instruments[i]][self.observables[i][k]]['yerr']), args=(self.results[self.observables[i][k]],self.data[self.instruments[i]][self.observables[i][k]]['y']-offset,self.data[self.instruments[i]][self.observables[i][k]]['yerr']), method='Nelder-Mead')
                        lnL=lnL-res.fun
                    
                    elif self.data[self.instruments[i]][self.observables[i][k]]['fix_jitter']:
                        jitter=0.0
                        newerror=np.sqrt(self.data[self.instruments[i]][self.observables[i][k]]['yerr']**2+jitter**2)
                        res=optimize.minimize(nbspectra.fit_only_linear_offset,np.mean(self.data[self.instruments[i]][self.observables[i][k]]['y'])-(np.mean(self.results[self.observables[i][k]])), args=(self.results[self.observables[i][k]],self.data[self.instruments[i]][self.observables[i][k]]['y'],newerror), method='Nelder-Mead')
                        lnL=lnL-res.fun
                    
                    else:
                        res=optimize.minimize(nbspectra.fit_linear_offset_jitter,[np.mean(self.data[self.instruments[i]][self.observables[i][k]]['y'])-(np.mean(self.results[self.observables[i][k]])),2*np.mean(self.data[self.instruments[i]][self.observables[i][k]]['yerr'])], args=(self.results[self.observables[i][k]],self.data[self.instruments[i]][self.observables[i][k]]['y'],self.data[self.instruments[i]][self.observables[i][k]]['yerr']), method='Nelder-Mead')
                        lnL=lnL-res.fun

    return lnL

def inversion_parallel(self,typ,i):
    best_map,best_lnL,DeltalnL,counter=inversion_PSO(self,typ,only_inversion=True)
    print('Inversion',i,'complete with a lnL of',best_lnL,'and a DeltalnL of the 1sigma best solutions at termination of',DeltalnL,'. Total iterations at termination',counter)
    return best_map, best_lnL




def inversion_PSO(self,typ,only_inversion=False):
    np.random.seed()
    #Compute models and other computationally expensive params for each instrument and RV o LC
    mode_lc_params=[]
    mode_rv_params=[]
    #First grid coordinates:
    Ngrids, Ngrid_in_ring, centres, amu, rs, alphas, xs, ys, zs, are, pare = nbspectra.generate_grid_coordinates_nb(self.n_grid_rings) #for this fast mode only use 5 circles
    theta, phi = np.arccos(zs*np.cos(-self.inclination)-xs*np.sin(-self.inclination)), np.arctan2(ys,xs*np.cos(-self.inclination)+zs*np.sin(-self.inclination))#coordinates in the star reference 
    rvel=self.vsini*np.sin(theta)*np.sin(phi)

    if len(self.instruments)==0:
        sys.exit('There is no external data loaded. Please use the starsim function ss.load_data(t,y,yerr,instrument,observable,wvmin,wvmax,offset,jitter) to do so.')

    for i in range(len(self.instruments)): #instrument i
        for j in np.unique(typ[i]): #type of observable 0-lc 1-rv
            if j==0:
                #Wv and filter of the instrument
                self.wavelength_lower_limit=self.data[self.instruments[i]]['wvmin']
                self.wavelength_upper_limit=self.data[self.instruments[i]]['wvmax']
                self.filter_name=self.data[self.instruments[i]]['filter']
                #Preload phoenix spectra
                acd, wvp_lc, flnp_lc =spectra.interpolate_Phoenix_mu_lc(self,self.temperature_photosphere,self.logg) #acd is the angles at which the model is computed. 
                _,_, flns_lc =spectra.interpolate_Phoenix_mu_lc(self,self.temperature_spot,self.logg)
                #preload de filter
                f_filt = spectra.interpolate_filter(self)

                #Append all the preloaded arrays to the list of lc preloaded arrrays.
                mode_lc_params.append([Ngrid_in_ring,acd,amu,pare,flnp_lc,flns_lc,f_filt(wvp_lc)])
            elif j==1:
                #WV limits of the instrument
                self.wavelength_lower_limit=self.data[self.instruments[i]]['wvmin']
                self.wavelength_upper_limit=self.data[self.instruments[i]]['wvmax']
                #preload synthetic spectrum, both HR and LR with limb darkening
                wv_rv, flnp_rv, _ =spectra.interpolate_Phoenix(self,self.temperature_photosphere,self.logg) #returns norm spectra and no normalized, interpolated at T and logg
                _, flns_rv, _ =spectra.interpolate_Phoenix(self,self.temperature_spot,self.logg)
                if self.facular_area_ratio>0:
                    _, flnf_rv, _ =spectra.interpolate_Phoenix(self,self.temperature_facula,self.logg)
                spec_ref = flnp_rv #reference spectrum to compute CCF. Normalized

                acd, wv_rv_LR, flpk_rv =spectra.interpolate_Phoenix_mu_lc(self,self.temperature_photosphere,self.logg)
                _, _, flsk_rv =spectra.interpolate_Phoenix_mu_lc(self,self.temperature_spot,self.logg)

                #CCFs
                rv = np.arange(-self.ccf_rv_range,self.ccf_rv_range+self.ccf_rv_step,self.ccf_rv_step)

                if self.ccf_template == 'model':
                    ccf_ph = nbspectra.cross_correlation_nb(rv,wv_rv,flnp_rv,wv_rv,spec_ref)
                    ccf_sp = nbspectra.cross_correlation_nb(rv,wv_rv,flns_rv,wv_rv,spec_ref)
                    if self.facular_area_ratio>0:            
                        ccf_fc = nbspectra.cross_correlation_nb(rv,wv_rv,flnf_rv,wv_rv,spec_ref)
                    else:
                        ccf_fc=ccf_ph*0.0

                elif self.ccf_template == 'mask':
                    ccf_ph = nbspectra.cross_correlation_mask(rv,np.asarray(wv_rv,dtype='float64'),np.asarray(flnp_rv,dtype='float64'),np.asarray(self.wvm,dtype='float64'),np.asarray(self.fm,dtype='float64'))
                    ccf_sp = nbspectra.cross_correlation_mask(rv,np.asarray(wv_rv,dtype='float64'),np.asarray(flnp_rv,dtype='float64'),np.asarray(self.wvm,dtype='float64'),np.asarray(self.fm,dtype='float64'))
                    if self.facular_area_ratio>0:
                        ccf_fc = nbspectra.cross_correlation_mask(rv,np.asarray(wv_rv,dtype='float64'),np.asarray(flnp_rv,dtype='float64'),np.asarray(self.wvm,dtype='float64'),np.asarray(self.fm,dtype='float64'))
                    else:
                        ccf_fc=ccf_ph*0.0


                #ADD BISECTORS TO RV
                fun_bis_ph = spectra.bisector_fit(self,rv,ccf_ph,plot_test=False,kind_interp=self.kind_interp)
                rv_ph = rv - fun_bis_ph(ccf_ph) #subtract the bisector from the CCF.
                fun_bis_sp = spectra.bisector_fit(self,rv,ccf_sp,plot_test=False,kind_interp=self.kind_interp)
                rv_sp = rv - fun_bis_ph(ccf_sp)
                rv_fc = rv_ph
                if self.facular_area_ratio>0:            
                    fun_bis_fc = spectra.bisector_fit(self,rv,ccf_fc,plot_test=False,kind_interp=self.kind_interp)        
                    rv_fc = rv - fun_bis_ph(ccf_fc)

                #IMMACULATE PHOTOSPHERE
                ccf_ph_g, flxph= spectra.compute_immaculate_photosphere_rv(self,Ngrid_in_ring,acd,amu,pare,flpk_rv,rv_ph,rv,ccf_ph,rvel) #return ccf of each grid, and also the integrated ccf
                ccf_ph_tot = np.sum(ccf_ph_g,axis=0) #CCF of immaculate rotating pphotosphere                

                RV0, C0, F0, B0=spectra.compute_ccf_params(self,rv,[ccf_ph_tot],plot_test=False)

                mode_rv_params.append([Ngrid_in_ring,acd,amu,pare,rv,rv_ph,rv_sp,rv_fc,ccf_ph_tot,ccf_ph,ccf_sp,ccf_fc,flxph,flpk_rv,flsk_rv,RV0,C0,F0,B0])


    #All the parameters that can be optimized, their bounds, priors and ooptimization or not
    N_spots = len(self.spot_map)

    fixed_spot_it=[self.spot_map[i][0] for i in range(N_spots)]
    fixed_spot_lt=[self.spot_map[i][1] for i in range(N_spots)]
    fixed_spot_lat=[self.spot_map[i][2] for i in range(N_spots)]
    fixed_spot_lon=[self.spot_map[i][3] for i in range(N_spots)]
    fixed_spot_c1=[self.spot_map[i][4] for i in range(N_spots)]
    fixed_spot_c2=[self.spot_map[i][5] for i in range(N_spots)]
    fixed_spot_c3=[self.spot_map[i][6] for i in range(N_spots)]

    PSOvparam=np.array([*fixed_spot_it,*fixed_spot_lt,*fixed_spot_lat,*fixed_spot_lon,*fixed_spot_c1,*fixed_spot_c2,*fixed_spot_c3])

    f_spot_it=[self.spot_map[i][7] for i in range(N_spots)]
    f_spot_lt=[self.spot_map[i][8]for i in range(N_spots)]
    f_spot_lat=[self.spot_map[i][9]for i in range(N_spots)]
    f_spot_lon=[self.spot_map[i][10] for i in range(N_spots)]
    f_spot_c1=[self.spot_map[i][11] for i in range(N_spots)]
    f_spot_c2=[self.spot_map[i][12] for i in range(N_spots)]
    f_spot_c3=[self.spot_map[i][13] for i in range(N_spots)]
    PSOfit=np.array([*f_spot_it,*f_spot_lt,*f_spot_lat,*f_spot_lon,*f_spot_c1,*f_spot_c2,*f_spot_c3])      

    bound_spot_it=np.array([[self.prior_spot_initial_time[1],self.prior_spot_initial_time[2]] for i in range(N_spots)])
    bound_spot_lt=np.array([[self.prior_spot_life_time[1],self.prior_spot_life_time[2]] for i in range(N_spots)])
    bound_spot_lat=np.array([[self.prior_spot_latitude[1],self.prior_spot_latitude[2]]for i in range(N_spots)])
    bound_spot_lon=np.array([[self.prior_spot_longitude[1],self.prior_spot_longitude[2]] for i in range(N_spots)])
    bound_spot_c1=np.array([[self.prior_spot_coeff_1[1],self.prior_spot_coeff_1[2]] for i in range(N_spots)])
    bound_spot_c2=np.array([[self.prior_spot_coeff_2[1],self.prior_spot_coeff_2[2]] for i in range(N_spots)])
    bound_spot_c3=np.array([[self.prior_spot_coeff_3[1],self.prior_spot_coeff_3[2]] for i in range(N_spots)])
    PSObounds=np.array([*bound_spot_it,*bound_spot_lt,*bound_spot_lat,*bound_spot_lon,*bound_spot_c1,*bound_spot_c2,*bound_spot_c3]) 

    prior_spot_it=[spectra.generate_prior(self.prior_spot_initial_time[3],self.prior_spot_initial_time[4],self.prior_spot_initial_time[5],self.nparticles) for i in range(N_spots)]
    prior_spot_lt=[spectra.generate_prior(self.prior_spot_life_time[3],self.prior_spot_life_time[4],self.prior_spot_life_time[5],self.nparticles) for i in range(N_spots)]
    prior_spot_lat=[spectra.generate_prior(self.prior_spot_latitude[3],self.prior_spot_latitude[4],self.prior_spot_latitude[5],self.nparticles)for i in range(N_spots)]
    prior_spot_lon=[spectra.generate_prior(self.prior_spot_longitude[3],self.prior_spot_longitude[4],self.prior_spot_longitude[5],self.nparticles) for i in range(N_spots)]
    prior_spot_c1=[spectra.generate_prior(self.prior_spot_coeff_1[3],self.prior_spot_coeff_1[4],self.prior_spot_coeff_1[5],self.nparticles) for i in range(N_spots)]
    prior_spot_c2=[spectra.generate_prior(self.prior_spot_coeff_2[3],self.prior_spot_coeff_2[4],self.prior_spot_coeff_2[5],self.nparticles) for i in range(N_spots)]
    prior_spot_c3=[spectra.generate_prior(self.prior_spot_coeff_3[3],self.prior_spot_coeff_3[4],self.prior_spot_coeff_3[5],self.nparticles) for i in range(N_spots)]
    PSOpriors=np.array([*prior_spot_it,*prior_spot_lt,*prior_spot_lat,*prior_spot_lon,*prior_spot_c1,*prior_spot_c2,*prior_spot_c3])


    logprior_spot_it=np.array([[self.prior_spot_initial_time[3],self.prior_spot_initial_time[4],self.prior_spot_initial_time[5]] for i in range(N_spots)])
    logprior_spot_lt=np.array([[self.prior_spot_life_time[3],self.prior_spot_life_time[4],self.prior_spot_life_time[5]] for i in range(N_spots)])
    logprior_spot_lat=np.array([[self.prior_spot_latitude[3],self.prior_spot_latitude[4],self.prior_spot_latitude[5]]for i in range(N_spots)])
    logprior_spot_lon=np.array([[self.prior_spot_longitude[3],self.prior_spot_longitude[4],self.prior_spot_longitude[5]] for i in range(N_spots)])
    logprior_spot_c1=np.array([[self.prior_spot_coeff_1[3],self.prior_spot_coeff_1[4],self.prior_spot_coeff_1[5]] for i in range(N_spots)])
    logprior_spot_c2=np.array([[self.prior_spot_coeff_2[3],self.prior_spot_coeff_2[4],self.prior_spot_coeff_2[5]] for i in range(N_spots)])
    logprior_spot_c3=np.array([[self.prior_spot_coeff_3[3],self.prior_spot_coeff_3[4],self.prior_spot_coeff_3[5]] for i in range(N_spots)])
    PSOlogpriors=np.array([*logprior_spot_it,*logprior_spot_lt,*logprior_spot_lat,*logprior_spot_lon,*logprior_spot_c1,*logprior_spot_c2,*logprior_spot_c3]) 

    boundvel_spot_it=np.array([[-0.125*(self.prior_spot_initial_time[2]-self.prior_spot_initial_time[1]),0.125*(self.prior_spot_initial_time[2]-self.prior_spot_initial_time[1])] for i in range(N_spots)])
    boundvel_spot_lt=np.array([[-0.125*(self.prior_spot_life_time[2]-self.prior_spot_life_time[1]),0.125*(self.prior_spot_life_time[2]-self.prior_spot_life_time[1])] for i in range(N_spots)])
    boundvel_spot_lat=np.array([[-0.125*(self.prior_spot_latitude[2]-self.prior_spot_latitude[1]),0.125*(self.prior_spot_latitude[2]-self.prior_spot_latitude[1])]for i in range(N_spots)])
    boundvel_spot_lon=np.array([[-0.125*(self.prior_spot_longitude[2]-self.prior_spot_longitude[1]),0.125*(self.prior_spot_longitude[2]-self.prior_spot_longitude[1])] for i in range(N_spots)])
    boundvel_spot_c1=np.array([[-0.125*(self.prior_spot_coeff_1[2]-self.prior_spot_coeff_1[1]),0.125*(self.prior_spot_coeff_1[2]-self.prior_spot_coeff_1[1])] for i in range(N_spots)])
    boundvel_spot_c2=np.array([[-0.125*(self.prior_spot_coeff_2[2]-self.prior_spot_coeff_2[1]),0.125*(self.prior_spot_coeff_2[2]-self.prior_spot_coeff_2[1])] for i in range(N_spots)])
    boundvel_spot_c3=np.array([[-0.125*(self.prior_spot_coeff_3[2]-self.prior_spot_coeff_3[1]),0.125*(self.prior_spot_coeff_3[2]-self.prior_spot_coeff_3[1])] for i in range(N_spots)])
    PSOboundvel=np.array([*boundvel_spot_it,*boundvel_spot_lt,*boundvel_spot_lat,*boundvel_spot_lon,*boundvel_spot_c1,*boundvel_spot_c2,*boundvel_spot_c3])


    vel_spot_it=np.array([np.random.uniform(-1,1,self.nparticles)*0.125*(self.prior_spot_initial_time[2]-self.prior_spot_initial_time[1]) for i in range(N_spots)])
    vel_spot_lt=np.array([np.random.uniform(-1,1,self.nparticles)*0.125*(self.prior_spot_life_time[2]-self.prior_spot_life_time[1]) for i in range(N_spots)])
    vel_spot_lat=np.array([np.random.uniform(-1,1,self.nparticles)*0.125*(self.prior_spot_latitude[2]-self.prior_spot_latitude[1])for i in range(N_spots)])
    vel_spot_lon=np.array([np.random.uniform(-1,1,self.nparticles)*0.125*(self.prior_spot_longitude[2]-self.prior_spot_longitude[1]) for i in range(N_spots)])
    vel_spot_c1=np.array([np.random.uniform(-1,1,self.nparticles)*0.125*(self.prior_spot_coeff_1[2]-self.prior_spot_coeff_1[1]) for i in range(N_spots)])
    vel_spot_c2=np.array([np.random.uniform(-1,1,self.nparticles)*0.125*(self.prior_spot_coeff_2[2]-self.prior_spot_coeff_2[1]) for i in range(N_spots)])
    vel_spot_c3=np.array([np.random.uniform(-1,1,self.nparticles)*0.125*(self.prior_spot_coeff_3[2]-self.prior_spot_coeff_3[1]) for i in range(N_spots)])
    PSOvel=np.array([*vel_spot_it,*vel_spot_lt,*vel_spot_lat,*vel_spot_lon,*vel_spot_c1,*vel_spot_c2,*vel_spot_c3])

    PSOvparamfit=np.array([])
    PSOboundfit=[]
    PSOpriors_fit=[]
    PSOlogpriors_fit=[]
    PSOvel_fit=[]
    PSOboundvel_fit=[]



    for i in range(len(PSOfit)):
      if PSOfit[i]==1:
        PSOvparamfit=np.append(PSOvparamfit,PSOvparam[i])
        PSOpriors_fit.append(PSOpriors[i])
        PSOlogpriors_fit.append(PSOlogpriors[i])
        PSOboundfit.append(PSObounds[i])
        PSOvel_fit.append(PSOvel[i])
        PSOboundvel_fit.append(PSOboundvel[i])
    PSOboundfit=np.asarray(PSOboundfit)
    PSOpriors_fit=np.asarray(PSOpriors_fit)
    PSOlogpriors_fit=np.asarray(PSOlogpriors_fit)
    PSOvel_fit=np.asarray(PSOvel_fit)
    PSOboundvel_fit=np.asarray(PSOboundvel_fit)

    ################ PSO OPTIONS ##########################
    ndim = len(PSOvparamfit)
    npart=self.nparticles#number of particles in the PSO
    maxfev=10000
    maxsteps=maxfev/npart #maxsteps is the maximum nuber of fevs (recommended 10000) divided by particles

    ######################################################

    X=PSOpriors_fit.T #initial position of the particles (Xi)
    V=PSOvel_fit.T #initial velocity of the particles (Vi)
    lnLs=np.zeros(npart)
    for i in range(npart): #PSO INIT STATE. EACH PARTICLE LOOP
        p_aux=np.zeros(len(PSOvparam))#auxiliar variable to store optimized and non-optimized variables
        ii=0
        for j in range(len(PSOfit)):
          if PSOfit[j]==0:
            p_aux[j]=PSOvparam[j] #non-optimized variable
          elif PSOfit[j]==1:
            p_aux[j]=X[i,ii] #optimized variable
            ii=ii+1

        for j in range(N_spots):
            self.spot_map[j,0]=p_aux[j+0*N_spots]   #set the variables of the new spot map
            self.spot_map[j,1]=p_aux[j+1*N_spots]
            self.spot_map[j,2]=p_aux[j+2*N_spots]
            self.spot_map[j,3]=p_aux[j+3*N_spots]
            self.spot_map[j,4]=p_aux[j+4*N_spots]
            self.spot_map[j,5]=p_aux[j+5*N_spots]
            self.spot_map[j,6]=p_aux[j+6*N_spots]
        #set the spot map
        lnLs[i]=lnposteriorPSO(self,p_aux,PSObounds,PSOlogpriors,typ,mode_lc_params,mode_rv_params) #log-likelihood of initial positions

    pbest=X #best coordinates of each particle (now set to the initial step)
    pbest_obj = lnLs #lnL of the pbests
    gbest=X[lnLs.argmax()] #the coordinate of the best particle (the one with the largest lnL)


    #PSO OPTIMIZATION.
    termination_condition=False
    counter=0

    lnLs=np.zeros(npart)
    while not termination_condition: #LOOP OF PSO UNTIL MAXFEV IS REACHED OR OTHER TERMINATIONS OCCUR

        V[PSOboundvel_fit[:,0]>V]=(PSOboundvel_fit[:,0]*np.ones([npart,ndim]))[PSOboundvel_fit[:,0]>V] #Velocities larger than Vmax are set to Vmax
        V[PSOboundvel_fit[:,1]<V]=(PSOboundvel_fit[:,1]*np.ones([npart,ndim]))[PSOboundvel_fit[:,1]<V] #Velocities lower than -Vmax are set to -Vmax
        #hyperparameters of PSO #Jayanti Prasad et al 2012, Piotrowski et al 2020 IILPSO
        w=0.6-0.2*counter/maxsteps #decreases from 0.6 to 0.4
        C1=2.5-2.0*counter/maxsteps #decreases from 2.5 to 0.5
        C2=0.5+2.0*counter/maxsteps #increases from 0.5 to 2.5


        r = np.random.random([2,npart,ndim])
        V = w*V + C1*r[0]*(pbest-X) + C2*r[1]*(gbest-X) #compute new velocity
        X = X + V #new position
        #CHECK IF THEY ARE INSIDE BOUNDS
        V[PSOboundfit[:,0]>X]=-V[PSOboundfit[:,0]>X] #if some X is < Xmin, change the sign of the velocity and set X=Xmin.
        X[PSOboundfit[:,0]>X]=(PSOboundfit[:,0]*np.ones([npart,ndim]))[PSOboundfit[:,0]>X]
        V[PSOboundfit[:,1]<X]=-V[PSOboundfit[:,1]<X] #if some X is > Xmax, change the sign of the velocity and set X=Xmax.
        X[PSOboundfit[:,1]<X]=(PSOboundfit[:,1]*np.ones([npart,ndim]))[PSOboundfit[:,1]<X]


        
        for i in range(npart): #PSO LOOP FOR EACH PARTICLE
            ii=0
            for j in range(len(PSOfit)):
              if PSOfit[j]==1:
                p_aux[j]=X[i,ii] #update only optimized variables
                ii=ii+1

            for j in range(N_spots):
                self.spot_map[j,0]=p_aux[j+0*N_spots]   #set the variables of the new spot map
                self.spot_map[j,1]=p_aux[j+1*N_spots]
                self.spot_map[j,2]=p_aux[j+2*N_spots]
                self.spot_map[j,3]=p_aux[j+3*N_spots]
                self.spot_map[j,4]=p_aux[j+4*N_spots]
                self.spot_map[j,5]=p_aux[j+5*N_spots]
                self.spot_map[j,6]=p_aux[j+6*N_spots]
            #set the spot map
            lnLs[i]=lnposteriorPSO(self,p_aux,PSObounds,PSOlogpriors,typ,mode_lc_params,mode_rv_params) #log-likelihood of each particle
            if lnLs[i]>pbest_obj[i]: #update pbest and its lnL
                pbest[i]=X[i]
                pbest_obj[i]=lnLs[i]

        gbest=pbest[pbest_obj.argmax()] #select the best pbest
        gbest_obj=pbest_obj.max()

        counter+=1

        if counter>maxsteps:
            termination_condition=True

        DeltalnL=gbest_obj-np.sort(pbest_obj)[-int(0.68*npart)] #difference between highest and 68%th highest lnL
        # if DeltalnL<2.0 and counter>(maxsteps/4):
        #     termination_condition=True

        if only_inversion:
            print(counter,DeltalnL,gbest_obj)

        #Write gbest to spotmap
        best_map=np.zeros([N_spots,7])
        p_aux=np.zeros(len(PSOfit))#auxiliar variable to store optimized and non-optimized variables
        ii=0
        for j in range(len(PSOfit)):
          if PSOfit[j]==0:
            p_aux[j]=PSOvparam[j] #non-optimized variable
          elif PSOfit[j]==1:
            p_aux[j]=gbest[ii] #optimized variable
            ii=ii+1
        for j in range(N_spots):
            best_map[j,0]=p_aux[j+0*N_spots]   #set the variables of the new spot map
            best_map[j,1]=p_aux[j+1*N_spots]
            best_map[j,2]=p_aux[j+2*N_spots]
            best_map[j,3]=p_aux[j+3*N_spots]
            best_map[j,4]=p_aux[j+4*N_spots]
            best_map[j,5]=p_aux[j+5*N_spots]
            best_map[j,6]=p_aux[j+6*N_spots]
  
    return best_map,gbest_obj,DeltalnL,counter




def generate_prior(flag,p1,p2,nw): #generate initial sample from priors
    if flag==0:
        prior=np.random.uniform(p1,p2,nw)

    if flag==1:
        prior=np.random.normal(p1,p2,nw)

    if flag==2:
        prior=np.exp(np.random.normal(p1,p2,nw))

    return prior



