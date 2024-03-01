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


    gbest,gbest_lnL=inversion_SA(self,typ,only_inversion=False)
    return p, gbest, gbest_lnL








def lnposteriorSA(self,P,pbound,logprior,typ,mode_lc_params,mode_rv_params):
    lp = lnpriorSA(self,P,pbound,logprior) #get the prior. Check if spots are overlapping
    # if the prior is not finite return a probability of zero (log probability of -inf), to avoid computing the likelihood and save time
    if not np.isfinite(lp):
        return -np.inf
    lnL=lnlikeSA(self,typ,mode_lc_params,mode_rv_params)
    # np.set_printoptions(precision=3,suppress=True)
    # print(P,lp,lnL,lp+lnL)
    # return the likeihood times the prior (log likelihood plus the log prior)
    return lp + lnL

def lnpriorSA(self,P,pbound,logprior):
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






def lnlikeSA(self,typ,mode_lc_params,mode_rv_params):
    #Compute the model for each instrument and observable, and the corresponding lnL
    lnL=0.0 
    l=0
    r=0
    for i in range(len(self.instruments)): #for each instrument

        for j in np.unique(typ[i]): #for each observable of the instrument

            if j==0: #photometric case
                t=self.data[self.instruments[i]]['lc']['t']
                _,FLUX,_,_,_,_=nbspectra.generate_rotating_photosphere_fast_lc(t,mode_lc_params[l][0],mode_lc_params[l][1],mode_lc_params[l][2],mode_lc_params[l][3],mode_lc_params[l][4],mode_lc_params[l][5],mode_lc_params[l][6],self.n_grid_rings,self.use_phoenix_limb_darkening,self.limb_darkening_law,self.limb_darkening_q1,self.limb_darkening_q2,self.spot_map,self.reference_time,self.rotation_period,self.differential_rotation,self.spots_evo_law,self.facular_area_ratio,self.inclination,self.temperature_photosphere,self.temperature_facula,self.simulate_planet,self.planet_esinw,self.planet_ecosw,self.planet_transit_t0,self.planet_period,self.planet_radius,self.planet_impact_param,self.planet_semi_major_axis,self.planet_spin_orbit_angle)
                data=self.data[self.instruments[i]]['lc']['y']
                error=self.data[self.instruments[i]]['lc']['yerr']

                if (self.data[self.instruments[i]]['lc']['fix_jitter'] and self.data[self.instruments[i]]['lc']['fix_offset']):
                    lnL=lnL-0.5*np.sum(((data-FLUX)/(error))**2.0+np.log(2.0*np.pi)+np.log(error**2))
                elif self.data[self.instruments[i]]['lc']['fix_offset']:
                    res=optimize.minimize(nbspectra.fit_only_jitter,2*np.mean(error), args=(FLUX,data,error), method='Nelder-Mead')
                    lnL=lnL-res.fun                
                elif self.data[self.instruments[i]]['lc']['fix_jitter']:
                    res=optimize.minimize(nbspectra.fit_only_multiplicative_offset,np.mean(data)/(np.mean(FLUX)+0.0001), args=(FLUX,data,error), method='Nelder-Mead')
                    lnL=lnL-res.fun
                else:
                    res=optimize.minimize(nbspectra.fit_multiplicative_offset_jitter,[np.mean(data)/(np.mean(FLUX)+0.0001),2*np.mean(error)], args=(FLUX,data,error), method='Nelder-Mead')
                    lnL=lnL-res.fun
                l+=1

            if j==1: #spectroscopic case
                idx_rv=np.where(np.array(typ[i])==1)[0] #indexs of observables that are rv bis or fwhm, contrast. Ideally only one

                t=self.data[self.instruments[i]][self.observables[i][idx_rv[0]]]['t']
                _,CCF,_,_,_,_=nbspectra.generate_rotating_photosphere_fast_rv(t,mode_rv_params[r][0],mode_rv_params[r][1],mode_rv_params[r][2],mode_rv_params[r][3],mode_rv_params[r][4],mode_rv_params[r][5],mode_rv_params[r][6],mode_rv_params[r][7],mode_rv_params[r][8],mode_rv_params[r][9],mode_rv_params[r][10],mode_rv_params[r][11],mode_rv_params[r][12],mode_rv_params[r][13],mode_rv_params[r][14],self.n_grid_rings,self.use_phoenix_limb_darkening,self.limb_darkening_law,self.limb_darkening_q1,self.limb_darkening_q2,self.spot_map,self.reference_time,self.rotation_period,self.differential_rotation,self.spots_evo_law,self.facular_area_ratio,self.inclination,self.vsini,self.convective_shift,self.temperature_photosphere,self.temperature_facula,self.simulate_planet,self.planet_esinw,self.planet_ecosw,self.planet_transit_t0,self.planet_period,self.planet_radius,self.planet_impact_param,self.planet_semi_major_axis,self.planet_spin_orbit_angle)
                if self.simulate_planet:
                    rvkepler = spectra.keplerian_orbit(t,[self.planet_period,self.planet_semi_amplitude,self.planet_esinw,self.planet_ecosw,self.planet_transit_t0])
                else:
                    rvkepler = 0.0

                ccf_params=spectra.compute_ccf_params(self,mode_rv_params[r][4],CCF,plot_test=False)
                self.results['rv']=ccf_params[0] + rvkepler - mode_rv_params[r][-4] #subtract rv of immaculate photosphere
                self.results['contrast']=ccf_params[1]/mode_rv_params[r][-3] #offsets
                self.results['fwhm']=ccf_params[2] #- mode_rv_params[r][-2]
                self.results['bis']=ccf_params[3] #- mode_rv_params[r][-1] 
                
                for k in idx_rv:
                    data=self.data[self.instruments[i]][self.observables[i][k]]['y']
                    error=self.data[self.instruments[i]][self.observables[i][k]]['yerr']
                    model=self.results[self.observables[i][k]]
                    
                    if self.data[self.instruments[i]][self.observables[i][k]]['offset_type'] =='multiplicative': #multiplicative offset
                        
                        if (self.data[self.instruments[i]][self.observables[i][k]]['fix_jitter'] and self.data[self.instruments[i]][self.observables[i][k]]['fix_offset']):
                            lnL=lnL-0.5*np.sum(((data-model)/(error))**2.0+np.log(2.0*np.pi)+np.log(error**2))
                        elif self.data[self.instruments[i]][self.observables[i][k]]['fix_offset']:
                            res=optimize.minimize(nbspectra.fit_only_jitter,2*np.mean(error), args=(model,data,error), method='Nelder-Mead')
                            lnL=lnL-res.fun                        
                        elif self.data[self.instruments[i]][self.observables[i][k]]['fix_jitter']:
                            res=optimize.minimize(nbspectra.fit_only_multiplicative_offset,np.mean(data)/(np.mean(model)+0.0001), args=(model,data,error), method='Nelder-Mead')
                            lnL=lnL-res.fun
                        else:
                            res=optimize.minimize(nbspectra.fit_multiplicative_offset_jitter,[np.mean(data)/(np.mean(model)+0.0001),2*np.mean(error)], args=(model,data,error), method='Nelder-Mead')
                            lnL=lnL-res.fun

                    else: #linear offset
                        
                        if (self.data[self.instruments[i]][self.observables[i][k]]['fix_jitter'] and self.data[self.instruments[i]][self.observables[i][k]]['fix_offset']):
                            lnL=lnL-0.5*np.sum(((data-model)/(error))**2.0+np.log(2.0*np.pi)+np.log(error**2))
                        elif self.data[self.instruments[i]][self.observables[i][k]]['fix_offset']:
                            res=optimize.minimize(nbspectra.fit_only_jitter,2*np.mean(error), args=(model,data,error), method='Nelder-Mead')
                            lnL=lnL-res.fun                        
                        elif self.data[self.instruments[i]][self.observables[i][k]]['fix_jitter']:
                            res=optimize.minimize(nbspectra.fit_only_linear_offset,np.mean(data)-np.mean(model), args=(model,data,error), method='Nelder-Mead')
                            lnL=lnL-res.fun
                        else:
                            res=optimize.minimize(nbspectra.fit_linear_offset_jitter,[np.mean(data)-np.mean(model),2*np.mean(error)], args=(model,data,error), method='Nelder-Mead')
                            lnL=lnL-res.fun

                        r+=1


            if j==2: #chromatic-spectroscopic case
                idx_crx=np.where(np.array(typ[i])==2)[0] #indexs of observables that are crx. Ideally only one
                self.wavelength_lower_limit=self.data[self.instruments[i]]['wvmin']
                self.wavelength_upper_limit=self.data[self.instruments[i]]['wvmax']
                self.compute_forward(observables=['crx'],t=self.data[self.instruments[i]][self.observables[i][idx_crx[0]]]['t'],inversion=True)

                for k in idx_crx:
                    data=self.data[self.instruments[i]][self.observables[i][k]]['y']
                    error=self.data[self.instruments[i]][self.observables[i][k]]['yerr']
                    model=self.results[self.observables[i][k]]
                    
                    if self.data[self.instruments[i]][self.observables[i][k]]['offset_type'] =='multiplicative': #multiplicative offset
                        
                        if (self.data[self.instruments[i]][self.observables[i][k]]['fix_jitter'] and self.data[self.instruments[i]][self.observables[i][k]]['fix_offset']):
                            lnL=lnL-0.5*np.sum(((data-model)/(error))**2.0+np.log(2.0*np.pi)+np.log(error**2))
                        elif self.data[self.instruments[i]][self.observables[i][k]]['fix_offset']:
                            res=optimize.minimize(nbspectra.fit_only_jitter,2*np.mean(error), args=(model,data,error), method='Nelder-Mead')
                            lnL=lnL-res.fun                        
                        elif self.data[self.instruments[i]][self.observables[i][k]]['fix_jitter']:
                            res=optimize.minimize(nbspectra.fit_only_multiplicative_offset,np.mean(data)/(np.mean(model)+0.0001), args=(model,data,error), method='Nelder-Mead')
                            lnL=lnL-res.fun
                        else:
                            res=optimize.minimize(nbspectra.fit_multiplicative_offset_jitter,[np.mean(data)/(np.mean(model)+0.0001),2*np.mean(error)], args=(model,data,error), method='Nelder-Mead')
                            lnL=lnL-res.fun

                    else: #linear offset
                        
                        if (self.data[self.instruments[i]][self.observables[i][k]]['fix_jitter'] and self.data[self.instruments[i]][self.observables[i][k]]['fix_offset']):
                            lnL=lnL-0.5*np.sum(((data-model)/(error))**2.0+np.log(2.0*np.pi)+np.log(error**2))
                        elif self.data[self.instruments[i]][self.observables[i][k]]['fix_offset']:
                            res=optimize.minimize(nbspectra.fit_only_jitter,2*np.mean(error), args=(model,data,error), method='Nelder-Mead')
                            lnL=lnL-res.fun                        
                        elif self.data[self.instruments[i]][self.observables[i][k]]['fix_jitter']:
                            res=optimize.minimize(nbspectra.fit_only_linear_offset,np.mean(data)-np.mean(model), args=(model,data,error), method='Nelder-Mead')
                            lnL=lnL-res.fun
                        else:
                            res=optimize.minimize(nbspectra.fit_linear_offset_jitter,[np.mean(data)-np.mean(model),2*np.mean(error)], args=(model,data,error), method='Nelder-Mead')
                            lnL=lnL-res.fun


    return lnL

def inversion_parallel(self,typ,i):
    best_map,best_lnL=inversion_SA(self,typ,only_inversion=True)
    print('Inversion',i,'complete with a lnL of',best_lnL)
    return best_map, best_lnL



def inversion_SA(self,typ,only_inversion=False):
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
                wv_rv, flnp_rv =spectra.interpolate_Phoenix(self,self.temperature_photosphere,self.logg) #returns norm spectra and no normalized, interpolated at T and logg
                _, flns_rv =spectra.interpolate_Phoenix(self,self.temperature_spot,self.logg)
                if self.facular_area_ratio>0:
                    _, flnf_rv =spectra.interpolate_Phoenix(self,self.temperature_facula,self.logg)
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
                    ccf_sp = nbspectra.cross_correlation_mask(rv,np.asarray(wv_rv,dtype='float64'),np.asarray(flns_rv,dtype='float64'),np.asarray(self.wvm,dtype='float64'),np.asarray(self.fm,dtype='float64'))
                    if self.facular_area_ratio>0:
                        ccf_fc = nbspectra.cross_correlation_mask(rv,np.asarray(wv_rv,dtype='float64'),np.asarray(flnf_rv,dtype='float64'),np.asarray(self.wvm,dtype='float64'),np.asarray(self.fm,dtype='float64'))
                    else:
                        ccf_fc=ccf_ph*0.0


                #ADD BISECTORS TO RV
                fun_bis_ph = spectra.bisector_fit(self,rv,ccf_ph,plot_test=False,kind_interp=self.kind_interp)
                rv_ph = rv - fun_bis_ph(ccf_ph) #subtract the bisector from the CCF.
                fun_bis_sp = spectra.bisector_fit(self,rv,ccf_sp,plot_test=False,kind_interp=self.kind_interp)
                rv_sp = rv - fun_bis_sp(ccf_sp)
                rv_fc = rv_ph
                if self.facular_area_ratio>0:            
                    fun_bis_fc = spectra.bisector_fit(self,rv,ccf_fc,plot_test=False,kind_interp=self.kind_interp)        
                    rv_fc = rv - fun_bis_fc(ccf_fc)

                #IMMACULATE PHOTOSPHERE
                ccf_ph_g, flxph= spectra.compute_immaculate_photosphere_rv(self,Ngrid_in_ring,acd,amu,pare,flpk_rv,rv_ph,rv,ccf_ph,rvel) #return ccf of each grid, and also the integrated ccf
                ccf_ph_tot = np.sum(ccf_ph_g,axis=0) #CCF of immaculate rotating pphotosphere                

                RV0, C0, F0, B0=spectra.compute_ccf_params(self,rv,[ccf_ph_tot],plot_test=False)

                mode_rv_params.append([Ngrid_in_ring,acd,amu,pare,rv,rv_ph,rv_sp,rv_fc,ccf_ph_tot,ccf_ph,ccf_sp,ccf_fc,flxph,flpk_rv,flsk_rv,RV0,C0,F0,B0])


    #All the parameters that can be optimized, their bounds, priors and ooptimization or not
    N_spots = len(self.spot_map)

    fixed_spot_it=[self.spot_map[i][0] for i in range(N_spots)]
    fixed_spot_lt=[self.spot_map[i][1] for i in range(N_spots)]
    fixed_spot_lat=[self.spot_map[i][2] for i in range(N_spots)] #uniform in sini
    fixed_spot_lon=[self.spot_map[i][3] for i in range(N_spots)]
    fixed_spot_c1=[self.spot_map[i][4] for i in range(N_spots)]
    fixed_spot_c2=[self.spot_map[i][5] for i in range(N_spots)]
    fixed_spot_c3=[self.spot_map[i][6] for i in range(N_spots)]

    ASvparam=np.array([*fixed_spot_it,*fixed_spot_lt,*fixed_spot_lat,*fixed_spot_lon,*fixed_spot_c1,*fixed_spot_c2,*fixed_spot_c3])

    f_spot_it=[self.spot_map[i][7] for i in range(N_spots)]
    f_spot_lt=[self.spot_map[i][8]for i in range(N_spots)]
    f_spot_lat=[self.spot_map[i][9]for i in range(N_spots)]
    f_spot_lon=[self.spot_map[i][10] for i in range(N_spots)]
    f_spot_c1=[self.spot_map[i][11] for i in range(N_spots)]
    f_spot_c2=[self.spot_map[i][12] for i in range(N_spots)]
    f_spot_c3=[self.spot_map[i][13] for i in range(N_spots)]
    ASfit=np.array([*f_spot_it,*f_spot_lt,*f_spot_lat,*f_spot_lon,*f_spot_c1,*f_spot_c2,*f_spot_c3])      

    bound_spot_it=np.array([[self.prior_spot_initial_time[1],self.prior_spot_initial_time[2]] for i in range(N_spots)])
    bound_spot_lt=np.array([[self.prior_spot_life_time[1],self.prior_spot_life_time[2]] for i in range(N_spots)])
    bound_spot_lat=np.array([[self.prior_spot_latitude[1],self.prior_spot_latitude[2]]for i in range(N_spots)]) #sin(colatmin) sin(colatmax)
    bound_spot_lon=np.array([[self.prior_spot_longitude[1],self.prior_spot_longitude[2]] for i in range(N_spots)])
    bound_spot_c1=np.array([[self.prior_spot_coeff_1[1],self.prior_spot_coeff_1[2]] for i in range(N_spots)])
    bound_spot_c2=np.array([[self.prior_spot_coeff_2[1],self.prior_spot_coeff_2[2]] for i in range(N_spots)])
    bound_spot_c3=np.array([[self.prior_spot_coeff_3[1],self.prior_spot_coeff_3[2]] for i in range(N_spots)])
    ASbounds=np.array([*bound_spot_it,*bound_spot_lt,*bound_spot_lat,*bound_spot_lon,*bound_spot_c1,*bound_spot_c2,*bound_spot_c3]) 

    prior_spot_it=[spectra.generate_prior(self.prior_spot_initial_time[3],self.prior_spot_initial_time[4],self.prior_spot_initial_time[5],1) for i in range(N_spots)]
    prior_spot_lt=[spectra.generate_prior(self.prior_spot_life_time[3],self.prior_spot_life_time[4],self.prior_spot_life_time[5],1) for i in range(N_spots)]
    prior_spot_lat=[spectra.generate_prior(self.prior_spot_latitude[3],self.prior_spot_latitude[4],self.prior_spot_latitude[5],1) for i in range(N_spots)]
    prior_spot_lon=[spectra.generate_prior(self.prior_spot_longitude[3],self.prior_spot_longitude[4],self.prior_spot_longitude[5],1) for i in range(N_spots)]
    prior_spot_c1=[spectra.generate_prior(self.prior_spot_coeff_1[3],self.prior_spot_coeff_1[4],self.prior_spot_coeff_1[5],1) for i in range(N_spots)]
    prior_spot_c2=[spectra.generate_prior(self.prior_spot_coeff_2[3],self.prior_spot_coeff_2[4],self.prior_spot_coeff_2[5],1) for i in range(N_spots)]
    prior_spot_c3=[spectra.generate_prior(self.prior_spot_coeff_3[3],self.prior_spot_coeff_3[4],self.prior_spot_coeff_3[5],1) for i in range(N_spots)]
    ASpriors=np.array([*prior_spot_it,*prior_spot_lt,*prior_spot_lat,*prior_spot_lon,*prior_spot_c1,*prior_spot_c2,*prior_spot_c3])

    logprior_spot_it=np.array([[self.prior_spot_initial_time[3],self.prior_spot_initial_time[4],self.prior_spot_initial_time[5]] for i in range(N_spots)])
    logprior_spot_lt=np.array([[self.prior_spot_life_time[3],self.prior_spot_life_time[4],self.prior_spot_life_time[5]] for i in range(N_spots)])
    logprior_spot_lat=np.array([[self.prior_spot_latitude[3],self.prior_spot_latitude[4],self.prior_spot_latitude[5]]for i in range(N_spots)])
    logprior_spot_lon=np.array([[self.prior_spot_longitude[3],self.prior_spot_longitude[4],self.prior_spot_longitude[5]] for i in range(N_spots)])
    logprior_spot_c1=np.array([[self.prior_spot_coeff_1[3],self.prior_spot_coeff_1[4],self.prior_spot_coeff_1[5]] for i in range(N_spots)])
    logprior_spot_c2=np.array([[self.prior_spot_coeff_2[3],self.prior_spot_coeff_2[4],self.prior_spot_coeff_2[5]] for i in range(N_spots)])
    logprior_spot_c3=np.array([[self.prior_spot_coeff_3[3],self.prior_spot_coeff_3[4],self.prior_spot_coeff_3[5]] for i in range(N_spots)])
    ASlogpriors=np.array([*logprior_spot_it,*logprior_spot_lt,*logprior_spot_lat,*logprior_spot_lon,*logprior_spot_c1,*logprior_spot_c2,*logprior_spot_c3]) 


    ASvparamfit=np.array([])
    ASboundfit=[]
    ASpriors_fit=[]
    ASlogpriors_fit=[]


    for i in range(len(ASfit)):
      if ASfit[i]==1:
        ASvparamfit=np.append(ASvparamfit,ASvparam[i])
        ASpriors_fit.append(ASpriors[i])
        ASlogpriors_fit.append(ASlogpriors[i])
        ASboundfit.append(ASbounds[i])
    ASboundfit=np.asarray(ASboundfit)
    ASpriors_fit=np.asarray(ASpriors_fit)
    ASlogpriors_fit=np.asarray(ASlogpriors_fit)

    ################ AS OPTIONS ##########################
    ndim = len(ASvparamfit)
    T0=self.N_data/3 #Initial temperature is N points/3. Is equivalent to the DeltalnL equivalent to a 1sigma deviation
    Tf=1e-20 #Final temperature.
    Niters=self.N_iters_SA
    alpha=(Tf/T0)**(1/Niters)
    ######################################################

    Xacc=ASpriors_fit.T[0] #initial position of the particles (Xi)
    Xprop=ASpriors_fit.T[0] 
    Xbest=ASpriors_fit.T[0] 


    #Compute the lnL of the first iteration
    p_aux=np.zeros(len(ASvparam))#auxiliar variable to store optimized and non-optimized variables
    ii=0
    for j in range(len(ASfit)):
      if ASfit[j]==0:
        p_aux[j]=ASvparam[j] #non-optimized variable
      elif ASfit[j]==1:
        p_aux[j]=Xacc[ii] #optimized variable
        ii=ii+1

    for j in range(N_spots):
        self.spot_map[j,0]=p_aux[j+0*N_spots]   #set the variables of the new spot map
        self.spot_map[j,1]=p_aux[j+1*N_spots] #lifetime
        self.spot_map[j,2]=p_aux[j+2*N_spots] #colat, in deg (convert sini to deg)
        self.spot_map[j,3]=p_aux[j+3*N_spots]%360 #longitude between 0 and 360
        self.spot_map[j,4]=p_aux[j+4*N_spots] #C1
        self.spot_map[j,5]=p_aux[j+5*N_spots] #C2
        self.spot_map[j,6]=p_aux[j+6*N_spots] #C3
    #set the spot map
    lnLacc=lnposteriorSA(self,p_aux,ASbounds,ASlogpriors,typ,mode_lc_params,mode_rv_params) #log-likelihood of initial positions
    lnLbest=lnLacc #set the best lnL

       
    acceptance=0 #compute acceptance fraction
    for k in range(Niters): #iterations on each temperature
        T=T0*alpha**(k) #reduce temperature following cooling sequence
        #Generate new state. Must beinside bounds ans non overlapping.
        alarm=0
        while True:
            v=np.random.uniform(0,1,ndim) #N random values 0 to 1
            q=np.sign(v-0.5)*(T/(T0))*((1+T0/T)**(np.abs(2*v-1))-1) #[-1,1], denser around 0 as T decreases

            Xprop=Xacc+q*0.5*(ASboundfit[:,1]-ASboundfit[:,0]) #new state


            #check if the parameters are outside bounds. If they are outside, use rebound rule.
            if np.any((ASboundfit[:,1]<Xprop)+(Xprop<ASboundfit[:,0])): 
                Xprop[ASboundfit[:,1]<Xprop]=2*ASboundfit[:,1][ASboundfit[:,1]<Xprop]-Xprop[ASboundfit[:,1]<Xprop]
                Xprop[Xprop<ASboundfit[:,0]]=2*ASboundfit[:,0][Xprop<ASboundfit[:,0]]-Xprop[Xprop<ASboundfit[:,0]]

            if alarm==100:
                # print('Warning, spots are overlapping.')
                #If alarm, set X to limit.
                break

            p_aux=np.zeros(len(ASvparam))#auxiliar variable to store optimized and non-optimized variables
            ii=0
            for j in range(len(ASfit)):
              if ASfit[j]==0:
                p_aux[j]=ASvparam[j] #non-optimized variable
              elif ASfit[j]==1:
                p_aux[j]=Xprop[ii] #optimized variable
                ii=ii+1


            for j in range(N_spots):
                self.spot_map[j,0]=p_aux[j+0*N_spots]   #set the variables of the new spot map
                self.spot_map[j,1]=p_aux[j+1*N_spots]
                self.spot_map[j,2]=p_aux[j+2*N_spots]
                self.spot_map[j,3]=p_aux[j+3*N_spots]%360
                self.spot_map[j,4]=p_aux[j+4*N_spots]
                self.spot_map[j,5]=p_aux[j+5*N_spots]
                self.spot_map[j,6]=p_aux[j+6*N_spots]
            #set the spot map

            if nbspectra.check_spot_overlap(self.spot_map,self.facular_area_ratio):
                alarm+=1
                continue

            break
        lnLprop=lnposteriorSA(self,p_aux,ASbounds,ASlogpriors,typ,mode_lc_params,mode_rv_params) #log-likelihood of new position
        DeltalnL=lnLprop-lnLacc
        if np.isnan(DeltalnL): #To avoid getting stuck at -inf
            DeltalnL=0.1

        if DeltalnL>0: #if the model is better, accept it
            Xacc=Xprop
            lnLacc=lnLprop
            acceptance+=1
            
            if lnLacc>lnLbest: #if the model is the best, save it
                lnLbest=lnLacc
                Xbest=Xacc
        else: #if the model is worse, accept it with probability exp(DeltalnL/T)
            if np.random.uniform(0)<np.exp(DeltalnL/T):
                Xacc=Xprop
                lnLacc=lnLprop
                acceptance+=1

        if k%1==0:
            if only_inversion:
                sys.stdout.write("\r Step={}/{}; Acc_frac={:.4f}; lnL={:.3f}".format(k,Niters,(acceptance)/(k+1),lnLbest))


        #Write gbest to spotmap
        best_map=np.zeros([N_spots,7])
        p_aux=np.zeros(len(ASfit))#auxiliar variable to store optimized and non-optimized variables
        ii=0
        for j in range(len(ASfit)):
          if ASfit[j]==0:
            p_aux[j]=ASvparam[j] #non-optimized variable
          elif ASfit[j]==1:
            p_aux[j]=Xbest[ii] #optimized variable
            ii=ii+1
        for j in range(N_spots):
            best_map[j,0]=p_aux[j+0*N_spots]   #set the variables of the new spot map
            best_map[j,1]=p_aux[j+1*N_spots]
            best_map[j,2]=p_aux[j+2*N_spots]
            best_map[j,3]=p_aux[j+3*N_spots]%360
            best_map[j,4]=p_aux[j+4*N_spots]
            best_map[j,5]=p_aux[j+5*N_spots]
            best_map[j,6]=p_aux[j+6*N_spots]
  
    return best_map,lnLbest




def generate_prior(flag,p1,p2,nw): #generate initial sample from priors
    if flag==0:
        prior=np.random.uniform(p1,p2,nw)

    if flag==1:
        prior=np.random.normal(p1,p2,nw)

    if flag==2:
        prior=np.exp(np.random.normal(p1,p2,nw))

    return prior



