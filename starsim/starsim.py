import tqdm
import numpy as np
import emcee
import gc
from pathlib import Path
from multiprocessing import Pool
import sys
import os
import corner
from configparser import ConfigParser
import matplotlib.pyplot as plt
import matplotlib as mpl
import math as m
import collections
from astropy.io import fits
import json
from scipy.interpolate import interp1d
from scipy import optimize
from . import spectra
from . import nbspectra
from . import SA
from numba import jit
import starsim
#initialize numba
nbspectra.dummy()

class StarSim(object): 
    """
    Main Starsim class. Reads the configuration file and store the options into variables.
    """
    def __init__(self,conf_file_path='starsim.conf'):
            self.path = Path(__file__).parent #path of working directory
            self.conf_file_path = self.path / conf_file_path
            self.conf_file = self.__conf_init() 
            #files
            self.filter_name =  str(self.conf_file.get('files','filter_name'))
            self.orders_CRX_filename= str(self.conf_file.get('files','orders_CRX_filename'))
            #general
            self.simulation_mode = str(self.conf_file.get('general','simulation_mode'))
            self.wavelength_lower_limit = float(self.conf_file.get('general','wavelength_lower_limit'))
            self.wavelength_upper_limit = float(self.conf_file.get('general','wavelength_upper_limit'))
            self.n_grid_rings = int(self.conf_file.get('general','n_grid_rings'))
            #star
            self.radius = float(self.conf_file.get('star','radius')) #Radius of the star in solar radii
            self.mass = float(self.conf_file.get('star','mass')) #Mass of the star in solar radii
            self.rotation_period = float(self.conf_file.get('star','rotation_period')) #Rotation period in days
            self.inclination = np.deg2rad(90-float(self.conf_file.get('star','inclination'))) #axis inclinations in rad (inc=0 has the axis pointing up). The input was in deg defined as usual.
            self.temperature_photosphere = float(self.conf_file.get('star','temperature_photosphere'))
            self.spot_T_contrast = float(self.conf_file.get('star','spot_T_contrast'))
            self.facula_T_contrast = float(self.conf_file.get('star','facula_T_contrast'))
            # self.temperature_spot = self.temperature_photosphere - float(self.conf_file.get('star','spot_T_contrast'))
            # self.temperature_facula = self.temperature_photosphere + float(self.conf_file.get('star','facula_T_contrast'))
            self.convective_shift = float(self.conf_file.get('star','convective_shift'))#CB in m/s
            self.logg = float(self.conf_file.get('star','logg'))
            self.facular_area_ratio = float(self.conf_file.get('star','facular_area_ratio'))
            self.differential_rotation = float(self.conf_file.get('star','differential_rotation'))
            #rv
            self.ccf_template = str(self.conf_file.get('rv','ccf_template'))
            self.ccf_mask = str(self.conf_file.get('rv','ccf_mask'))
            self.ccf_weight_lines = int(self.conf_file.get('rv','ccf_weight_lines'))
            self.path_weight_lines = str(self.conf_file.get('rv','path_weight_lines'))
            self.ccf_rv_range= float(self.conf_file.get('rv','ccf_rv_range'))*1000 #in m/s
            self.ccf_rv_step= float(self.conf_file.get('rv','ccf_rv_step'))*1000 #in m/s
            self.kind_interp = 'cubic'#str(self.conf_file.get('rv','ccf_interpolation_spectra'))
            #limb-darkening
            self.use_phoenix_limb_darkening = int(self.conf_file.get('LD','use_phoenix_limb_darkening'))
            self.limb_darkening_law = str(self.conf_file.get('LD','limb_darkening_law'))
            self.limb_darkening_q1= float(self.conf_file.get('LD','limb_darkening_q1'))
            self.limb_darkening_q2= float(self.conf_file.get('LD','limb_darkening_q2'))
            #spots
            self.spots_evo_law = str(self.conf_file.get('spots','spots_evo_law'))
            self.plot_grid_map = int(self.conf_file.get('spots','plot_grid_map'))
            self.reference_time = float(self.conf_file.get('spots','reference_time'))
            #planet
            self.planet_period = float(self.conf_file.get('planet','planet_period')) #in days
            self.planet_transit_t0 = float(self.conf_file.get('planet','planet_transit_t0')) #in days
            self.planet_radius = float(self.conf_file.get('planet','planet_radius')) #in R* units
            self.planet_impact_param = float(self.conf_file.get('planet','planet_impact_param')) #from 0 to 1
            self.planet_spin_orbit_angle = float(self.conf_file.get('planet','planet_spin_orbit_angle'))*np.pi/180 #in deg
            self.simulate_planet=int(self.conf_file.get('planet','simulate_planet'))
            self.planet_semi_amplitude = float(self.conf_file.get('planet','planet_semi_amplitude')) #in m/s
            self.planet_esinw = float(self.conf_file.get('planet','planet_esinw')) 
            self.planet_ecosw = float(self.conf_file.get('planet','planet_ecosw')) 
            # self.inc_pl = np.arccos((self.planet_impact_param/self.planet_semi_major_axis)*(1+self.planet_esinw)/(1-np.sqrt(self.planet_esinw**2+self.planet_ecosw**2))) #inclination of the planet orbit

            #optimization
            self.prior_spot_initial_time = json.loads(self.conf_file.get('optimization','prior_spot_initial_time'))
            self.prior_spot_life_time = json.loads(self.conf_file.get('optimization','prior_spot_life_time'))
            self.prior_spot_latitude = json.loads(self.conf_file.get('optimization','prior_spot_colatitude'))
            self.prior_spot_longitude = json.loads(self.conf_file.get('optimization','prior_spot_longitude'))
            self.prior_spot_coeff_1 = json.loads(self.conf_file.get('optimization','prior_spot_coeff_1'))
            self.prior_spot_coeff_2 = json.loads(self.conf_file.get('optimization','prior_spot_coeff_2'))
            self.prior_spot_coeff_3 = json.loads(self.conf_file.get('optimization','prior_spot_coeff_3'))
            self.prior_t_eff_ph = json.loads(self.conf_file.get('optimization','prior_t_eff_ph'))
            self.prior_spot_T_contrast = json.loads(self.conf_file.get('optimization','prior_spot_T_contrast'))
            self.prior_facula_T_contrast = json.loads(self.conf_file.get('optimization','prior_facula_T_contrast'))
            self.prior_q_ratio = json.loads(self.conf_file.get('optimization','prior_q_ratio'))
            self.prior_convective_blueshift = json.loads(self.conf_file.get('optimization','prior_convective_blueshift'))
            self.prior_p_rot = json.loads(self.conf_file.get('optimization','prior_p_rot'))
            self.prior_inclination = json.loads(self.conf_file.get('optimization','prior_inclination'))
            self.prior_Rstar = json.loads(self.conf_file.get('optimization','prior_stellar_radius'))
            self.prior_LD1 = json.loads(self.conf_file.get('optimization','prior_limb_darkening_q1'))
            self.prior_LD2 = json.loads(self.conf_file.get('optimization','prior_limb_darkening_q2'))
            self.prior_Pp = json.loads(self.conf_file.get('optimization','prior_period_planet'))
            self.prior_T0p = json.loads(self.conf_file.get('optimization','prior_time_transit_planet'))
            self.prior_Kp = json.loads(self.conf_file.get('optimization','prior_semi_amplitude_planet'))
            self.prior_esinwp = json.loads(self.conf_file.get('optimization','prior_esinw_planet'))
            self.prior_ecoswp = json.loads(self.conf_file.get('optimization','prior_ecosw_planet'))
            self.prior_Rp = json.loads(self.conf_file.get('optimization','prior_radius_planet'))
            self.prior_bp = json.loads(self.conf_file.get('optimization','prior_impact_parameter_planet'))
            self.prior_alp = json.loads(self.conf_file.get('optimization','prior_spin_orbit_planet'))           

            self.nwalkers = int(self.conf_file.get('optimization','N_walkers'))
            self.steps = int(self.conf_file.get('optimization','N_steps'))
            self.planet_impact_paramurns = int(self.conf_file.get('optimization','N_burns'))
            self.N_cpus = int(self.conf_file.get('optimization','N_cpus'))
            self.N_iters_SA = int(self.conf_file.get('optimization','N_iters_SA'))


            #FUNCTIONS USED TO ADD BISECTORS TO THE PHOTOSPHERE AT DIFF ANGLES
            self.fun_coeff_bisectors_amu = spectra.cifist_coeff_interpolate
            self.fun_coeff_bisector_spots = spectra.dumusque_coeffs
            self.fun_coeff_bisector_faculae = spectra.dumusque_coeffs


            #initialize other variables to store results
            self.data = collections.defaultdict(dict) #dictionary to store input data
            self.instruments = []
            self.observables=[]
            self.results = {} #initialize results attribute. It will be a dictionary containing the results of the method forward (maybe also inverse)
            self.name_params = {'rv': 'RV\n[m/s]', 'contrast': 'CCF$_{{cont}}$', 'fwhm':'CCF$_{{FWHM}}$\n[m/s]', 'bis':'CCF$_{{BIS}}$\n[m/s]', 
            'lc':'Norm. flux', 'ff_sp': 'ff$_{{spot}}$ \n[%]','ff_ph': 'ff$_{{phot}}$ \n[%]','ff_fc': 'ff$_{{fac}}$ \n[%]','ff_pl': 'ff$_{{pl}}$ \n[%]',
            'crx':'CRX \n[m/s/Np]','ccx':'C$_{{Cont}}$X \n[1/Np]','cfx':'C$_{{FWHM}}$X \n[m/s/Np]','cbx':'C$_{{BIS}}$X \n[m/s/Np]'}
            self.rvo = None #initialize
            self.conto = None #initialize
            self.fwhmo = None #initialize
            self.planet_impact_paramiso = None #initialize

            #read and check spotmap
            pathspots = self.path / 'spotmap.dat' #path relatve to working directory 
            self.spot_map=np.loadtxt(pathspots)

            if self.spot_map.ndim == 1:
                self.spot_map = np.array([self.spot_map]) #to avoid future errors
            elif self.spot_map.ndim == 0:
                sys.exit('The spot map file spotmap.dat is empty')

            #select mode
            if self.simulation_mode == 'grid':
                pass
            elif self.simulation_mode == 'fast':
                pass
            else: 
                sys.exit('simulation_mode in configuration file is not valid. Valid modes are "fast" or "grid".')

            #mode to select the template used to compute the ccf. Model are Phoenix models, mask are custom masks. 
            if self.ccf_template == 'model': #use phoenix models
                pass
            elif self.ccf_template == 'mask': #use maske
                pathmask = self.path / 'masks' / self.ccf_mask
                try:
                    d = np.loadtxt(pathmask,unpack=True)
                    if len(d) == 2:
                        self.wvm = d[0]
                        self.fm = d[1]
                    elif len(d) == 3:
                        self.wvm = spectra.air2vacuum((d[0]+d[1])/2) #HARPS mask ar in air, not vacuum
                        self.fm = d[2]
                    else:
                        sys.exit('Mask format not valid. Must have two (wv and weight) or three columns (wv1 wv2 weight).')

                except:
                    sys.exit('Mask file not found. Save it inside the masks folder.')


                if self.ccf_weight_lines:
                    pathweight = self.path / 'masks' / self.path_weight_lines
                    try:
                        order, order_weight, order_wvi, order_wvf = np.loadtxt(pathweight,unpack=True)
                        self.wvm, self.fm = nbspectra.weight_mask(order_wvi,order_wvf,order_weight,self.wvm,self.fm)
                    except:
                        sys.exit('File containing the weights of each order cold not be found/read. Make sure the file is inside the masks folder, and that it have 4 columns: order num, weight, initial wavelength, final wavelength')

                #Finally, set the wavlength range around the available lines
                self.wavelength_upper_limit=self.wvm.max()+1
                self.wavelength_lower_limit=self.wvm.min()-1

            else:
                sys.exit('ccf_template in configuration file is not valid. Valid modes are "model" or "mask".')

    @property
    def temperature_spot(self):
        return self.temperature_photosphere - self.spot_T_contrast

    @property
    def temperature_facula(self):
        return self.temperature_photosphere + self.facula_T_contrast

    @property
    def vsini(self):
        return 1000*2*np.pi*(self.radius*696342)*np.cos(self.inclination)/(self.rotation_period*86400) #vsini in m/s

    @property
    def planet_semi_major_axis(self):
        return 4.2097*self.planet_period**(2/3)*self.mass**(1/3) #semi major axis in stellar radius units 


    def __conf_init(self):
        """creates an instance of class ConfigParser, and read the .conf file. Returns the object created
        ConfigParser, and read the .conf file. Returns the object created
        """
        conf_file_Object = ConfigParser(inline_comment_prefixes='#')
        if not conf_file_Object.read([self.conf_file_path]):
            print("The configuration file in" + str(self.conf_file_path) + " could not be read, please check that the format and/or path is are correct")
            sys.exit()
        else:
            return conf_file_Object


    def set_stellar_parameters(self,p):
        """Set the stellar parameters that have been optimized.
        """
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
        self.planet_radius = p[15]
        self.planet_impact_param = p[16]
        self.planet_spin_orbit_angle = p[17]*np.pi/180 #deg2rad    


    def compute_forward(self,observables=['lc'],t=None,inversion=False):

        if inversion==False:
            self.wavelength_lower_limit = float(self.conf_file.get('general','wavelength_lower_limit')) #Repeat this just in case CRX has modified the values
            self.wavelength_upper_limit = float(self.conf_file.get('general','wavelength_upper_limit'))


        if t is None:
            sys.exit('Please provide a valid time in compute_forward(observables,t=time)')

        self.obs_times = t


        Ngrids, Ngrid_in_ring, centres, amu, rs, alphas, xs, ys, zs, are, pare = nbspectra.generate_grid_coordinates_nb(self.n_grid_rings)

        vec_grid = np.array([xs,ys,zs]).T #coordinates in cartesian
        theta, phi = np.arccos(zs*np.cos(-self.inclination)-xs*np.sin(-self.inclination)), np.arctan2(ys,xs*np.cos(-self.inclination)+zs*np.sin(-self.inclination))#coordinates in the star reference 


        #Main core of the method. Dependingon the observables you want, use lowres or high res spectra
        if 'lc' in observables: #use LR templates. Interpolate for temperatures and logg for different elements. Cut to desired wavelength.
   
            #Interpolate PHOENIX intensity models, only spot and photosphere
            acd, wvp_lc, flnp_lc =spectra.interpolate_Phoenix_mu_lc(self,self.temperature_photosphere,self.logg) #acd is the angles at which the model is computed. 
            acd, wvp_lc, flns_lc =spectra.interpolate_Phoenix_mu_lc(self,self.temperature_spot,self.logg)

            #Read filter and interpolate it in order to convolve it with the spectra
            f_filt = spectra.interpolate_filter(self)

             
            
            if self.simulation_mode == 'grid':
                brigh_grid_ph, flx_ph = spectra.compute_immaculate_lc(self,Ngrid_in_ring,acd,amu,pare,flnp_lc,f_filt,wvp_lc) #returns spectrum of grid in ring N, its brightness, and the total flux
                brigh_grid_sp, flx_sp = spectra.compute_immaculate_lc(self,Ngrid_in_ring,acd,amu,pare,flns_lc,f_filt,wvp_lc) #returns spectrum of grid in ring N, its brightness, and the total flux
                brigh_grid_fc, flx_fc = brigh_grid_sp, flx_sp #if there are no faculae
                if self.facular_area_ratio>0:
                    brigh_grid_fc, flx_fc = spectra.compute_immaculate_facula_lc(self,Ngrid_in_ring,acd,amu,pare,flnp_lc,f_filt,wvp_lc) #returns spectrum of grid in ring N, its brightness, and the total flux

                t,FLUX,ff_ph,ff_sp,ff_fc,ff_pl=spectra.generate_rotating_photosphere_lc(self,Ngrid_in_ring,pare,amu,brigh_grid_ph,brigh_grid_sp,brigh_grid_fc,flx_ph,vec_grid,inversion,plot_map=self.plot_grid_map)

            #FAST MODE ONLY WORKS FOR NON-OVERLAPPING SPOTS. NOT FACULAE AND NOT PLANETS YET.
            elif self.simulation_mode == 'fast': #in fast mode only immaculate photosphere is computed
                t,FLUX,ff_ph,ff_sp,ff_fc,ff_pl=nbspectra.generate_rotating_photosphere_fast_lc(self.obs_times,Ngrid_in_ring,acd,amu,pare,flnp_lc,flns_lc,f_filt(wvp_lc),self.n_grid_rings,self.use_phoenix_limb_darkening,self.limb_darkening_law,self.limb_darkening_q1,self.limb_darkening_q2,self.spot_map,self.reference_time,self.rotation_period,self.differential_rotation,self.spots_evo_law,self.facular_area_ratio,self.inclination,self.temperature_photosphere,self.temperature_facula,self.simulate_planet,self.planet_esinw,self.planet_ecosw,self.planet_transit_t0,self.planet_period,self.planet_radius,self.planet_impact_param,self.planet_semi_major_axis,self.planet_spin_orbit_angle)


            self.results['time']=t
            self.results['lc']=FLUX
            self.results['ff_ph']=ff_ph
            self.results['ff_sp']=ff_sp
            self.results['ff_pl']=ff_pl
            self.results['ff_fc']=ff_fc


        if 'rv' in observables or 'bis' in observables or 'fwhm' in observables or 'contrast' in observables: #use HR templates. Interpolate for temperatures and logg for different elements. Cut to desired wavelength.
            rvel=self.vsini*np.sin(theta)*np.sin(phi)#*np.cos(self.inclination) #radial velocities of each grid

            wv_rv, flnp_rv, flp_rv =spectra.interpolate_Phoenix(self,self.temperature_photosphere,self.logg) #returns norm spectra and no normalized, interpolated at T and logg
            wv_rv, flns_rv, fls_rv =spectra.interpolate_Phoenix(self,self.temperature_spot,self.logg)
            if self.facular_area_ratio>0:
                wv_rv, flnf_rv, flf_rv =spectra.interpolate_Phoenix(self,self.temperature_facula,self.logg)
            spec_ref = flnp_rv #reference spectrum to compute CCF. Normalized

            #Interpolate also Phoenix intensity models to the Phoenix wavelength. 
            # if self.use_phoenix_limb_darkening: 
            acd, wv_rv_LR, flpk_rv =spectra.interpolate_Phoenix_mu_lc(self,self.temperature_photosphere,self.logg) #acd is the angles at which the model is computed. 
            acd, wv_rv_LR, flsk_rv =spectra.interpolate_Phoenix_mu_lc(self,self.temperature_spot,self.logg)

            #Compute the CCF of the spectrum of each element againts the reference template (photosphere)
            rv = np.arange(-self.ccf_rv_range,self.ccf_rv_range+self.ccf_rv_step,self.ccf_rv_step)
            #CCF with phoenix model  
            if self.ccf_template == 'model':
                ccf_ph = nbspectra.cross_correlation_nb(rv,wv_rv,flnp_rv,wv_rv,spec_ref)
                ccf_sp = nbspectra.cross_correlation_nb(rv,wv_rv,flns_rv,wv_rv,spec_ref)
                if self.facular_area_ratio>0:            
                    ccf_fc = nbspectra.cross_correlation_nb(rv,wv_rv,flnf_rv,wv_rv,spec_ref)
                else:
                    ccf_fc=ccf_ph*0.0

            elif self.ccf_template == 'mask':
                if wv_rv.max()<(self.wvm.max()+1) or wv_rv.min()>(self.wvm.min()-1):
                    sys.exit('Selected wavelength must cover all the mask wavelength range, including 1A overhead covering RV shifts. Units in Angstroms.')

                ccf_ph = nbspectra.cross_correlation_mask(rv,np.asarray(wv_rv,dtype='float64'),np.asarray(flnp_rv,dtype='float64'),np.asarray(self.wvm,dtype='float64'),np.asarray(self.fm,dtype='float64'))
                ccf_sp = nbspectra.cross_correlation_mask(rv,np.asarray(wv_rv,dtype='float64'),np.asarray(flnp_rv,dtype='float64'),np.asarray(self.wvm,dtype='float64'),np.asarray(self.fm,dtype='float64'))
                if self.facular_area_ratio>0:
                    ccf_fc = nbspectra.cross_correlation_mask(rv,np.asarray(wv_rv,dtype='float64'),np.asarray(flnp_rv,dtype='float64'),np.asarray(self.wvm,dtype='float64'),np.asarray(self.fm,dtype='float64'))
                else:
                    ccf_fc=ccf_ph*0.0

            #Compute the bisector of the three reference CCF and return a cubic spline f fiting it, such that rv=f(ccf).
            fun_bis_ph = spectra.bisector_fit(self,rv,ccf_ph,plot_test=False,kind_interp=self.kind_interp)
            rv_ph = rv - fun_bis_ph(ccf_ph) #subtract the bisector from the CCF.
            fun_bis_sp = spectra.bisector_fit(self,rv,ccf_sp,plot_test=False,kind_interp=self.kind_interp)
            rv_sp = rv - fun_bis_ph(ccf_sp)
            rv_fc = rv_ph
            if self.facular_area_ratio>0:            
                fun_bis_fc = spectra.bisector_fit(self,rv,ccf_fc,plot_test=False,kind_interp=self.kind_interp)        
                rv_fc = rv - fun_bis_ph(ccf_fc)

            
            if self.simulation_mode == 'grid':

                ccf_ph_g, flxph= spectra.compute_immaculate_photosphere_rv(self,Ngrid_in_ring,acd,amu,pare,flpk_rv,rv_ph,rv,ccf_ph,rvel) #return ccf of each grid, and also the integrated ccf
                ccf_ph_tot = np.sum(ccf_ph_g,axis=0)
                ccf_sp_g = spectra.compute_immaculate_spot_rv(self,Ngrid_in_ring,acd,amu,pare,flsk_rv,rv_sp,rv,ccf_sp,flxph,rvel)
                ccf_fc_g = ccf_ph_g #to avoid errors, not used
                if self.facular_area_ratio>0:
                    # print('Computing facula. Limb brightening is hard coded. Luke Johnson 2021 maybe is better millor.')
                    ccf_fc_g = spectra.compute_immaculate_facula_rv(self,Ngrid_in_ring,acd,amu,pare,flpk_rv,rv_fc,rv,ccf_fc,flxph,rvel)

                RV0, C0, F0, B0=spectra.compute_ccf_params(self,rv,[ccf_ph_tot],plot_test=False) #compute 0 point of immaculate photosphere

                #integrate the ccfs with doppler shifts at each time stamp
                t,CCF,ff_ph,ff_sp,ff_fc,ff_pl=spectra.generate_rotating_photosphere_rv(self,Ngrid_in_ring,pare,amu,rv,ccf_ph_tot,ccf_ph_g,ccf_sp_g,ccf_fc_g,vec_grid,inversion,plot_map=self.plot_grid_map) 


            #FAST MODE ONLY WORKS FOR NON-OVERLAPPING SPOTS. 
            elif self.simulation_mode == 'fast': #in fast mode only immaculate photosphere is computed
                ccf_ph_g, flxph= spectra.compute_immaculate_photosphere_rv(self,Ngrid_in_ring,acd,amu,pare,flpk_rv,rv_ph,rv,ccf_ph,rvel) #return ccf of each grid, and also the integrated ccf
                ccf_ph_tot = np.sum(ccf_ph_g,axis=0) #CCF of immaculate rotating pphotosphere
                RV0, C0, F0, B0=spectra.compute_ccf_params(self,rv,[ccf_ph_tot],plot_test=False) #compute 0 point of immaculate photosphere
                
                t,CCF,ff_ph,ff_sp,ff_fc,ff_pl=nbspectra.generate_rotating_photosphere_fast_rv(self.obs_times,Ngrid_in_ring,acd,amu,pare,rv,rv_ph,rv_sp,rv_fc,ccf_ph_tot,ccf_ph,ccf_sp,ccf_fc,flxph,flpk_rv,flsk_rv,self.n_grid_rings,self.use_phoenix_limb_darkening,self.limb_darkening_law,self.limb_darkening_q1,self.limb_darkening_q2,self.spot_map,self.reference_time,self.rotation_period,self.differential_rotation,self.spots_evo_law,self.facular_area_ratio,self.inclination,self.vsini,self.convective_shift,self.temperature_photosphere,self.temperature_facula,self.simulate_planet,self.planet_esinw,self.planet_ecosw,self.planet_transit_t0,self.planet_period,self.planet_radius,self.planet_impact_param,self.planet_semi_major_axis,self.planet_spin_orbit_angle)


            if self.simulate_planet:
                rvkepler = spectra.keplerian_orbit(t,[self.planet_period,self.planet_semi_amplitude,self.planet_esinw,self.planet_ecosw,self.planet_transit_t0])
            else:
                rvkepler = 0.0

            ccf_params=spectra.compute_ccf_params(self,rv,CCF,plot_test=False)
            self.results['time']=self.obs_times
            self.results['rv']=ccf_params[0] - RV0 + rvkepler #subtract rv of immaculate photosphere
            self.results['contrast']=ccf_params[1]/C0
            self.results['fwhm']=ccf_params[2]
            self.results['bis']=ccf_params[3]
            self.results['ff_ph']=ff_ph
            self.results['ff_sp']=ff_sp
            self.results['ff_pl']=ff_pl
            self.results['ff_fc']=ff_fc
            self.results['CCF']=np.vstack((rv,CCF))



        if 'crx' in observables: #use HR templates in different wavelengths to compute chromatic index. Interpolate for temperatures and logg for different elements. Cut to desired wavelength.
            rvel=self.vsini*np.sin(theta)*np.sin(phi) #radial velocities of each grid

            pathorders = self.path / 'orders_CRX' / self.orders_CRX_filename
            # print('Reading the file in',pathorders,'containing the wavelengthranges of each echelle order,to compute the CRX')
            try:
                orders, wvmins, wvmaxs = np.loadtxt(pathorders,unpack=True)
            except:
                sys.exit('Please, provide a valid file containing the order number and wavelength range, inside the folder orders_CRX')

            rvso=np.zeros([len(self.obs_times),len(orders)])
            conto=np.zeros([len(self.obs_times),len(orders)])
            fwhmo=np.zeros([len(self.obs_times),len(orders)])
            biso=np.zeros([len(self.obs_times),len(orders)])
            for i in range(len(orders)):
                # print('\nOrder: {:.0f}, wv range: {:.1f}-{:.1f} nm'.format(orders[i],wvmins[i],wvmaxs[i]))

                self.wavelength_lower_limit, self.wavelength_upper_limit = wvmins[i], wvmaxs[i]

                wv_rv, flnp_rv, flp_rv =spectra.interpolate_Phoenix(self,self.temperature_photosphere,self.logg) #returns norm spectra and no normalized, interpolated at T and logg
                wv_rv, flns_rv, fls_rv =spectra.interpolate_Phoenix(self,self.temperature_spot,self.logg)
                if self.facular_area_ratio>0:
                    wv_rv, flnf_rv, flf_rv =spectra.interpolate_Phoenix(self,self.temperature_facula,self.logg)
                spec_ref = flnp_rv #reference spectrum to compute CCF. Normalized


                acd, wv_rv_LR, flpk_rv =spectra.interpolate_Phoenix_mu_lc(self,self.temperature_photosphere,self.logg) #acd is the angles at which the model is computed. 
                acd, wv_rv_LR, flsk_rv =spectra.interpolate_Phoenix_mu_lc(self,self.temperature_spot,self.logg)


                #Compute the CCF of the spectrum of each element againts the reference template (photosphere)
                rv = np.arange(-self.ccf_rv_range,self.ccf_rv_range+self.ccf_rv_step,self.ccf_rv_step)
                ccf_ph = nbspectra.cross_correlation_nb(rv,wv_rv,flnp_rv,wv_rv,spec_ref)
                ccf_sp = nbspectra.cross_correlation_nb(rv,wv_rv,flns_rv,wv_rv,spec_ref)
                if self.facular_area_ratio>0:            
                    rv = np.arange(-self.ccf_rv_range,self.ccf_rv_range+self.ccf_rv_step,self.ccf_rv_step)
                    ccf_fc = nbspectra.cross_correlation_nb(rv,wv_rv,flnf_rv,wv_rv,spec_ref)
                else:
                    ccf_fc = ccf_ph*0.0

                #Compute the bisector of the three reference CCF and return a cubic spline f fiting it, such that rv=f(ccf).
                fun_bis_ph = spectra.bisector_fit(self,rv,ccf_ph,plot_test=False,kind_interp=self.kind_interp)
                rv_ph = rv - fun_bis_ph(ccf_ph) #subtract the bisector from the CCF.
                fun_bis_sp = spectra.bisector_fit(self,rv,ccf_sp,plot_test=False,kind_interp=self.kind_interp)
                rv_sp = rv - fun_bis_ph(ccf_sp)
                rv_fc = rv_ph

                if self.facular_area_ratio>0:            
                    fun_bis_fc = spectra.bisector_fit(self,rv,ccf_fc,plot_test=False,kind_interp=self.kind_interp)        
                    rv_fc = rv - fun_bis_ph(ccf_fc)


                if self.simulation_mode == 'grid':
                    #COMPUTE CCFS of each ring of a non-rotating IMMACULATE PHOTOSPHERE, and total flux of the immaculate star
                    # print('Computing photosphere')

                    ccf_ph_g, flxph= spectra.compute_immaculate_photosphere_rv(self,Ngrid_in_ring,acd,amu,pare,flpk_rv,rv_ph,rv,ccf_ph,rvel) #return ccf of each grid, and also the integrated ccf
                    ccf_ph_tot = np.sum(ccf_ph_g,axis=0)
                    # print('Computing spot')
                    ccf_sp_g = spectra.compute_immaculate_spot_rv(self,Ngrid_in_ring,acd,amu,pare,flsk_rv,rv_sp,rv,ccf_sp,flxph,rvel)
                    ccf_fc_g = ccf_ph_g #to avoid errors, not used
                    if self.facular_area_ratio>0:
                        # print('Computing facula. Limb brightening is hard coded. Luke Johnson 2021 maybe is better millor.')
                        ccf_fc_g = spectra.compute_immaculate_facula_rv(self,Ngrid_in_ring,acd,amu,pare,flpk_rv,rv_fc,rv,ccf_fc,flxph,rvel)
                    
                    RV0, C0, F0, B0=spectra.compute_ccf_params(self,rv,[ccf_ph_tot],plot_test=False) #compute 0 point of immaculate photosphere
                    
                    #integrate the ccfs with doppler shifts at each time stamp
                    t,CCF,ff_ph,ff_sp,ff_fc,ff_pl=spectra.generate_rotating_photosphere_rv(self,Ngrid_in_ring,pare,amu,rv,ccf_ph_tot,ccf_ph_g,ccf_sp_g,ccf_fc_g,vec_grid,inversion,plot_map=self.plot_grid_map) 


                #FAST MODE ONLY WORKS FOR NON-OVERLAPPING SPOTS. 
                elif self.simulation_mode == 'fast': #in fast mode only immaculate photosphere is computed
                    ccf_ph_g, flxph= spectra.compute_immaculate_photosphere_rv(self,Ngrid_in_ring,acd,amu,pare,flpk_rv,rv_ph,rv,ccf_ph,rvel) #return ccf of each grid, and also the integrated ccf
                    ccf_ph_tot = np.sum(ccf_ph_g,axis=0) #CCF of immaculate rotating pphotosphere
                       
                    RV0, C0, F0, B0=spectra.compute_ccf_params(self,rv,[ccf_ph_tot],plot_test=False) #compute 0 point of immaculate photosphere
                    
                    t,CCF,ff_ph,ff_sp,ff_fc,ff_pl=nbspectra.generate_rotating_photosphere_fast_rv(self.obs_times,Ngrid_in_ring,acd,amu,pare,rv,rv_ph,rv_sp,rv_fc,ccf_ph_tot,ccf_ph,ccf_sp,ccf_fc,flxph,flpk_rv,flsk_rv,self.n_grid_rings,self.use_phoenix_limb_darkening,self.limb_darkening_law,self.limb_darkening_q1,self.limb_darkening_q2,self.spot_map,self.reference_time,self.rotation_period,self.differential_rotation,self.spots_evo_law,self.facular_area_ratio,self.inclination,self.vsini,self.convective_shift,self.temperature_photosphere,self.temperature_facula,self.simulate_planet,self.planet_esinw,self.planet_ecosw,self.planet_transit_t0,self.planet_period,self.planet_radius,self.planet_impact_param,self.planet_semi_major_axis,self.planet_spin_orbit_angle)


                if self.simulate_planet:
                    rvkepler = spectra.keplerian_orbit(t,[self.planet_period,self.planet_semi_amplitude,self.planet_esinw,self.planet_ecosw,self.planet_transit_t0])
                else:
                    rvkepler = 0.0


                ccf_params=spectra.compute_ccf_params(self,rv,CCF,plot_test=False)
                
                rvso[:,i]=ccf_params[0] + rvkepler #do not subtract offsets, could bias crx
                conto[:,i]=ccf_params[1]
                fwhmo[:,i]=ccf_params[2]
                biso[:,i]=ccf_params[3]

            lambdas = (np.log(wvmaxs)+np.log(wvmins))/2 #natural log of the central wavelength
            crx=np.zeros(len(self.obs_times))
            ccx=np.zeros(len(self.obs_times))
            cfx=np.zeros(len(self.obs_times))
            cbx=np.zeros(len(self.obs_times))
            for i in range(len(self.obs_times)): #compute crx for each time
                crx[i]=np.polyfit(lambdas,rvso[i,:],deg=1)[0] #crx is the slope of the rv as a function of the central wavelength
                ccx[i]=np.polyfit(lambdas,conto[i,:],deg=1)[0]
                cfx[i]=np.polyfit(lambdas,fwhmo[i,:],deg=1)[0]
                cbx[i]=np.polyfit(lambdas,biso[i,:],deg=1)[0]

            self.results['time']=self.obs_times
            self.rvo=rvso
            self.conto=conto
            self.fwhm=fwhmo
            self.planet_impact_paramiso=biso
            self.results['ccx']=ccx
            self.results['cfx']=cfx
            self.results['cbx']=cbx
            self.results['crx']=crx
            self.results['ff_ph']=ff_ph
            self.results['ff_sp']=ff_sp
            self.results['ff_pl']=ff_pl
            self.results['ff_fc']=ff_fc

        return 





    #Optimizes ALL the parameters (including spots) using an MCMC. Use only for 1-2 spots.
    def optimize_MCMC(self):
        os.environ["OMP_NUM_THREADS"] = "1"


        print('\nUsing data from the instruments:')
        self.instruments=[]
        self.observables=[]
        typ=[]

        N_obs=0
        for ins in self.data.keys():
            print('-',ins,', with the observables:')
            self.instruments.append(ins)
            o=[]
            ty=[]
            for obs in self.data[ins].keys():
                if obs in ['lc']:
                    print('\t-',obs)
                    o.append(obs)
                    ty.append(0)
                elif obs in ['rv','fwhm','bis','contrast']:
                    print('\t-',obs)
                    o.append(obs)
                    ty.append(1)
                if obs in ['crx']:
                    print('\t-',obs)
                    o.append(obs)
                    ty.append(2)
            N_obs+=len(o)
            self.observables.append(o)
            typ.append(ty)



        N_spots = len(self.spot_map) #number of spots in spot_map

        fixed_spot_it=[self.spot_map[i][0] for i in range(N_spots)]
        fixed_spot_lt=[self.spot_map[i][1] for i in range(N_spots)]
        fixed_spot_lat=[self.spot_map[i][2] for i in range(N_spots)]
        fixed_spot_lon=[self.spot_map[i][3] for i in range(N_spots)]
        fixed_spot_c1=[self.spot_map[i][4] for i in range(N_spots)]
        fixed_spot_c2=[self.spot_map[i][5] for i in range(N_spots)]
        fixed_spot_c3=[self.spot_map[i][6] for i in range(N_spots)]
        fixed_T = self.temperature_photosphere
        fixed_sp_T = self.spot_T_contrast
        fixed_fc_T = self.facula_T_contrast
        fixed_Q = self.facular_area_ratio
        fixed_CB = self.convective_shift
        fixed_Prot = self.rotation_period
        fixed_inc = np.rad2deg(np.pi/2-self.inclination) #in deg. 0 is pole-on
        fixed_R = self.radius
        fixed_LD1 = self.limb_darkening_q1
        fixed_LD2 = self.limb_darkening_q2
        fixed_Pp = self.planet_period
        fixed_T0p = self.planet_transit_t0
        fixed_Kp = self.planet_semi_amplitude
        fixed_esinwp = self.planet_esinw
        fixed_ecoswp = self.planet_ecosw
        fixed_Rp =  self.planet_radius
        fixed_bp = self.planet_impact_param
        fixed_alp =  self.planet_spin_orbit_angle    #spin-orbit angle

        self.vparam=np.array([fixed_T,fixed_sp_T,fixed_fc_T,fixed_Q,fixed_CB,fixed_Prot,fixed_inc,fixed_R,fixed_LD1,fixed_LD2,fixed_Pp,fixed_T0p,fixed_Kp,fixed_esinwp,fixed_ecoswp,fixed_Rp,fixed_bp,fixed_alp,*fixed_spot_it,*fixed_spot_lt,*fixed_spot_lat,*fixed_spot_lon,*fixed_spot_c1,*fixed_spot_c2,*fixed_spot_c3])#

        name_spot_it=['spot_{0}_it'.format(i) for i in range(N_spots)]
        name_spot_lt=['spot_{0}_lt'.format(i) for i in range(N_spots)]
        name_spot_lat=['spot_{0}_lat'.format(i) for i in range(N_spots)]
        name_spot_lon=['spot_{0}_lon'.format(i) for i in range(N_spots)]
        name_spot_c1=['spot_{0}_c1'.format(i) for i in range(N_spots)]
        name_spot_c2=['spot_{0}_c2'.format(i) for i in range(N_spots)]
        name_spot_c3=['spot_{0}_c3'.format(i) for i in range(N_spots)]
        name_T ='T$_{{eff}}$'
        name_sp_T ='$\\Delta$ T$_{{sp}}$'
        name_fc_T ='$\\Delta$ T$_{{fc}}$'
        name_Q ='Fac-spot ratio'
        name_CB ='CS'
        name_Prot ='P$_{{rot}}$'
        name_inc ='inc'
        name_R ='R$_*$'
        name_LD1 = 'q$_1$'
        name_LD2 = 'q$_2$'
        name_Pp = 'P$_{{pl}}$'
        name_T0p = 'T$_{{0,pl}}$'
        name_Kp = 'K$_{{pl}}$'
        name_esinwp = 'esinw'
        name_ecoswp = 'ecosw'
        name_Rp =  'R$_{{pl}}$'
        name_bp = 'b'
        name_alp = '$\\lambda$'  

        lparam=np.array([name_T,name_sp_T,name_fc_T,name_Q,name_CB,name_Prot,name_inc,name_R,name_LD1,name_LD2,name_Pp,name_T0p,name_Kp,name_esinwp,name_ecoswp,name_Rp,name_bp,name_alp,*name_spot_it,*name_spot_lt,*name_spot_lat,*name_spot_lon,*name_spot_c1,*name_spot_c2,*name_spot_c3])

        f_spot_it=[self.spot_map[i][7] for i in range(N_spots)]
        f_spot_lt=[self.spot_map[i][8]for i in range(N_spots)]
        f_spot_lat=[self.spot_map[i][9]for i in range(N_spots)]
        f_spot_lon=[self.spot_map[i][10] for i in range(N_spots)]
        f_spot_c1=[self.spot_map[i][11] for i in range(N_spots)]
        f_spot_c2=[self.spot_map[i][12] for i in range(N_spots)]
        f_spot_c3=[self.spot_map[i][13] for i in range(N_spots)]
        f_T = self.prior_t_eff_ph[0]
        f_sp_T = self.prior_spot_T_contrast[0] 
        f_fc_T = self.prior_facula_T_contrast[0] 
        f_Q = self.prior_q_ratio[0]   
        f_CB = self.prior_convective_blueshift[0]   
        f_Prot = self.prior_p_rot[0] 
        f_inc = self.prior_inclination[0]   
        f_R = self.prior_Rstar[0]
        f_LD1 = self.prior_LD1[0]
        f_LD2 = self.prior_LD2[0]
        f_Pp = self.prior_Pp[0]
        f_T0p = self.prior_T0p[0]
        f_Kp = self.prior_Kp[0]
        f_esinwp = self.prior_esinwp[0]
        f_ecoswp = self.prior_ecoswp[0]
        f_Rp = self.prior_Rp[0]
        f_bp = self.prior_bp[0]
        f_alp = self.prior_alp[0]       
        self.fit=np.array([f_T,f_sp_T,f_fc_T,f_Q,f_CB,f_Prot,f_inc,f_R,f_LD1,f_LD2,f_Pp,f_T0p,f_Kp,f_esinwp,f_ecoswp,f_Rp,f_bp,f_alp ,*f_spot_it,*f_spot_lt,*f_spot_lat,*f_spot_lon,*f_spot_c1,*f_spot_c2,*f_spot_c3])       

        bound_spot_it=np.array([[self.prior_spot_initial_time[1],self.prior_spot_initial_time[2]] for i in range(N_spots)])
        bound_spot_lt=np.array([[self.prior_spot_life_time[1],self.prior_spot_life_time[2]] for i in range(N_spots)])
        bound_spot_lat=np.array([[self.prior_spot_latitude[1],self.prior_spot_latitude[2]]for i in range(N_spots)])
        bound_spot_lon=np.array([[self.prior_spot_longitude[1],self.prior_spot_longitude[2]] for i in range(N_spots)])
        bound_spot_c1=np.array([[self.prior_spot_coeff_1[1],self.prior_spot_coeff_1[2]] for i in range(N_spots)])
        bound_spot_c2=np.array([[self.prior_spot_coeff_2[1],self.prior_spot_coeff_2[2]] for i in range(N_spots)])
        bound_spot_c3=np.array([[self.prior_spot_coeff_3[1],self.prior_spot_coeff_3[2]] for i in range(N_spots)])
        bound_T = np.array([self.prior_t_eff_ph[1],self.prior_t_eff_ph[2]])
        bound_sp_T = np.array([self.prior_spot_T_contrast[1],self.prior_spot_T_contrast[2]]) 
        bound_fc_T = np.array([self.prior_facula_T_contrast[1],self.prior_facula_T_contrast[2]]) 
        bound_Q = np.array([self.prior_q_ratio[1],self.prior_q_ratio[2]])   
        bound_CB = np.array([self.prior_convective_blueshift[1],self.prior_convective_blueshift[2]])   
        bound_Prot = np.array([self.prior_p_rot[1],self.prior_p_rot[2]]) 
        bound_inc = np.array([self.prior_inclination[1],self.prior_inclination[2]])   
        bound_R = np.array([self.prior_Rstar[1],self.prior_Rstar[2]])
        bound_LD1 = np.array([self.prior_LD1[1],self.prior_LD1[2]])
        bound_LD2 = np.array([self.prior_LD2[1],self.prior_LD2[2]])
        bound_Pp = np.array([self.prior_Pp[1],self.prior_Pp[2]])
        bound_T0p = np.array([self.prior_T0p[1],self.prior_T0p[2]])
        bound_Kp = np.array([self.prior_Kp[1],self.prior_Kp[2]])
        bound_esinwp = np.array([self.prior_esinwp[1],self.prior_esinwp[2]])
        bound_ecoswp = np.array([self.prior_ecoswp[1],self.prior_ecoswp[2]])
        bound_Rp = np.array([self.prior_Rp[1],self.prior_Rp[2]])
        bound_bp = np.array([self.prior_bp[1],self.prior_bp[2]])
        bound_alp = np.array([self.prior_alp[1],self.prior_alp[2]])

        bounds=np.array([bound_T,bound_sp_T,bound_fc_T,bound_Q,bound_CB,bound_Prot,bound_inc,bound_R,bound_LD1,bound_LD2,bound_Pp,bound_T0p,bound_Kp,bound_esinwp,bound_ecoswp,bound_Rp,bound_bp,bound_alp,*bound_spot_it,*bound_spot_lt,*bound_spot_lat,*bound_spot_lon,*bound_spot_c1,*bound_spot_c2,*bound_spot_c3])#,*np.array(bound_offset),*np.array(bound_jitter)]) 

        prior_spot_it=[spectra.generate_prior(self.prior_spot_initial_time[3],self.prior_spot_initial_time[4],self.prior_spot_initial_time[5],self.nwalkers) for i in range(N_spots)]
        prior_spot_lt=[spectra.generate_prior(self.prior_spot_life_time[3],self.prior_spot_life_time[4],self.prior_spot_life_time[5],self.nwalkers) for i in range(N_spots)]
        prior_spot_lat=[spectra.generate_prior(self.prior_spot_latitude[3],self.prior_spot_latitude[4],self.prior_spot_latitude[5],self.nwalkers)for i in range(N_spots)]
        prior_spot_lon=[spectra.generate_prior(self.prior_spot_longitude[3],self.prior_spot_longitude[4],self.prior_spot_longitude[5],self.nwalkers) for i in range(N_spots)]
        prior_spot_c1=[spectra.generate_prior(self.prior_spot_coeff_1[3],self.prior_spot_coeff_1[4],self.prior_spot_coeff_1[5],self.nwalkers) for i in range(N_spots)]
        prior_spot_c2=[spectra.generate_prior(self.prior_spot_coeff_2[3],self.prior_spot_coeff_2[4],self.prior_spot_coeff_2[5],self.nwalkers) for i in range(N_spots)]
        prior_spot_c3=[spectra.generate_prior(self.prior_spot_coeff_3[3],self.prior_spot_coeff_3[4],self.prior_spot_coeff_3[5],self.nwalkers) for i in range(N_spots)]
        prior_T = spectra.generate_prior(self.prior_t_eff_ph[3],self.prior_t_eff_ph[4],self.prior_t_eff_ph[5],self.nwalkers)
        prior_sp_T = spectra.generate_prior(self.prior_spot_T_contrast[3],self.prior_spot_T_contrast[4],self.prior_spot_T_contrast[5],self.nwalkers) 
        prior_fc_T = spectra.generate_prior(self.prior_facula_T_contrast[3],self.prior_facula_T_contrast[4],self.prior_facula_T_contrast[5],self.nwalkers) 
        prior_Q = spectra.generate_prior(self.prior_q_ratio[3],self.prior_q_ratio[4],self.prior_q_ratio[5],self.nwalkers)   
        prior_CB = spectra.generate_prior(self.prior_convective_blueshift[3],self.prior_convective_blueshift[4],self.prior_convective_blueshift[5],self.nwalkers)   
        prior_Prot = spectra.generate_prior(self.prior_p_rot[3],self.prior_p_rot[4],self.prior_p_rot[5],self.nwalkers) 
        prior_inc = spectra.generate_prior(self.prior_inclination[3],self.prior_inclination[4],self.prior_inclination[5],self.nwalkers)   
        prior_R = spectra.generate_prior(self.prior_Rstar[3],self.prior_Rstar[4],self.prior_Rstar[5],self.nwalkers)
        prior_LD1 = spectra.generate_prior(self.prior_LD1[3],self.prior_LD1[4],self.prior_LD1[5],self.nwalkers)
        prior_LD2 = spectra.generate_prior(self.prior_LD2[3],self.prior_LD2[4],self.prior_LD2[5],self.nwalkers)
        prior_Pp = spectra.generate_prior(self.prior_Pp[3],self.prior_Pp[4],self.prior_Pp[5],self.nwalkers)
        prior_T0p = spectra.generate_prior(self.prior_T0p[3],self.prior_T0p[4],self.prior_T0p[5],self.nwalkers)
        prior_Kp = spectra.generate_prior(self.prior_Kp[3],self.prior_Kp[4],self.prior_Kp[5],self.nwalkers)
        prior_esinwp = spectra.generate_prior(self.prior_esinwp[3],self.prior_esinwp[4],self.prior_esinwp[5],self.nwalkers)
        prior_ecoswp = spectra.generate_prior(self.prior_ecoswp[3],self.prior_ecoswp[4],self.prior_ecoswp[5],self.nwalkers)
        prior_Rp = spectra.generate_prior(self.prior_Rp[3],self.prior_Rp[4],self.prior_Rp[5],self.nwalkers)
        prior_bp = spectra.generate_prior(self.prior_bp[3],self.prior_bp[4],self.prior_bp[5],self.nwalkers)
        prior_alp = spectra.generate_prior(self.prior_alp[3],self.prior_alp[4],self.prior_alp[5],self.nwalkers)

        priors=np.array([prior_T,prior_sp_T,prior_fc_T,prior_Q,prior_CB,prior_Prot,prior_inc,prior_R,prior_LD1,prior_LD2,prior_Pp,prior_T0p,prior_Kp,prior_esinwp,prior_ecoswp,prior_Rp,prior_bp,prior_alp,*prior_spot_it,*prior_spot_lt,*prior_spot_lat,*prior_spot_lon,*prior_spot_c1,*prior_spot_c2,*prior_spot_c3])#,*prior_offset,*prior_jitter]) 


        logprior_spot_it=np.array([[self.prior_spot_initial_time[3],self.prior_spot_initial_time[4],self.prior_spot_initial_time[5]] for i in range(N_spots)])
        logprior_spot_lt=np.array([[self.prior_spot_life_time[3],self.prior_spot_life_time[4],self.prior_spot_life_time[5]] for i in range(N_spots)])
        logprior_spot_lat=np.array([[self.prior_spot_latitude[3],self.prior_spot_latitude[4],self.prior_spot_latitude[5]]for i in range(N_spots)])
        logprior_spot_lon=np.array([[self.prior_spot_longitude[3],self.prior_spot_longitude[4],self.prior_spot_longitude[5]] for i in range(N_spots)])
        logprior_spot_c1=np.array([[self.prior_spot_coeff_1[3],self.prior_spot_coeff_1[4],self.prior_spot_coeff_1[5]] for i in range(N_spots)])
        logprior_spot_c2=np.array([[self.prior_spot_coeff_2[3],self.prior_spot_coeff_2[4],self.prior_spot_coeff_2[5]] for i in range(N_spots)])
        logprior_spot_c3=np.array([[self.prior_spot_coeff_3[3],self.prior_spot_coeff_3[4],self.prior_spot_coeff_3[5]] for i in range(N_spots)])
        logprior_T=np.array([self.prior_t_eff_ph[3],self.prior_t_eff_ph[4],self.prior_t_eff_ph[5]])
        logprior_sp_T=np.array([self.prior_spot_T_contrast[3],self.prior_spot_T_contrast[4],self.prior_spot_T_contrast[5]]) 
        logprior_fc_T=np.array([self.prior_facula_T_contrast[3],self.prior_facula_T_contrast[4],self.prior_facula_T_contrast[5]]) 
        logprior_Q=np.array([self.prior_q_ratio[3],self.prior_q_ratio[4],self.prior_q_ratio[5]])   
        logprior_CB=np.array([self.prior_convective_blueshift[3],self.prior_convective_blueshift[4],self.prior_convective_blueshift[5]])   
        logprior_Prot=np.array([self.prior_p_rot[3],self.prior_p_rot[4],self.prior_p_rot[5]]) 
        logprior_inc=np.array([self.prior_inclination[3],self.prior_inclination[4],self.prior_inclination[5]])   
        logprior_R=np.array([self.prior_Rstar[3],self.prior_Rstar[4],self.prior_Rstar[5]])
        logprior_LD1=np.array([self.prior_LD1[3],self.prior_LD1[4],self.prior_LD1[5]])
        logprior_LD2=np.array([self.prior_LD2[3],self.prior_LD2[4],self.prior_LD2[5]])
        logprior_Pp = np.array([self.prior_Pp[3],self.prior_Pp[4],self.prior_Pp[5]])
        logprior_T0p = np.array([self.prior_T0p[3],self.prior_T0p[4],self.prior_T0p[5]])
        logprior_Kp = np.array([self.prior_Kp[3],self.prior_Kp[4],self.prior_Kp[5]])
        logprior_esinwp = np.array([self.prior_esinwp[3],self.prior_esinwp[4],self.prior_esinwp[5]])
        logprior_ecoswp = np.array([self.prior_ecoswp[3],self.prior_ecoswp[4],self.prior_ecoswp[5]])
        logprior_Rp = np.array([self.prior_Rp[3],self.prior_Rp[4],self.prior_Rp[5]])
        logprior_bp = np.array([self.prior_bp[3],self.prior_bp[4],self.prior_bp[5]])
        logprior_alp = np.array([self.prior_alp[3],self.prior_alp[4],self.prior_alp[5]])
        logpriors=np.array([logprior_T,logprior_sp_T,logprior_fc_T,logprior_Q,logprior_CB,logprior_Prot,logprior_inc,logprior_R,logprior_LD1,logprior_LD2,logprior_Pp,logprior_T0p,logprior_Kp,logprior_esinwp,logprior_ecoswp,logprior_Rp,logprior_bp,logprior_alp,*logprior_spot_it,*logprior_spot_lt,*logprior_spot_lat,*logprior_spot_lon,*logprior_spot_c1,*logprior_spot_c2,*logprior_spot_c3])#,*logprior_offset,*logprior_jitter]) 


        vparamfit=np.array([])
        self.lparamfit=np.array([])
        boundfit=[]
        priors_fit=[]
        logpriors_fit=[]

        for i in range(len(self.fit)):
          if self.fit[i]==1:
            vparamfit=np.append(vparamfit,self.vparam[i])
            self.lparamfit=np.append(self.lparamfit,lparam[i])
            priors_fit.append(priors[i])
            logpriors_fit.append(logpriors[i])
            boundfit.append(bounds[i])
        boundfit=np.asarray(boundfit)
        priors_fit=np.asarray(priors_fit)
        logpriors_fit=np.asarray(logpriors_fit)

        
        ndim = len(self.lparamfit)
        p0=priors_fit.T
        
        print('MCMC uncertainties estimation')
        print('Total parameters to optimize:',ndim)

        preburns=self.planet_impact_paramurns
        burns=self.planet_impact_paramurns
        steps=self.steps
        nwalkers=self.nwalkers


        with Pool(self.N_cpus) as pool:
        #EMCEE

            p1=np.zeros([preburns,nwalkers,ndim])
            lp=np.zeros([preburns,nwalkers,1])

            postot=np.zeros([preburns+burns+steps,nwalkers,ndim])
            lptot=np.zeros([preburns+burns+steps,nwalkers])

            sampler = emcee.EnsembleSampler(nwalkers, ndim, spectra.lnposterior,args=(boundfit,logpriors_fit,self.vparam,self.fit,typ,self),pool=pool,moves=[(emcee.moves.DEMove(), 0.2),(emcee.moves.StretchMove(), 0.8)])
            
            print("Running first burn-in...")
            sampler.run_mcmc(p0,preburns,progress=True)
            p1, lp= sampler.get_chain(flat=False), sampler.get_log_prob(flat=False)
            postot[0:preburns,:,:], lptot[0:preburns,:]= sampler.get_chain(flat=False), sampler.get_log_prob(flat=False)

            print("Running second burn-in...")
            # p2= sampler.get_last_sample()
            sampler = emcee.EnsembleSampler(nwalkers, ndim, spectra.lnposterior,args=(boundfit,logpriors_fit,self.vparam,self.fit,typ,self),pool=pool,moves=[(emcee.moves.DEMove(), 0.6),(emcee.moves.DESnookerMove(), 0.2),(emcee.moves.StretchMove(), 0.2)])
            p2 = p1[np.unravel_index(lp.argmax(), lp.shape)[0:2]] + 1e-1*(np.max(priors_fit,1)-np.min(priors_fit,1)) * np.random.randn(nwalkers, ndim)
            sampler.reset()
            sampler.run_mcmc(p2,burns,progress=True)
            postot[preburns:preburns+burns,:,:], lptot[preburns:preburns+burns,:]= sampler.get_chain(flat=False), sampler.get_log_prob(flat=False)              
            p3= sampler.get_last_sample()

            print("Running production...")
            sampler.reset()
            sampler.run_mcmc(p3,steps,progress=True)
            postot[preburns+burns:preburns+burns+steps,:,:], lptot[preburns+burns:preburns+burns+steps:]= sampler.get_chain(flat=False), sampler.get_log_prob(flat=False)

        sampler.get_autocorr_time(quiet=True)

        #END OF EMCEE

        self.samples=postot
        self.logs=lptot

        gc.collect()
        sampler.pool.terminate()

        samples_burned=postot[preburns+burns::,:,:].reshape((-1,ndim))
        logs_burned=lptot[preburns+burns::,:].reshape((-2))

        self.planet_impact_paramestparams=samples_burned[np.argmax(logs_burned),:]
        self.planet_impact_paramestmean=[np.median(samples_burned[:,i]) for i in range(len(samples_burned[0]))]
        self.planet_impact_parameststd=[np.std(samples_burned[:,i]) for i in range(len(samples_burned[0]))]
        self.planet_impact_paramestup=[np.quantile(samples_burned[:,i],0.84135)-np.median(samples_burned[:,i]) for i in range(len(samples_burned[0]))]
        self.planet_impact_paramestbot=[np.median(samples_burned[:,i])-np.quantile(samples_burned[:,i],0.15865) for i in range(len(samples_burned[0]))] 

        param_inv=[]
        # print(P)
        ii=0
        for i in range(len(self.fit)):
          if self.fit[i]==0:
            param_inv.append(np.array(self.vparam[i]))
          elif self.fit[i]==1:
            param_inv.append(np.array(samples_burned[:,ii]))
            ii=ii+1
        
        vsini_inv= 2*np.pi*(param_inv[7]*696342)*np.cos(np.deg2rad(90-param_inv[6]))/(param_inv[5]*86400) #in km/s
        if self.limb_darkening_law == 'linear':
            a_LD=param_inv[8]
            b_LD=param_inv[8]*0
        elif self.limb_darkening_law == 'quadratic':
            a_LD=2*np.sqrt(param_inv[8])*param_inv[9]
            b_LD=np.sqrt(param_inv[8])*(1-2*param_inv[9])
        elif self.limb_darkening_law == 'sqrt':
            a_LD=np.sqrt(param_inv[8])*(1-2*param_inv[9]) 
            b_LD=2*np.sqrt(param_inv[8])*param_inv[9]
        elif self.limb_darkening_law == 'log':
            a_LD=param_inv[9]*param_inv[8]**2+1
            b_LD=param_inv[8]**2-1


        s='Results of the inversion process\n'
        print('Results of the inversion process:')
        s+='    -Mean and 1 sigma confidence interval:\n'
        print('\t -Mean and 1 sigma confidence interval:')
        for ip in range(len(self.vparam)):
          if self.fit[ip]==1:
            s+='        {} = {:.5f}+{:.5f}-{:.5f}\n'.format(lparam[ip],np.median(param_inv[ip]),np.quantile(param_inv[ip],0.84135)-np.median(param_inv[ip]),np.median(param_inv[ip])-np.quantile(param_inv[ip],0.15865))
            print('\t \t {} = {:.5f}+{:.5f}-{:.5f}'.format(lparam[ip],np.median(param_inv[ip]),np.quantile(param_inv[ip],0.84135)-np.median(param_inv[ip]),np.median(param_inv[ip])-np.quantile(param_inv[ip],0.15865)))
          else:
            s+='        {} = {:.5f} (fixed)\n'.format(lparam[ip],self.vparam[ip])
            print('\t \t',lparam[ip],' = ',self.vparam[ip],'(fixed) ') 
        s+='        $vsini$ = {:.5f}+{:.5f}-{:.5f}\n'.format(np.median(vsini_inv),np.quantile(vsini_inv,0.84135)-np.median(vsini_inv),np.median(vsini_inv)-np.quantile(vsini_inv,0.15865))
        s+='        LD_a = {:.5f}+{:.5f}-{:.5f}\n'.format(np.median(a_LD),np.quantile(a_LD,0.84135)-np.median(a_LD),np.median(a_LD)-np.quantile(a_LD,0.15865))
        s+='        LD_b = {:.5f}+{:.5f}-{:.5f}\n'.format(np.median(b_LD),np.quantile(b_LD,0.84135)-np.median(b_LD),np.median(b_LD)-np.quantile(b_LD,0.15865)) 

        s+='    -Mean and standard deviation:\n'
        print('\t -Mean and standard deviation:')
        for ip in range(len(self.vparam)):
          if self.fit[ip]==1:
            s+='        {} = {:.5f}+-{:.5f}\n'.format(lparam[ip],np.median(param_inv[ip]),np.std(param_inv[ip]))
            print('\t \t {} = {:.5f}+-{:.5f}'.format(lparam[ip],np.median(param_inv[ip]),np.std(param_inv[ip])))
        s+='    -Best solution, with maximum log-likelihood of {:.5f}\n'.format(np.max(self.logs))
        print('\t -Best solution, with maximum log-likelihood of',np.max(self.logs))  
        for ip in range(len(self.vparam)):
          if self.fit[ip]==1:
            s+='        {} = {:.5f}\n'.format(lparam[ip],param_inv[ip][np.argmax(logs_burned)])
            print('\t \t {} = {:.5f}'.format(lparam[ip],param_inv[ip][np.argmax(logs_burned)]))


        fig = plt.figure(figsize=(6,10))
        plt.annotate(s, xy=(0.0, 1.0),ha='left',va='top')
        plt.axis('off')
        plt.tight_layout()
        ofilename = self.path / 'plots' / 'results_inversion.png'
        plt.savefig(ofilename,dpi=200)
            



        
    #Optimize the stellar map. The spot map is optimized using Simulated annealing.
    def compute_inverseSA(self,N_inversions):
        N_spots = len(self.spot_map) #number of spots in spot_map
        # self.n_grid_rings = 5 
        self.simulation_mode = 'fast' #must work in fast mode

        print('Computing',N_inversions,'inversions of',N_spots,'spots each.')
        print('\nUsing data from the instruments:')
        self.instruments=[]
        self.observables=[]
        typ=[]

        N_obs=0
        self.N_data=0 #total number of points
        for ins in self.data.keys():

            print('-',ins,', with the observables:')
            self.instruments.append(ins)
            o=[]
            ty=[]
            for obs in self.data[ins].keys():
                if obs in ['lc']:
                    print('\t-',obs)
                    o.append(obs)
                    ty.append(0)
                    self.N_data=self.N_data+len(self.data[ins][obs]['t'])
                elif obs in ['rv','fwhm','bis','contrast']:
                    print('\t-',obs)
                    o.append(obs)
                    ty.append(1)
                    self.N_data=self.N_data+len(self.data[ins][obs]['t'])
                if obs in ['crx']:
                    print('\t-',obs)
                    o.append(obs)
                    ty.append(2)
                    self.N_data=self.N_data+len(self.data[ins][obs]['t'])
            N_obs+=len(o)
            self.observables.append(o)
            typ.append(ty)


        with Pool(processes=self.N_cpus) as pool:
            res=pool.starmap(SA.inversion_parallel,tqdm.tqdm([(self,typ,i) for i in range(N_inversions)],total=N_inversions),chunksize=1)

        best_maps = np.asarray(res,dtype='object')[:,0]
        lnLs = np.asarray(res,dtype='object')[:,1]


        ofilename = self.path / 'results' / 'inversion_stats.npy'
        np.save(ofilename,np.array([lnLs,best_maps],dtype='object'),allow_pickle=True)

        return best_maps, lnLs



    #Optimize the stellar parameters. For each configuration of the MCMC, the spot map is optimized using SA.
    def optimize_inversion_SA(self):
        os.environ["OMP_NUM_THREADS"] = "1"

        N_spots = len(self.spot_map) #number of spots in spot_map
        # self.n_grid_rings = 5 
        self.simulation_mode = 'fast' #must work in fast mode

        print('\nUsing data from the instruments:')
        self.instruments=[]
        self.observables=[]
        typ=[]

        N_obs=0
        self.N_data=0
        for ins in self.data.keys():
            print('-',ins,', with the observables:')
            self.instruments.append(ins)
            o=[]
            ty=[]
            for obs in self.data[ins].keys():
                if obs in ['lc']:
                    print('\t-',obs)
                    o.append(obs)
                    ty.append(0)
                    self.N_data=self.N_data+len(self.data[ins][obs]['t'])
                elif obs in ['rv','fwhm','bis','contrast']:
                    print('\t-',obs)
                    o.append(obs)
                    ty.append(1)
                    self.N_data=self.N_data+len(self.data[ins][obs]['t'])
                if obs in ['crx']:
                    print('\t-',obs)
                    o.append(obs)
                    ty.append(2)
                    self.N_data=self.N_data+len(self.data[ins][obs]['t'])
            N_obs+=len(o)
            self.observables.append(o)
            typ.append(ty)



        fixed_T = self.temperature_photosphere
        fixed_sp_T = self.spot_T_contrast
        fixed_fc_T = self.facula_T_contrast
        fixed_Q = self.facular_area_ratio
        fixed_CB = self.convective_shift
        fixed_Prot = self.rotation_period
        fixed_inc = np.rad2deg(np.pi/2-self.inclination) 
        fixed_R = self.radius
        fixed_LD1 = self.limb_darkening_q1
        fixed_LD2 = self.limb_darkening_q2
        fixed_Pp = self.planet_period
        fixed_T0p = self.planet_transit_t0
        fixed_Kp = self.planet_semi_amplitude
        fixed_esinwp = self.planet_esinw
        fixed_ecoswp = self.planet_ecosw
        fixed_Rp =  self.planet_radius
        fixed_bp = self.planet_impact_param
        fixed_alp =  self.planet_spin_orbit_angle    #spin-orbit angle

        self.vparam=np.array([fixed_T,fixed_sp_T,fixed_fc_T,fixed_Q,fixed_CB,fixed_Prot,fixed_inc,fixed_R,fixed_LD1,fixed_LD2,fixed_Pp,fixed_T0p,fixed_Kp,fixed_esinwp,fixed_ecoswp,fixed_Rp,fixed_bp,fixed_alp])

        name_T ='T$_{{eff}}$'
        name_sp_T ='$\\Delta$ T$_{{sp}}$'
        name_fc_T ='$\\Delta$ T$_{{fc}}$'
        name_Q ='Fac-spot ratio'
        name_CB ='CS'
        name_Prot ='P$_{{rot}}$'
        name_inc ='inc'
        name_R ='R$_*$'
        name_LD1 = 'q$_1$'
        name_LD2 = 'q$_2$'
        name_Pp = 'P$_{{pl}}$'
        name_T0p = 'T$_{{0,pl}}$'
        name_Kp = 'K$_{{pl}}$'
        name_esinwp = 'esinw'
        name_ecoswp = 'ecosw'
        name_Rp =  'R$_{{pl}}$'
        name_bp = 'b'
        name_alp = '$\\lambda$'  

        self.lparam=np.array([name_T,name_sp_T,name_fc_T,name_Q,name_CB,name_Prot,name_inc,name_R,name_LD1,name_LD2,name_Pp,name_T0p,name_Kp,name_esinwp,name_ecoswp,name_Rp,name_bp,name_alp])

        f_T = self.prior_t_eff_ph[0]
        f_sp_T = self.prior_spot_T_contrast[0] 
        f_fc_T = self.prior_facula_T_contrast[0] 
        f_Q = self.prior_q_ratio[0]   
        f_CB = self.prior_convective_blueshift[0]   
        f_Prot = self.prior_p_rot[0] 
        f_inc = self.prior_inclination[0]   
        f_R = self.prior_Rstar[0]
        f_LD1 = self.prior_LD1[0]
        f_LD2 = self.prior_LD2[0]
        f_Pp = self.prior_Pp[0]
        f_T0p = self.prior_T0p[0]
        f_Kp = self.prior_Kp[0]
        f_esinwp = self.prior_esinwp[0]
        f_ecoswp = self.prior_ecoswp[0]
        f_Rp = self.prior_Rp[0]
        f_bp = self.prior_bp[0]
        f_alp = self.prior_alp[0]     

        self.fit=np.array([f_T,f_sp_T,f_fc_T,f_Q,f_CB,f_Prot,f_inc,f_R,f_LD1,f_LD2,f_Pp,f_T0p,f_Kp,f_esinwp,f_ecoswp,f_Rp,f_bp,f_alp])       

        bound_T = np.array([self.prior_t_eff_ph[1],self.prior_t_eff_ph[2]])
        bound_sp_T = np.array([self.prior_spot_T_contrast[1],self.prior_spot_T_contrast[2]]) 
        bound_fc_T = np.array([self.prior_facula_T_contrast[1],self.prior_facula_T_contrast[2]]) 
        bound_Q = np.array([self.prior_q_ratio[1],self.prior_q_ratio[2]])   
        bound_CB = np.array([self.prior_convective_blueshift[1],self.prior_convective_blueshift[2]])   
        bound_Prot = np.array([self.prior_p_rot[1],self.prior_p_rot[2]]) 
        bound_inc = np.array([self.prior_inclination[1],self.prior_inclination[2]])   
        bound_R = np.array([self.prior_Rstar[1],self.prior_Rstar[2]])
        bound_LD1 = np.array([self.prior_LD1[1],self.prior_LD1[2]])
        bound_LD2 = np.array([self.prior_LD2[1],self.prior_LD2[2]])
        bound_Pp = np.array([self.prior_Pp[1],self.prior_Pp[2]])
        bound_T0p = np.array([self.prior_T0p[1],self.prior_T0p[2]])
        bound_Kp = np.array([self.prior_Kp[1],self.prior_Kp[2]])
        bound_esinwp = np.array([self.prior_esinwp[1],self.prior_esinwp[2]])
        bound_ecoswp = np.array([self.prior_ecoswp[1],self.prior_ecoswp[2]])
        bound_Rp = np.array([self.prior_Rp[1],self.prior_Rp[2]])
        bound_bp = np.array([self.prior_bp[1],self.prior_bp[2]])
        bound_alp = np.array([self.prior_alp[1],self.prior_alp[2]])

        bounds=np.array([bound_T,bound_sp_T,bound_fc_T,bound_Q,bound_CB,bound_Prot,bound_inc,bound_R,bound_LD1,bound_LD2,bound_Pp,bound_T0p,bound_Kp,bound_esinwp,bound_ecoswp,bound_Rp,bound_bp,bound_alp]) 

        prior_T = spectra.generate_prior(self.prior_t_eff_ph[3],self.prior_t_eff_ph[4],self.prior_t_eff_ph[5],self.steps)
        prior_sp_T = spectra.generate_prior(self.prior_spot_T_contrast[3],self.prior_spot_T_contrast[4],self.prior_spot_T_contrast[5],self.steps) 
        prior_fc_T = spectra.generate_prior(self.prior_facula_T_contrast[3],self.prior_facula_T_contrast[4],self.prior_facula_T_contrast[5],self.steps) 
        prior_Q = spectra.generate_prior(self.prior_q_ratio[3],self.prior_q_ratio[4],self.prior_q_ratio[5],self.steps)   
        prior_CB = spectra.generate_prior(self.prior_convective_blueshift[3],self.prior_convective_blueshift[4],self.prior_convective_blueshift[5],self.steps)   
        prior_Prot = spectra.generate_prior(self.prior_p_rot[3],self.prior_p_rot[4],self.prior_p_rot[5],self.steps) 
        prior_inc = spectra.generate_prior(self.prior_inclination[3],self.prior_inclination[4],self.prior_inclination[5],self.steps)   
        prior_R = spectra.generate_prior(self.prior_Rstar[3],self.prior_Rstar[4],self.prior_Rstar[5],self.steps)
        prior_LD1 = spectra.generate_prior(self.prior_LD1[3],self.prior_LD1[4],self.prior_LD1[5],self.steps)
        prior_LD2 = spectra.generate_prior(self.prior_LD2[3],self.prior_LD2[4],self.prior_LD2[5],self.steps)
        prior_Pp = spectra.generate_prior(self.prior_Pp[3],self.prior_Pp[4],self.prior_Pp[5],self.steps)
        prior_T0p = spectra.generate_prior(self.prior_T0p[3],self.prior_T0p[4],self.prior_T0p[5],self.steps)
        prior_Kp = spectra.generate_prior(self.prior_Kp[3],self.prior_Kp[4],self.prior_Kp[5],self.steps)
        prior_esinwp = spectra.generate_prior(self.prior_esinwp[3],self.prior_esinwp[4],self.prior_esinwp[5],self.steps)
        prior_ecoswp = spectra.generate_prior(self.prior_ecoswp[3],self.prior_ecoswp[4],self.prior_ecoswp[5],self.steps)
        prior_Rp = spectra.generate_prior(self.prior_Rp[3],self.prior_Rp[4],self.prior_Rp[5],self.steps)
        prior_bp = spectra.generate_prior(self.prior_bp[3],self.prior_bp[4],self.prior_bp[5],self.steps)
        prior_alp = spectra.generate_prior(self.prior_alp[3],self.prior_alp[4],self.prior_alp[5],self.steps)

        priors=np.array([prior_T,prior_sp_T,prior_fc_T,prior_Q,prior_CB,prior_Prot,prior_inc,prior_R,prior_LD1,prior_LD2,prior_Pp,prior_T0p,prior_Kp,prior_esinwp,prior_ecoswp,prior_Rp,prior_bp,prior_alp]) 


        logprior_T=np.array([self.prior_t_eff_ph[3],self.prior_t_eff_ph[4],self.prior_t_eff_ph[5]])
        logprior_sp_T=np.array([self.prior_spot_T_contrast[3],self.prior_spot_T_contrast[4],self.prior_spot_T_contrast[5]]) 
        logprior_fc_T=np.array([self.prior_facula_T_contrast[3],self.prior_facula_T_contrast[4],self.prior_facula_T_contrast[5]]) 
        logprior_Q=np.array([self.prior_q_ratio[3],self.prior_q_ratio[4],self.prior_q_ratio[5]])   
        logprior_CB=np.array([self.prior_convective_blueshift[3],self.prior_convective_blueshift[4],self.prior_convective_blueshift[5]])   
        logprior_Prot=np.array([self.prior_p_rot[3],self.prior_p_rot[4],self.prior_p_rot[5]]) 
        logprior_inc=np.array([self.prior_inclination[3],self.prior_inclination[4],self.prior_inclination[5]])   
        logprior_R=np.array([self.prior_Rstar[3],self.prior_Rstar[4],self.prior_Rstar[5]])
        logprior_LD1=np.array([self.prior_LD1[3],self.prior_LD1[4],self.prior_LD1[5]])
        logprior_LD2=np.array([self.prior_LD2[3],self.prior_LD2[4],self.prior_LD2[5]])
        logprior_Pp = np.array([self.prior_Pp[3],self.prior_Pp[4],self.prior_Pp[5]])
        logprior_T0p = np.array([self.prior_T0p[3],self.prior_T0p[4],self.prior_T0p[5]])
        logprior_Kp = np.array([self.prior_Kp[3],self.prior_Kp[4],self.prior_Kp[5]])
        logprior_esinwp = np.array([self.prior_esinwp[3],self.prior_esinwp[4],self.prior_esinwp[5]])
        logprior_ecoswp = np.array([self.prior_ecoswp[3],self.prior_ecoswp[4],self.prior_ecoswp[5]])
        logprior_Rp = np.array([self.prior_Rp[3],self.prior_Rp[4],self.prior_Rp[5]])
        logprior_bp = np.array([self.prior_bp[3],self.prior_bp[4],self.prior_bp[5]])
        logprior_alp = np.array([self.prior_alp[3],self.prior_alp[4],self.prior_alp[5]])

        logpriors=np.array([logprior_T,logprior_sp_T,logprior_fc_T,logprior_Q,logprior_CB,logprior_Prot,logprior_inc,logprior_R,logprior_LD1,logprior_LD2,logprior_Pp,logprior_T0p,logprior_Kp,logprior_esinwp,logprior_ecoswp,logprior_Rp,logprior_bp,logprior_alp]) 


        vparamfit=np.array([])
        self.lparamfit=np.array([])
        boundfit=[]
        priors_fit=[]
        logpriors_fit=[]

        for i in range(len(self.fit)):
          if self.fit[i]==1:
            vparamfit=np.append(vparamfit,self.vparam[i])
            self.lparamfit=np.append(self.lparamfit,self.lparam[i])
            priors_fit.append(priors[i])
            logpriors_fit.append(logpriors[i])
            boundfit.append(bounds[i])
        boundfit=np.asarray(boundfit)
        priors_fit=np.asarray(priors_fit)
        logpriors_fit=np.asarray(logpriors_fit)

        
        ndim = len(self.lparamfit)
        p0=priors_fit.T
        
        print('Searching random grid for best stellar parameters. Optimizing spotmap at each step.')
        print('Total parameters to optimize:',ndim)

        steps=self.steps

        with Pool(processes=self.N_cpus) as pool:
            res=pool.starmap(SA.inversion_parallel_MCMC,tqdm.tqdm([(self,p0,boundfit,logpriors_fit,typ,i) for i in range(steps)], total=steps),chunksize=1)


        p_used = np.asarray(res,dtype='object')[:,0]
        best_maps = np.asarray(res,dtype='object')[:,1]
        lnLs = np.asarray(res,dtype='object')[:,2]

        ofilename = self.path / 'results' / 'optimize_inversion_SA_stats.npy'
        np.save(ofilename,np.array([lnLs,p_used,best_maps],dtype='object'),allow_pickle=True)














    def load_data(self,filename=None,t=None,y=None,yerr=None,instrument=None,observable=None,wvmin=None,wvmax=None,filter_name=None,offset=None,fix_offset=False,jitter=0.0,fix_jitter=False):
    
        
        if observable not in ['lc','rv','bis','fwhm','contrast','crx']:
            sys.exit('Observable not valid. Use one of the following: lc, rv, bis, fwhm, contrast or crx')

        if wvmin==None and wvmax==None:
            print('Wavelength range of the instrument not specified. Using the values in the file starsim.conf, ',self.wavelength_lower_limit,'and ',self.wavelength_upper_limit)

        if observable=='lc' and filter_name== None:
            print('Filter file neam not specified. Using the values in ',self.filter_name,'. Filters can be retrieved from http://svo2.cab.inta-csic.es/svo/theory/fps3/')
            filter_name = self.filter_name

        self.data[instrument][observable]={}
        self.data[instrument]['wvmin']=wvmin
        self.data[instrument]['wvmax']=wvmax
        self.data[instrument]['filter']=filter_name
        self.data[instrument][observable]['offset']=offset
        self.data[instrument][observable]['jitter']=jitter
        self.data[instrument][observable]['fix_offset']=fix_offset
        self.data[instrument][observable]['fix_jitter']=fix_jitter

        if filename != None:
            filename = self.path / filename
            self.data[instrument][observable]['t'], self.data[instrument][observable]['y'], self.data[instrument][observable]['yerr'] = np.loadtxt(filename,unpack=True)              
        else:
            self.data[instrument][observable]['t'], self.data[instrument][observable]['y'], self.data[instrument][observable]['yerr'] = t, y, yerr
            if t is None:
                sys.exit('Please provide a valid filename with the input data')


        if observable in ['lc','fwhm','contrast']:
            if offset == 0.0:
                sys.exit("Error in the input offset of the observable:",observable,". It is a multiplicative offset, can't be 0")
            if offset is None:
                offset=1.0
            self.data[instrument][observable]['yerr']=np.sqrt(self.data[instrument][observable]['yerr']**2+jitter**2)/offset
            self.data[instrument][observable]['y']=self.data[instrument][observable]['y']/offset
            self.data[instrument][observable]['offset_type']='multiplicative'
        else:
            if offset is None:
                offset=0.0
            self.data[instrument][observable]['y']=self.data[instrument][observable]['y'] - offset
            self.data[instrument][observable]['yerr']=np.sqrt(self.data[instrument][observable]['yerr']**2+jitter**2)
            self.data[instrument][observable]['offset_type']='linear'



    ###########################################################
    ################ PLOTS ####################################
    ###########################################################
    def plot_forward_results(self):
        '''method for plotting the results from forward method
        '''
        fig, ax = plt.subplots(len(self.results.keys())-2,figsize=(6,8),sharex=True)
        k=0
        for i, name in enumerate(self.results.keys()):
            if name=='time':
                k-=1
            elif name=='CCF':
                k-=1
            else:
                ax[k].plot(self.results['time'],self.results[name],'.')
                ax[k].set_ylabel(self.name_params[name])
                ax[k].minorticks_on()
                ax[k].tick_params(axis='both',which='both',direction='inout')
            k+=1


        ax[-1].set_xlabel('Obs. time [days]')

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.0)

        ofilename = self.path  / 'plots' / 'forward_results.png'
        plt.savefig(ofilename,dpi=200)
        #plt.show(block=True)
        plt.close()

    def plot_MCMCoptimization_chain(self):
        ndim=len(self.lparamfit)
        fig1 = plt.figure(figsize=(15,12))
        xstep=np.arange(len(self.samples[:,0,0]))
        for ip in range(ndim):
          plt.subplot(m.ceil(ndim/4),4,ip+1)
          for iw in range(0,self.nwalkers):
            ystep=self.samples[:,iw,ip]
            plt.plot(xstep,ystep,'-k',alpha=0.07)
          plt.xlabel('MCMC step')
          plt.ylabel(self.lparamfit[ip])

        ofilename = self.path  / 'plots' / 'MCMCoptimization_chains.png'
        plt.savefig(ofilename,dpi=200)
        #plt.show(block=True)
        plt.close()

    def plot_MCMCoptimization_likelihoods(self):
        xtot=self.samples.reshape((-1,len(self.lparamfit)))
        ytot=self.logs.reshape((-2))
        ndim=len(self.lparamfit)
        fig1 = plt.figure(figsize=(15,12))
        
        for ip in range(ndim):
          plt.subplot(m.ceil(ndim/4),4,ip+1)
          plt.plot(xtot[:,ip],ytot,'k,')
          plt.axhline(np.max(ytot)-15,color='r',ls=':')
          plt.ylabel('lnL')
          left=np.min(xtot[ytot>(np.max(ytot)-15),ip])
          right=np.max(xtot[ytot>(np.max(ytot)-15),ip])
          plt.ylim([np.max(ytot)-30,np.max(ytot)+2])
          plt.xlim([left-(right-left)*0.2,right+(right-left)*0.2])
          plt.xlabel(self.lparamfit[ip])
        
        ofilename = self.path  / 'plots' / 'MCMCoptimization_likelihoods.png'
        plt.savefig(ofilename,dpi=200)
        #plt.show(block=True)
        plt.close()

    def plot_MCMCoptimization_corner(self):
        fig2, axes = plt.subplots(len(self.lparamfit),len(self.lparamfit), figsize=(2.3*len(self.lparamfit),2.3*len(self.lparamfit)))
        corner.corner(self.samples[-self.steps::,:,:].reshape((-1,len(self.lparamfit))),bins=20,plot_contours=False,fig=fig2,max_n_ticks=2,labels=self.lparamfit,label_kwargs={'fontsize':13},quantiles=(0.16,0.5,0.84),show_titles=True)
        
        ofilename = self.path / 'plots' / 'MCMCoptimization_cornerplot.png'
        plt.savefig(ofilename,dpi=200)
        #plt.show(block=True)
        plt.close()



    def plot_MCMCoptimization_results(self,Nsamples=100,t=None,fold=True):

        sample=self.samples[-self.steps::,:,:].reshape((-1,len(self.lparamfit)))
        num_obs=len(sum(self.observables,[]))

        fig2, ax = plt.subplots(num_obs,1,figsize=(8,12))
        if num_obs == 1:
            ax= [ax]

        stack_dic={}
        for k in range(Nsamples):
            sys.stdout.write("\r [{}/{}]".format(k,Nsamples))
            P=sample[np.random.randint(len(sample))]
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
            self.planet_radius = p[15]
            self.planet_impact_param = p[16]
            self.planet_spin_orbit_angle = p[17]*np.pi/180 #deg2rad           

            N_spots=len(self.spot_map)
            for i in range(N_spots):
                self.spot_map[i][0]=p[18+i]
                self.spot_map[i][1]=p[18+N_spots+i]
                self.spot_map[i][2]=p[18+2*N_spots+i]
                self.spot_map[i][3]=p[18+3*N_spots+i]
                self.spot_map[i][4]=p[18+4*N_spots+i]
                self.spot_map[i][5]=p[18+5*N_spots+i]
                self.spot_map[i][6]=p[18+6*N_spots+i]


            #np.round(t/step)*step
            #Compute the model for each instrument and observable, and the corresponding lnL
            l=0
            for i in range(len(self.instruments)):
                for j in self.observables[i]:
                    if k==0:
                        stack_dic['{}_{}'.format(i,j)]=[]
                    self.wavelength_lower_limit=self.data[self.instruments[i]]['wvmin']
                    self.wavelength_upper_limit=self.data[self.instruments[i]]['wvmax']
                    self.filter_name=self.data[self.instruments[i]]['filter']
                    self.compute_forward(observables=j,t=self.data[self.instruments[i]][j]['t'],inversion=True)
                    
                    if self.data[self.instruments[i]][j]['offset_type']=='multiplicative': #j=='lc':
                        
                        if (self.data[self.instruments[i]][j]['fix_jitter'] and self.data[self.instruments[i]][j]['fix_offset']):
                            offset=1.0
                            jitter=0.0

                        elif self.data[self.instruments[i]][j]['fix_offset']:
                            offset=1.0
                            res=optimize.minimize(nbspectra.fit_only_jitter,2*np.mean(self.data[self.instruments[i]][j]['yerr']), args=(self.results[j],self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                            jitter=res.x[0]
                            
                        
                        elif self.data[self.instruments[i]][j]['fix_jitter']:
                            jitter=0.0
                            res=optimize.minimize(nbspectra.fit_only_multiplicative_offset,np.mean(self.data[self.instruments[i]][j]['y'])/(np.mean(self.results[j])+0.0001), args=(self.results[j],self.data[self.instruments[i]][j]['y'],np.sqrt(self.data[self.instruments[i]][j]['yerr']**2+jitter**2)), method='Nelder-Mead')
                            offset=res.x[0]

                        else:
                            res=optimize.minimize(nbspectra.fit_multiplicative_offset_jitter,[np.mean(self.data[self.instruments[i]][j]['y'])/(np.mean(self.results[j])+0.0001),2*np.mean(self.data[self.instruments[i]][j]['yerr'])], args=(self.results[j],self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                            offset=res.x[0]
                            jitter=res.x[1]

                        self.compute_forward(observables=j,t=t,inversion=True)
                        stack_dic['{}_{}'.format(i,j)].append(self.results[j]*offset)
                        l+=1
                    
                    
                    else: #linear offset

                        if (self.data[self.instruments[i]][j]['fix_jitter'] and self.data[self.instruments[i]][j]['fix_offset']):
                            offset=0.0
                            jitter=0.0

                        elif self.data[self.instruments[i]][j]['fix_offset']:
                            offset=0.0
                            res=optimize.minimize(nbspectra.fit_only_jitter,2*np.mean(self.data[self.instruments[i]][j]['yerr']), args=(self.results[j],self.data[self.instruments[i]][j]['y']-offset,self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                            jitter=res.x[0]
                        
                        elif self.data[self.instruments[i]][j]['fix_jitter']:
                            jitter=0.0
                            res=optimize.minimize(nbspectra.fit_only_linear_offset,np.mean(self.data[self.instruments[i]][j]['y'])-np.mean(self.results[j]), args=(self.results[j],self.data[self.instruments[i]][j]['y'],np.sqrt(jitter**2+self.data[self.instruments[i]][j]['yerr']**2)), method='Nelder-Mead')
                            offset=res.x[0]

                        else:
                            res=optimize.minimize(nbspectra.fit_linear_offset_jitter,[np.mean(self.data[self.instruments[i]][j]['y'])-np.mean(self.results[j]),2*np.mean(self.data[self.instruments[i]][j]['yerr'])], args=(self.results[j],self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                            offset=res.x[0]
                            jitter=res.x[1]

                        self.compute_forward(observables=j,t=t,inversion=True)
                        stack_dic['{}_{}'.format(i,j)].append(self.results[j]+offset)
                        l+=1
        
        if fold is True:
            t=t/self.rotation_period%1*self.rotation_period
        idxsrt=np.argsort(t)

        #Plot the data
        l=0
        for i in range(len(self.instruments)):
            for j in self.observables[i]:
                ax[l].fill_between(t[idxsrt],np.mean(stack_dic['{}_{}'.format(i,j)],axis=0)[idxsrt]-np.std(stack_dic['{}_{}'.format(i,j)],axis=0)[idxsrt],np.mean(stack_dic['{}_{}'.format(i,j)],axis=0)[idxsrt]+np.std(stack_dic['{}_{}'.format(i,j)],axis=0)[idxsrt],color='k',alpha=0.3)
                ax[l].plot(t[idxsrt],np.mean(stack_dic['{}_{}'.format(i,j)],axis=0)[idxsrt],'k')
                if fold is True:
                    ax[l].errorbar(self.data[self.instruments[i]][j]['t']/self.rotation_period%1*self.rotation_period,self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr'],fmt='bo',ecolor='lightblue')
                else:
                    ax[l].errorbar(self.data[self.instruments[i]][j]['t'],self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr'],fmt='bo',ecolor='lightblue')
                ax[l].set_ylabel('{}_{}'.format(self.instruments[i],j))
                l+=1

        ofilename = self.path  / 'plots' / 'MCMCoptimization_timeseries_result.png'
        plt.savefig(ofilename,dpi=200)
        plt.close()
        # plt.show(block=True)



    def plot_inversion_results(self,best_maps,lnLs,Npoints=200):

        self.instruments=[]
        self.observables=[]
        typ=[]
        tmax=-3000000
        tmin=3000000

        for ins in self.data.keys():
            self.instruments.append(ins)
            o=[]
            ty=[]
            for obs in self.data[ins].keys():
                if obs in ['lc']:
                    o.append(obs)
                    ty.append(0)
                    if self.data[ins]['lc']['t'].min()<tmin: tmin=self.data[ins]['lc']['t'].min()
                    if self.data[ins]['lc']['t'].max()>tmax: tmax=self.data[ins]['lc']['t'].max()
                elif obs in ['rv','fwhm','bis','contrast']:
                    o.append(obs)
                    ty.append(1)
                    if self.data[ins][obs]['t'].min()<tmin: tmin=self.data[ins][obs]['t'].min()
                    if self.data[ins][obs]['t'].max()>tmax: tmax=self.data[ins][obs]['t'].max()
                if obs in ['crx']:
                    o.append(obs)
                    ty.append(2)
                    if self.data[ins]['crx']['t'].min()<tmin: tmin=self.data[ins]['crx']['t'].min()
                    if self.data[ins]['crx']['t'].max()>tmax: tmax=self.data[ins]['crx']['t'].max()
            self.observables.append(o)

        num_obs=len(sum(self.observables,[]))

        t=np.linspace(tmin,tmax,Npoints)

        fig, ax = plt.subplots(num_obs,1,figsize=(12,12))
        if num_obs == 1:
            ax= [ax]



        store_results = np.zeros([len(best_maps),num_obs,Npoints])


        bestlnL=np.argmax(lnLs)
        for k in range(len(best_maps)):
            self.spot_map[:,0:7] = best_maps[k]
                        
            #Plot the data
            l=0
            for i in range(len(self.instruments)):
                for j in self.observables[i]:
                    self.wavelength_lower_limit=self.data[self.instruments[i]]['wvmin']
                    self.wavelength_upper_limit=self.data[self.instruments[i]]['wvmax']
                    self.filter_name=self.data[self.instruments[i]]['filter']
                    self.compute_forward(observables=j,t=self.data[self.instruments[i]][j]['t'],inversion=True)

                    if self.data[self.instruments[i]][j]['offset_type']=='multiplicative':
                        
                        if (self.data[self.instruments[i]][j]['fix_jitter'] and self.data[self.instruments[i]][j]['fix_offset']):
                            offset=1.0
                            jitter=0.0

                        elif self.data[self.instruments[i]][j]['fix_offset']:
                            offset=1.0
                            res=optimize.minimize(nbspectra.fit_only_jitter,2*np.mean(self.data[self.instruments[i]][j]['yerr']), args=(self.results[j],self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                            jitter=res.x[0]                           
                        
                        elif self.data[self.instruments[i]][j]['fix_jitter']:
                            jitter=0.0
                            res=optimize.minimize(nbspectra.fit_only_multiplicative_offset,np.mean(self.data[self.instruments[i]][j]['y'])/(np.mean(self.results[j])+0.0001), args=(self.results[j],self.data[self.instruments[i]][j]['y'],np.sqrt(self.data[self.instruments[i]][j]['yerr']**2+jitter**2)), method='Nelder-Mead')
                            offset=res.x[0]

                        else:
                            res=optimize.minimize(nbspectra.fit_multiplicative_offset_jitter,[np.mean(self.data[self.instruments[i]][j]['y'])/(np.mean(self.results[j])+0.0001),2*np.mean(self.data[self.instruments[i]][j]['yerr'])], args=(self.results[j],self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                            offset=res.x[0]
                            jitter=res.x[1]

                        self.compute_forward(observables=j,t=t,inversion=True)
                        if k==bestlnL:
                            ax[l].plot(t,self.results[j]*offset,'r--',zorder=11,label='Offset={:.5f}, Jitter={:.5f}'.format(offset,jitter))
                            ax[l].errorbar(self.data[self.instruments[i]][j]['t'],self.data[self.instruments[i]][j]['y'],np.sqrt(self.data[self.instruments[i]][j]['yerr']**2+jitter**2),fmt='bo',ecolor='lightblue',zorder=10)                        
                            ax[l].set_ylabel('{}_{}'.format(self.instruments[i],j))
                            ax[l].legend()
                        
                        store_results[k,l,:]=self.results[j]*offset
                        # ax[l].plot(t,self.results[j]*offset,c=cmap.to_rgba(lnLs[k]),alpha=0.5)
                        l+=1
                    

                    else: #linear offset

                        if (self.data[self.instruments[i]][j]['fix_jitter'] and self.data[self.instruments[i]][j]['fix_offset']):
                            offset=0.0
                            jitter=0.0

                        elif self.data[self.instruments[i]][j]['fix_offset']:
                            offset=0.0
                            res=optimize.minimize(nbspectra.fit_only_jitter,2*np.mean(self.data[self.instruments[i]][j]['yerr']), args=(self.results[j],self.data[self.instruments[i]][j]['y']-offset,self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                            jitter=res.x[0]
                        
                        elif self.data[self.instruments[i]][j]['fix_jitter']:
                            jitter=0.0
                            res=optimize.minimize(nbspectra.fit_only_linear_offset,np.mean(self.data[self.instruments[i]][j]['y'])-np.mean(self.results[j]), args=(self.results[j],self.data[self.instruments[i]][j]['y'],np.sqrt(jitter**2+self.data[self.instruments[i]][j]['yerr']**2)), method='Nelder-Mead')
                            offset=res.x[0]

                        else:
                            res=optimize.minimize(nbspectra.fit_linear_offset_jitter,[np.mean(self.data[self.instruments[i]][j]['y'])-np.mean(self.results[j]),2*np.mean(self.data[self.instruments[i]][j]['yerr'])], args=(self.results[j],self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                            offset=res.x[0]
                            jitter=res.x[1]

                        self.compute_forward(observables=j,t=t,inversion=True)
                        store_results[k,l,:]=self.results[j]+offset
                        if k==bestlnL:
                            ax[l].plot(t,self.results[j]+offset,'r--',zorder=11,label='Offset={:.5f}, Jitter={:.5f}'.format(offset,jitter))
                            ax[l].errorbar(self.data[self.instruments[i]][j]['t'],self.data[self.instruments[i]][j]['y'],np.sqrt(self.data[self.instruments[i]][j]['yerr']**2+jitter**2),fmt='bo',ecolor='lightblue',zorder=10)
                            ax[l].set_ylabel('{}_{}'.format(self.instruments[i],j))
                            ax[l].legend()
                        l+=1

        for i in range(num_obs):
            ax[i].plot(t,np.mean(store_results[:,i],axis=0),'k')
            ax[i].fill_between(t,np.mean(store_results[:,i],axis=0)-np.std(store_results[:,i],axis=0),np.mean(store_results[:,i],axis=0)+np.std(store_results[:,i],axis=0),color='k',alpha=0.2)
            



        ofilename = self.path  / 'plots' / 'inversion_timeseries_result.png'
        plt.savefig(ofilename,dpi=200)
        # plt.show(block=True)
        plt.close()

    
    def plot_spot_map(self,best_maps,tref=None):

        N_div = 100
        Ngrids, Ngrid_in_ring, centres, amu, rs, alphas, xs, ys, zs, are, pare = nbspectra.generate_grid_coordinates_nb(N_div)
        vec_grid = np.array([xs,ys,zs]).T #coordinates in cartesian
        theta, phi = np.arccos(zs*np.cos(-self.inclination)-xs*np.sin(-self.inclination)), np.arctan2(ys,xs*np.cos(-self.inclination)+zs*np.sin(-self.inclination))#coordinates in the star reference 

        

        if tref is None:
            tref=best_maps[0][0][0]
        elif len(tref)==1:
            tref=[tref]

        for t in tref:
            Surface=np.zeros(len(vec_grid[:,0])) #len Ngrids
            for k in range(len(best_maps)):

                self.spot_map[:,0:7]=best_maps[k]
                spot_pos=spectra.compute_spot_position(self,t) #return colat, longitude and raddii in radians
             
                vec_spot=np.zeros([len(self.spot_map),3])
                xspot = np.cos(self.inclination)*np.sin(spot_pos[:,0])*np.cos(spot_pos[:,1])+np.sin(self.inclination)*np.cos(spot_pos[:,0])
                yspot = np.sin(spot_pos[:,0])*np.sin(spot_pos[:,1])
                zspot = np.cos(spot_pos[:,0])*np.cos(self.inclination)-np.sin(self.inclination)*np.sin(spot_pos[:,0])*np.cos(spot_pos[:,1])
                vec_spot[:,:]=np.array([xspot,yspot,zspot]).T #spot center in cartesian
                
                for s in range(len(best_maps[k])):
                    if spot_pos[s,2]==0:
                        continue

                    for i in range(len(vec_grid[:,0])):
                        dist=m.acos(np.dot(vec_spot[s],vec_grid[i]))
                        if dist < spot_pos[s,2]:
                            Surface[i]+=1

                    

            cm = plt.cm.get_cmap('afmhot_r')
            #make figure
            fig = plt.figure(1,figsize=(6,6))
            plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
            ax = fig.add_subplot(111)
            ax.axis('off')
            ax.plot(np.cos(np.linspace(0,2*np.pi,100)),np.sin(np.linspace(0,2*np.pi,100)),'k') #circumference
            x=np.linspace(-0.999,0.999,1000)
            h=np.sqrt((1-x**2)/(np.tan(self.inclination)**2+1))
            plt.plot(x,h,'k--')
            spotmap = ax.scatter(vec_grid[:,1],vec_grid[:,2], marker='o', c=Surface/len(best_maps), s=5.0, edgecolors='none', cmap=cm,vmax=(Surface.max()+0.1)/len(best_maps),vmin=-0.2*(Surface.max()+0.1)/len(best_maps))
            # cb = plt.colorbar(spotmap,ax=ax, fraction=0.035, pad=0.05, aspect=20)
            ofilename = self.path  / 'plots' / 'inversion_spotmap_t_{:.4f}.png'.format(t)
            plt.savefig(ofilename,dpi=200)
            # plt.show()
            plt.close()

    def plot_active_longitudes(self,best_maps,tini=None,tfin=None,N_obs=100):

        N_div = 500
        

        if tini is None:
            tini=best_maps[0][0][0]
        if tfin is None:
            tfin=best_maps[0][0][0]+1.0

        tref=np.linspace(tini,tfin,N_obs)
        longs=np.linspace(0,2*np.pi,N_div)
        Surface=np.zeros([N_obs,N_div])
        for j in range(N_obs):
            for k in range(len(best_maps)):
                self.spot_map[:,0:7]=best_maps[k]
                spot_pos=spectra.compute_spot_position(self,tref[j]) #return colat, longitude and raddii in radians

        #update longitude adding diff rotation
                for s in range(len(best_maps[k])):
                    ph_s=(spot_pos[s,1]-((tref[j]-self.reference_time)/self.rotation_period%1*360)*np.pi/180)%(2*np.pi) #longitude
                    r_s=spot_pos[s,2] #radius
                    if r_s==0.0:
                        continue

                    for i in range(N_div):
                        dist=np.abs(longs[i]-ph_s) #distance to spot centre
                        
                        if dist < r_s:
                            Surface[j,i]+=1

        X,Y = np.meshgrid(longs*180/np.pi,tref)
        fig = plt.figure(1,figsize=(6,6))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.98, bottom=0.1)
        ax = fig.add_subplot(111)
        cm = plt.cm.get_cmap('afmhot_r')
        spotmap=ax.contourf(X,Y,Surface/len(best_maps), 25, cmap=cm,vmax=(Surface.max()+0.1)/len(best_maps),vmin=-0.2*(Surface.max()+0.1)/len(best_maps))
        cb = plt.colorbar(spotmap,ax=ax)   
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Time [d]")

        ofilename = self.path / 'plots' / 'active_map.png'
        plt.savefig(ofilename,dpi=200)
        # plt.show()
        plt.close()

    def plot_optimize_inversion_SA_results(self,DeltalnL):

        fixed_T = self.temperature_photosphere
        fixed_sp_T = self.spot_T_contrast
        fixed_fc_T = self.facula_T_contrast
        fixed_Q = self.facular_area_ratio
        fixed_CB = self.convective_shift
        fixed_Prot = self.rotation_period
        fixed_inc = np.rad2deg(np.pi/2-self.inclination) 
        fixed_R = self.radius
        fixed_LD1 = self.limb_darkening_q1
        fixed_LD2 = self.limb_darkening_q2
        fixed_Pp = self.planet_period
        fixed_T0p = self.planet_transit_t0
        fixed_Kp = self.planet_semi_amplitude
        fixed_esinwp = self.planet_esinw
        fixed_ecoswp = self.planet_ecosw
        fixed_Rp =  self.planet_radius
        fixed_bp = self.planet_impact_param
        fixed_alp =  self.planet_spin_orbit_angle    #spin-orbit angle

        self.vparam=np.array([fixed_T,fixed_sp_T,fixed_fc_T,fixed_Q,fixed_CB,fixed_Prot,fixed_inc,fixed_R,fixed_LD1,fixed_LD2,fixed_Pp,fixed_T0p,fixed_Kp,fixed_esinwp,fixed_ecoswp,fixed_Rp,fixed_bp,fixed_alp])

        name_T ='T$_{{eff}}$'
        name_sp_T ='$\\Delta$ T$_{{sp}}$'
        name_fc_T ='$\\Delta$ T$_{{fc}}$'
        name_Q ='Fac-spot ratio'
        name_CB ='CS'
        name_Prot ='P$_{{rot}}$'
        name_inc ='inc'
        name_R ='R$_*$'
        name_LD1 = 'q$_1$'
        name_LD2 = 'q$_2$'
        name_Pp = 'P$_{{pl}}$'
        name_T0p = 'T$_{{0,pl}}$'
        name_Kp = 'K$_{{pl}}$'
        name_esinwp = 'esinw'
        name_ecoswp = 'ecosw'
        name_Rp =  'R$_{{pl}}$'
        name_bp = 'b'
        name_alp = '$\\lambda$'  

        self.lparam=np.array([name_T,name_sp_T,name_fc_T,name_Q,name_CB,name_Prot,name_inc,name_R,name_LD1,name_LD2,name_Pp,name_T0p,name_Kp,name_esinwp,name_ecoswp,name_Rp,name_bp,name_alp])

        f_T = self.prior_t_eff_ph[0]
        f_sp_T = self.prior_spot_T_contrast[0] 
        f_fc_T = self.prior_facula_T_contrast[0] 
        f_Q = self.prior_q_ratio[0]   
        f_CB = self.prior_convective_blueshift[0]   
        f_Prot = self.prior_p_rot[0] 
        f_inc = self.prior_inclination[0]   
        f_R = self.prior_Rstar[0]
        f_LD1 = self.prior_LD1[0]
        f_LD2 = self.prior_LD2[0]
        f_Pp = self.prior_Pp[0]
        f_T0p = self.prior_T0p[0]
        f_Kp = self.prior_Kp[0]
        f_esinwp = self.prior_esinwp[0]
        f_ecoswp = self.prior_ecoswp[0]
        f_Rp = self.prior_Rp[0]
        f_bp = self.prior_bp[0]
        f_alp = self.prior_alp[0]       
        self.fit=np.array([f_T,f_sp_T,f_fc_T,f_Q,f_CB,f_Prot,f_inc,f_R,f_LD1,f_LD2,f_Pp,f_T0p,f_Kp,f_esinwp,f_ecoswp,f_Rp,f_bp,f_alp])       

        self.lparamfit=np.array([])
        for i in range(len(self.fit)):
          if self.fit[i]==1:
            self.lparamfit=np.append(self.lparamfit,self.lparam[i])


        #read the results
        filename = self.path / 'results' / 'optimize_inversion_SA_stats.npy'
        res = np.load(filename,allow_pickle=True)

        lnLs=res[0]
        params=np.vstack(res[1]).T
        best_maps=res[2]
        ndim=np.sum(self.fit)

        p=np.zeros([ndim,len(lnLs)])
        # print(P)
        ii=0
        for i in range(len(self.fit)):
          if self.fit[i]==1:
            p[ii,:]=params[i,:]
            ii+=1


        
        fig1 = plt.figure(figsize=(15,12))
        
        for ip in range(ndim):
          plt.subplot(m.ceil(ndim/4),4,ip+1)
          plt.plot(p[ip],lnLs,'k.')
          plt.axhline(np.max(lnLs)-DeltalnL,color='r',ls=':')
          plt.ylabel('lnL')
          # left=np.min(xtot[ytot>(np.max(ytot)-15),ip])
          # right=np.max(xtot[ytot>(np.max(ytot)-15),ip])
          plt.ylim([np.max(lnLs)-DeltalnL*3,np.max(lnLs)+DeltalnL/10])
          # plt.xlim([left-(right-left)*0.2,right+(right-left)*0.2])
          plt.xlabel(self.lparamfit[ip])
        
        ofilename = self.path / 'plots' / 'inversion_MCMCSA_likelihoods.png'
        plt.savefig(ofilename,dpi=200)
        #plt.show(block=True)
        plt.close()

        paramsnew=res[1][lnLs>(np.nanmax(lnLs)-DeltalnL)]
        pcorner=p[:,lnLs>(np.nanmax(lnLs)-DeltalnL)]
        lnLsnew=lnLs[lnLs>(np.nanmax(lnLs)-DeltalnL)]
        best_maps_new=best_maps[lnLs>(np.nanmax(lnLs)-DeltalnL)]


        fig2, axes = plt.subplots(int(ndim),int(ndim), figsize=(2.3*int(ndim),2.3*int(ndim)))
        corner.corner(pcorner.T,bins=10,plot_contours=False,fig=fig2,max_n_ticks=2,labels=self.lparamfit,label_kwargs={'fontsize':13},quantiles=(0.16,0.5,0.84),show_titles=True)
        
        ofilename = self.path / 'plots' / 'inversion_MCMCSA_cornerplot.png'
        plt.savefig(ofilename,dpi=200)
        #plt.show(block=True)
        plt.close()


        param_inv=[]
        # print(P)
        ii=0
        for i in range(len(self.fit)):
          if self.fit[i]==0:
            param_inv.append(np.array(self.vparam[i]))
          elif self.fit[i]==1:
            param_inv.append(np.array(pcorner[ii]))
            ii=ii+1
        
        vsini_inv= 2*np.pi*(param_inv[7]*696342)*np.cos(np.deg2rad(90-param_inv[6]))/(param_inv[5]*86400) #in km/s
        if self.limb_darkening_law == 'linear':
            a_LD=param_inv[8]
            b_LD=param_inv[8]*0
        elif self.limb_darkening_law == 'quadratic':
            a_LD=2*np.sqrt(param_inv[8])*param_inv[9]
            b_LD=np.sqrt(param_inv[8])*(1-2*param_inv[9])
        elif self.limb_darkening_law == 'sqrt':
            a_LD=np.sqrt(param_inv[8])*(1-2*param_inv[9]) 
            b_LD=2*np.sqrt(param_inv[8])*param_inv[9]
        elif self.limb_darkening_law == 'log':
            a_LD=param_inv[9]*param_inv[8]**2+1
            b_LD=param_inv[8]**2-1


        s='Results of the inversion process with DeltalnL<{:.1f} \n'.format(DeltalnL)
        print('Results of the inversion process:')
        s+='    -Mean and 1 sigma confidence interval:\n'
        print('\t -Mean and 1 sigma confidence interval:')
        for ip in range(len(self.vparam)):
          if self.fit[ip]==1:
            s+='        {} = {:.5f}+{:.5f}-{:.5f}\n'.format(self.lparam[ip],np.median(param_inv[ip]),np.quantile(param_inv[ip],0.84135)-np.median(param_inv[ip]),np.median(param_inv[ip])-np.quantile(param_inv[ip],0.15865))
            print('\t \t {} = {:.5f}+{:.5f}-{:.5f}'.format(self.lparam[ip],np.median(param_inv[ip]),np.quantile(param_inv[ip],0.84135)-np.median(param_inv[ip]),np.median(param_inv[ip])-np.quantile(param_inv[ip],0.15865)))
          else:
            s+='        {} = {:.5f} (fixed)\n'.format(self.lparam[ip],self.vparam[ip])
            print('\t \t',self.lparam[ip],' = ',self.vparam[ip],'(fixed) ') 
        s+='        $vsini$ = {:.5f}+{:.5f}-{:.5f}\n'.format(np.median(vsini_inv),np.quantile(vsini_inv,0.84135)-np.median(vsini_inv),np.median(vsini_inv)-np.quantile(vsini_inv,0.15865))
        s+='        LD_a = {:.5f}+{:.5f}-{:.5f}\n'.format(np.median(a_LD),np.quantile(a_LD,0.84135)-np.median(a_LD),np.median(a_LD)-np.quantile(a_LD,0.15865))
        s+='        LD_b = {:.5f}+{:.5f}-{:.5f}\n'.format(np.median(b_LD),np.quantile(b_LD,0.84135)-np.median(b_LD),np.median(b_LD)-np.quantile(b_LD,0.15865)) 

        s+='    -Mean and standard deviation:\n'
        print('\t -Mean and standard deviation:')
        for ip in range(len(self.vparam)):
          if self.fit[ip]==1:
            s+='        {} = {:.5f}+-{:.5f}\n'.format(self.lparam[ip],np.median(param_inv[ip]),np.std(param_inv[ip]))
            print('\t \t {} = {:.5f}+-{:.5f}'.format(self.lparam[ip],np.median(param_inv[ip]),np.std(param_inv[ip])))
        s+='    -Best solution, with maximum log-likelihood of {:.5f}\n'.format(np.max(lnLsnew))
        print('\t -Best solution, with maximum log-likelihood of',np.max(lnLsnew))  
        for ip in range(len(self.vparam)):
          if self.fit[ip]==1:
            s+='        {} = {:.5f}\n'.format(self.lparam[ip],param_inv[ip][np.argmax(lnLsnew)])
            print('\t \t {} = {:.5f}'.format(self.lparam[ip],param_inv[ip][np.argmax(lnLsnew)]))

        fig = plt.figure(figsize=(6,10))
        plt.annotate(s, xy=(0.0, 1.0),ha='left',va='top')
        plt.axis('off')
        plt.tight_layout()
        ofilename = self.path / 'plots' / 'inversion_MCMCSA_results.png'
        plt.savefig(ofilename,dpi=200)
        plt.close()



        self.instruments=[]
        self.observables=[]
        tmax=-3000000
        tmin=3000000
        Npoints=1000

        for ins in self.data.keys():
            self.instruments.append(ins)
            o=[]
            ty=[]
            for obs in self.data[ins].keys():
                if obs in ['lc']:
                    o.append(obs)
                    ty.append(0)
                    if self.data[ins]['lc']['t'].min()<tmin: tmin=self.data[ins]['lc']['t'].min()
                    if self.data[ins]['lc']['t'].max()>tmax: tmax=self.data[ins]['lc']['t'].max()
                elif obs in ['rv','fwhm','bis','contrast']:
                    o.append(obs)
                    ty.append(1)
                    if self.data[ins][obs]['t'].min()<tmin: tmin=self.data[ins][obs]['t'].min()
                    if self.data[ins][obs]['t'].max()>tmax: tmax=self.data[ins][obs]['t'].max()
                if obs in ['crx']:
                    o.append(obs)
                    ty.append(2)
                    if self.data[ins]['crx']['t'].min()<tmin: tmin=self.data[ins]['crx']['t'].min()
                    if self.data[ins]['crx']['t'].max()>tmax: tmax=self.data[ins]['crx']['t'].max()
            self.observables.append(o)

        num_obs=len(sum(self.observables,[]))

        t=np.linspace(tmin,tmax,Npoints)

        

        fig, ax = plt.subplots(num_obs,1,figsize=(12,12))
        if num_obs == 1:
            ax= [ax]

        store_results = np.zeros([len(best_maps_new),num_obs,Npoints])


        bestlnL=np.argmax(lnLsnew)
        for k in range(len(best_maps_new)):
            self.spot_map = best_maps_new[k]
            self.temperature_photosphere = paramsnew[k][0]
            self.spot_T_contrast = paramsnew[k][1]
            self.facula_T_contrast = paramsnew[k][2]
            self.facular_area_ratio = paramsnew[k][3]
            self.convective_shift = paramsnew[k][4]
            self.rotation_period = paramsnew[k][5]
            self.inclination = np.deg2rad(90-paramsnew[k][6]) #axis inclinations in rad (inc=0 has the axis pointing up). The input was in deg defined as usual.
            self.radius = paramsnew[k][7] #in Rsun
            self.limb_darkening_q1 = paramsnew[k][8]
            self.limb_darkening_q2 = paramsnew[k][9]
            self.planet_period = paramsnew[k][10]
            self.planet_transit_t0 = paramsnew[k][11]
            self.planet_semi_amplitude = paramsnew[k][12]
            self.planet_esinw = paramsnew[k][13]
            self.planet_ecosw = paramsnew[k][14]
            self.planet_radius = paramsnew[k][15]
            self.planet_impact_param = paramsnew[k][16]
            self.planet_spin_orbit_angle = paramsnew[k][17]*np.pi/180 #deg2rad   

            #Plot the data
            l=0
            for i in range(len(self.instruments)):
                for j in self.observables[i]:
                    self.wavelength_lower_limit=self.data[self.instruments[i]]['wvmin']
                    self.wavelength_upper_limit=self.data[self.instruments[i]]['wvmax']
                    self.filter_name=self.data[self.instruments[i]]['filter']
                    self.compute_forward(observables=j,t=self.data[self.instruments[i]][j]['t'],inversion=True)

                    if self.data[self.instruments[i]][j]['offset_type']=='multiplicative':
                        
                        if (self.data[self.instruments[i]][j]['fix_jitter'] and self.data[self.instruments[i]][j]['fix_offset']):
                            offset=1.0
                            jitter=0.0

                        elif self.data[self.instruments[i]][j]['fix_offset']:
                            offset=1.0
                            res=optimize.minimize(nbspectra.fit_only_jitter,2*np.mean(self.data[self.instruments[i]][j]['yerr']), args=(self.results[j],self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                            jitter=res.x[0]                           
                        
                        elif self.data[self.instruments[i]][j]['fix_jitter']:
                            jitter=0.0
                            res=optimize.minimize(nbspectra.fit_only_multiplicative_offset,np.mean(self.data[self.instruments[i]][j]['y'])/(np.mean(self.results[j])+0.0001), args=(self.results[j],self.data[self.instruments[i]][j]['y'],np.sqrt(self.data[self.instruments[i]][j]['yerr']**2+jitter**2)), method='Nelder-Mead')
                            offset=res.x[0]

                        else:
                            res=optimize.minimize(nbspectra.fit_multiplicative_offset_jitter,[np.mean(self.data[self.instruments[i]][j]['y'])/(np.mean(self.results[j])+0.0001),2*np.mean(self.data[self.instruments[i]][j]['yerr'])], args=(self.results[j],self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                            offset=res.x[0]
                            jitter=res.x[1]

                        self.compute_forward(observables=j,t=t,inversion=True)
                        if k==bestlnL:
                            ax[l].plot(t,self.results[j]*offset,'r--',zorder=11,label='Offset={:.5f}, Jitter={:.5f}'.format(offset,jitter))
                            ax[l].errorbar(self.data[self.instruments[i]][j]['t'],self.data[self.instruments[i]][j]['y'],np.sqrt(self.data[self.instruments[i]][j]['yerr']**2+jitter**2),fmt='bo',ecolor='lightblue',zorder=10)                        
                            ax[l].set_ylabel('{}_{}'.format(self.instruments[i],j))
                            ax[l].legend()
                        store_results[k,l,:]=self.results[j]*offset
                        l+=1
                    

                    else: #linear offset

                        if (self.data[self.instruments[i]][j]['fix_jitter'] and self.data[self.instruments[i]][j]['fix_offset']):
                            offset=0.0
                            jitter=0.0

                        elif self.data[self.instruments[i]][j]['fix_offset']:
                            offset=0.0
                            res=optimize.minimize(nbspectra.fit_only_jitter,2*np.mean(self.data[self.instruments[i]][j]['yerr']), args=(self.results[j],self.data[self.instruments[i]][j]['y']-offset,self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                            jitter=res.x[0]
                        
                        elif self.data[self.instruments[i]][j]['fix_jitter']:
                            jitter=0.0
                            res=optimize.minimize(nbspectra.fit_only_linear_offset,np.mean(self.data[self.instruments[i]][j]['y'])-np.mean(self.results[j]), args=(self.results[j],self.data[self.instruments[i]][j]['y'],np.sqrt(jitter**2+self.data[self.instruments[i]][j]['yerr']**2)), method='Nelder-Mead')
                            offset=res.x[0]

                        else:
                            res=optimize.minimize(nbspectra.fit_linear_offset_jitter,[np.mean(self.data[self.instruments[i]][j]['y'])-np.mean(self.results[j]),2*np.mean(self.data[self.instruments[i]][j]['yerr'])], args=(self.results[j],self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                            offset=res.x[0]
                            jitter=res.x[1]

                        self.compute_forward(observables=j,t=t,inversion=True)
                        store_results[k,l,:]=self.results[j]+offset
                        if k==bestlnL:
                            ax[l].plot(t,self.results[j]+offset,'r--',zorder=11,label='Offset={:.5f}, Jitter={:.5f}'.format(offset,jitter))
                            ax[l].errorbar(self.data[self.instruments[i]][j]['t'],self.data[self.instruments[i]][j]['y'],np.sqrt(self.data[self.instruments[i]][j]['yerr']**2+jitter**2),fmt='bo',ecolor='lightblue',zorder=10)
                            ax[l].set_ylabel('{}_{}'.format(self.instruments[i],j))
                            ax[l].legend()
                        l+=1

        for i in range(num_obs):
            ax[i].plot(t,np.mean(store_results[:,i],axis=0),'k')
            ax[i].fill_between(t,np.mean(store_results[:,i],axis=0)-np.std(store_results[:,i],axis=0),np.mean(store_results[:,i],axis=0)+np.std(store_results[:,i],axis=0),color='k',alpha=0.2)
            


        ofilename = self.path  / 'plots' / 'inversion_timeseries_result.png'
        plt.savefig(ofilename,dpi=200)
        # plt.show(block=True)
        plt.close()


    def plot_data_and_model(self,spot_map,stellar_params,Npoints=200):


        #set spot map and stellar params
        self.spot_map = spot_map
        self.set_stellar_parameters(stellar_params)


        self.instruments=[]
        self.observables=[]
        typ=[]
        tmax=-3000000
        tmin=3000000

        for ins in self.data.keys():
            self.instruments.append(ins)
            o=[]
            ty=[]
            for obs in self.data[ins].keys():
                if obs in ['lc']:
                    o.append(obs)
                    ty.append(0)
                    if self.data[ins]['lc']['t'].min()<tmin: tmin=self.data[ins]['lc']['t'].min()
                    if self.data[ins]['lc']['t'].max()>tmax: tmax=self.data[ins]['lc']['t'].max()
                elif obs in ['rv','fwhm','bis','contrast']:
                    o.append(obs)
                    ty.append(1)
                    if self.data[ins][obs]['t'].min()<tmin: tmin=self.data[ins][obs]['t'].min()
                    if self.data[ins][obs]['t'].max()>tmax: tmax=self.data[ins][obs]['t'].max()
                if obs in ['crx']:
                    o.append(obs)
                    ty.append(2)
                    if self.data[ins]['crx']['t'].min()<tmin: tmin=self.data[ins]['crx']['t'].min()
                    if self.data[ins]['crx']['t'].max()>tmax: tmax=self.data[ins]['crx']['t'].max()
            self.observables.append(o)

        num_obs=len(sum(self.observables,[]))

        t=np.linspace(tmin,tmax,Npoints)

        fig, ax = plt.subplots(num_obs,1,figsize=(12,12))
        if num_obs == 1:
            ax= [ax]





                    
        #Plot the data
        l=0
        for i in range(len(self.instruments)):
            for j in self.observables[i]:
                self.wavelength_lower_limit=self.data[self.instruments[i]]['wvmin']
                self.wavelength_upper_limit=self.data[self.instruments[i]]['wvmax']
                self.filter_name=self.data[self.instruments[i]]['filter']
                self.compute_forward(observables=j,t=self.data[self.instruments[i]][j]['t'],inversion=True)

                if self.data[self.instruments[i]][j]['offset_type']=='multiplicative':
                    
                    if (self.data[self.instruments[i]][j]['fix_jitter'] and self.data[self.instruments[i]][j]['fix_offset']):
                        offset=1.0
                        jitter=0.0

                    elif self.data[self.instruments[i]][j]['fix_offset']:
                        offset=1.0
                        res=optimize.minimize(nbspectra.fit_only_jitter,2*np.mean(self.data[self.instruments[i]][j]['yerr']), args=(self.results[j],self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                        jitter=res.x[0]                           
                    
                    elif self.data[self.instruments[i]][j]['fix_jitter']:
                        jitter=0.0
                        res=optimize.minimize(nbspectra.fit_only_multiplicative_offset,np.mean(self.data[self.instruments[i]][j]['y'])/(np.mean(self.results[j])+0.0001), args=(self.results[j],self.data[self.instruments[i]][j]['y'],np.sqrt(self.data[self.instruments[i]][j]['yerr']**2+jitter**2)), method='Nelder-Mead')
                        offset=res.x[0]

                    else:
                        res=optimize.minimize(nbspectra.fit_multiplicative_offset_jitter,[np.mean(self.data[self.instruments[i]][j]['y'])/(np.mean(self.results[j])+0.0001),2*np.mean(self.data[self.instruments[i]][j]['yerr'])], args=(self.results[j],self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                        offset=res.x[0]
                        jitter=res.x[1]

                    self.compute_forward(observables=j,t=t,inversion=True)
                    
                    ax[l].plot(t,self.results[j]*offset,'k',zorder=11,label='Offset={:.5f}, Jitter={:.5f}'.format(offset,jitter))
                    ax[l].errorbar(self.data[self.instruments[i]][j]['t'],self.data[self.instruments[i]][j]['y'],np.sqrt(self.data[self.instruments[i]][j]['yerr']**2+jitter**2),fmt='bo',ecolor='lightblue',zorder=10)                        
                    ax[l].set_ylabel('{}_{}'.format(self.instruments[i],j))
                    ax[l].legend()
                    

                else: #linear offset

                    if (self.data[self.instruments[i]][j]['fix_jitter'] and self.data[self.instruments[i]][j]['fix_offset']):
                        offset=0.0
                        jitter=0.0

                    elif self.data[self.instruments[i]][j]['fix_offset']:
                        offset=0.0
                        res=optimize.minimize(nbspectra.fit_only_jitter,2*np.mean(self.data[self.instruments[i]][j]['yerr']), args=(self.results[j],self.data[self.instruments[i]][j]['y']-offset,self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                        jitter=res.x[0]
                    
                    elif self.data[self.instruments[i]][j]['fix_jitter']:
                        jitter=0.0
                        res=optimize.minimize(nbspectra.fit_only_linear_offset,np.mean(self.data[self.instruments[i]][j]['y'])-np.mean(self.results[j]), args=(self.results[j],self.data[self.instruments[i]][j]['y'],np.sqrt(jitter**2+self.data[self.instruments[i]][j]['yerr']**2)), method='Nelder-Mead')
                        offset=res.x[0]

                    else:
                        res=optimize.minimize(nbspectra.fit_linear_offset_jitter,[np.mean(self.data[self.instruments[i]][j]['y'])-np.mean(self.results[j]),2*np.mean(self.data[self.instruments[i]][j]['yerr'])], args=(self.results[j],self.data[self.instruments[i]][j]['y'],self.data[self.instruments[i]][j]['yerr']), method='Nelder-Mead')
                        offset=res.x[0]
                        jitter=res.x[1]

                    self.compute_forward(observables=j,t=t,inversion=True)

                    ax[l].plot(t,self.results[j]+offset,'k',zorder=11,label='Offset={:.5f}, Jitter={:.5f}'.format(offset,jitter))
                    ax[l].errorbar(self.data[self.instruments[i]][j]['t'],self.data[self.instruments[i]][j]['y'],np.sqrt(self.data[self.instruments[i]][j]['yerr']**2+jitter**2),fmt='bo',ecolor='lightblue',zorder=10)
                    ax[l].set_ylabel('{}_{}'.format(self.instruments[i],j))
                    ax[l].legend()

                l+=1




        ofilename = self.path  / 'plots' / 'data_and_model.png'
        plt.savefig(ofilename,dpi=200)
        # plt.show(block=True)
        plt.close()