#NUMBA ############################################
import numba as nb
import numpy as np
import math as m

@nb.njit
def dummy():
    return None

@nb.njit(cache=True,error_model='numpy')
def fit_multiplicative_offset_jitter(x0,f,y,dy):
    off=x0[0]
    jit=x0[1]
    newerr=np.sqrt((dy)**2+jit**2)/off
    lnL=-0.5*np.sum(((y/off-f)/(newerr))**2.0+np.log(2.0*np.pi)+np.log(newerr**2))
    return -lnL

@nb.njit(cache=True,error_model='numpy')
def fit_only_multiplicative_offset(x0,f,y,dy):
    off=x0
    lnL=-0.5*np.sum(((y/off-f)/(dy/off))**2.0+np.log(2.0*np.pi)+np.log((dy/off)**2))
    return -lnL

@nb.njit(cache=True,error_model='numpy')
def fit_linear_offset_jitter(x0,f,y,dy):
    off=x0[0]
    jit=x0[1]
    lnL=-0.5*np.sum(((y-off-f)/(np.sqrt(dy**2+jit**2)))**2.0+np.log(2.0*np.pi)+np.log(dy**2+jit**2))
    return -lnL

@nb.njit(cache=True,error_model='numpy')
def fit_only_linear_offset(x0,f,y,dy):
    off=x0
    lnL=-0.5*np.sum(((y-off-f)/(dy))**2.0+np.log(2.0*np.pi)+np.log(dy**2))
    return -lnL

@nb.njit(cache=True,error_model='numpy')
def fit_only_jitter(x0,f,y,dy):
    jit=x0
    lnL=-0.5*np.sum(((y-f)/(np.sqrt(dy**2+jit**2)))**2.0+np.log(2.0*np.pi)+np.log(dy**2+jit**2))
    return -lnL


@nb.njit(cache=True,error_model='numpy')
def _coeff_mat(x, deg):
    mat_ = np.zeros(shape=(x.shape[0],deg + 1))
    const = np.ones_like(x)
    mat_[:,0] = const
    mat_[:, 1] = x
    if deg > 1:
        for n in range(2, deg + 1):
            mat_[:, n] = x**n
    return mat_
    
@nb.njit(cache=True,error_model='numpy')
def _fit_x(a, b):
    # linalg solves ax = b
    det_ = np.linalg.lstsq(a, b)[0]
    return det_
 
@nb.njit(cache=True,error_model='numpy')
def fit_poly(x, y, deg,w):
    a = _coeff_mat(x, deg)*w.reshape(-1,1)
    p = _fit_x(a, y*w)
    # Reverse order so p[0] is coefficient of highest order
    return p[::-1]
#####################################################
############# UTILITIES #############################
#####################################################
@nb.njit(cache=True,error_model='numpy')
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-(x-mean)**2/(2*stddev**2))

@nb.njit(cache=True,error_model='numpy')
def gaussian2(x, amplitude, mean, stddev,C):
    return C + amplitude * np.exp(-(x-mean)**2/(2*stddev**2))


@nb.njit(cache=True,error_model='numpy')
def normalize_spectra_nb(bins,wavelength,flux):

    x_bin=np.zeros(len(bins)-1)
    y_bin=np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
        idxup = wavelength>bins[i]
        idxdown= wavelength<bins[i+1]
        idx=idxup & idxdown
        y_bin[i]=flux[idx].max()
        x_bin[i]=wavelength[idx][np.argmax(flux[idx])]
    #divide by 6th deg polynomial

    return x_bin, y_bin


@nb.njit(cache=True,error_model='numpy')
def interpolation_nb(xp,x,y,left=0,right=0):

    # Create result array
    yp=np.zeros(len(xp))
    minx=x[0]
    maxx=x[-1]
    lastidx=1 

    for i,xi in enumerate(xp):
        if xi<minx: #extrapolate left
            yp[i]=left
        elif xi>maxx: #extrapolate right
            yp[i]=right
        else:
            for j in range(lastidx,len(x)): #per no fer el loop sobre tota la x, ja que esta sorted sempre comenso amb lanterior.
                if x[j]>xi:
                    #Trobo el primer x mes gran que xj. llavors utilitzo x[j] i x[j-1] per interpolar
                    yp[i]=y[j-1]+(xi-x[j-1])*(y[j]-y[j-1])/(x[j]-x[j-1])
                    lastidx=j
                    break

    return yp


@nb.njit(cache=True,error_model='numpy')
def cross_correlation_nb(rv,wv,flx,wv_ref,flx_ref):
    #Compute the CCF against the reference spectrum. Can be optimized.
    ccf=np.zeros(len(rv)) #initialize ccf
    lenf=len(flx_ref)
    for i in range(len(rv)):
        wvshift=wv_ref*(1.0+rv[i]/2.99792458e8) #shift ref spectrum, in m/s
        # fshift=np.interp(wvshift,wv,flx)
        fshift = interpolation_nb(wvshift,wv,flx,left=0,right=0)
        ccf[i]=np.sum(flx_ref*fshift)/lenf #compute ccf

    return (ccf-np.min(ccf))/np.max((ccf-np.min(ccf)))


@nb.njit(cache=True,error_model='numpy')
def cross_correlation_mask(rv,wv,f,wvm,fm):
    ccf = np.zeros(len(rv))
    lenm = len(wvm)
    wvmin=wv[0]

    for i in range(len(rv)):
        wvshift=wvm*(1.0+rv[i]/2.99792458e8) #shift ref spectrum, in m/s
        #for each mask line
        for j in range(lenm):
            #find wavelengths right and left of the line.
            wvline=wvshift[j]

            if wvline<3000.0:
                idxlf = int((wvline-wvmin)/0.1)

            elif wvline<4999.986:
                if wvmin<3000.0:
                    idxlf = np.round(int((3000.0-wvmin)/0.1)) + int((wvline-3000.0)/0.006) 
                else:
                    idxlf = int((wvline-wvmin)/0.006)

            elif wvline<5000.0:
                if wvmin<3000.0:
                    idxlf = np.round(int((3000.0-wvmin)/0.1)) + int((4999.986-3000.0)/0.006) + 1
                else:
                    idxlf = int((4999.986-wvmin)/0.006) + 1

            elif wvline<10000.0:
                if wvmin<3000.0:
                    idxlf = np.round(int((3000.0-wvmin)/0.1)) + int((4999.986-3000.0)/0.006) + 1 + int((wvline-5000.0)/0.01)
                elif wvmin<4999.986:
                    idxlf = int((4999.986-wvmin)/0.006) + 1 + int((wvline-5000.0)/0.01)
                else:
                    idxlf = int((wvline-wvmin)/0.01) 

            elif wvline<15000.0:
                if wvmin<3000.0:
                    idxlf = np.round(int((3000.0-wvmin)/0.1)) + int((4999.986-3000.0)/0.006) + 1 + int((10000.0-5000.0)/0.01) + int((wvline-10000.0)/0.02)
                elif wvmin<4999.986:
                    idxlf = int((4999.986-wvmin)/0.006) + 1 + int((10000-5000.0)/0.01) + int((wvline-10000.0)/0.02)
                elif wvmin<10000.0:
                    idxlf = int((10000.0-wvmin)/0.01) + int((wvline-10000.0)/0.02)
                else:
                    idxlf = int((wvline-wvmin)/0.02)

            else:
                if wvmin<3000.0:
                    idxlf = np.round(int((3000.0-wvmin)/0.1)) + int((4999.986-3000.0)/0.006) + 1 + int((10000.0-5000.0)/0.01) + int((15000.0-10000.0)/0.02) + int((wvline-15000.0)/0.03)
                elif wvmin<4999.986:
                    idxlf = int((4999.986-wvmin)/0.006) + 1 + int((10000-5000.0)/0.01) + int((15000-10000.0)/0.02) + int((wvline-15000.0)/0.03)
                elif wvmin<10000.0:
                    idxlf = int((10000.0-wvmin)/0.01) + int((15000-10000.0)/0.02) + int((wvline-15000.0)/0.03)
                elif wvmin<15000.0:
                    idxlf = int((15000-wvmin)/0.02) + int((wvline-15000.0)/0.03)
                else:
                    idxlf = int((wvline-wvmin)/0.03)

            idxrg = idxlf + 1

            diffwv=wv[idxrg]-wv[idxlf] #pixel size in wavelength
            midpix=(wv[idxrg]+wv[idxlf])/2 #wavelength between the two pixels
            leftmask = wvline - diffwv/2 #left edge of the mask
            rightmask = wvline + diffwv/2 #right edge of the mask
            frac1 = (midpix - leftmask)/diffwv #fraction of the mask ovelapping the left pixel
            frac2 = (rightmask - midpix)/diffwv #fraction of the mask overlapping the right pixel
            midleft = (leftmask + midpix)/2 #central left overlapp
            midright = (rightmask + midpix)/2 #central wv right overlap
            f1 = f[idxlf] + (midleft-wv[idxlf])*(f[idxrg]-f[idxlf])/(diffwv)
            f2 = f[idxlf] + (midright-wv[idxlf])*(f[idxrg]-f[idxlf])/(diffwv)

            ccf[i]=ccf[i] - f1*fm[j]*frac1 - f2*fm[j]*frac2

    return (ccf-np.min(ccf))/np.max((ccf-np.min(ccf)))

@nb.njit(cache=True,error_model='numpy')
def weight_mask(wvi,wvf,o_weight,wvm,fm):
    j=0
    maxj=len(wvi)

    for i in range(len(wvm)):

        if wvm[i]<wvi[j]:
            fm[i]=0.0
        elif wvm[i]>=wvi[j] and wvm[i]<=wvf[j]:
            fm[i]=fm[i]*o_weight[j]
        elif wvm[i]>wvf[j]:
            j+=1
            if j>=maxj:
                fm[i]=0.0
                break
            else:
                i-=1

    return wvm, fm

@nb.njit(cache=True,error_model='numpy')
def polar2colatitude_nb(r,a,i):
    '''Enters the polars coordinates and the inclination i (with respect to the north pole, i=0 makes transits, 90-(inclination defined in exoplanets))
    Returns the colatitude in the star (90-latitude)
    '''
    a=a*m.pi/180.
    i=-i #negative to make the rotation toward the observer.
    theta=m.acos(r*m.sin(a)*m.cos(i)-m.sin(i)*m.sqrt(1-r*r))
    return theta

@nb.njit(cache=True,error_model='numpy')
def polar2longitude_nb(r,a,i):
    '''Enters the polars coordinates and the inclination i (with respect to the north pole, i=0 makes transits, 90-(inclination defined in exoplanets))
    Returns the longitude in the star (from -90 to 90)
    '''
    a=a*m.pi/180.
    i=-i #negative to make the rotation toward the observer.
    h=m.sqrt((1.-(r*m.cos(a))**2.)/(m.tan(i)**2.+1.)) #heigh of the terminator (long=pi/2)
    if r*np.sin(a)>h:
        phi=m.asin(-r*m.cos(a)/m.sqrt(1.-(r*m.sin(a)*m.cos(i)-m.sin(i)*m.sqrt(1.-r*r))**2.))+m.pi #to correct for mirroring of longitudes in the terminator

    else:
        phi=m.asin(r*m.cos(a)/m.sqrt(1.-(r*m.sin(a)*m.cos(i)-m.sin(i)*m.sqrt(1.-r*r))**2.))

    return phi

@nb.njit(cache=True,error_model='numpy')
def speed_bisector_nb(rv,ccf,integrated_bis):
    ''' Fit the bisector of the CCF with a 5th deg polynomial
    '''
    idxmax=ccf.argmax()
    maxccf=ccf[idxmax]
    maxrv=rv[idxmax]

    xnew = rv
    ynew = ccf


    cutleft=0
    cutright=len(ynew)-1
    # if not integrated_bis: #cut the CCF at the minimum of the wings only for reference CCF, if not there are errors.
    for i in range(len(ynew)):
        if xnew[i]>maxrv:
            if ynew[i]>ynew[i-1]:
                cutright=i
                break

    for i in range(len(ynew)):
        if xnew[-1-i]<maxrv:
            if ynew[-1-i]>ynew[-i]:
                cutleft=len(ynew)-i
                break

    xnew=xnew[cutleft:cutright]
    ynew=ynew[cutleft:cutright]
    
    minright=np.min(ynew[xnew>maxrv])
    minleft=np.min(ynew[xnew<maxrv])
    minccf=np.max(np.array([minright,minleft]))
    ybis=np.linspace(minccf+0.01*(maxccf-minccf),0.999*maxccf,50) #from 5% to maximum
    xbis=np.zeros(len(ybis))


    for i in range(len(ybis)):
        for j in range(len(ynew)-1):
            if ynew[j]<ybis[i] and ynew[j+1]>ybis[i] and xnew[j]<maxrv:
                rv1=xnew[j]+(xnew[j+1]-xnew[j])*(ybis[i]-ynew[j])/(ynew[j+1]-ynew[j])
            if ynew[j]>ybis[i] and ynew[j+1]<ybis[i] and xnew[j+1]>maxrv:
                rv2=xnew[j]+(xnew[j+1]-xnew[j])*(ybis[i]-ynew[j])/(ynew[j+1]-ynew[j])
        xbis[i]=(rv1+rv2)/2.0 #bisector
    # xbis[-1]=maxrv #at the top should be max RV

    return cutleft,cutright,xbis,ybis


@nb.njit(cache=True,error_model='numpy')
def limb_darkening_law(LD_law,LD1,LD2,amu):

    if LD_law == 'linear':
        mu=1-LD1*(1-amu)

    elif LD_law == 'quadratic':
        a=2*np.sqrt(LD1)*LD2
        b=np.sqrt(LD1)*(1-2*LD2)
        mu=1-a*(1-amu)-b*(1-amu)**2

    elif LD_law == 'sqrt':
        a=np.sqrt(LD1)*(1-2*LD2) 
        b=2*np.sqrt(LD1)*LD2
        mu=1-a*(1-amu)-b*(1-np.sqrt(amu))

    elif LD_law == 'log':
        a=LD2*LD1**2+1
        b=LD1**2-1
        mu=1-a*(1-amu)-b*amu*(1-np.log(amu))

    else:
        print('LD law not valid.')

    return mu

@nb.njit(cache=True,error_model='numpy')
def compute_spot_position(t,spot_map,ref_time,Prot,diff_rot,Revo,Q):
    pos=np.zeros((len(spot_map),4))

    for i in range(len(spot_map)):
        tini = spot_map[i][0] #time of spot apparence
        dur = spot_map[i][1] #duration of the spot
        tfin = tini + dur #final time of spot
        colat = spot_map[i][2] #colatitude
        lat = 90 - colat #latitude
        longi = spot_map[i][3] #longitude
        Rcoef = spot_map[i][4:7] #coefficients for the evolution od the radius. Depends on the desired law.

        pht = longi + (t-ref_time)/Prot%1*360
        #update longitude adding diff rotation
        phsr= pht + (t-ref_time)*diff_rot*(1.698*m.sin(np.deg2rad(lat))**2+2.346*m.sin(np.deg2rad(lat))**4)


        if Revo == 'constant':
            if t>=tini and t<=tfin: 
                rad=Rcoef[0] 
            else:
                rad=0.0
        elif Revo == 'linear':
            if t>=tini and t<=tfin:
                rad=Rcoef[0]+(t-tini)*(Rcoef[1]-Rcoef[0])/dur
            else:
                rad=0.0
        elif Revo == 'quadratic':
            if t>=tini and t<=tfin:
                rad=-4*Rcoef[0]/(dur*(1-2*tini))*(t-tini)*(t-tini-dur)
            else:
                rad=0.0
        
        else:
            print('Spot evolution law not implemented yet. Only constant and linear are implemented.')
        
        if Q!=0.0: #to speed up the code when no fac are present
            rad_fac=np.deg2rad(rad)*m.sqrt(1+Q) 
        else: rad_fac=0.0

        pos[i]=np.array([np.deg2rad(colat), np.deg2rad(phsr), np.deg2rad(rad), rad_fac])
        #return position and radii of spots at t in radians.

    return pos


@nb.njit(cache=True,error_model='numpy')
def compute_planet_pos(t,esinw,ecosw,T0p,Pp,rad_pl,b,a,alp):
    
    if(esinw==0 and ecosw==0):
       ecc=0
       omega=0
    else:
       ecc=m.sqrt(esinw**2+ecosw**2)
       omega=m.atan2(esinw,ecosw)

    t_peri = Ttrans_2_Tperi(T0p,Pp, ecc, omega)
    sinf,cosf=true_anomaly(t,Pp,ecc,t_peri)


    cosftrueomega=cosf*m.cos(omega+m.pi/2)-sinf*m.sin(omega+np.pi/2) #cos(f+w)=cos(f)*cos(w)-sin(f)*sin(w)
    sinftrueomega=cosf*m.sin(omega+m.pi/2)+sinf*m.cos(omega+np.pi/2) #sin(f+w)=cos(f)*sin(w)+sin(f)*cos(w)

    if cosftrueomega>0.0: return np.array([1+rad_pl*2, 0.0, rad_pl]) #avoid secondary transits

    cosi = (b/a)*(1+esinw)/(1-ecc**2) #cosine of planet inclination (i=90 is transit)

    rpl=a*(1-ecc**2)/(1+ecc*cosf)
    xpl=rpl*(-m.cos(alp)*sinftrueomega-m.sin(alp)*cosftrueomega*cosi)
    ypl=rpl*(m.sin(alp)*sinftrueomega-m.cos(alp)*cosftrueomega*cosi)

    rhopl=m.sqrt(ypl**2+xpl**2)
    thpl=m.atan2(ypl,xpl)
    pos=np.array([rhopl, thpl, rad_pl],dtype=np.float64) #rho, theta, and radii (in Rstar) of the planet
    return pos


@nb.njit(cache=True,error_model='numpy')
def Ttrans_2_Tperi(T0, P, e, w):

    f = m.pi/2 - w
    E = 2 * m.atan(m.tan(f/2.) * m.sqrt((1.-e)/(1.+e)))  # eccentric anomaly
    Tp = T0 - P/(2*np.pi) * (E - e*m.sin(E))      # time of periastron

    return Tp

@nb.njit(cache=True,error_model='numpy')
def true_anomaly(x,period,ecc,tperi):
    fmean=2.0*m.pi*(x-tperi)/period
    #Solve by Newton's method x(n+1)=x(n)-f(x(n))/f'(x(n))
    fecc=fmean
    diff=1.0
    while(diff>1.0E-6):
        fecc_0=fecc
        fecc=fecc_0-(fecc_0-ecc*m.sin(fecc_0)-fmean)/(1.0-ecc*m.cos(fecc_0))
        diff=m.fabs(fecc-fecc_0)
    sinf=m.sqrt(1.0-ecc*ecc)*m.sin(fecc)/(1.0-ecc*m.cos(fecc))
    cosf=(m.cos(fecc)-ecc)/(1.0-ecc*m.cos(fecc))
    return sinf, cosf
########################################################################################
########################################################################################
#                              SPECTROSCOPY FUNCTIONS  FOR SPHERICAL GRID              #
########################################################################################
########################################################################################



#with this the x and y width of each grid is the same, thus the area of the grids is similar in all the sphere, avoiding an over/under sampling of the poles/center
@nb.njit(cache=True,error_model='numpy')
def generate_grid_coordinates_nb(N):

    Nt=2*N-1 #N is number og concentric rings. Nt is counting them two times minus the center one.
    width=180.0/(2*N-1) #width of one grid element.

    centres=np.append(0,np.linspace(width,90-width/2,N-1)) #latitudes of the concentric grids
    anglesout=np.linspace(0,360-width,2*Nt) #longitudes of the grid in the equator. The pole ofthe grid faces the observer.
    
    radi=np.sin(np.pi*centres/180) #projected polar radius of the ring.
    amu=np.cos(np.pi*centres/180) #amus

    ts=[0.0] #central grid
    alphas=[0.0] #central grid

    area=[2.0*np.pi*(1.0-np.cos(width*np.pi/360.0))] #area central element
    parea=[np.pi*np.sin(width*np.pi/360.0)**2]

    Ngrid_in_ring=[1]

    for i in range(1,len(amu)): #for each ring except firs
        Nang=int(round(len(anglesout)*(radi[i]))) #Number of longitudes to have grids of same width
        w=360/Nang #width i angles
        Ngrid_in_ring.append(Nang)

        angles=np.linspace(0,360-w,Nang)
        area.append(radi[i]*width*w*np.pi*np.pi/(180*180)) #area of each grid
        parea.append(amu[i]*area[-1]) #PROJ. AREA OF THE GRID

        for j in range(Nang):
            ts.append(centres[i]) #latitude
            alphas.append(angles[j]) #longitude


    alphas=np.array(alphas) #longitude of grid (pole faces observer)
    ts=np.array(ts) #colatitude of grid
    Ngrids=len(ts)  #number of grids

    rs = np.sin(np.pi*ts/180) #projected polar radius of grid

    xs = np.cos(np.pi*ts/180) #grid elements in cartesian coordinates. Note that pole faces the observer.
    ys = rs*np.sin(np.pi*alphas/180)
    zs = -rs*np.cos(np.pi*alphas/180)

    return Ngrids,Ngrid_in_ring, centres, amu, rs, alphas, xs, ys, zs, area, parea



@nb.njit(cache=True,error_model='numpy')
def loop_compute_immaculate_nb(N,Ngrid_in_ring,ccf_tot,rvel,rv,rvs_ring,ccf_ring):
    #CCF of each pixel, adding doppler and interpolating
    iteration=0
    #Compute the position of the grid projected on the sphere and its radial velocity.
    for i in range(0,N): #Loop for each ring.
        for j in range(Ngrid_in_ring[i]): #loop for each grid in the ring
            ccf_tot[iteration,:]=ccf_tot[iteration,:]+interpolation_nb(rv,rvs_ring[i,:] + rvel[iteration],ccf_ring[i,:],ccf_ring[i,0],ccf_ring[i,-1]) 
            iteration=iteration+1

    return ccf_tot



@nb.njit(cache=True,error_model='numpy')
def loop_generate_rotating_nb(N,Ngrid_in_ring,pare,amu,spot_pos,vec_grid,vec_spot,simulate_planet,planet_pos,ccf_ph,ccf_sp,ccf_fc,ccf_ph_tot,vis):
    #define things
    width=np.pi/(2*N-1) #width of one grid element, in radiants
    ccf_tot = ccf_ph_tot


    vis_spots_idx=[]
    for i in range(len(vis)-1):
        if vis[i]==1.0:
            vis_spots_idx.append(i)
    ###################### CENTRAL GRID ###############################
    #Central grid is different since it is a circle. initialize values.
    ####################################################################
    dsp=0.0 #fraction covered by each spot
    dfc=0.0
    asp=0.0 #fraction covered by all spots
    afc=0.0
    apl=0.0
    iteration = 0
    
    for l in vis_spots_idx: #for each spot

        if spot_pos[l][2]==0.0:
            continue

        dist=m.acos(np.dot(vec_grid[iteration],vec_spot[l])) #compute the distance to the grid

        if dist>(width/2+spot_pos[l][2]):
            dsp=0.0
        else:
            if (width/2)<spot_pos[l][2]: #if the spot can cover completely the grid, two cases:
                if dist<=spot_pos[l][2]-(width/2):  #grid completely covered
                    dsp=1.0
                else:  #grid partially covered
                    dsp=-(dist-spot_pos[l][2]-width/2)/width

            else: #the grid can completely cover the spot, two cases:
                if dist<=(width/2-spot_pos[l][2]): #all the spot is inside the grid
                    dsp=(2*spot_pos[l][2]/width)**2                 
                else: #grid partially covered
                    dsp=-2*spot_pos[l][2]*(dist-width/2-spot_pos[l][2])/width**2


        asp+=dsp
        #FACULA
        if spot_pos[l][3]==0.0: #if radius=0, there is no facula, jump to next spot with continue
            continue 

        if dist>(width/2+spot_pos[l][3]):
            dfc=0.0
        else:
            if (width/2)<spot_pos[l][3]: #if the spot can cover completely the grid, two cases:
                if dist<=spot_pos[l][3]-(width/2):  #grid completely covered
                    dfc=1.0 - dsp
                else:  #grid partially covered
                    dfc=-(dist-spot_pos[l][3]-width/2)/width - dsp

            else: #if the grid can completely cover the spot, two cases:
                if dist<=(width/2-spot_pos[l][3]): #all the spot is inside the grid
                    dfc=(2*spot_pos[l][3]/width)**2 - dsp                
                else: #grid partially covered
                    dfc =-2*spot_pos[l][3]*(dist-width/2-spot_pos[l][3])/width**2 - dsp

        afc+=dfc


    #PLANET
    if simulate_planet:
        if vis[-1]==1.0:
            dist=m.sqrt((planet_pos[0]*m.cos(planet_pos[1]) - vec_grid[iteration,1])**2 + ( planet_pos[0]*m.sin(planet_pos[1]) - vec_grid[iteration,2] )**2) #grid-planet distance
            
            width2=2*m.sin(width/2)
            if dist>width2/2+planet_pos[2]: apl=0.0
            elif dist<planet_pos[2]-width2/2: apl=1.0
            else: apl=-(dist-planet_pos[2]-width2/2)/width2

    
    if afc>1.0:
        afc=1.0

    if asp>1.0:
        asp=1.0
        afc=0.0

    if apl>0.0:
        asp=asp*(1-apl)
        afc=afc*(1-apl)

    aph=1-asp-afc-apl           

    #add the corresponding ccf to the total CCF
    ccf_tot = ccf_tot  - (1-aph)*ccf_ph[iteration] + asp*ccf_sp[iteration] + afc*ccf_fc[iteration] 


    Aph=aph*pare[0]
    Asp=asp*pare[0]
    Afc=afc*pare[0]
    Apl=apl*pare[0]
    typ=[[aph,asp,afc,apl]]


    ############### OTHER GRIDS #######################
    # NOW DO THE SAME FOR THE REST OF GRIDS
    ###################################################
    for i in range(1,N): #Loop for each ring.
        for j in range(Ngrid_in_ring[i]): #Loop for each grid
            iteration+=1

            dsp=0.0 #fraction covered by each spot
            dfc=0.0
            asp=0.0 #fraction covered by all spots
            afc=0.0
            apl=0.0

            for l in vis_spots_idx:
                
                if spot_pos[l][2]==0.0: #if radius=0, there is no spot, jump to next spot with continue
                    continue 

                dist=m.acos(np.dot(vec_grid[iteration],vec_spot[l])) #distance between spot centre and grid,multiplying two unit vectors


                #SPOT
                if dist>(width/2+spot_pos[l][2]): #grid not covered 
                    dsp=0.0
                
                else:
                    if (width/m.sqrt(2))<spot_pos[l][2]: #if the spot can cover completely the grid, two cases:
                        if dist<=(m.sqrt(spot_pos[l][2]**2-(width/2)**2)-width/2):  #grid completely covered
                            dsp=1.0
                        else:  #grid partially covered
                            dsp=-(dist-spot_pos[l][2]-width/2)/(width+spot_pos[l][2]-m.sqrt(spot_pos[l][2]**2-(width/2)**2))

                    elif (width/2)>spot_pos[l][2]: #if the grid can completely cover the spot, two cases:
                        if dist<=(width/2-spot_pos[l][2]): #all the spot is inside the grid
                            dsp=(np.pi/4)*(2*spot_pos[l][2]/width)**2                 
                        else: #grid partially covered
                            dsp=(np.pi/4)*((2*spot_pos[l][2]/width)**2-(2*spot_pos[l][2]/width**2)*(dist-width/2+spot_pos[l][2]))

                    else: #if the spot is larger than the grid but not enough to cover it, grid partially covered by the spot 
                        A1=(width/2)*m.sqrt(spot_pos[l][2]**2-(width/2)**2)
                        A2=(spot_pos[l][2]**2/2)*(m.pi/2-2*m.asin(m.sqrt(spot_pos[l][2]**2-(width/2)**2)/spot_pos[l][2]))
                        Ar=4*(A1+A2)/width**2
                        dsp=-Ar*(dist-width/2-spot_pos[l][2])/(width/2+spot_pos[l][2])

                asp+=dsp
                #FACULA
                if spot_pos[l][3]==0.0: #if radius=0, there is no facula, jump to next spot with continue
                    continue 
                
                if dist>(width/2+spot_pos[l][3]): #grid not covered by faculae
                    dfc=0.0
                
                else:
                    if (width/m.sqrt(2))<spot_pos[l][3]: #if the spot can cover completely the grid, two cases:
                        if dist<=(m.sqrt(spot_pos[l][3]**2-(width/2)**2)-width/2):  #grid completely covered
                            dfc=1.0-dsp #subtract spot
                        else:  #grid partially covered
                            dfc=-(dist-spot_pos[l][3]-width/2)/(width+spot_pos[l][3]-m.sqrt(spot_pos[l][3]**2-(width/2)**2))-dsp

                    elif (width/2)>spot_pos[l][3]: #if the grid can completely cover the spot, two cases:
                        if dist<=(width/2-spot_pos[l][3]): #all the spot is inside the grid
                            dfc=(np.pi/4)*(2*spot_pos[l][3]/width)**2-dsp               
                        else: #grid partially covered
                            dfc=(np.pi/4)*((2*spot_pos[l][3]/width)**2-(2*spot_pos[l][3]/width**2)*(dist-width/2+spot_pos[l][3]))-dsp

                    else: #if the spot is larger than the grid but not enough to cover it, grid partially covered by the spot 
                        A1=(width/2)*m.sqrt(spot_pos[l][3]**2-(width/2)**2)
                        A2=(spot_pos[l][3]**2/2)*(m.pi/2-2*m.asin(m.sqrt(spot_pos[l][3]**2-(width/2)**2)/spot_pos[l][3]))
                        Ar=4*(A1+A2)/width**2
                        dfc=-Ar*(dist-width/2-spot_pos[l][3])/(width/2+spot_pos[l][3])-dsp

                afc+=dfc


            #PLANET
            if simulate_planet:
                if vis[-1]==1.0:
                    dist=m.sqrt((planet_pos[0]*m.cos(planet_pos[1]) - vec_grid[iteration,1])**2 + ( planet_pos[0]*m.sin(planet_pos[1]) - vec_grid[iteration,2] )**2) #grid-planet distance
                    
                    width2=amu[i]*width
                    if dist>width2/2+planet_pos[2]: apl=0.0
                    elif dist<planet_pos[2]-width2/2: apl=1.0
                    else: apl=-(dist-planet_pos[2]-width2/2)/width2


            if afc>1.0:
                afc=1.0

            if asp>1.0:
                asp=1.0
                afc=0.0

            if apl>0.0:
                asp=asp*(1-apl)
                afc=afc*(1-apl)

            aph=1-asp-afc-apl           
  
            #add the corresponding ccf to the total CCF
            ccf_tot = ccf_tot  - (1-aph)*ccf_ph[iteration] + asp*ccf_sp[iteration] + afc*ccf_fc[iteration] 

            Aph=Aph+aph*pare[i]
            Asp=Asp+asp*pare[i]
            Afc=Afc+afc*pare[i]
            Apl=Apl+apl*pare[i]
            typ.append([aph,asp,afc,apl])

            

    return ccf_tot,typ, Aph, Asp, Afc, Apl



@nb.njit(cache=True,error_model='numpy')
def loop_generate_rotating_lc_nb(N,Ngrid_in_ring,pare,amu,spot_pos,vec_grid,vec_spot,simulate_planet,planet_pos,bph,bsp,bfc,flxph,vis):
 

    #define things
    width=np.pi/(2*N-1) #width of one grid element, in radiants
    flux = flxph


    vis_spots_idx=[]
    for i in range(len(vis)-1):
        if vis[i]==1.0:
            vis_spots_idx.append(i)
    ###################### CENTRAL GRID ###############################
    #Central grid is different since it is a circle. initialize values.
    ####################################################################
    dsp=0.0 #fraction covered by each spot
    dfc=0.0
    asp=0.0 #fraction covered by all spots
    afc=0.0
    apl=0.0
    iteration = 0
    
    for l in vis_spots_idx: #for each spot

        if spot_pos[l][2]==0.0:
            continue

        dist=m.acos(np.dot(vec_grid[iteration],vec_spot[l])) #compute the distance to the grid

        if dist>(width/2+spot_pos[l][2]):
            dsp=0.0
        else:
            if (width/2)<spot_pos[l][2]: #if the spot can cover completely the grid, two cases:
                if dist<=spot_pos[l][2]-(width/2):  #grid completely covered
                    dsp=1.0
                else:  #grid partially covered
                    dsp=-(dist-spot_pos[l][2]-width/2)/width

            else: #the grid can completely cover the spot, two cases:
                if dist<=(width/2-spot_pos[l][2]): #all the spot is inside the grid
                    dsp=(2*spot_pos[l][2]/width)**2                 
                else: #grid partially covered
                    dsp=-2*spot_pos[l][2]*(dist-width/2-spot_pos[l][2])/width**2


        asp+=dsp
        #FACULA
        if spot_pos[l][3]==0.0: #if radius=0, there is no facula, jump to next spot with continue
            continue 

        if dist>(width/2+spot_pos[l][3]):
            dfc=0.0
        else:
            if (width/2)<spot_pos[l][3]: #if the spot can cover completely the grid, two cases:
                if dist<=spot_pos[l][3]-(width/2):  #grid completely covered
                    dfc=1.0 - dsp
                else:  #grid partially covered
                    dfc=-(dist-spot_pos[l][3]-width/2)/width - dsp

            else: #if the grid can completely cover the spot, two cases:
                if dist<=(width/2-spot_pos[l][3]): #all the spot is inside the grid
                    dfc=(2*spot_pos[l][3]/width)**2 - dsp                
                else: #grid partially covered
                    dfc =-2*spot_pos[l][3]*(dist-width/2-spot_pos[l][3])/width**2 - dsp

        afc+=dfc


    #PLANET
    if simulate_planet:
        if vis[-1]==1.0:
            dist=m.sqrt((planet_pos[0]*m.cos(planet_pos[1]) - vec_grid[iteration,1])**2 + ( planet_pos[0]*m.sin(planet_pos[1]) - vec_grid[iteration,2] )**2) #grid-planet distance
            
            width2=2*m.sin(width/2)
            if dist>width2/2+planet_pos[2]: apl=0.0
            elif dist<planet_pos[2]-width2/2: apl=1.0
            else: apl=-(dist-planet_pos[2]-width2/2)/width2

    
    if afc>1.0:
        afc=1.0

    if asp>1.0:
        asp=1.0
        afc=0.0

    if apl>0.0:
        asp=asp*(1-apl)
        afc=afc*(1-apl)

    aph=1-asp-afc-apl           

    #add the corresponding flux to the total flux
    flux = flux - (1-aph)*bph[i]+asp*bsp[i]+bfc[i]*afc


    Aph=aph*pare[0]
    Asp=asp*pare[0]
    Afc=afc*pare[0]
    Apl=apl*pare[0]
    typ=[[aph,asp,afc,apl]]


    ############### OTHER GRIDS #######################
    # NOW DO THE SAME FOR THE REST OF GRIDS
    ###################################################
    for i in range(1,N): #Loop for each ring.
        for j in range(Ngrid_in_ring[i]): #Loop for each grid
            iteration+=1

            dsp=0.0 #fraction covered by each spot
            dfc=0.0
            asp=0.0 #fraction covered by all spots
            afc=0.0
            apl=0.0

            for l in vis_spots_idx:
                
                if spot_pos[l][2]==0.0: #if radius=0, there is no spot, jump to next spot with continue
                    continue 

                dist=m.acos(np.dot(vec_grid[iteration],vec_spot[l])) #distance between spot centre and grid,multiplying two unit vectors


                #SPOT
                if dist>(width/2+spot_pos[l][2]): #grid not covered 
                    dsp=0.0
                
                else:
                    if (width/m.sqrt(2))<spot_pos[l][2]: #if the spot can cover completely the grid, two cases:
                        if dist<=(m.sqrt(spot_pos[l][2]**2-(width/2)**2)-width/2):  #grid completely covered
                            dsp=1.0
                        else:  #grid partially covered
                            dsp=-(dist-spot_pos[l][2]-width/2)/(width+spot_pos[l][2]-m.sqrt(spot_pos[l][2]**2-(width/2)**2))

                    elif (width/2)>spot_pos[l][2]: #if the grid can completely cover the spot, two cases:
                        if dist<=(width/2-spot_pos[l][2]): #all the spot is inside the grid
                            dsp=(np.pi/4)*(2*spot_pos[l][2]/width)**2                 
                        else: #grid partially covered
                            dsp=(np.pi/4)*((2*spot_pos[l][2]/width)**2-(2*spot_pos[l][2]/width**2)*(dist-width/2+spot_pos[l][2]))

                    else: #if the spot is larger than the grid but not enough to cover it, grid partially covered by the spot 
                        A1=(width/2)*m.sqrt(spot_pos[l][2]**2-(width/2)**2)
                        A2=(spot_pos[l][2]**2/2)*(m.pi/2-2*m.asin(m.sqrt(spot_pos[l][2]**2-(width/2)**2)/spot_pos[l][2]))
                        Ar=4*(A1+A2)/width**2
                        dsp=-Ar*(dist-width/2-spot_pos[l][2])/(width/2+spot_pos[l][2])

                asp+=dsp
                #FACULA
                if spot_pos[l][3]==0.0: #if radius=0, there is no facula, jump to next spot with continue
                    continue 
                
                if dist>(width/2+spot_pos[l][3]): #grid not covered by faculae
                    dfc=0.0
                
                else:
                    if (width/m.sqrt(2))<spot_pos[l][3]: #if the spot can cover completely the grid, two cases:
                        if dist<=(m.sqrt(spot_pos[l][3]**2-(width/2)**2)-width/2):  #grid completely covered
                            dfc=1.0-dsp #subtract spot
                        else:  #grid partially covered
                            dfc=-(dist-spot_pos[l][3]-width/2)/(width+spot_pos[l][3]-m.sqrt(spot_pos[l][3]**2-(width/2)**2))-dsp

                    elif (width/2)>spot_pos[l][3]: #if the grid can completely cover the spot, two cases:
                        if dist<=(width/2-spot_pos[l][3]): #all the spot is inside the grid
                            dfc=(np.pi/4)*(2*spot_pos[l][3]/width)**2-dsp               
                        else: #grid partially covered
                            dfc=(np.pi/4)*((2*spot_pos[l][3]/width)**2-(2*spot_pos[l][3]/width**2)*(dist-width/2+spot_pos[l][3]))-dsp

                    else: #if the spot is larger than the grid but not enough to cover it, grid partially covered by the spot 
                        A1=(width/2)*m.sqrt(spot_pos[l][3]**2-(width/2)**2)
                        A2=(spot_pos[l][3]**2/2)*(m.pi/2-2*m.asin(m.sqrt(spot_pos[l][3]**2-(width/2)**2)/spot_pos[l][3]))
                        Ar=4*(A1+A2)/width**2
                        dfc=-Ar*(dist-width/2-spot_pos[l][3])/(width/2+spot_pos[l][3])-dsp

                afc+=dfc


            #PLANET
            if simulate_planet:
                if vis[-1]==1.0:
                    dist=m.sqrt((planet_pos[0]*m.cos(planet_pos[1]) - vec_grid[iteration,1])**2 + ( planet_pos[0]*m.sin(planet_pos[1]) - vec_grid[iteration,2] )**2) #grid-planet distance
                    
                    width2=amu[i]*width
                    if dist>width2/2+planet_pos[2]: apl=0.0
                    elif dist<planet_pos[2]-width2/2: apl=1.0
                    else: apl=-(dist-planet_pos[2]-width2/2)/width2


            if afc>1.0:
                afc=1.0

            if asp>1.0:
                asp=1.0
                afc=0.0

            if apl>0.0:
                asp=asp*(1-apl)
                afc=afc*(1-apl)

            aph=1-asp-afc-apl           
  
            #add the corresponding ccf to the total CCF
            flux = flux - (1-aph)*bph[i]+asp*bsp[i]+bfc[i]*afc

            Aph=Aph+aph*pare[i]
            Asp=Asp+asp*pare[i]
            Afc=Afc+afc*pare[i]
            Apl=Apl+apl*pare[i]
            typ.append([aph,asp,afc,apl])
            

    return flux ,typ, Aph, Asp, Afc, Apl




################################################################
# FAST MODE ROUTINES 
################################################################
#PHOTOMETRY
@nb.njit(cache=True,error_model='numpy')
def generate_rotating_photosphere_fast_lc(obs_times,Ngrid_in_ring,acd,amu,pare,flnp,flns,filter_trans,N,use_phoenix_mu,LD_law,LD1,LD2,spot_map,ref_time,Prot,diff_rot,Revo,Q,inc,temp_ph,temp_fc,simulate_planet,esinw,ecosw,T0p,Pp,Rpl,b,a,alp):
    flxph = 0.0 #initialze flux of photosphere
    sflp=np.zeros(N) #brightness of ring
    flp=np.zeros((N,len(filter_trans))) #spectra of each ring convolved by filter


    ################### IMMACULATE FLUX ###########################
    #Computing flux of immaculate photosphere and of every pixel
    for i in range(0,N): #Loop for each ring, to compute the flux of the star.   

        #Interpolate Phoenix intensity models to correct projected ange:
        if use_phoenix_mu:
            idx_upp=len(acd)-1-np.searchsorted(np.flip(acd),amu[i]*0.999999999,side='right') #acd is sorted inversely
            idx_low=idx_upp+1
            dlp = flnp[idx_low]+(flnp[idx_upp]-flnp[idx_low])*(amu[i]-acd[idx_low])/(acd[idx_upp]-acd[idx_low]) #spectra of the projected angle. includes limb darkening
        
        else: #or use a specified limb darkening law to multiply central spectra
            dlp = flnp[0]*limb_darkening_law(LD_law,LD1,LD2,amu[i])


        flp[i,:]=dlp*pare[i]/(4*np.pi)*filter_trans #spectra of one grid in ring N multiplied by the filter.
        sflp[i]=np.sum(flp[i,:]) #brightness of onegrid in ring N.  
        flxph=flxph+sflp[i]*Ngrid_in_ring[i] #total BRIGHTNESS of the immaculate photosphere

    ######################## ROTATE PHOTSPHERE FOR EACH TIME ################################
    flux=flxph+np.zeros((len(obs_times))) #initialize total flux at each timestamp
    filling_sp=0.0+np.zeros(len(obs_times))
    filling_ph=m.pi+np.zeros(len(obs_times))
    filling_pl=0.0+np.zeros(len(obs_times))
    filling_fc=0.0+np.zeros(len(obs_times))

    for k,t in enumerate(obs_times):
        
        if simulate_planet:        
            planet_pos=compute_planet_pos(t,esinw,ecosw,T0p,Pp,Rpl,b,a,alp)#compute the planet position at current time. In polar coordinates!! 
        else:
            planet_pos = np.array([2.0,0.0,0.0],dtype=np.float64)

        if spot_map.size==0:
            spot_pos=np.expand_dims(np.array([m.pi/2,-m.pi,0.0,0.0]),axis=0)
        else:
            spot_pos=compute_spot_position(t,spot_map,ref_time,Prot,diff_rot,Revo,Q) #compute the position of all spots at the current time. Returns theta and phi of each spot.      

        #convert latitude/longitude of spot centre to XYZ
        vec_spot=np.zeros((len(spot_map),3))
        for i in range(len(spot_map)):
            xspot = m.cos(inc)*m.sin(spot_pos[i,0])*m.cos(spot_pos[i,1])+m.sin(inc)*m.cos(spot_pos[i,0])
            yspot = m.sin(spot_pos[i,0])*m.sin(spot_pos[i,1])
            zspot = m.cos(spot_pos[i,0])*m.cos(inc)-m.sin(inc)*m.sin(spot_pos[i,0])*m.cos(spot_pos[i,1])
            vec_spot[i,:]=np.array([xspot,yspot,zspot]).T #spot center in cartesian

        #Loop for each spot.
        for i in range(len(vec_spot)):
            
            if spot_pos[i][2]==0.0: #if radius is 0, go to next spot
                continue

            dist=m.acos(np.dot(vec_spot[i],np.array([1.,0.,0.]))) #Angle center spot to center of star. 

            if (dist-spot_pos[i,2]*np.sqrt(1.0+Q)) > (m.pi/2): #spot & facula not visible. Jump to next spot.
                continue
            
            beta=np.pi/2-dist #angle of the spot with the edge of the star
            alpha=spot_pos[i,2] #angle of the radii of the spot
            ############ FACULA PROJECTED AREA ##################
            if Q>0.0: #facula
                alphaf=spot_pos[i,2]*m.sqrt(1.0+Q) #angle of the radii of the faculae
                #CASE 1: FACULA OUTSIDE OUTSIDE-> NULL, ALREADY EVALUATED BEFORE
                
                #CASE 2: ALL FACULAE INSIDE -> ELLIPSE 
                if 0.0 < alphaf <= beta:
                    ay=m.sin(alphaf) #semiminor axis ellipse
                    ax=ay*m.sin(beta) #semimajor axis ellipse

                    Ape=ax*ay*m.pi/4.0
                    pare_fac =  2.0*2.0*Ape #area of ellipse (projected area of spot)
                    amu_fac=m.cos(dist)

                #CASE 3: MOST OF THE FACULA INSIDE -> ELLIPSE + LUNE
                elif 0.0 <= beta < alphaf:
                    ay=m.sin(alphaf) #semiminor axis ellipse
                    ax=ay*m.sin(beta) #semimajor axis ellipse
                                    
                    yl=m.sqrt(1.0-(1.0-ay**2)/m.cos(beta)**2)   

                    ylay=min(1.0,max(yl/ay,-1.0)) #yl/ay, to avoid getting values >1 and errors in asin, due to floating points

                    Ape=ax*ay*m.pi/4.0 #y=ay
                    Apep= ax*ay*0.5*((ylay)*m.sqrt(1.0-(ylay)**2)+m.asin((ylay)))# y''=yl
                    Apcp= 0.5*(yl*m.sqrt(1.0-yl**2)+m.asin(yl)) - yl*m.cos(alphaf)*m.cos(beta)

                    pare_fac = 2.0*(2.0*Ape + Apcp - Apep) #proj. area is described in Urena et al. Stratified Sampling of Projected Spherical Caps
                    amu_fac=m.sin((beta+alphaf)/2.0) #representative mu as the mean point between edge of spot and of star


                #CASE 4: MOST OF THE FACULA OUTSIDE-> LUNE
                elif 0.0 < (-beta) < alphaf:
                    ay=m.sin(alphaf) #semiminor axis ellipse
                    ax=ay*m.sin(beta) #semimajor axis ellipse
                                    
                    yl=m.sqrt(1.0-(1.0-ay**2)/m.cos(beta)**2)  #intesection x and y between ellipse and lune 

                    ylay=min(1.0,max(yl/ay,-1.0)) #yl/ay, to avoid getting values >1 and errors in asin, due to floating points

                    Apep= ax*ay*0.5*((ylay)*m.sqrt(1-(ylay)**2)+m.asin((ylay)))# y''=yl
                    Apcp= 0.5*(yl*m.sqrt(1.0-yl**2)+m.asin(yl)) - yl*m.cos(alphaf)*m.cos(beta)

                    pare_fac = 2.0*(Apep + Apcp) #proj. area is area of lune  
                    amu_fac=m.sin((beta+alphaf)/2.0)


            ######### SPOT PROJECTED AREA ############
            #CASE 1: SPOT OUTSIDE-> NULL
            if 0.0 <= alpha <= (-beta):
                pare_spot=0.0
                amu_spot=0.0
            
            #CASE 2: ALL SPOT INSIDE -> ELLIPSE 
            elif 0.0 < alpha <= beta:
                ay=m.sin(alpha) #semiminor axis ellipse
                ax=ay*m.sin(beta) #semimajor axis ellipse

                Ape=ax*ay*m.pi/4.0
                pare_spot =  2.0*2.0*Ape #area of ellipse (projected area of spot)
                amu_spot=m.cos(dist)

            #CASE 3: MOST OF THE SPOT INSIDE -> ELLIPSE + LUNE
            elif 0.0 <= beta < alpha:
                ay=m.sin(alpha) #semiminor axis ellipse
                ax=ay*m.sin(beta) #semimajor axis ellipse
                                
                yl=m.sqrt(1.0-(1.0-ay**2)/m.cos(beta)**2)   

                ylay=min(1.0,max(yl/ay,-1.0)) #yl/ay, to avoid getting values >1 and errors in asin, due to floating points

                Ape=ax*ay*m.pi/4.0 #y=ay
                Apep= ax*ay*0.5*((ylay)*m.sqrt(1.0-(ylay)**2)+m.asin((ylay)))# y''=yl
                Apcp= 0.5*(yl*m.sqrt(1.0-yl**2)+m.asin(yl)) - yl*m.cos(alpha)*m.cos(beta)

                pare_spot = 2.0*(2.0*Ape + Apcp - Apep) #proj. area is described in Urena et al. Stratified Sampling of Projected Spherical Caps
                amu_spot=m.sin((beta+alpha)/2.0) #representative mu as the mean point between edge of spot and of star

            #CASE 4: MOST OF THE SPOT OUTSIDE-> LUNE
            elif 0.0 < (-beta) < alpha:
                ay=m.sin(alpha) #semiminor axis ellipse
                ax=ay*m.sin(beta) #semimajor axis ellipse
                                
                yl=m.sqrt(1.0-(1.0-ay**2)/m.cos(beta)**2)  #intesection x and y between ellipse and lune 

                ylay=min(1.0,max(yl/ay,-1.0)) #yl/ay, to avoid getting values >1 and errors in asin, due to floating points

                Apep= ax*ay*0.5*((ylay)*m.sqrt(1-(ylay)**2)+m.asin((ylay)))# y''=yl
                Apcp= 0.5*(yl*m.sqrt(1.0-yl**2)+m.asin(yl)) - yl*m.cos(alpha)*m.cos(beta)

                pare_spot = 2.0*(Apep + Apcp) #proj. area is area of lune  
                amu_spot=m.sin((beta+alpha)/2.0)
        

            #Spot, photosphere, and faculae flux at the angle of the spot

            if use_phoenix_mu:
                
                idx_upp=len(acd)-1-np.searchsorted(np.flip(acd),amu_spot*0.999999999,side='right') #acd is sorted inversely
                idx_low=idx_upp+1
                dlp = flnp[idx_low]+(flnp[idx_upp]-flnp[idx_low])*(amu_spot-acd[idx_low])/(acd[idx_upp]-acd[idx_low]) #limb darkening #limb darkening
                dls = flns[idx_low]+(flns[idx_upp]-flns[idx_low])*(amu_spot-acd[idx_low])/(acd[idx_upp]-acd[idx_low]) #limb darkening

            else: #or use a specified limb darkening law
                ld=limb_darkening_law(LD_law,LD1,LD2,amu_spot)
                dlp = flnp[0]*ld
                dls = flns[0]*ld


            flux_phsp=np.sum(dlp*pare_spot/(4*np.pi)*filter_trans) #flux of the photosphere occuppied by the spot.
            flux_sp=np.sum(dls*pare_spot/(4*np.pi)*filter_trans) #flux of the spot



            if Q>0.0:

                pare_facula= pare_fac - pare_spot

                if use_phoenix_mu:                   
                    idx_upp=len(acd)-1-np.searchsorted(np.flip(acd),amu_fac*0.999999999,side='right') #acd is sorted inversely
                    idx_low=idx_upp+1
                    dlp = flnp[idx_low]+(flnp[idx_upp]-flnp[idx_low])*(amu_fac-acd[idx_low])/(acd[idx_upp]-acd[idx_low]) #limb darkening #limb darkening

                else: #or use a specified limb darkening law
                    ld=limb_darkening_law(LD_law,LD1,LD2,amu_fac)
                    dlp = flnp[0]*ld

                flux_phfc=np.sum(dlp*pare_fac/(4*np.pi)*filter_trans) #flux of the photosphere occuppied by the spot.

                dtfmu=250.9-407.4*amu_fac+190.9*amu_fac**2 #(T_fac-T_ph) multiplied by a factor depending on the 
                flux_fc=np.sum(dlp*pare_fac/(4*np.pi)*filter_trans)*((temp_ph+dtfmu)/(temp_fc))**4 #flux of the spot

            else:
                flux_phfc = 0.0
                flux_fc = 0.0
                pare_facula = 0.0


            flux[k] = flux[k] - flux_phsp + flux_sp - flux_phfc + flux_fc #total flux - photosphere + spot
            filling_sp[k] = filling_sp[k] + pare_spot
            filling_ph[k] = filling_ph[k] - pare_spot - pare_facula
            filling_fc[k] = filling_fc[k] + pare_facula
        


        ################### PLANETARY TRANSIT PROJECTED AREA ########################

        if simulate_planet:
            if planet_pos[0]-planet_pos[2]>= 1.0: #all planet outside
                pare_pl = 0.0
                amu_pl = 0.0
                block = 'none'

            elif planet_pos[0]+planet_pos[2] <= 1.0: #all planet inside
                pare_pl = m.pi*planet_pos[2]**2 #area of a circle
                amu_pl = m.sqrt(1-planet_pos[0]**2) #cos(mu)**2=1-sin(mu)**2=1-r**2

                block='ph'
                for i in range(len(vec_spot)): #check if planet is over a spot or photosphere or faculae
                    distsp=m.acos(np.dot(vec_spot[i],np.array([1.,0.,0.]))) #Angle center spot to center of star. 
                    if (distsp-spot_pos[i,2]*np.sqrt(1.0+Q)) >= (m.pi/2): #spot & facula not visible. Jump to next spot.
                        block='ph'
                        continue 

                    dist=m.acos(np.dot(np.array([m.cos(m.asin(planet_pos[0])),planet_pos[0]*m.cos(planet_pos[1]),planet_pos[0]*m.sin(planet_pos[1])]),vec_spot[i])) #spot-planet centers distance
                    
                    if dist < spot_pos[i,2]: #if the distance is lower than spot radius, most of the planet is inside the spot
                        if (distsp-spot_pos[i,2]) >= (m.pi/2): #if spot is not visible
                            block='ph'
                        else:
                            block = 'sp'
                    elif dist < spot_pos[i,2]*m.sqrt(1+Q): #if the distance is lower than facula radius, most of the planet is inside the facula
                            block = 'fc'
                    elif (block != 'sp') and (block != 'fc'): #if the planet is not blocking a spot or a facula, then its blocking ph
                        block = 'ph' #else, the planet is blocking photosphere

            else: #the planet is partially covering the star.
                
                d1=(1-planet_pos[2]**2+planet_pos[0]**2)/(2*planet_pos[0]) #dist from star centre to centre of intersection
                d2=planet_pos[0]-d1 #dist from centre of planet to centre of intersection
                dedge = 1-(1+planet_pos[2]-planet_pos[0])/2 #dist from centre star to centre intersection
                pare_pl = m.acos(d1) - d1*m.sqrt(1-d1**2) + planet_pos[2]**2*m.acos(d2/planet_pos[2]) - d2*m.sqrt(planet_pos[2]**2-d2**2) #area of intersection star-planet
                amu_pl = m.sqrt(1-(dedge)**2) #amu is represented by the mean point of the intersection.
                
                block='ph'
                for i in range(len(vec_spot)): #check if planet is over a spot or photosphere or faculae
                    distsp=m.acos(np.dot(vec_spot[i],np.array([1.,0.,0.]))) #Angle center spot to center of star. 
                    if (distsp-spot_pos[i,2]*np.sqrt(1.0+Q)) >= (m.pi/2): #spot & facula not visible. Jump to next spot.
                        block='ph'
                        continue 

                    dist=m.acos(np.dot(np.array([m.cos(m.asin(dedge)),dedge*m.cos(planet_pos[1]),dedge*m.sin(planet_pos[1])]),vec_spot[i])) #distance from spot to centre of planet-star intersection

                    if dist < spot_pos[i,2]: #if the distance is lower than spot radius, most of the planet is inside the spot
                        if (distsp-spot_pos[i,2]) > (m.pi/2): #if spot is not visible
                            block='ph'
                        else:
                            block = 'sp'
                    elif dist < spot_pos[i,2]*m.sqrt(1+Q): #if the distance is lower than facula radius, most of the planet is inside the facula
                        block = 'fc'
                    elif (block != 'sp') and (block != 'fc'): #if the planet is not blocking a spot or a facula, then its blocking ph
                        block = 'ph' #else, the planet is blocking photosphere


            #compute and subtract flux blocked by the planet
            if block == 'ph': 
                if use_phoenix_mu:
                    idx_upp=len(acd)-1-np.searchsorted(np.flip(acd),amu_pl*0.999999999,side='right') #acd is sorted inversely
                    idx_low=idx_upp+1
                    dlp = flnp[idx_low]+(flnp[idx_upp]-flnp[idx_low])*(amu_pl-acd[idx_low])/(acd[idx_upp]-acd[idx_low]) #limb darkening #limb darkening
                else: #or use a specified limb darkening law
                    ld=limb_darkening_law(LD_law,LD1,LD2,amu_pl)
                    dlp = flnp[0]*ld

                flux_pl=np.sum(dlp*pare_pl/(4*np.pi)*filter_trans) #flux of the photosphere occuppied by the planet.
                flux[k] = flux[k] - flux_pl #total flux - flux blocked
                filling_ph[k] = filling_ph[k] - pare_pl
                filling_pl[k] = filling_pl[k] + pare_pl


            if block == 'sp': #flux blocked by the planet
                if use_phoenix_mu:
                    idx_upp=len(acd)-1-np.searchsorted(np.flip(acd),amu_pl*0.999999999,side='right') #acd is sorted inversely
                    idx_low=idx_upp+1
                    dls = flns[idx_low]+(flns[idx_upp]-flns[idx_low])*(amu_pl-acd[idx_low])/(acd[idx_upp]-acd[idx_low]) #limb darkening #limb darkening
                else: #or use a specified limb darkening law
                    ld=limb_darkening_law(LD_law,LD1,LD2,amu_pl)
                    dls = flns[0]*ld

                flux_pl=np.sum(dls*pare_pl/(4*np.pi)*filter_trans) #flux of the photosphere occuppied by the planet.
                flux[k] = flux[k] - flux_pl #total flux - flux blocked
                filling_sp[k] = filling_sp[k] - pare_pl
                filling_pl[k] = filling_pl[k] + pare_pl

            if block == 'fc': #flux blocked by the spot
                if use_phoenix_mu:
                    idx_upp=len(acd)-1-np.searchsorted(np.flip(acd),amu_pl*0.999999999,side='right') #acd is sorted inversely
                    idx_low=idx_upp+1
                    dlp = flnp[idx_low]+(flnp[idx_upp]-flnp[idx_low])*(amu_pl-acd[idx_low])/(acd[idx_upp]-acd[idx_low]) #limb darkening #limb darkening
                else: #or use a specified limb darkening law
                    ld=limb_darkening_law(LD_law,LD1,LD2,amu_pl)
                    dlp = flnp[0]*ld

                dtfmu=250.9-407.4*amu_pl+190.9*amu_pl**2 #(T_fac-T_ph) multiplied by a factor depending on the 
                flux_pl=np.sum(dlp*pare_pl/(4*np.pi)*filter_trans)*((temp_ph+dtfmu)/(temp_fc))**4 #flux of the facula occuppied by the planet.
                flux[k] = flux[k] - flux_pl #total flux - flux blocked
                filling_fc[k] = filling_fc[k] - pare_pl
                filling_pl[k] = filling_pl[k] + pare_pl


        filling_ph[k]=100*filling_ph[k]/m.pi
        filling_sp[k]=100*filling_sp[k]/m.pi
        filling_fc[k]=100*filling_fc[k]/m.pi
        filling_pl[k]=100*filling_pl[k]/m.pi    
    
    return obs_times, flux/flxph, filling_ph, filling_sp, filling_fc, filling_pl


###############
#CCFS
###############
# @nb.njit(cache=True,error_model='numpy') 
# def fun_spot_bisect(ccf):
#     rv=-0.61095587*ccf**5 -0.27009652*ccf**4 + 3.20415179*ccf**3 -4.12503903*ccf**2 + 1.82468626*ccf + 0.19032404 #Polynomial fit to ccf in Fig 2 of Dumusque 2014
#     return rv

@nb.njit(cache=True,error_model='numpy') 
def fun_cifist(ccf,amu):
    '''Interpolate the cifist bisectors as a function of the projected angle
    '''
    # amv=np.arange(1,0.0,-0.1) #list of angles defined in cfist
    amv=np.arange(0.0,1.01,0.1) #list of angles defined in cfist

    idx_upp=np.searchsorted(amv,amu*0.999999999,side='right')
    idx_low=idx_upp-1


    cxm=np.zeros((len(amv),7)) #coeff of the bisectors. NxM, N is number of angles, M=7, the degree of the polynomial
    #PARAMS COMPUTED WITH HARPS MASK
    cxm[10,:]=np.array([-3.51974861,11.1702017,-13.22368296,6.67694456,-0.63201573,-0.44695616,-0.36838495]) #1.0
    cxm[9,:]=np.array([-4.05903967,13.21901003,-16.47215949,9.51023171,-2.13104764,-0.05153799,-0.36973749]) #0.9
    cxm[8,:]=np.array([-3.92153131,12.76694663,-15.96958217,9.39599116,-2.34394028,0.12546611,-0.42092905]) #0.8
    cxm[7,:]=np.array([-3.81892968,12.62209118,-16.06973368,9.71487198,-2.61439945,0.25356088,-0.43310756]) #0.7
    cxm[6,:]=np.array([-5.37213406,17.6604689,-22.52477323,13.91461247,-4.13186181,0.60271171,-0.46427559]) #0.6
    cxm[5,:]=np.array([-6.35351933,20.92046705,-26.83933359,16.86220487,-5.28285592,0.90643187,-0.47696283]) #0.5
    cxm[4,:]=np.array([-7.67270144,25.60866105,-33.4381214,21.58855269,-7.1527039,1.35990694,-0.48001707]) #0.4
    cxm[3,:]=np.array([-9.24152009,31.09337903,-41.07410957,27.04196984,-9.32910982,1.89291407,-0.455407]) #0.3
    cxm[2,:]=np.array([-11.62006536,39.30962189,-52.38161244,34.98243089,-12.40650704,2.57940618,-0.37337442]) #0.2
    cxm[1,:]=np.array([-14.14768805,47.9566719,-64.20294114,43.23156971,-15.57423374,3.13318175,-0.14451226]) #0.1
    cxm[0,:]=np.array([-16.67531074,56.60372191,-76.02426984,51.48070853,-18.74196044,3.68695732,0.0843499 ]) #0.0

    #interpolate
    cxu=cxm[idx_low]+(cxm[idx_upp]-cxm[idx_low])*(amu-amv[idx_low])/(amv[idx_upp]-amv[idx_low])

    rv = cxu[0]*ccf**6 + cxu[1]*ccf**5 + cxu[2]*ccf**4 + cxu[3]*ccf**3 + cxu[4]*ccf**2 + cxu[5]*ccf + cxu[6]
    return rv

@nb.njit(cache=True,error_model='numpy') 
def generate_rotating_photosphere_fast_rv(obs_times,Ngrid_in_ring,acd,amu,pare,rv,rv_ph,rv_sp,rv_fc,ccf_ph_tot,ccf_ph,ccf_sp,ccf_fc,fluxph,flpk,flsk,N,use_phoenix_mu,LD_law,LD1,LD2,spot_map,ref_time,Prot,diff_rot,Revo,Q,inc,vsini,CB,temp_ph,temp_fc,simulate_planet,esinw,ecosw,T0p,Pp,Rpl,b,a,alp):

    ######################## ROTATE PHOTSPHERE FOR EACH TIME ################################

    ccf=ccf_ph_tot*np.ones((len(obs_times),len(ccf_ph_tot))) #initialize total flux at each timestamp
    filling_sp=0.0+np.zeros(len(obs_times))
    filling_ph=m.pi+np.zeros(len(obs_times))
    filling_pl=0.0+np.zeros(len(obs_times))
    filling_fc=0.0+np.zeros(len(obs_times))

    for k,t in enumerate(obs_times):
        
        if simulate_planet:        
            planet_pos=compute_planet_pos(t,esinw,ecosw,T0p,Pp,Rpl,b,a,alp)#compute the planet position at current time. In polar coordinates!! 
        else:
            planet_pos = np.array([2.0,0.0,0.0],dtype=np.float64)

        if spot_map.size==0:
            spot_pos=np.expand_dims(np.array([m.pi/2,-m.pi,0.0,0.0]),axis=0)
        else:
            spot_pos=compute_spot_position(t,spot_map,ref_time,Prot,diff_rot,Revo,Q) #compute the position of all spots at the current time. Returns theta and phi of each spot.      

        #convert latitude/longitude of spot centre to XYZ
        vec_spot=np.zeros((len(spot_map),3))
        for i in range(len(spot_map)):
            xspot = m.cos(inc)*m.sin(spot_pos[i,0])*m.cos(spot_pos[i,1])+m.sin(inc)*m.cos(spot_pos[i,0])
            yspot = m.sin(spot_pos[i,0])*m.sin(spot_pos[i,1])
            zspot = m.cos(spot_pos[i,0])*m.cos(inc)-m.sin(inc)*m.sin(spot_pos[i,0])*m.cos(spot_pos[i,1])
            vec_spot[i,:]=np.array([xspot,yspot,zspot]).T #spot center in cartesian


        #Loop for each spot.
        for i in range(len(vec_spot)):
            
            if spot_pos[i][2]==0.0: #if radius is 0, go to next spot
                continue

            dist=m.acos(np.dot(vec_spot[i],np.array([1.,0.,0.]))) #Angle center spot to center of star. 

            if (dist-spot_pos[i,2]*np.sqrt(1.0+Q)) > (m.pi/2): #spot & facula not visible. Jump to next spot.
                continue
            
            beta=np.pi/2-dist #angle of te spot with the edge of the star
            alpha=spot_pos[i,2] #angle of the radii of the spot
            
            ############ FACULA PROJECTED AREA ##################
            if Q>0.0: #facula
                alphaf=spot_pos[i,2]*m.sqrt(1.0+Q) #angle of the radii of the faculae
                #CASE 1: FACULA OUTSIDE OUTSIDE-> NULL, ALREADY EVALUATED BEFORE
                
                #CASE 2: ALL FACULAE INSIDE -> ELLIPSE 
                if 0.0 < alphaf <= beta:
                    ay=m.sin(alphaf) #semiminor axis ellipse
                    ax=ay*m.sin(beta) #semimajor axis ellipse

                    Ape=ax*ay*m.pi/4.0
                    pare_fac =  2.0*2.0*Ape #area of ellipse (projected area of spot)
                    amu_fac=m.cos(dist)
                    rvel_fac=vsini*m.sin(spot_pos[i,0])*m.sin(spot_pos[i,1])

                #CASE 3: MOST OF THE FACULA INSIDE -> ELLIPSE + LUNE
                elif 0.0 <= beta < alphaf:
                    ay=m.sin(alphaf) #semiminor axis ellipse
                    ax=ay*m.sin(beta) #semimajor axis ellipse
                                    
                    yl=m.sqrt(1.0-(1.0-ay**2)/m.cos(beta)**2)   

                    ylay=min(1.0,max(yl/ay,-1.0)) #yl/ay, to avoid getting values >1 and errors in asin, due to floating points

                    Ape=ax*ay*m.pi/4.0 #y=ay
                    Apep= ax*ay*0.5*((ylay)*m.sqrt(1.0-(ylay)**2)+m.asin((ylay)))# y''=yl
                    Apcp= 0.5*(yl*m.sqrt(1.0-yl**2)+m.asin(yl)) - yl*m.cos(alphaf)*m.cos(beta)

                    pare_fac = 2.0*(2.0*Ape + Apcp - Apep) #proj. area is described in Urena et al. Stratified Sampling of Projected Spherical Caps
                    amu_fac=m.sin((beta+alphaf)/2.0) #representative mu as the mean point between edge of spot and of star
                    #position in polar:
                    r_fac=m.cos((beta+alphaf)/2.0)
                    t_fac=m.atan2(vec_spot[i,2],vec_spot[i,1])
                    #in spherical
                    x_fac=amu_fac
                    y_fac=r_fac*m.cos(t_fac)
                    z_fac=r_fac*m.sin(t_fac)
                    #in star coords
                    colat_fac, lon_fac = m.acos(z_fac*m.cos(-inc)-x_fac*m.sin(-inc)), m.atan2(y_fac,x_fac*m.cos(-inc)+z_fac*m.sin(-inc))
                    #rvel of spot
                    rvel_fac=vsini*m.sin(colat_fac)*m.sin(lon_fac)
                #CASE 4: MOST OF THE FACULA OUTSIDE-> LUNE
                elif 0.0 < (-beta) < alphaf:
                    ay=m.sin(alphaf) #semiminor axis ellipse
                    ax=ay*m.sin(beta) #semimajor axis ellipse
                                    
                    yl=m.sqrt(1.0-(1.0-ay**2)/m.cos(beta)**2)  #intesection x and y between ellipse and lune 

                    ylay=min(1.0,max(yl/ay,-1.0)) #yl/ay, to avoid getting values >1 and errors in asin, due to floating points

                    Apep= ax*ay*0.5*((ylay)*m.sqrt(1-(ylay)**2)+m.asin((ylay)))# y''=yl
                    Apcp= 0.5*(yl*m.sqrt(1.0-yl**2)+m.asin(yl)) - yl*m.cos(alphaf)*m.cos(beta)

                    pare_fac = 2.0*(Apep + Apcp) #proj. area is area of lune  
                    amu_fac=m.sin((beta+alphaf)/2.0)
                    #position in polar:
                    r_fac=m.cos((beta+alphaf)/2.0)
                    t_fac=m.atan2(vec_spot[i,2],vec_spot[i,1])
                    #in spherical
                    x_fac=amu_fac
                    y_fac=r_fac*m.cos(t_fac)
                    z_fac=r_fac*m.sin(t_fac)
                    #in star coords
                    colat_fac, lon_fac = m.acos(z_fac*m.cos(-inc)-x_fac*m.sin(-inc)), m.atan2(y_fac,x_fac*m.cos(-inc)+z_fac*m.sin(-inc))
                    #rvel of spot
                    rvel_fac=vsini*m.sin(colat_fac)*m.sin(lon_fac)

            ######### SPOT PROJECTED AREA ############
            #CASE 1: SPOT OUTSIDE-> NULL
            if 0.0 <= alpha <= (-beta):
                pare_spot=0.0
                amu_spot=0.0
            
            #CASE 2: ALL SPOT INSIDE -> ELLIPSE 
            elif 0.0 < alpha <= beta:
                ay=m.sin(alpha) #semiminor axis ellipse
                ax=ay*m.sin(beta) #semimajor axis ellipse

                Ape=ax*ay*m.pi/4.0
                pare_spot =  2.0*2.0*Ape #area of ellipse (projected area of spot)
                amu_spot=m.cos(dist)
                rvel_spot=vsini*m.sin(spot_pos[i,0])*m.sin(spot_pos[i,1])

            #CASE 3: MOST OF THE SPOT INSIDE -> ELLIPSE + LUNE
            elif 0.0 <= beta < alpha:
                ay=m.sin(alpha) #semiminor axis ellipse
                ax=ay*m.sin(beta) #semimajor axis ellipse
                                
                yl=m.sqrt(1.0-(1.0-ay**2)/m.cos(beta)**2)   

                ylay=min(1.0,max(yl/ay,-1.0)) #yl/ay, to avoid getting values >1 and errors in asin, due to floating points

                Ape=ax*ay*m.pi/4.0 #y=ay
                Apep= ax*ay*0.5*((ylay)*m.sqrt(1.0-(ylay)**2)+m.asin((ylay)))# y''=yl
                Apcp= 0.5*(yl*m.sqrt(1.0-yl**2)+m.asin(yl)) - yl*m.cos(alpha)*m.cos(beta)

                pare_spot = 2.0*(2.0*Ape + Apcp - Apep) #proj. area is described in Urena et al. Stratified Sampling of Projected Spherical Caps
                amu_spot=m.sin((beta+alpha)/2.0) #representative mu as the mean point between edge of spot and of star

                #position in polar:
                r_spot=m.cos((beta+alpha)/2.0)
                t_spot=m.atan2(vec_spot[i,2],vec_spot[i,1])
                #in spherical
                x_spot=amu_spot
                y_spot=r_spot*m.cos(t_spot)
                z_spot=r_spot*m.sin(t_spot)
                #in star coords
                colat_spot, lon_spot = m.acos(z_spot*m.cos(-inc)-x_spot*m.sin(-inc)), m.atan2(y_spot,x_spot*m.cos(-inc)+z_spot*m.sin(-inc))
                #rvel of spot
                rvel_spot=vsini*m.sin(colat_spot)*m.sin(lon_spot)
            #CASE 4: MOST OF THE SPOT OUTSIDE-> LUNE
            elif 0.0 < (-beta) < alpha:
                ay=m.sin(alpha) #semiminor axis ellipse
                ax=ay*m.sin(beta) #semimajor axis ellipse
                                
                yl=m.sqrt(1.0-(1.0-ay**2)/m.cos(beta)**2)  #intesection x and y between ellipse and lune 

                ylay=min(1.0,max(yl/ay,-1.0)) #yl/ay, to avoid getting values >1 and errors in asin, due to floating points

                Apep= ax*ay*0.5*((ylay)*m.sqrt(1-(ylay)**2)+m.asin((ylay)))# y''=yl
                Apcp= 0.5*(yl*m.sqrt(1.0-yl**2)+m.asin(yl)) - yl*m.cos(alpha)*m.cos(beta)

                pare_spot = 2.0*(Apep + Apcp) #proj. area is area of lune  
                amu_spot=m.sin((beta+alpha)/2.0)
                #position in polar:
                r_spot=m.cos((beta+alpha)/2.0)
                t_spot=m.atan2(vec_spot[i,2],vec_spot[i,1])
                #in spherical
                x_spot=amu_spot
                y_spot=r_spot*m.cos(t_spot)
                z_spot=r_spot*m.sin(t_spot)
                #in star coords
                colat_spot, lon_spot = m.acos(z_spot*m.cos(-inc)-x_spot*m.sin(-inc)), m.atan2(y_spot,x_spot*m.cos(-inc)+z_spot*m.sin(-inc))
                #rvel of spot
                rvel_spot=vsini*m.sin(colat_spot)*m.sin(lon_spot)       


            #Compute CCF ofspot an ph at amu
            if use_phoenix_mu:
                
                idx_upp=len(acd)-1-np.searchsorted(np.flip(acd),amu_spot*0.999999999,side='right') #acd is sorted inversely
                idx_low=idx_upp+1
                dlp = flpk[idx_low]+(flpk[idx_upp]-flpk[idx_low])*(amu_spot-acd[idx_low])/(acd[idx_upp]-acd[idx_low]) #limb darkening #limb darkening
                dls = flsk[idx_low]+(flsk[idx_upp]-flsk[idx_low])*(amu_spot-acd[idx_low])/(acd[idx_upp]-acd[idx_low]) #limb darkening

            else: #or use a specified limb darkening law
                ld=limb_darkening_law(LD_law,LD1,LD2,amu_spot)
                dlp = flpk[0]*ld
                dls = flsk[0]*ld


            flux_phsp=np.sum(dlp*pare_spot/(4*np.pi)) #flux of the photosphere occuppied by the spot.
            flux_spph=np.sum(dls*pare_spot/(4*np.pi)) #flux of the spot


            rv_phsp = rv_ph + rvel_spot + fun_cifist(ccf_ph,amu_spot)*1000.0*CB
            rv_spph = rv_sp + rvel_spot 
            ccf_phsp=interpolation_nb(rv,rv_phsp,ccf_ph,ccf_ph[0],ccf_ph[-1]) #still normalized ccf.
            ccf_spph=interpolation_nb(rv,rv_spph,ccf_sp,ccf_sp[0],ccf_sp[-1]) #still normalized ccf.
            #Compute RVshift, shift CCF, and iterpolate the CCF values. 
            CCF_phsp = ccf_phsp*flux_phsp/fluxph #the ccf of the element photosphere is the CCF weighted by the flux of the element over all the flux.
            CCF_spph = ccf_spph*flux_spph/fluxph

            if Q>0.0:

                pare_facula= pare_fac - pare_spot

                if use_phoenix_mu:                   
                    idx_upp=len(acd)-1-np.searchsorted(np.flip(acd),amu_fac*0.999999999,side='right') #acd is sorted inversely
                    idx_low=idx_upp+1
                    dlp = flsk[idx_low]+(flsk[idx_upp]-flsk[idx_low])*(amu_fac-acd[idx_low])/(acd[idx_upp]-acd[idx_low]) #limb darkening #limb darkening

                else: #or use a specified limb darkening law
                    ld=limb_darkening_law(LD_law,LD1,LD2,amu_fac)
                    dlp = flsk[0]*ld

                flux_phfc=np.sum(dlp*pare_fac/(4*np.pi)) #flux of the photosphere occuppied by the spot.

                dtfmu=250.9-407.4*amu_fac+190.9*amu_fac**2 #(T_fac-T_ph) multiplied by a factor depending on the 
                flux_fcph=np.sum(dlp*pare_fac/(4*np.pi))*((temp_ph+dtfmu)/(temp_fc))**4 #flux of the spot

                #Compute RVshift, shift CCF, and iterpolate the CCF values.                
                rv_phfc = rv_ph + rvel_fac + fun_cifist(ccf_ph,amu_fac)*1000.0*CB
                rv_fcph = rv_fc + rvel_fac 
                ccf_phfc=interpolation_nb(rv,rv_phfc,ccf_ph,ccf_ph[0],ccf_ph[-1]) #still normalized ccf.
                ccf_fcph=interpolation_nb(rv,rv_fcph,ccf_fc,ccf_fc[0],ccf_fc[-1]) #still normalized ccf.
                #Compute RVshift, shift CCF, and iterpolate the CCF values. 
                CCF_phfc = ccf_phfc*flux_phfc/fluxph #the ccf of the element photosphere is the CCF weighted by the flux of the element over all the flux.
                CCF_fcph = ccf_fcph*flux_fcph/fluxph

            else:
                CCF_phfc = ccf_ph*0.0 #the ccf of the element photosphere is the CCF weighted by the flux of the element over all the flux.
                CCF_fcph = ccf_fc*0.0  
                pare_facula = 0.0


            ccf[k] = ccf[k] - CCF_phsp + CCF_spph - CCF_phfc + CCF_fcph #total CCF - photosphere_spot + spot - photosphere_fac + facula
            filling_sp[k] = filling_sp[k] + pare_spot
            filling_ph[k] = filling_ph[k] - pare_spot - pare_facula
            filling_fc[k] = filling_fc[k] + pare_facula
        


        ################### PLANETARY TRANSIT PROJECTED AREA ########################

        if simulate_planet:
            if planet_pos[0]-planet_pos[2]>= 1.0: #all planet outside
                pare_pl = 0.0
                amu_pl = 0.0
                block = 'none'

            elif planet_pos[0]+planet_pos[2] <= 1.0: #all planet inside
                pare_pl = m.pi*planet_pos[2]**2 #area of a circle
                amu_pl = m.sqrt(1-planet_pos[0]**2) #cos(mu)**2=1-sin(mu)**2=1-r**2
                
                x_pl=amu_pl
                y_pl=planet_pos[0]*m.cos(planet_pos[1])
                z_pl=planet_pos[0]*m.sin(planet_pos[1])
                #in star coords
                colat_pl, lon_pl = m.acos(z_pl*m.cos(-inc)-x_pl*m.sin(-inc)), m.atan2(y_pl,x_pl*m.cos(-inc)+z_pl*m.sin(-inc))
                #rvel of spot
                rvel_pl=vsini*m.sin(colat_pl)*m.sin(lon_pl)     


                block='ph'
                for i in range(len(vec_spot)): #check if planet is over a spot or photosphere or faculae
                    distsp=m.acos(np.dot(vec_spot[i],np.array([1.,0.,0.]))) #Angle center spot to center of star. 
                    if (distsp-spot_pos[i,2]*np.sqrt(1.0+Q)) >= (m.pi/2): #spot & facula not visible. Jump to next spot.
                        block='ph'
                        continue 

                    dist=m.acos(np.dot(np.array([m.cos(m.asin(planet_pos[0])),planet_pos[0]*m.cos(planet_pos[1]),planet_pos[0]*m.sin(planet_pos[1])]),vec_spot[i])) #spot-planet centers distance
                    
                    if dist < spot_pos[i,2]: #if the distance is lower than spot radius, most of the planet is inside the spot
                        if (distsp-spot_pos[i,2]) >= (m.pi/2): #if spot is not visible
                            block='ph'
                        else:
                            block = 'sp'
                    elif dist < spot_pos[i,2]*m.sqrt(1+Q): #if the distance is lower than facula radius, most of the planet is inside the facula
                            block = 'fc'
                    elif (block != 'sp') and (block != 'fc'): #if the planet is not blocking a spot or a facula, then its blocking ph
                        block = 'ph' #else, the planet is blocking photosphere

            else: #the planet is partially covering the star.                
                d1=(1-planet_pos[2]**2+planet_pos[0]**2)/(2*planet_pos[0]) #dist from star centre to  intersection point
                d2=planet_pos[0]-d1 #dist from centre of planet to centre of intersection
                dedge = 1-(1+planet_pos[2]-planet_pos[0])/2 #dist from centre star to centre intersection
                pare_pl = m.acos(d1) - d1*m.sqrt(1-d1**2) + planet_pos[2]**2*m.acos(d2/planet_pos[2]) - d2*m.sqrt(planet_pos[2]**2-d2**2) #area of intersection star-planet
                amu_pl = m.sqrt(1-(dedge)**2) #amu is represented by the mean point of the intersection.
                #spherical coords of planet and corresponding RV of the surface
                x_pl=amu_pl
                y_pl=dedge*m.cos(planet_pos[1])
                z_pl=dedge*m.sin(planet_pos[1])
                #in star coords
                colat_pl, lon_pl = m.acos(z_pl*m.cos(-inc)-x_pl*m.sin(-inc)), m.atan2(y_pl,x_pl*m.cos(-inc)+z_pl*m.sin(-inc))
                #rvel of surface
                rvel_pl=vsini*m.sin(colat_pl)*m.sin(lon_pl)     

    
                block='ph'
                for i in range(len(vec_spot)): #check if planet is over a spot or photosphere or faculae
                    distsp=m.acos(np.dot(vec_spot[i],np.array([1.,0.,0.]))) #Angle center spot to center of star. 
                    if (distsp-spot_pos[i,2]*np.sqrt(1.0+Q)) >= (m.pi/2): #spot & facula not visible. Jump to next spot.
                        block='ph'
                        continue 

                    dist=m.acos(np.dot(np.array([m.cos(m.asin(dedge)),dedge*m.cos(planet_pos[1]),dedge*m.sin(planet_pos[1])]),vec_spot[i])) #distance from spot to centre of planet-star intersection

                    if dist < spot_pos[i,2]: #if the distance is lower than spot radius, most of the planet is inside the spot
                        if (distsp-spot_pos[i,2]) > (m.pi/2): #if spot is not visible
                            block='ph'
                        else:
                            block = 'sp'
                    elif dist < spot_pos[i,2]*m.sqrt(1+Q): #if the distance is lower than facula radius, most of the planet is inside the facula
                        block = 'fc'
                    elif (block != 'sp') and (block != 'fc'): #if the planet is not blocking a spot or a facula, then its blocking ph
                        block = 'ph' #else, the planet is blocking photosphere


            #compute and subtract flux blocked by the planet
            if block == 'ph': 
                if use_phoenix_mu:
                    idx_upp=len(acd)-1-np.searchsorted(np.flip(acd),amu_pl*0.999999999,side='right') #acd is sorted inversely
                    idx_low=idx_upp+1
                    dlp = flpk[idx_low]+(flpk[idx_upp]-flpk[idx_low])*(amu_pl-acd[idx_low])/(acd[idx_upp]-acd[idx_low]) #limb darkening #limb darkening
                else: #or use a specified limb darkening law
                    ld=limb_darkening_law(LD_law,LD1,LD2,amu_pl)
                    dlp = flpk[0]*ld

                flux_phpl=np.sum(dlp*pare_pl/(4*np.pi)) #flux of the photosphere occuppied by the planet.

                rv_phpl = rv_ph + rvel_pl + fun_cifist(ccf_ph,amu_pl)*1000.0*CB
                ccf_phpl=interpolation_nb(rv,rv_phpl,ccf_ph,ccf_ph[0],ccf_ph[-1]) #still normalized ccf.
                #Compute RVshift, shift CCF, and iterpolate the CCF values. 
                CCF_phpl = ccf_phpl*flux_phpl/fluxph #the ccf of the element photosphere is the CCF weighted by the flux of the element over all the flux.


                ccf[k] = ccf[k] - CCF_phpl #total flux - flux blocked
                filling_ph[k] = filling_ph[k] - pare_pl
                filling_pl[k] = filling_pl[k] + pare_pl


            if block == 'sp': #flux blocked by the planet
                if use_phoenix_mu:
                    idx_upp=len(acd)-1-np.searchsorted(np.flip(acd),amu_pl*0.999999999,side='right') #acd is sorted inversely
                    idx_low=idx_upp+1
                    dls = flsk[idx_low]+(flsk[idx_upp]-flsk[idx_low])*(amu_pl-acd[idx_low])/(acd[idx_upp]-acd[idx_low]) #limb darkening #limb darkening
                else: #or use a specified limb darkening law
                    ld=limb_darkening_law(LD_law,LD1,LD2,amu_pl)
                    dls = flsk[0]*ld

                flux_sppl=np.sum(dls*pare_pl/(4*np.pi)) #flux of the photosphere occuppied by the planet.

                rv_sppl = rv_sp + rvel_pl
                ccf_sppl=interpolation_nb(rv,rv_sppl,ccf_sp,ccf_sp[0],ccf_sp[-1]) #still normalized ccf.
                #Compute RVshift, shift CCF, and iterpolate the CCF values. 
                CCF_sppl = ccf_sppl*flux_sppl/fluxph #the ccf of the element photosphere is the CCF weighted by the flux of the element over all the flux.


                ccf[k] = ccf[k] - CCF_sppl #total flux - flux blocked
                filling_sp[k] = filling_sp[k] - pare_pl
                filling_pl[k] = filling_pl[k] + pare_pl

            if block == 'fc': #flux blocked by the spot
                if use_phoenix_mu:
                    idx_upp=len(acd)-1-np.searchsorted(np.flip(acd),amu_pl*0.999999999,side='right') #acd is sorted inversely
                    idx_low=idx_upp+1
                    dlp = flpk[idx_low]+(flpk[idx_upp]-flpk[idx_low])*(amu_pl-acd[idx_low])/(acd[idx_upp]-acd[idx_low]) #limb darkening #limb darkening
                else: #or use a specified limb darkening law
                    ld=limb_darkening_law(LD_law,LD1,LD2,amu_pl)
                    dlp = flpk[0]*ld

                dtfmu=250.9-407.4*amu_pl+190.9*amu_pl**2 #(T_fac-T_ph) multiplied by a factor depending on the 
                flux_fcpl=np.sum(dlp*pare_pl/(4*np.pi))*((temp_ph+dtfmu)/(temp_fc))**4 #flux of the facula occuppied by the planet.
                
                rv_fcpl = rv_fc + rvel_pl
                ccf_fcpl=interpolation_nb(rv,rv_fcpl,ccf_fc,ccf_fc[0],ccf_fc[-1]) #still normalized ccf.
                #Compute RVshift, shift CCF, and iterpolate the CCF values. 
                CCF_fcpl = ccf_fcpl*flux_fcpl/fluxph #the ccf of the element photosphere is the CCF weighted by the flux of the element over all the flux.

                ccf[k] = ccf[k] - CCF_fcpl #total ccf - ccf blocked
                filling_fc[k] = filling_fc[k] - pare_pl
                filling_pl[k] = filling_pl[k] + pare_pl


        filling_ph[k]=100*filling_ph[k]/m.pi
        filling_sp[k]=100*filling_sp[k]/m.pi
        filling_fc[k]=100*filling_fc[k]/m.pi
        filling_pl[k]=100*filling_pl[k]/m.pi    
    
    return obs_times, ccf, filling_ph, filling_sp, filling_fc, filling_pl






@nb.njit(cache=True,error_model='numpy') 
def check_spot_overlap(spot_map,Q): 
#False if there is no overlap between spots
    N_spots=len(spot_map)
    for i in range(N_spots):
        for j in range(i+1,N_spots):
            t_ini_0 = spot_map[i][0]
            t_ini = spot_map[j][0]
            t_fin_0 = t_ini_0 + spot_map[i][1]
            t_fin = t_ini + spot_map[j][1]
            r_0 = np.max(spot_map[i][4:6])
            r = np.max(spot_map[j][4:6])
            th_0 = m.pi/2-spot_map[i][2]*m.pi/180 #latitude in radians
            th = m.pi/2-spot_map[j][2]*m.pi/180 #latitude in radians
            ph_0 = spot_map[i][3]*m.pi/180 #longitude in radians
            ph = spot_map[j][3]*m.pi/180 #longitude in radians
            

            dist = m.acos(m.sin(th_0)*m.sin(th) + m.cos(th_0)*m.cos(th)*m.cos(m.fabs(ph_0 - ph)))*180/m.pi #in

            if (dist<m.sqrt(Q+1)*(r_0+r)) and not ((t_ini>t_fin_0) or (t_ini_0>t_fin)): #if they touch and coincide in time
                return True
            
    return False