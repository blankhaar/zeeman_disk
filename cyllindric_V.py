#Boy Lankhaar
#03/04/23
#
#Library of functions that are relevant to circular
#polarization ray tracing of a cyllindrically 
#symmetric simulation. 
#
#Requires external library: numpy
#
#List of functions:
###########LINE RADIATIVE TRANSFER FUNCTIONS###########
#phi_v_nonorm
#black_body_norm
#background_rad_norm
#S_cg2
#get_phi
#get_A
#prop_ray
#prop_ray_dust
#prop_ray_iso
#prop_ray_iso_dust
#rad_fac_meter
#rad_fac_pop_meter
#rad_fac_pop_au
#get_b
#source_fun
#CYLLINDRICAL COORINATES AND THE RAY TRACING FUNCTIONS#
#proj_z
#proj_psi
#proj_rc
#proj_pol
####### MAGNETIC AND VELOCITY FIELD PROPERTIES ########
#mag_props
#proj_mag
#eta_mag
#mag_strength
#proj_vel
##### READING ANS SETTIN UP SIMULATION PROPERTIES #####
#prep_zeeman
#prop_dust
######## CARVING A PATH THROUGH THE SIMULATION ########
#abc_form
#genesis
#get_h
#divide_steps
######### RAY TRACING THROUGH THE SIMULATION ##########
#get_props
#get_props_dust
#get_image
#
#
#
#
#
#######################################################
#######################################################
#######################################################
###########LINE RADIATIVE TRANSFER FUNCTIONS###########
#######################################################
#######################################################
#######################################################
#
#
#normalized (maximum to 1) Gaussian profile, centered at 
#v0 and with width b.
#
def phi_v_nonorm(v,v0,b):
#input:
#v      : velocity  
#v0     : line centre
#b      : line width
#
#output:
#       : normalized Gaussian profile
#
  return np.exp(-np.square(np.divide(np.subtract(v,v0),b)))
#
#
#Planck function in photon occupation number units
#
def black_body_norm(nu,T):
#input:
#nu     : frequency
#T      : temperature
#
#output:
#       : Planck function in photon occupation number units
#
  c2 = np.divide(4.79924e-11,T)
  fac = np.subtract(np.exp(np.multiply(c2,nu)),1.0)
  return np.divide(1.0,fac)
#
#
#Background radiation field (photon occ numbers) corresponding 
#to the CMB
#
def background_rad_norm(nu):
#input:
#nu     : frequency
#
#output:
#       : CMB Planck function in photon occupation number units 
#
  return black_body_norm(nu,2.73)
#
#
#
#line strength factors calculated using analytical expressions for
#the relevant Clebsch-Gordan coefficients. See also Eq. (3.16) of
#Landi Degl'Innocenti's book Polarization in Spectral Lines.
#
def S_cg2(j1,dJ,m,q):
#input:
#j1     : initial state angular momentum
#dJ     : angular momentum change in transition
#m      : initial state magnetic projection
#q      : projection change in transition
#
#output:
#
#       : line strength factor
#
    nj1 = 2.0*j1+1.0
    lj1 = j1*(j1+1.0)
#
    if dJ == 0:
        if q == 0:
            cg2 = m*m / (lj1)
        else:
            cg2 = (j1-q*m+1.0)*(j1+q*m) / (2.0*lj1)
    if dJ == 1:
        if q == 0:
            cg2 = (j1-m+1.0)*(j1+m+1.0) / (nj1*(j1+1.0))
        else:
            cg2 = (j1+q*m+1.0)*(j1+q*m) / (nj1*(2.0*j1+2.0))
    if dJ == -1:
        if q == 0:
            cg2 = (j1-m)*(j1+m) / (j1*nj1)
        else:
            cg2 = (j1-q*m+1.0)*(j1-q*m) / (2*j1*nj1)
#
    nJ = 2.0*(j1+dJ)+1.0
    return 3.0*cg2/nJ
#
#
#full line profile Zeeman splitted line
#see Landi 2004, Eqs.~(9.6) and (3.16) for a definition
#
def get_phi(fl,fu,xl,xu,vr,b,v0):
#input:
#fl,fu  : Angular momentum of lower and upper level 
#x1,x2  : Lande g-factors of lower and upper level in Doppler units
#x      : Doppler spectrum  
#
#output:
#phi    : line profiles for the sigma and pi transitions 
#
#
    f_m = np.zeros(vr.size)    
    f_0 = np.zeros(vr.size)    
    f_p = np.zeros(vr.size)    
#
#now loop over all transitions
    for nf in range(int(2*fl+1)):
        ml = 1.0*nf - fl
#
        mu = ml + 1         # m=-1 transitions 
        if abs(mu) <= fu:
            dE  = xu * mu - xl * ml                #E shift
            f_m += S_cg2(fl,fu-fl,mu,1)*phi_v_nonorm(vr,dE+v0,b)    
#
        mu = ml             # m=0 transitions 
        if abs(mu) <= fu:
            dE  = xu * mu - xl * ml                #E shift
            f_0 += S_cg2(fl,fu-fl,mu,0)*phi_v_nonorm(vr,dE+v0,b)
# 
        mu = ml - 1         # m=-1 transitions 
        if abs(mu) <= fu:
            dE = xu * mu - xl * ml
            f_p += S_cg2(fl,fu-fl,mu,-1)*phi_v_nonorm(vr,dE+v0,b)
    return f_m,f_0,f_p 
#
#polarized propagation matrices for the \sigma^{\pm} and \pi^0
#transitions. These will be multiplied by the relevant line
#profiles to be later inserted in the polarized radiative transfer
#equation.
#
def get_A(cth,eta):
#input:
#cth        : projection of propagation direction onto magnetic field
#eta        : position angle of magnetic field on the plane of the sky
#
#output:
#A_{p/0/m}  : propagation matrices for the \sigma^{\pm} and \pi^0 transitions
#
    A_p,A_0,A_m = np.zeros((4,4)),np.zeros((4,4)),np.zeros((4,4))
#
    cth2 = np.square(cth)
    sth2 = np.subtract(1.0,cth2)
#
    cet2 = np.cos(2.0 * eta)
    set2 = np.sin(2.0 * eta)
#
    A_p,A_m = (1.0+cth2)*np.eye(4),(1.0+cth2)*np.eye(4)
    A_0     = sth2 * np.eye(4) 
    A_p[0,1],A_m[0,1] = -cet2*sth2,-cet2*sth2
    A_p[0,2],A_m[0,2] = -set2*sth2,-set2*sth2
    A_p[0,3] = 2.0*cth
    A_m[0,3] = -2.0*cth
    A_p[1,0],A_p[2,0],A_p[3,0] = A_p[0,1],A_p[0,2],A_p[0,3]
    A_m[1,0],A_m[2,0],A_m[3,0] = A_m[0,1],A_m[0,2],A_m[0,3]
    A_0[0,1] = cet2 * sth2
    A_0[0,2] = set2 * sth2
    A_0[1,0],A_0[2,0] = A_0[0,1],A_0[0,2]
#
    return A_p,A_0,A_m 
#
#propagation of an incoming radiation field I0, through an isothermal
#slab of optical depth t and source function S. Propagation set up for
#an array of transitions and velocities. Transitions are Zeeman split
#by xl and xu.
#
def prop_ray(t,S,b,xl,xu,jl,ju,v0,b0,be,vr,I0):
#input:
#t      : optical depth
#S      : source function
#xl,xu  : Zeeman splitting lower/upper level
#jl,ju  : Angular momenum lower/upper level
#v0     : velocity of isothermal slab
#b0,be  : magnetic field projection and position angle
#vr     : velocity grid
#I0     : incoming radiation field
#
#output:
#       : outgoing radiation field
#

    A_p,A_0,A_m = get_A(b0,be)
#
    f_m,f_0,f_p =  get_phi(jl,ju,xl,xu,vr,b,v0)
    tt = np.multiply(np.ones((f_0.shape[-1])).T,t).T
# 
    t_0 = np.multiply(np.multiply(tt,f_0).T,np.ones((f_0.shape[-1],4,4)).T).T
    t_p = np.multiply(np.multiply(tt,f_p).T,np.ones((f_0.shape[-1],4,4)).T).T
    t_m = np.multiply(np.multiply(tt,f_m).T,np.ones((f_0.shape[-1],4,4)).T).T
#
    K   = np.add(np.add(np.multiply(t_0,A_0)/2.0,np.multiply(t_p,A_p)/4.0),np.multiply(t_m,A_m)/4.0)
#
    n = divide_steps(np.amax(t),1.0/50.0,0.0,0.0)     #divide up propagation steps into max(t)<0.01
    IS = np.zeros((f_0.shape[-1],4))
    IS[:,0] = np.multiply(np.ones((f_0.shape[-1])).T,S).T
#
    I_new = np.zeros((f_0.shape[-1],4))
    for i in range(n):
        if i > 0:
            I0 = I_new
        for j in range(4):
            I_new[:,j] = np.subtract(I0[:,j],np.sum(np.multiply(K[:,j,:]/n,np.subtract(I0,IS)),axis=-1))
#
    return I_new
#
#
#propagation of an incoming radiation field I0, through an isothermal
#slab of optical depth t and source function S. Propagation adapted
#to the dusty midplane 
#
#
def prop_ray_dust(t,S,vr,I0):
#input:
#t      : optical depth
#S      : source function
#vr     : velocity grid
#I0     : incoming radiation field
#
#output:
#       : outgoing radiation field
#
#
    t_d = np.multiply(np.multiply(t.T,np.ones((vr.size,4,4)).T).T,np.eye(4))
    K   = t_d
#
    IS = np.zeros((vr.size,4))
    IS[:,0] = np.multiply(np.ones((vr.size)).T,S).T
#
    n = divide_steps(np.amax(t),1.0/50.0,0.0,0.0)     #divide up propagation steps into max(t)<0.01
    IS = np.zeros((vr.size,4))
    IS[:,0] = np.multiply(np.ones((vr.size)).T,S).T
#
    I_new = np.zeros((vr.size,4))
    for i in range(n):
        if i > 0:
            I0 = I_new
        for j in range(4):
            I_new[:,j] = np.subtract(I0[:,j],np.sum(np.multiply(K[:,j,:]/n,np.subtract(I0,IS)),axis=-1))
#
    return I_new
#
#
#propagation of an incoming radiation field I0, through an isothermal
#slab of optical depth t and source function S. Propagation adapted
#to be unpolarized 
#
def prop_ray_iso(t,S,I0):
#input:
#t      : optical depth
#S      : source function
#I0     : incoming radiation field
#
#output:
#       : outgoing radiation field
#
  de_t = np.subtract(1.0,np.exp(-t))
  e_   = np.subtract(S.T,I0.T).T
  I    = np.multiply(e_,de_t)
  return np.add(I,I0)
#
#
#propagation of an incoming radiation field I0, through an isothermal
#slab of optical depth t and source function S. Propagation adapted
#to be unpolarized and for the dusty midplane
#
def prop_ray_iso_dust(t,S,I0,t_dust,s_dust):
#input:
#t      : optical depth
#t_dust : dust optical depth
#S      : source function
#s_dust : dust source function
#I0     : incoming radiation field
#
#output:
#       : outgoing radiation field
# 
  tt = np.add(t,t_dust)
  ss = np.add(np.multiply(np.divide(t,tt).T,S.T),np.multiply(np.divide(t_dust,tt).T,s_dust.T)).T 
  return prop_ray_iso(tt,ss,I0) 
#
#
#radiation factor used to compute the opacity/optical depth
#for a range of transitions
#
def rad_fac_meter(A,nu):
#input:
#A      : Einstein A-coefficent * gu 
#nu     : Frequency in Hz
#
#output:
#       : radiation factor
#
  lam = np.divide(2.998e8,nu)
  h_nu   = np.multiply(nu,6.626e-34)
  k0 = np.divide(np.multiply(np.power(lam,3.0),A),8.0*np.pi)
  return k0
#
#opacity for a range of transitions. Opacity units in per meter
#
def rad_fac_pop_meter(A,nu,N):
#input:
#A      : Einstein A-coefficent * gu 
#nu     : Frequency in Hz
#N      : populations
# 
#output:
#       : opacity 
#
  k0 = rad_fac_meter(A,nu)
  return np.multiply(k0,N.T).T
#
#opacity for a range of transitions. Opacity units in per au 
#
def rad_fac_pop_au(A,nu,N):
#input:
#A      : Einstein A-coefficent * gu 
#nu     : Frequency in Hz
#N      : populations
# 
#output:
#       : opacity in au-1 
#
  return rad_fac_pop_meter(A,nu,N)*1.496e11
#
#compute the total Doppler b parameter from the microturbulence
#and the temperature
#
def get_b(T,mass,bturb):
#input:
#T      : temperature 
#mass   : mass of the particel 
#bturb  : turbulent Doppler parameter
# 
#output:
#       : total Doppler parameter 
#
  btherm2 = np.multiply(1.66289e4,np.divide(T,mass))         #2kT/m 
  return np.sqrt(np.add(btherm2,np.square(bturb)))
#
#source function (normalized units) for a range of transitions
#
def source_fun(Nl,Nu):
#input:
#Nl,Nu  : lower and upper populations / gl,gu 
#
#output:
#       : source function normalized
#
  return np.divide(Nu,np.subtract(Nl,Nu))
#
#
#######################################################
#######################################################
#######################################################
#CYLLINDRICAL COORINATES AND THE RAY TRACING FUNCTIONS#
#######################################################
#######################################################
#######################################################
#
#
#projection of the z-direction onto the ray-tracing direction. 
def proj_z(th):
#input:
#th     : inclination angle
#
  return np.cos(th)
#
#
#projection of the psi-direction onto the ray-tracing direction. 
def proj_psi(th,ph,psi):
#input:
#th,ph      : ray-tracing direction angles (inclination,azimuth)
#psi        : atan2(y,x)=psi -- cyllindrical angle of the position.
#
#  return np.multiply(np.sin(th),np.sin(ph-psi))
  return np.multiply(np.sin(th),np.cos(psi-ph))
#
#
#projection of the rc-direction onto the ray-tracing direction. 
def proj_rc(th,ph,psi):
#input:
#th,ph      : ray-tracing direction angles (inclination,azimuth)
#psi        : atan2(y,x)=psi -- cyllindrical angle of the position.
#
  return np.multiply(np.sin(th),np.sin(psi-ph))
#  return np.multiply(-np.sin(th),np.cos(ph-psi))
#
#projection of poloidal direction onto the ray-tracing direction 
def proj_pol(th,ph,rn,pn,zn):
#input:
#th,ph      : ray-tracing direction angles (inclination,azimuth)
#pn         : atan2(y,x)=pn -- cyllindrical angle of the position.
#rn         : cyllindrical radius (x^2+y^2) of the position
#zn         : z-coordinate
#
#output:
#           : projection onto the poloidal direction
# 
#  R = np.sqrt(np.add(np.square(rn),np.square(zn)))
#  r = np.divide(rn,R)
#  z = np.divide(zn,R)
#  return np.add(np.multiply(r,proj_z(th)),np.multiply(z,proj_rc(th,ph,pn)))
  return proj_z(th)
#
#
#######################################################
#######################################################
#######################################################
####### MAGNETIC AND VELOCITY FIELD PROPERTIES ########
#######################################################
#######################################################
#######################################################
#
#
#define the magnetic properties. Magnetic field divided into a
#toroidal and vertical component:
#B = Bt * (r/1 AU)^pt * \hat{\phi} + Bv * (r/1 AU)^pv *\hat{z}
#
def mag_props():
#
#
#output:
#Bt,pt      : toroidal magnetic field parameters
#Bv,pv      : vertical magnetic field parameters
#
  Bt = 2.5*3.55355339e3
  pt = -1.5
  Bv = 0.8*3.55355339e2
  pv = -1.5
  return Bt,pt,Bv,pv
#
#
#compute the magnetic field projection from the location in the simulation
#and the ray-tracing direction, while calling the magnetic field properties.
#
def proj_mag(th,ph,pn,rn,zn):
#input:
#pn         : atan2(y,x)=pn -- cyllindrical angle of the position.
#rn         : cyllindrical radius (x^2+y^2) of the position
#zn         : z-coordinate
#Bt,Bv      : toroidal and vertical magnetic field
#pt,pv      : toroidal and vertical magnetic field r-dependance (power law)   
#
#output:
#p          : magnetic field projection 
#
#
  Bt,pt,Bv,pv = mag_props()
#
  bt = np.multiply(Bt,np.power(rn,pt))
  bv = np.multiply(Bv,np.power(rn,pv))
  B  = np.sqrt(np.add(np.square(bt),np.square(bv)))
#
  p = np.add(np.multiply(np.divide(bt,B),proj_psi(th,ph,pn)),np.multiply(np.divide(bv,B),proj_pol(th,ph,rn,pn,np.abs(zn))))
#
#  p = np.where(zn>=0.0,p,-p)    #where zn < 0, return -p
#new algorithm where only toroidal field changes sign 
  p = np.where(zn>=0.0,p,np.add(-np.multiply(np.divide(bt,B),proj_psi(th,ph,pn)),np.multiply(np.divide(bv,B),proj_pol(th,ph,rn,pn,np.abs(zn)))))    #where zn < 0, return -p for toroidal field
  p = np.where(np.isinf(p),0.0,p)   #where p = inf, return 0
#
  return p
#
#
#compute the magnetic field position angle from the location in the simulation
#and the ray-tracing direction, while calling the magnetic field properties.
#
def eta_mag(th,ph,pn,rn,zn):
#input:
#pn         : atan2(y,x)=pn -- cyllindrical angle of the position.
#rn         : cyllindrical radius (x^2+y^2) of the position
#zn         : z-coordinate
#Bt,Bv      : toroidal and vertical magnetic field
#pt,pv      : toroidal and vertical magnetic field r-dependance (power law)   
#
#output:
#eta        : angle between POS B and reference axis  
#
#
  Bt,pt,Bv,pv = mag_props()
#
  bt = np.multiply(Bt,np.power(rn,pt))
  bv = np.multiply(Bv,np.power(rn,pv))
  B  = np.sqrt(np.add(np.square(bt),np.square(bv)))
#
#  v_xPOS = -np.sin(th)
#  v_yPOS = 0.0
##  t_xPOS = -np.cos(th)*np.sin(ph-pn)
##  t_yPOS = -np.cos(ph+pn)
#  t_xPOS = np.cos(th)*np.sin(ph-pn)
#  t_yPOS = -np.cos(ph-pn)
#
#Eq. (13)
  v_yPOS = -np.sin(th)
  v_xPOS = 0.0
#  t_xPOS = -np.cos(th)*np.sin(ph-pn)
#  t_yPOS = -np.cos(ph+pn)
  t_yPOS = np.cos(th)*np.cos(pn-ph)
  t_xPOS = -np.sin(pn-ph)
#
  Bx = np.add(np.multiply(np.divide(bt,B),t_xPOS),np.multiply(np.divide(bv,B),v_xPOS)) 
  By = np.add(np.multiply(np.divide(bt,B),t_yPOS),np.multiply(np.divide(bv,B),v_yPOS))
#
#  Bx = np.where(zn>=0.0,Bx,-Bx)
#  By = np.where(zn>=0.0,By,-By)
#new algorithm where only toroidal field changes sign 
  Bx = np.where(zn>=0.0,Bx,np.add(-np.multiply(np.divide(bt,B),t_xPOS),np.multiply(np.divide(bv,B),v_xPOS)))
  By = np.where(zn>=0.0,By,np.add(-np.multiply(np.divide(bt,B),t_yPOS),np.multiply(np.divide(bv,B),v_yPOS)))
#
  eta = np.arctan2(By,Bx)
  eta = np.where(np.isinf(eta),0.0,eta)
#
  return eta
#
#
#compute the total magnetic field strength from the location in the simulation
#
def mag_strength(pn,rn,zn):
#input:
#pn         : atan2(y,x)=pn -- cyllindrical angle of the position.
#rn         : cyllindrical radius (x^2+y^2) of the position
#zn         : z-coordinate
#Bt,Bv      : toroidal and vertical magnetic field
#pt,pv      : toroidal and vertical magnetic field r-dependance (power law)   
#
#output:
#           : magnetic field strength 
#
  Bt,pt,Bv,pv = mag_props()
  bt = np.multiply(Bt,np.power(rn,pt))
  bv = np.multiply(Bv,np.power(rn,pv))
#
  return np.where(np.sqrt(np.add(np.square(bt),np.square(bv)))>1e5,1e5,np.sqrt(np.add(np.square(bt),np.square(bv))))    #set a cap on 100 Gauss 
#
#
#compute the projection of the velocity onto the ray-tracing direction. 
#
def proj_vel(th,ph,pn,rn,zn):
#input:
#th,ph      : ray-tracing direction angles (inclination,azimuth)
#pn         : atan2(y,x)=pn -- cyllindrical angle of the position.
#rn         : cyllindrical radius (x^2+y^2) of the position
#zn         : z-coordinate
#
#output:
#           : velocity vector projected onto the ray-tracing direction.
# 
  R  = np.sqrt(np.add(np.square(rn),np.square(zn)))     #in AU
  rf = np.multiply(rn,np.power(R,-1.5))                 #in AU
#
  M_star = 0.8                                          #in solar masses 
#
  v = np.multiply(29780.0,np.multiply(rf,np.sqrt(M_star)))      #in m/s
#
  return np.multiply(-v,proj_psi(th,ph,pn))
#
#
#######################################################
#######################################################
#######################################################
##### READING ANS SETTIN UP SIMULATION PROPERTIES #####
#######################################################
#######################################################
#######################################################
#
#compute the Zeeman shifts for a grid of coordinates using the
#input magnetic field and a range of Zeeman coefficients.
#
def prep_zeeman(rvec,zvec,vz):
#input:
#rvec,zvec  : grid of radius and height 
#vz         : Zeeman coefficients in m/s / mG 
#
#output:
#           : array of Zeeman splittings (t,n_rvec,n_zvec)
#
    Bt,pt,Bv,pv = mag_props() 
    rr,zz = np.meshgrid(rvec,zvec,indexing='ij')
    B0 = mag_strength(0.0,rr,0.0)
    dvz = np.multiply(np.ones((vz.size,*B0.shape)),B0)
    return np.multiply(dvz.T,vz).T 
#
#compute the dust properties (tau and source function) as a function of
#the cyllindrical distance.
#
def prop_dust(rc,Td,band):
#input:
#rc       : cyllindrical radius
#Td       : dust temperature as a function of radius
#
#output:
#t_dust     : dust optical depth (per au) at coordinates  
#
    td_50 = 0.7
#
    td = td_50 * np.power(rc/50.0,-1.0)       #approximate power law for Macias results 
    td = np.where(np.isinf(td),1e3,td) 

    if band == 3:
        nu = 113e9        #band 3
    elif band == 6:
        nu = 226e9        #band 6
    elif band == 7:       
        nu = 340e9        #band 7
    s_dust = black_body_norm(nu,Td) 
    return td,s_dust

#######################################################
#######################################################
#######################################################
######## CARVING A PATH THROUGH THE SIMULATION ########
#######################################################
#######################################################
#######################################################
#
#
#function:  abc_form    -- returns the roots of the equation c + b x + a x^2 = 0   
#
def abc_form(a,b,c):
#input:
#abc    :   see above
#
#output:
#       : see above
#
  a = np.multiply(2.0,a)
  c = np.multiply(2.0,c)
#
  d = np.subtract(np.square(b),np.multiply(a,c))
  D = np.divide(np.sqrt(d),a)
  ba = np.divide(-b,a)
  xp = np.add(ba,D)
  xm = np.subtract(ba,D)
  return xp,xm
#
#
#function genesis   -- point of introduction of light in the simulation
def genesis(x_POS,y_POS,h0,psi,th,ph):
#input:
#x_POS,y_POS: mesh of image coordinates 
#th,ph      : photon ray inclination and azimuth
#h0,psi     : height parameters for the emission surface 
#
#output:
#X0,Y0,Z0   : initial coordinates
#Xm,Ym,Zm   : midplane coordinates
#Xu,Yu,Zu   : front emission surface coordinates
#Xl,Yl,Zl   : back emission surface  coordinates
#
  f_z = proj_z(th)
#
  sth  = np.sin(th)
  cth  = np.cos(th)
  sth2 = np.square(sth)
  cth2 = np.square(cth)
#
  sph  = np.sin(ph)
  cph  = np.cos(ph)
  s2ph = np.sin(2.0 * ph)
#
#coordinates of the plane of the sky from the origin
  xp = np.subtract(np.multiply(x_POS,cph),np.multiply(y_POS,cth*sph))
  yp = np.add(np.multiply(x_POS,sph),np.multiply(y_POS,cth*cph))
  zp = np.multiply(y_POS,-sth)
#
#-ds to the end of the vertical height
  if cth == 0.0:
    zs = 1e30
  else:
    zs = -np.divide(np.add(100.0,zp),cth)  
#
#-ds to the end of the cyllindrical radius
  a = sth2
  b = 2.0 * np.multiply(sth,np.add(np.multiply(-sph,xp),np.multiply(cph,yp))) 
  c = np.subtract(np.add(np.square(xp),np.square(yp)),np.square(200.0))
  if a == 0.0:
    s_p,s_m = 1e30,-1e30 
  else:
    s_p,s_m = abc_form(a,b,c)
#
#if theres no solutions to s_m (=nan), then the ray doesn't cut through the simulation.
  ds = np.maximum(s_m,zs)
#
#check if we're inside the simulation -- otherwise !nan!   (r_cyll)
  ds = np.where(ds<s_p,ds,np.nan)
#
  X0 = np.add(xp,np.multiply(ds,-sth*sph))
  Y0 = np.add(yp,np.multiply(ds,sth*cph))
  Z0 = np.add(zp,np.multiply(ds,cth))
#
#check if we're inside the simulation -- otherwise !nan!   (z)
  X0 = np.where(np.abs(Z0)<=100.0,X0,np.nan)
  Y0 = np.where(np.abs(Z0)<=100.0,Y0,np.nan)
  Z0 = np.where(np.abs(Z0)<=100.0,Z0,np.nan)
#
#now we compute the associated coordinates of the midplane
  s_ = -Z0 / cth
  Xm = X0 - s_ * sth * sph 
  Ym = Y0 + s_ * sth * cph 
  Zm = Z0 + s_ * cth 
#
#we find the lower emission surface between the midplane and the initial coordinates 
  Xu = np.zeros_like(Xm)
  Yu = np.zeros_like(Xm)
  Zu = np.zeros_like(Xm)
  Xl = np.zeros_like(Xm)
  Yl = np.zeros_like(Xm)
  Zl = np.zeros_like(Xm)
#
  for (i,j),x in np.ndenumerate(Xm):
    y = Ym[i,j]
    z = Zm[i,j]
    h_max = 2.0 * get_h(np.sqrt(x*x+y*y),h0,psi) / cth
    s_upper = np.linspace(0.0,h_max,num=5000) 
    s_lower = np.linspace(-h_max,0.0,num=5000) 
    x_u = x - sth * sph * s_upper 
    x_l = x - sth * sph * s_lower
    y_u = y + sth * cph * s_upper 
    y_l = y + sth * cph * s_lower
    z_u = z + cth * s_upper 
    z_l = z + cth * s_lower
    h_u = get_h(np.sqrt(x_u*x_u+y_u*y_u),h0,psi)
    h_l = -get_h(np.sqrt(x_l*x_l+y_l*y_l),h0,psi)
    s_u_min = s_upper[np.argmin(np.abs(h_u-z_u))]
    s_l_min = s_lower[np.argmin(np.abs(h_l-z_l))]
    Xu[i,j] = x - sth * sph * s_u_min
    Yu[i,j] = y + sth * cph * s_u_min
    Zu[i,j] = z + cth * s_u_min
    Xl[i,j] = x - sth * sph * s_l_min
    Yl[i,j] = y + sth * cph * s_l_min
    Zl[i,j] = z + cth * s_l_min
#
  return X0,Y0,Z0,Xm,Ym,Zm,Xu,Yu,Zu,Xl,Yl,Zl
#
#get the height of the emission surface
#
def get_h(rc,h0,psi_h):
#input:
#rc         : cyllindrical radius 
#h0,psi     : height parameters for the emission surface 
#
#output:
#           : emission surface height 
#
  return h0 * pow(rc,psi_h)
#
#
#function:  divide_steps    -- considers, from the difference in velocity and magnetic
#field angles, if the propagation step has to be divided into multiple steps. 
#
def divide_steps(dv,b,db,dbc):
#input:
#dv,b   : velocity difference and Dopple b-parameter
#db,dbc : magnetic field projection and rejection difference
#
#output:
#n      : number of steps the propagation has to be divided in  
#
  nv = np.rint(np.divide(np.absolute(dv),0.2*b))   #at least a step of 0.2*b  
  nb = np.rint(np.divide(np.absolute(db),0.1))     #at least a step of 0.1 radian 
  nc = np.rint(np.divide(np.absolute(dbc),0.1))    #at least a step of 0.1 radian 
  n  = np.amax([nv,nb,nc])
  if (n == 0.0):
    n = 1.0
  return np.intc(n)
#
#######################################################
#######################################################
#######################################################
######### RAY TRACING THROUGH THE SIMULATION ##########
#######################################################
#######################################################
#######################################################
#
#collect all the propagation properties that we require
#to perform the polarized radiative transfer through the
#slabs characterized by their positions X,Y,Z. 
#
def get_props(X,Y,Z,th,ph,rc_f,tau,Su,b,xxl,xxu):
#input:
#X,Y,Z      : slab positions 
#th,ph      : photon ray inclination and azimuth
#rc_f       : fitting grid for tau,Su,b
#tau,Su     : tau and Su of all the transitions as a function of r_c 
#b          : Doppler parameters  
#xxl,xxu    : Zeeman coefficients of the transitions
#
#output:    
#TAU,S      : optical depth and source function of slabs
#bc         : Doppler parameter of slabs
#xl,xu      : Zeeman shifts of slabs
#v0         : velocities of slabs
#b0,be      : magnetic field projection and position angle of slabs
#
    Rc  = np.ravel(np.sqrt(X*X + Y*Y))
    Psi = np.ravel(np.arctan2(Y,X))
    Zn  = np.ravel(Z)
#
    v0    = np.reshape(proj_vel(th,ph,Psi,Rc,Zn),X.shape)
    b0    = np.reshape(proj_mag(th,ph,Psi,Rc,Zn),X.shape)
    be    = np.reshape(eta_mag(th,ph,Psi,Rc,Zn),X.shape)
    Bnorm = np.reshape(mag_strength(Psi,Rc,Zn),X.shape)
#
    TAU = np.zeros((xxl.size,*X.shape))
    S   = np.zeros((xxl.size,*X.shape))
    xl  = np.zeros((xxl.size,*X.shape))
    xu  = np.zeros((xxl.size,*X.shape))
#
    bc = np.reshape(np.interp(Rc,rc_f,b),X.shape)
    for i in range(xxl.shape[0]):
        TAU[i,:,:] = np.reshape(np.interp(Rc,rc_f,tau[i,:]),X.shape)
        S[i,:,:] = np.reshape(np.interp(Rc,rc_f,Su[i,:]),X.shape)
        xl[i,:,:] = xxl[i] * Bnorm
        xu[i,:,:] = xxu[i] * Bnorm
    return TAU,S,bc,xl,xu,v0,b0,be 
#
#
#collect all the propagation properties that we require
#to perform the radiative transfer through the dusty midplane
#characterized by their positions X,Y,Z. 
#
def get_props_dust(X,Y,Z,rc_f,tau,Sd):
#input:
#X,Y,Z      : slab positions 
#th,ph      : photon ray inclination and azimuth
#rc_f       : fitting grid for tau,Su,b
#tau,Sd     : tau and Source function of the dust as a function of r_c 
#
#output:    
#TAU,S      : optical depth and source function of slabs
#
    Rc  = np.ravel(np.sqrt(X*X + Y*Y))
#
    TAU = np.reshape(np.interp(Rc,rc_f,tau),X.shape)
    S  = np.reshape(np.interp(Rc,rc_f,Sd),X.shape)
    return TAU,S
#
#
#function:  get_image   -- get total and polarized intensity image 
def get_image(x_POS,y_POS,th,ph,rc_f,tau,Su,b,tau_dust,Sd,xxl,xxu,jl,ju,vr,B0,h0,psi,dust):
#input:
#x_POS,y_POS: mesh of image coordinates 
#th,ph      : photon ray inclination and azimuth
#rc_f       : fitting grid for tau,Su,b
#tau,Su     : tau and Su of all the transitions as a function of r_c (from LIME) 
#tau_dust,Sd: tau and Su of the dust as a function of r_c (from LIME) 
#b          : Doppler parameters  
#xxl,xxu    : Zeeman coefficients of the transitions
#jl,ju      : angular momenta of the transition states 
#vr         : velocity grid
#dvz        : possible additional velocity shift (for instance through Zeeman splitting) 
#B0         : incoming radiation field strength.
#h0,psi     : height parameters for the emission surface 
#dust       : switch to consider dust or not
#
#output:    
#I_im       : image of the outgoing light (Stokes I) at the mesh of image coordinates 
#Q_im       : image of the outgoing light (Stokes Q) at the mesh of image coordinates 
#U_im       : image of the outgoing light (Stokes U) at the mesh of image coordinates 
#V_im       : image of the outgoing light (Stokes V) at the mesh of image coordinates 
#I0_im      : image of the outgoing light (Stokes I) at the mesh of image coordinates 
#             assuming no magnetic field present
#
    debug = False 
    if debug:
        for i in range(B0.size):
            print('i,xl,xu: ',i,xxl[i],xxu[i])
#
    X0,Y0,Z0,Xm,Ym,Zm,Xu,Yu,Zu,Xl,Yl,Zl = genesis(x_POS,y_POS,h0,psi,th,ph)
#
    I_im = np.zeros((B0.size,vr.size,*x_POS.shape))
    Q_im = np.zeros((B0.size,vr.size,*x_POS.shape))
    U_im = np.zeros((B0.size,vr.size,*x_POS.shape))
    V_im = np.zeros((B0.size,vr.size,*x_POS.shape))
    I0_im = np.zeros((B0.size,vr.size,*x_POS.shape))
    tau_im = np.zeros((B0.size,vr.size,*x_POS.shape))
#
#if X0,Y0,Z0 are nan, we don't need to ray-trace, as the ray will miss the 
#simulation.
#
#get the relevant line properties
    TAU_l,S_l,l_b,l_xl,l_xu,l_v0,l_b0,l_be = get_props(Xl,Yl,Zl,th,ph,rc_f,tau,Su,b,xxl,xxu)
    TAU_u,S_u,u_b,u_xl,u_xu,u_v0,u_b0,u_be = get_props(Xu,Yu,Zu,th,ph,rc_f,tau,Su,b,xxl,xxu)
    if debug:
        for i in range(B0.size):
            print('transition number, i: ',i)
            Q = np.argwhere(~np.isnan(X0))
            for j in range(Q.shape[0]):
                jx = Q[j,0]  
                jy = Q[j,1] 
                print('i,jx,jy: ',i,jx,jy) 
                print('tau, S, b, lower: ',TAU_l[i,jx,jy],S_l[i,jx,jy],l_b[jx,jy]) 
                print('xl/b,xu/b lower: ',l_xl[i,jx,jy]/l_b[jx,jy],l_xu[i,jx,jy]/l_b[jx,jy]) 
                print('cos(th),eta lower: ',l_b0[jx,jy],l_be[jx,jy]) 
                print('tau, S, b, upper: ',TAU_u[i,jx,jy],S_u[i,jx,jy],u_b[jx,jy])
                print('xl/b,xu/b upper: ',u_xl[i,jx,jy]/u_b[jx,jy],u_xu[i,jx,jy]/u_b[jx,jy]) 
                print('cos(th),eta upper: ',u_b0[jx,jy],u_be[jx,jy]) 
            print(' ')
#
#if we run a dust simulation, get the relevant dust properties
    if dust:
        TAU_d,S_d = get_props_dust(Xm,Ym,Zm,rc_f,tau_dust,Sd)
#
#if X0,Y0,Z0 are nan, we don't need to ray-trace, as the ray will miss the 
#simulation.
#
    Q = np.argwhere(~np.isnan(X0))
    for i in range(Q.shape[0]):
        ix = Q[i,0]
        iy = Q[i,1]
        if debug:
            print('position (ix,iy):',ix,iy)
        for j in range(xxl.size):
#
            if debug:
                print('transition number, i: ',j)
            I = np.zeros((vr.size,4))
            I[:,0] = B0[j]
            I0 = np.zeros((vr.size))
            I0[:] = B0[j]
#            if debug:
#                print('input')
#                print(*I[:,0])	
#                print(*I[:,1])	
#                print(*I[:,2])	
#                print(*I[:,3])	
#
#propagation with polarization
            I = prop_ray(TAU_l[j,ix,iy],S_l[j,ix,iy],l_b[ix,iy],l_xl[j,ix,iy],l_xu[j,ix,iy],jl[j],ju[j],l_v0[ix,iy],l_b0[ix,iy],l_be[ix,iy],vr,I)
            I0 = prop_ray_iso(TAU_l[j,ix,iy]*phi_v_nonorm(vr,l_v0[ix,iy],l_b[ix,iy]),S_l[j,ix,iy],I0)
            if debug:
#                print('first')
#                print(*I[:,0])   
#                print(*I[:,1])   
#                print(*I[:,2])   
#                print(*I[:,3])   
#                print('test FWHM')
                g1,g2_0,g2_1 = get_zeeman_facs(l_xl[j,ix,iy],l_xu[j,ix,iy],jl[j],ju[j])
                q_av  = (g2_0+g2_1)/(g1*g1)
                q_del = (g2_1-g2_0)/(g1*g1)
                I[:,0] = I[:,0] - np.amin(I[:,0])
                I0 = I0 - np.amin(I0)
                print('test FWHM: ',(get_FWHM(vr,I[:,0])-get_FWHM(vr,I0))/l_b[ix,iy],0.84*(g1/l_b[ix,iy])*(g1/l_b[ix,iy])*(q_av + q_del*l_b0[ix,iy]*l_b0[ix,iy]))	
                print('test pV: ',get_pV(I[:,0],I[:,3]),0.86*(g1/l_b[ix,iy])*l_b0[ix,iy])	
                print('test pQ: ',get_pV(I[:,0],np.cos(2.0*l_be[jx,jy])*I[:,1]+np.sin(-2.0*l_be[jx,jy])*I[:,2]),0.83*(g1/l_b[ix,iy])*(g1/l_b[ix,iy])*(q_del*(1.0-l_b0[ix,iy]*l_b0[ix,iy])))	
                print('g1,q_av,q_del: ',g1,q_av,q_del)	
            if dust:
                I = prop_ray_dust(TAU_d[ix,iy],S_d[ix,iy],vr,I)
                I0 = prop_ray_iso(TAU_d[ix,iy]*np.ones(vr.size),S_d[ix,iy],I0)
#            if debug:
#                print('post dust')
#                print(*I[:,0])   
#                print(*I[:,1])   
#                print(*I[:,2])   
#                print(*I[:,3])   
            I = prop_ray(TAU_u[j,ix,iy],S_u[j,ix,iy],u_b[ix,iy],u_xl[j,ix,iy],u_xu[j,ix,iy],jl[j],ju[j],u_v0[ix,iy],u_b0[ix,iy],u_be[ix,iy],vr,I)
            I0 = prop_ray_iso(TAU_u[j,ix,iy]*phi_v_nonorm(vr,u_v0[ix,iy],u_b[ix,iy]),S_u[j,ix,iy],I0)
#
#            if debug:
#                print('last')
#                print(*I[:,0])
#                print(*I[:,1])
#                print(*I[:,2])
#                print(*I[:,3])

            I_im[j,:,ix,iy] = I[:,0]
            Q_im[j,:,ix,iy] = I[:,1] 
            U_im[j,:,ix,iy] = I[:,2] 
            V_im[j,:,ix,iy] = I[:,3] 
#
            I0_im[j,:,ix,iy] = I0 

#            if debug:
#                print('I0')
#                print(*I0)
#
    return I_im,Q_im,U_im,V_im,I0_im

