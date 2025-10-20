# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:55:01 2022

@author: php20jo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import splev, splrep



def conv(x,height,position,std,A,l1,B,C,l3):
    l2=(1/0.125)
    g=height*np.exp(-(x-position)**2/(2*std**2))
    e=(A * (np.exp(-(l1 * x))) + B * (np.exp(-(l2 * x))) + C * (np.exp(-(l3 * x))))# + D)
    return (np.convolve(g,e,mode='full') / sum(e) )[:667]

def flat(x,D):
    y=D
    return y

#%%load sample

sample = 46

#files = np.loadtxt('callum_sample_list.txt',dtype=str)
#files = np.loadtxt('Simon_sample_list.txt',dtype=str)


files = np.loadtxt('samples.txt',dtype=str)
                                                                       
filename = files[sample] + '_T1'

label = filename.partition('/')#[2] #2 for soaked #0 for original (removes filename)
print(label)
#%%
file = np.loadtxt('240328-10-47_T1' + '.dat', comments = 'S')
#%%
a = len(file[3:])

time = np.linspace(-10,50,a)

N = np.zeros(a)

for i in range(len(N)):
    N[i] = file[i+3]
    
Nerr=(np.sqrt(N))

#%%Initial guess

#exp initial guess
iA = 1000000
il1=1/0.3
iB = 1000000
iC = 1700
il3 = 1/1.6
iD = 25

#gauss initial guess
iheight= 36651
iposition=1.2
istd=0.21

#%%find background

zero = 333   #fit background before rise (before zero)

xb = time[:zero]  #only use x & y & error before zero
yb = N[:zero] 
Nerrb = Nerr[:zero]
back, backpcov = curve_fit(flat,xb,yb,p0=(iD),sigma=Nerrb)  #fit flat backgound

Nb = N-back #remove background from all y data

#%%useful data

#fit from 0 - 25 ns

start = 333
stop = 1000

y = Nb[start:stop]  #useful section of background removed y data
x = time[start:stop] 

#error needs to be derived from the original value for N
##Nerr defined at top
Nberr=Nerr[start:stop]

#%%fit convolution function
popt, pcov = curve_fit(conv,x,y,p0=(iheight,iposition,istd,iA,il1,iB,iC,il3),bounds=(0,np.inf),sigma=Nberr)

height,position,std,A,l1,B,C,l3 = popt #define optimised variables
l2 = 1/0.125 #define L2 because it is fixed in the function

t3 = 1/l3 #calculate the oPs lifetime

#%%calculate FV

TaoEldrup=np.load('TaoEldrup.npy') #import TaoEldrup bubble model with 10,000 points (calculated from equation)

TaoEldrup_x=TaoEldrup[0,:] 
TaoEldrup_y=TaoEldrup[1,:]

spl = splrep(TaoEldrup_x, TaoEldrup_y)

FV = splev(t3,spl) # calculate the FV for the measured oPs lifetime

#%%Print results

print('height= ',height,'position= ',position,'std= ',std,'\nA= ',A,'\nB= ',B,'\nC= ',C,'\nD= ',back,'\nt1= ',1/l1,'\nt2= ',1/l2,'\nt3= ',1/l3)

FWHM = 2*np.sqrt(2*np.log(2))*std
FWHM = FWHM * 1000





perr = np.sqrt(np.diag(pcov))

#error

errsd = perr[2]
errFWHM = (errsd/std)*FWHM
print('FWHM = ',FWHM,'+/-',errFWHM)


print('l3 err = ',perr[-1])

print('Free volume = ', FV)

FVe = (perr[-1]/l3)*FV

print('free volume error = ', FVe)

#%%Intensity

A1 = A/l1     #Area under exponential decay
A2 = B/l2
A3 = C/l3
At = A1+A2+A3
I1 = A1/At    #Intensity is proportional to area under curve: Klym, H.; Annihilation Lifetime Spectroscopy Insight on Free Volume Conversion of Nanostructured MgAl2O4 Ceramics. Nanomaterials 2021, 11, 3373
I2 = A2/At
I3 = A3/At
It = I1+I2+I3
I3e = (perr[-1]/l3)*I3

print('I1 = %f I2 = %f I3 = %f It = %f' %(I1,I2,I3,It))
print('I3 err = ', I3e)

#%%FFT

FFV3 = I3*FV
FFV3e = (perr[-1]/l3)*FFV3

print('FFV = ',FFV3)
print('FFV error = ',FFV3e)

#%%Resolution
#fit data

# =============================================================================
# def gaus (x,a,x0,sigma):
#     return a*np.exp(-(x-x0)**2/(2*sigma**2))
# 
# 
# #just the data i want to model (the rise, beteen 0 and the peak a 1.2)
# #go from 0-2 (333-400)
# # -10-50 = 2000 points 
# # 0.44 = 347, 1.6 = 386.6 (387)
# start = 345
# stop = 387
# 
# FWHM = 2*np.sqrt(2*np.log(2))*std
# FWHM = FWHM * 1000
# #FWHM = str(FWHM)
# #error
# perrsd = perr[2]
# perrFWHM = (perrsd/std)*FWHM
# #perrsd = errsd*1000
# #perrsd = str(errsd)
# 
# print('FWHM = ', FWHM, 'ps +/- ', perrFWHM )
# 
# =============================================================================


#%%plot fit

# =============================================================================
# plt.figure()
# plt.errorbar(x,y,yerr=Nberr,label='data')
# plt.plot(x,conv(x,height,position,std,A,l1,B,C,l3),label='fit')
# plt.title(label)
# plt.xlabel('Time [ns]')
# plt.ylabel('Counts')
# plt.legend()
# 
# =============================================================================
#%%plot all lines

# =============================================================================
# 
# plt.figure()
# plt.plot(x,y,label='data')
# plt.plot(x,conv(x,height,position,std,A,l1,B,C,l3),label='fit')
# plt.plot(x,gaus(x,height,position,std),label='gaus')
# plt.plot(x,single(x,A,l1),label='free positron')
# plt.plot(x,single(x,B,l2),label='pPs')
# plt.plot(x,single(x,C,l3),label='oPs')
# plt.xlabel('Time [ns]')
# plt.ylabel('Counts')
# plt.title('DGEBF')
# plt.legend()
# 
# =============================================================================

