from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import h5py
import scidata.carpet.hdf5 as hdf5
# Import math Library
import math
from pylab import *
import scipy.interpolate
import pandas as pd
import seaborn as sbn
import sys
import csv

rc('font', weight='bold')
#constants
gammac=4.68*(10**(-19))
delta=6
x_1=189/(367*((math.pi)**2))
x_2=21/((367*math.pi)**4)
x_3=3/((1835*math.pi)**6)


#This function uses scidata-routines in order to
#read the data. It takes as parameters the file, iteration,
#refinement-level (which grid) and the component.
def get_data(df, it, tl, rl):
    grid = df.get_reflevel (iteration=it, timelevel=tl, reflevel=rl)
    xxx, yxx, zxx = grid.mesh ()
    #    xxy,yxy=grid.mesh()
    datxx = df.get_reflevel_data (grid, iteration=it)
    #    datxy=df.get_reflevel_data(grid,iteration=it)
    return xxx, yxx, zxx, datxx


######################################################################
#The data can be  find einstein toolkit page
datafilerho = hdf5.dataset ("rho.xyz.all.h5")
datafiletemp = hdf5.dataset ("temperature.xyz.all.h5")
datafileye = hdf5.dataset ("Y_e.xyz.all.h5")

ktt = []
rate = []
rl = 4;
tl = 0;
it=0;
old=0;
tt=1024*38




# We can see evolution over time by utilizing the while loop.
while it < tt:
    it=it+1024;
    #    kt+=1
    kt=((it*0.028235)*4.926)*0.001
    ktt.append(kt)
    time = float("{0:.2f}".format(kt))
    old=old+1
    dff = pd.read_csv ('{}'.format(old), sep='\s+')
    xxx, yxx, zxx, rho = get_data (datafilerho, it, tl, rl)  # 6.column
    xxx, yxx, zxx, temp = get_data (datafiletemp, it, tl, rl)
    xxx, yxx, zxx, ye = get_data (datafileye, it, tl, rl)
    
    
    #This method converts array data to a dat frame so that we can  apply certain restrictions to the data using pandas.
    When convert function convert temperature (tempn), density(rhon)  and electron fraction(yen),
    convertt function  convert x and y axes to data frame.
    The difference beetwen axes and other parameters is while parameters keep as  a masked array on 3 dimension axes values
    keeping just array.
    
    #explanitation of the units can be find on the referans [1]
    
    in order to
#read the data.
    def convert(data):
        data = data[data.mask == False]
        adata = array (data)
        ss = pd.Series (adata)
#    ssn = 369.52 * ss
        ssdd = ss.to_frame ()
    #    xxy,yxy=grid.mesh()
    #    datxy=df.get_reflevel_data(grid,iteration=it)
        return ssdd

    rhon=convert(rho)*369.52
    tempn=convert(temp)
    yen=convert(ye)
    
    def convertt(axes):
        ax=axes.flatten ()
        axr = array (ax)
        ex = pd.Series (axr)
        exx = ex.to_frame ()
        return exx
    xax=convertt(xxx)
    yax=convertt(yxx)


    rhonn = rhon.columns = ['rho']
    tempnn = tempn.columns = ['temp']
    yenn = yen.columns = ['ye']
    xaxn = xax.columns = ['xx']
    yaxn = yax.columns = ['yy']



    # constructing a data frame consisting of temperature, density electron fraction x, and axis
    pdm = pd.concat ([tempn, rhon, yen, xax, yax], axis=1)
 

    # Applying the constraint here. It's mentioned in the readme's references [3].
    use = pdm[(pdm['rho'] > 1.00e-12) & (pdm['rho'] < 326) & \
                (pdm['temp'] > 1.00e-01) & (pdm['temp'] < 1.00e+01) & \
                (pdm['ye'] > 1.00e-02) & (pdm['ye'] < 6.00e-01)]


#    print(dff.shape)
    #To convert a baryon density in nuclear saturation density units from a mass density in geometrized units.
    bden=array (dff[dff.columns[1]])*(2300)
    Temp=array (dff[dff.columns[0]])
    dmu = array (dff[dff.columns[4]])
    bsum=-gammac*(Temp**6)*dmu
    
    # Using referance [2], determine the net equilibration rate
    rated=bsum*(1+(x_1*((dmu/Temp)*1)+x_2*((dmu/Temp)*2)+x_3*((dmu/Temp)*3)))


    use = use[:-1]
    print(use.shape)
    print(dff.shape)
    rrx = array (use[use.columns[3]])
    rry = array (use[use.columns[4]])


    print(type(rry))
    

    print(rate.shape)
#    print(xax.shape)

    print(rry.shape)
   # Using contourf, create a two-dimensional rate plot.
    import pandas as pd
    import numpy as np
    from scipy.interpolate import griddata
    import matplotlib.pyplot as plt
    
    plt.clf ()

    resolution = str (50) + 'j'
    X, Y = np.mgrid[min (rrx):max (rrx):complex (resolution), min (rry):max (rry):complex (resolution)]
    points = [[a, b] for a, b in zip (rrx, rry)]

    M = griddata (points, rate, (X, Y), method='linear')

    CS = plt.contourf (X, Y, M, 15, cmap='cividis')
    plt.tick_params (axis="x", labelsize=12)
    plt.tick_params (axis="y", labelsize=12)
    plt.xlabel ("$x$", fontsize=13)
    plt.ylabel ("$y$", fontsize=13)
    cbar = plt.colorbar ()
    plt.title(''r'$\Gamma$ ({} ms)'.format(time),fontweight="bold")

    plt.savefig('rate{}.png'.format(time))
    
    
