# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 10:13:21 2021

@author: Balu
"""

import os
import math
import scipy.odr
import scipy.optimize
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from matplotlib import cm
from matplotlib.colors import ListedColormap #, LinearSegmentedColormap

FOLDER='spectra'

#astronomical constants
DISTANCE_TO_GALAXY=[780,40] #kpc

#universal constants
LINE_FREQ = 1420.405751768 #Hz
SPEED_OF_LIGHT=299792.458 #km/s
GRAVITATIONAL_CONSTANT=6.67408*10**(-11)
SOLAR_MASS_UNIT=1.989*10**30

#user threshold settings for integration
INT_RANGE=(-710,50)
THR=0.2
ZEROTH_MOMENT_LIM=15

#Fitted parameters, now used as constants
CENTRAL_SPECTRUM=74
POSITION_ANGLE=-37
INCLINATION=77

#notes of some outliers
fosok=[102,111,112,36,45,47, 54, 55,56, 65]
fosok2=[35,110,85,72]
sus=[85,17,26,66,72] #85 csak közel 74-eshez, (66 és 72 is?), 16,26 just mu1 thershold
sus2=[63,110,65] #Na ezek wth, kicsi error is
scale = 1
#scale = (5/3.25)

#list outliers
outliers = {"65":-506.64621277,"85":-295.35748} #str(CENTRAL_SPECTRUM):0
# ,"85":-295.35748

COLORMAPS1=["hot","RdBu","coolwarm","bwr","seismic"]

def data_reader(file_name, folder_name):
    """
    This function opens a file and converts it
    into 2 numpy arrays (header, spectrum).
    Both arrays have two columns.

    Parameters
    ----------
    file_name : string
        The name of the file.
    folder_name : string
        The name of the folder (or folders)
        realtive to the location of the program.

    Returns
    -------
    header : numpy array
        Contains strings and floats, which descirbe
        important informations and parameters about the spectrum.
    spectrum : numpy array
        Only contains floats.
        Each row is a data point of the form: [intensity, channel number].

    """

    header=np.array(['START','START'])
    spectrum=np.array([0.0,0])

    path = os.getcwd()+"\\" + folder_name+"\\"+file_name
    with open(path, 'r+') as input_data:
        lines=input_data.readlines()

        for line in lines:

            if line == 'DATA\n':
                break
            split_up = line.split('=')

            if len(split_up) == 2:
                header=np.vstack((header,[split_up[0].strip(),
                                          split_up[1].strip()]))

        for line in lines:
            while line != 'DATA\n':
                break
            split_up = line.split('/')
            if len(split_up) == 2:
                spectrum=np.vstack((spectrum,[float(split_up[0]),
                                              int(split_up[1])]))
    header=np.delete(header,0,0)
    spectrum=np.delete(spectrum,0,0)
    header[0][1]=header[0][1].strip("'").strip()

    return header, spectrum

def list_files(foldername):
    """
    This function lists the files in a certain folder.

    Parameters
    ----------
    foldername : string
        The name of the folder (or folders)
        realtive to the location of the program
    Returns
    The list of filenames inside the folder
    -------
    filenames : string

    """

    filenames=[]
    for filename in os.listdir(os.getcwd()+"\\"+foldername):
        filenames=np.append(filenames,filename)
    return filenames

def dec2rad(x):
    """
    This function converts declination into radians

    Parameters
    ----------
    x : float
        The declination in the form (deg.arcmin).

    Returns
    -------
    float
        The declination in radians.
    """

    arc_min=np.round(np.mod(x,1),2)

    return np.deg2rad(x-arc_min+arc_min*10/6)

def ra2rad(x):

    """
    This function converts right ascension into radians

    Parameters
    ----------
    x : float
        The declination in the form (h.min).

    Returns
    -------
    float
        The declination in radians.
    """
    return np.deg2rad((x-np.mod(x,1))*15 + np.mod(x,1)*25)

def dec_converter(x):
    """

    This function converts the declination into a string,
    which can be used as an axis label
    Parameters
    ----------
    x : float
        in form deg.arcmin.

    Returns
    -------
    string
        in the form: "deg°arcmin'".

    """
    arc_min=np.round(np.mod(x,1),2)
    degree=int((x-arc_min))
    return str(degree)+'°'+str(int(arc_min*100))+"'"

def ra_converter(x):

    """

    This function converts the right ascension into a string,
    which can be used as an axis label
    Parameters
    ----------
    x : float
        in form h.min.

    Returns
    -------
    string
        in the form: "hʰminᵐ'".
    """
    minute=np.mod(x,1)
    minute *= 100
    return str(int(minute))+'ᵐ'

def scientific_formatting(Value,uncertainty):
    magnitude = math.floor(np.log(Value)/np.log(10))
    return f"{Value/10**magnitude:.2f} ± {uncertainty/10**magnitude:.2f} e{magnitude}"


class Spectrum:
    def __init__(self,file_name,folder_name):
        """
        This function initializes an object in the class of Spectrum.
        based on the file's location and names,
        by using the data reader_function.'

        Parameters
        ----------
        file_name : string
            The name of the file.
        folder_name : string
            The name of the folder (or folders)
            realtive to the location of the program.
        Returns
        Spectrum object
        -------
        The object has many attributres,collected from the header and spectrum,
        the main attributes are the:

        The source: self.source,
            name of the file (without the file extension)

        Spectrum (self.spec),
            An array in the form [intensity, velocity].
            In units of [K] (kelvin) and [kms⁻¹].

        Header (self.hdr),
            The array containing the useful informaitons,
            the important parameters have their own attributes (can be seen below).

        The right ascension and declination (self.ra and self.dec),
            these are both floats in the form:
            (h.min and deg.arcmin) respectively.

        The step size between channels (self.dv, self.dnu),
            in both frequency and velocity
            These are floats in the unit of [kms⁻¹] and [s⁻¹]:

        The azimuthal angle and elevation: (self.phi and self.theta):
            These are basicly the right ascension and the declination,
            converted into radians using the dec2rad and ra2rad functions.

        The length of the spectra (self.len):
            The number of data poins
        """

        hdr, spec = data_reader(file_name,folder_name)
        self.hdr=hdr
        self.source=hdr[0][1]
        self.number = int(self.source[4:])
        self.ra = round(float(hdr[np.where(hdr=='RA')[0][0]][1]),2)
        self.dec = round(float(hdr[np.where(hdr=='DEC')[0][0]][1]),1)
        self.ch1 = float(hdr[np.where(hdr=='%V_OF_FIRST_CHANNEL')[0][0]][1])
        self.dv = float(hdr[np.where(hdr=='DV')[0][0]][1])*scale
        self.dnu = float(hdr[np.where(hdr=='DNU')[0][0]][1])*scale
        self.el = float(hdr[np.where(hdr=='EL')[0][0]][1])
        self.phi=ra2rad(self.ra)
        self.theta=dec2rad(self.dec)
        spec[:,1] = self.ch1 + self.dv*(spec[:,1]-1)
        self.spec=spec
        self.len=len(spec)


    def plot(self):
        """
        This function plots the spectrum
        Returns
        -------
        plot
            The plot of the spctrum:
            Where the x-axis is velocity [kms⁻¹],
            and the y-axis is intensity [K].

        """
        data=np.transpose(self.spec)
        plt.plot(data[1], data[0])
        plt.title(self.source)
        return plt.show()

    def zeroth_moment(self,int_range=INT_RANGE,threshold=THR):
        """
        This function calculates the zeroth moment of the spectrum

        Parameters
        ----------
        int_range : tuple, optional
            The range of the integration. The default is INT_RANGE.
        threshold : float, optional
            The threshold of the integration,
            values with absolute value below this limit will be ignored.
            The default is THR.

        Returns
        -------
        numpy array
            The zeroth moment and the ucertainty in a numpy array:
            [zeroth_moment, uncertainty] both have unit [K kms⁻¹].

        """
        summ=0.00001

        for row in self.spec:
            if int_range[0] <= row[1] <= int_range[1]\
            and np.abs(row[0]) >= np.abs(threshold):
                summ += row[0]

        mu_0  = summ*self.dv
        sigma = threshold*np.sqrt(self.len)*self.dv
        return np.array([mu_0, sigma])

    def first_moment(self,int_range=INT_RANGE,threshold=THR):
        """
        This function calculates the first moment of the spectrum

        Parameters
        ----------
        int_range : tuple, optional
            The range of the integration. The default is INT_RANGE.
        threshold : float, optional
            The threshold of the integration,
            values with absolute value below this limit will be ignored.
            The default is THR.

        Returns
        -------
        numpy array
            The first moment and the ucertainty in a numpy array:
            [first_moment, uncertainty] both are in [kms⁻¹].

        """
        summ=0.00001
        v2summ=0.00001

        #ch_n=0 unused line variable?
        for row in self.spec:
            if int_range[0] <= row[1] <= int_range[1]\
            and np.abs(row[0]) >= np.abs(threshold):
                summ += row[0]*row[1]
                v2summ += row[1]**2

        zeroth_moment=self.zeroth_moment(int_range, threshold)
        first_moment=self.dv*summ/zeroth_moment[0]
        sigma2sum=(self.dv*summ*zeroth_moment[0])**2+v2summ*((threshold*summ)**2)

        uncertainty=np.sqrt((first_moment**2*(sigma2sum/summ**2
                                      +(zeroth_moment[1]/
                                        zeroth_moment[0])**2))**(1/2))

        uncertainty=np.sqrt((self.dv**2+(v2summ*(threshold/zeroth_moment[0])**2)\
                     +(summ*zeroth_moment[1]/zeroth_moment[0]**2)**2))
            
        if (num:=str(self.number)) in outliers:
            first_moment = outliers[num]
        #print("UC:",(self.source,first_moment,uncertainty))
        return np.array([first_moment,uncertainty])


    def column_density(self):
        """
        The column density of the spectrum.
        The number of atoms in the angular area of the spectrum:

        Returns
        -------
        numpy array
            An array in the form [column_density, uncertainty]
            both in the unit of [atoms per cm²].

        """

        return 3.848*10**20*self.zeroth_moment(INT_RANGE,THR)*self.dnu/self.dv

    def rel_coord(self,spectrum_0):
        """
        Returns the coordinates relative to another spectrum in the plane
        of the projection of the galaxy.
        Parameters
        ----------
        spectrum_0 : Spectrum type object
            The spectrum in the origo of the new coordinates,
            (the new coordinates are measured with respect to this spectrum).

        Returns
        -------
        Two numpy array
            The coordinates in the form [x,y],
            where x and y are in kiloparsec [kpc].
            And the corresponding uncertainties.

        """
        r=DISTANCE_TO_GALAXY[0]
        delta_phi=self.phi-spectrum_0.phi
        delta_theta=self.theta-spectrum_0.theta
        avg_theta= np.mean((self.theta,spectrum_0.theta))
        delta_x=delta_phi*np.cos(avg_theta)*r
        delta_y=delta_theta*r
        coordinate=np.array([delta_x,delta_y])
        
        sigma_r = DISTANCE_TO_GALAXY[1]
        sigma_theta = 0
        sgima2_cos = (sigma_theta*np.sin(avg_theta))**2
        sigma2_x = delta_x**2*((sigma_r/r)**2 + sgima2_cos/((np.cos(avg_theta))**2))
        sigma2_y = (delta_theta*sigma_r)**2
        uncertainty=np.array([sigma2_x,sigma2_y])

        return coordinate, uncertainty
    
    
class Spectra:

    def __init__(self,folder_name):
        """
        This function initializes an object in the class of Spectra.
        based on the folders location.

        Parameters
        ----------

        folder_name : string
            The name of the folder (or folders)
            realtive to the location of the program.
        Returns
        Spectra type object,
        This object collects multiple spectrums
        -------
        The object has three attributres:

            The folder name, and filenames (self.foldername, self.filenames),
                The name of the folder.
                A list of the filenames in a folder
                (uses the list_files function)

            spectra (self.spectra),
                returns an array of the spectrum objects from the folder.

        """

        self.filenames=list_files(folder_name)
        self.foldername=folder_name
        spectra=[]
        for name in self.filenames:
            spectra.append(Spectrum(name,folder_name))
        self.spectra=spectra


    def get_spectrum(self,spec_number):
        """


        Parameters
        ----------
        spec_number : int
            The spectrums number.
            (the spectrum number specifies ## in "M31P##")
        Returns
        -------
        Spectrum object
            The Spectrum with the number.

        """
        filename='M31P' + str(spec_number) + '.ASC'
        if np.isin(self.filenames, filename).any():
            return self.spectra[np.where(self.filenames  == filename)[0][0]]
        else:
            raise FileNotFoundError(f"No spectrum named {filename} found!")
            #return None #just for clarity, function without return by default return None

    def number_of_atoms(self):
        """
        This method calculates the number of
        neutral hydrogen atoms in the galaxy,
        by integrating the column densities over the solid angle.


        Returns
        -------
        numpy array
            An array containing the value and the uncertainty:
            [# of atoms, uncertainty].

        """
        n=0
        for spec in self.spectra:
            dn=spec.column_density()*np.sin(spec.theta)\
                *dec2rad(0.2)*ra2rad(0.02)\
                *(DISTANCE_TO_GALAXY[0]*3.0857*10**21)**2
            n+=dn
        result=n[0]
        uncertainty=np.sqrt(n[1]**2+(2*n[0]
                            *DISTANCE_TO_GALAXY[1]/DISTANCE_TO_GALAXY[0])**2)
        return np.array([result,uncertainty])

    def mass(self):
        """
        This method derives the mass of the neutral hydrogen in the galaxy

        Returns
        -------
        numpy array
            An array containing the value and the uncertainty:
            [mass, uncertainty] in unit of solar mass.

        """
        return self.number_of_atoms()*8.411172625*10**(-58)

    def average_velocity(self):
        """
        This method calculates the average velocity of the galaxy:

        Returns
        -------
        average velocity : numpy array
            [avg, uncertainty] both are in [kms⁻¹].

        """
        sum_1 = 0
        n = 0
        sigma2_sum = 0
        for spectrum in self.spectra:      
            sum_1 += spectrum.first_moment()[0]
            sigma2_sum += spectrum.first_moment()[1]
            n += 1
        avg = sum_1/n
        sigma=np.sqrt(sigma2_sum)/n
        return np.array([avg,sigma])

    def relative_coordinates(self,spectrum_0):
        """
        Returns the coordinates of each spectrum
        in the plane of the projection of the galaxy,
        relative to the central spectrum .
        Parameters
        ----------
        spectrum_0 : Spectrum type object
            The spectrum in the origo of the new coordinates,
            (the new coordinates are measured with respect to this spectrum).

        Returns
        -------
        2 numpy array

            The relative coordinates of each spectrum
            The corresponding uncertainties of each spectrum

           (All values are in kiloparsec [kpc]).

        """


        coordinates=np.array([[0,0]])
        uncertainties=np.array([[0,0]])

        for spectrum in self.spectra:
            if np.abs(spectrum.zeroth_moment()[0]) >= ZEROTH_MOMENT_LIM:

                coordinate, uncertainty = spectrum.rel_coord(spectrum_0)
    
    
                coordinates=np.vstack((coordinates, coordinate))
                uncertainties=np.vstack((uncertainties, uncertainty))

        return coordinates[1:], uncertainties[1:]

    def relative_velocities(self, spectrum_0):


        """
        This method calculates the component of velocity
        in the direction of the optical axis, relative to the central spectrum,
        for each spectrum.

        Parameters
        ----------
        spectrum_0 : Spectrum type object
            The central velocity.

        Returns
        -------
        velocities : numpy array
            The velocities in [kms⁻¹].
        sigma_vel : TYPE
            The velocities in [kms⁻¹].

        """
        r_velocity=spectrum_0.first_moment()

        velocities=np.array([])
        sigma2_vel=np.array([])
        
        for spectrum in self.spectra:
            if np.abs(spectrum.zeroth_moment()[0]) >= ZEROTH_MOMENT_LIM:
                first_moment = spectrum.first_moment()
                velocity=first_moment[0]-r_velocity[0]
    
                velocities=np.append(velocities, velocity)
                sigma2_vel=np.append(sigma2_vel, first_moment[1]**2\
                                     +r_velocity[1]**2)
         
        return velocities, sigma2_vel

    def deproject(self,center,pa,incl):
        """
        This method deprojects the coordinates of each spectrum,
        simple linear transformations.

        Parameters
        ----------
        center : Spectrum object
           The central spectrum.
        pa : float
            Major axis position angle (North Eastwards).
        incl : float
            Inclination between line of sight and polar axis of a galaxy
.

        Returns
        -------
        new_coordinates : numpy array
            An array containing the transformed coordinates.
        new_uncertainties : numpy array
             An array containing the uncertainties
             on the transformed coordinates.

        """

        coordinates, uncertainties = self.relative_coordinates(center)

        alpha=np.deg2rad(pa)
        beta=np.deg2rad(incl)

        rot_mat=np.array([[np.cos(alpha),-np.sin(alpha)],
                          [np.sin(alpha),np.cos(alpha)]])
        
        stretch_mat=np.array([[np.cos(beta),0],[0,1]])

        inverse_transformation=np.linalg.inv(rot_mat.dot(stretch_mat))
        
        inverse_uc_transformation=inverse_transformation**2

        new_coordinates=np.array([[0,0]])
        new_uncertainties=np.array([[0,0]])

        for coord,uc in zip(coordinates, uncertainties):
            new_coord=inverse_transformation.dot(coord)
            new_coordinates=np.vstack((new_coordinates,new_coord))
            new_uc = inverse_uc_transformation.dot(uc)
            new_uncertainties=np.vstack((new_uncertainties,new_uc))

        return new_coordinates[1:], new_uncertainties[1:]

    def rotation_curve(self,center,pa,incl):
        """
        This method calculates the data for the rotation curve:

        Parameters
        ----------
        center : Spectrum object
           The central spectrum.
        pa : float
            Major axis position angle (North Eastwards).
        incl : float
            Inclination between line of sight and polar axis of a galaxy

        Returns
        -------
        radii : numpy array
            Array containing the radial distance
            from the distance of each spectrum.
        sigma_radii : numpy array
            Contains the uncertainty of each radius.
        rotation_velocities : numpy array
            Contains the magnitude of the rotational velocities
            for each spectrum.
        sigma_rotvel : numpy array
            The uncertainty of each rotational velocity.

        """

        velocities, sigma2_vel = self.relative_velocities(center)
        coordinates, sigma2_xy=self.deproject(center,pa,incl)
        data = zip(velocities, sigma2_vel, coordinates, sigma2_xy )
    
        radii=np.array([])
        sigma2_radii=np.array([])
        rotation_velocities=np.array([])
        sigma2_rotvel=np.array([])

        for velocity,velocity_sigma2,coordinate,coordinate_sigma2 in data:
            if coordinate[0]!=0 and coordinate[1]!=0:
            
                x=coordinate[0]
                y=coordinate[1]
                radius=(x**2 + y**2)**(0.5)
                sigma2_radius=4*(((x/radius)**2)*coordinate_sigma2[0]\
                                +((y/radius)**2)*coordinate_sigma2[1])
                #print(coordinate_sigma2,sigma2_radius)
                
                velocity_mag=np.abs(velocity/(y/radius))
                sigma2_vel_mag=velocity_mag**2*( (velocity_sigma2/(velocity_mag**2))\
                                                +(sigma2_radius/((radius)**2))\
                                                +(coordinate_sigma2[1]/(y**2)) )
    
                radii=np.append(radii,radius)
                sigma2_radii=np.append(sigma2_radii,sigma2_radius)
    
                rotation_velocities=np.append(rotation_velocities, velocity_mag)
                sigma2_rotvel=np.append(sigma2_rotvel,sigma2_vel_mag)
            
        sigma2_radii = np.sqrt(np.abs(sigma2_radii))
        sigma2_rotvel = np.sqrt(np.abs(sigma2_rotvel))
        
        print("RAD 20",radii[20])
        
        return radii, sigma2_radii, rotation_velocities, sigma2_rotvel
    
def custom_cmap(data, avg):
    minimum = np.amin(np.array(data))
    maximum = np.amax(np.array(data))
    #avg = (minimum+maximum)/2
    diff = maximum-minimum
    
    top = cm.get_cmap('Blues', 128)
    middle = cm.get_cmap('binary',128)
    bottom = cm.get_cmap('Oranges_r', 128)
    
    #print(minimum,maximum,diff)
    w = 0.2*(diff)
    n = int((avg-minimum-(w/2))*256/diff)
    o = int((w)*256/diff)
    m = int((maximum-avg-(w/2))*256/diff)
    #print(n,o,m)
    newcolors = np.vstack((middle(np.linspace(0,0,0)),
                           top(np.linspace(1, 0, n)),
                           middle(np.linspace(0,0,o)),
                           bottom(np.linspace(1, 0, m))))
    
    newcmp = ListedColormap(newcolors, name='OrangeBlue')
    return newcmp

def contour_plot(spectra,n,colormap="hot",custom_colors=True,\
                 controrcolors=None,labels=False):

    mesh=np.zeros((12,9))
    x_axis=np.array([])
    y_axis=np.array([])

    for spectrum in spectra.spectra:
        x_axis=np.append(x_axis, spectrum.ra)
        y_axis=np.append(y_axis, spectrum.dec)
        number = spectrum.number
        x_index=int(np.mod(number,10))
        y_index=int((120-(number-x_index))/10)

        if n==0:
            mesh[y_index,x_index]=spectrum.zeroth_moment(INT_RANGE,THR)[0]
        else:
            mesh[y_index,x_index]=spectrum.first_moment(INT_RANGE,THR)[0]
        if labels:
            plt.text(x_index,y_index,spectrum.number)

    x_axis=np.unique(x_axis)
    y_axis=np.unique(y_axis)

    x_axis=reversed(list(map(ra_converter,x_axis)))
    y_axis=reversed(list(map(dec_converter,y_axis)))

    plt.xticks(np.arange(0,9),x_axis,size=6)
    plt.yticks(np.arange(0,12),y_axis,size=6)

    plt.xlabel("Right-ascension")
    plt.ylabel("Declination")
    plt.title("Controur plot of M31")

    c = plt.contour(mesh,10,colors=controrcolors)  
    
    avg_v = M31.average_velocity()[0]
    mesh[mesh==0] = avg_v
    mesh[mesh==1] = avg_v
    if custom_colors and n==1:
        colour_data = np.array(mesh) #- M31.average_velocity()[0]
        custom_colormap = custom_cmap(colour_data,avg_v)
        plt.imshow(mesh,cmap=custom_colormap)
    else:
        plt.imshow(mesh,cmap=colormap)
    
    #plt.contourf(X, Y, mesh)
    
    return plt.show(),c



#rotation curve models
#def model_0(B,x):
#    return (B[0]*x/(1+B[1]*x))

def model_0(B,x):
    """if B[0]==0: B[0]=0.01
    if B[1]==0: B[1]=0.01
    for idd,j in enumerate(x):
        if x[idd]==0: x[idd]=0.1"""
    return np.sqrt(np.abs(B[0]*(B[1]**2)*(1-(B[1]/x)*np.arctan(x/B[1]))))

def model_1(B,r):
    GM=B[0]
    r_0=B[1]
    b=B[2]
    v=(GM/r*(1+b*(1+r/r_0)))**0.5
    return v

def model_2(B,r):
    GM=B[0]
    R_0=B[1]
    b=B[2]
    r_c=B[3]
    beta=B[4]
    v=(GM/r*((R_0/r_c)**0.5*r/(r+r_c))**(3*beta)*(1+b*(1+r/R_0)))**0.5
    return v
  
def results_0(B):
    M=B[1]*B[0]/(GRAVITATIONAL_CONSTANT)
    return M, (B[0]/B[1], 0)

def results_0(B,SD_B):
    M = 0
    v = np.sqrt(np.abs(B[0]*(B[1]**2)))
    err_v = SD_B[1]*1/(4*np.sqrt(B[0]))
    print(err_v)
    return M, (v,err_v)

def results_1(B):
    
    M=B[0]/(GRAVITATIONAL_CONSTANT*SOLAR_MASS_UNIT)
    v_f=(B[0]*B[2]/B[1])**0.5
    
    return M, v_f

def results_2(B):
    
    M=B[0]/(GRAVITATIONAL_CONSTANT*SOLAR_MASS_UNIT)
    v_f=(B[0]*B[2]/B[1]*(B[1]/B[3])**(1.5*B[4]))**0.5
    
    return M, v_f

models=[model_0,model_1, model_2]
model_results=[results_0,results_1, results_2]

#rotation curve and angle fitting
def fit_rotation(function,RADII,VEL,SIGMA_R,SIGMA_VEL):
    modelll = scipy.odr.Model(function)
    mydata = scipy.odr.RealData(RADII,VEL,SIGMA_R,SIGMA_VEL)
    myodr = scipy.odr.ODR(mydata, modelll, beta0=[1,1,1,1,1])
    myoutput = myodr.run()
    #myoutput.pprint()
    
    return myoutput.beta,myoutput.sd_beta,myoutput.res_var
  
def fit_rot_parametric(alpha,beta,model=model_1,center=CENTRAL_SPECTRUM):
    RADII, SIGMA_R, VEL, SIGMA_VEL = M31.rotation_curve(M31.\
                    get_spectrum(center),alpha, beta)
    #print(model,center)
    beta,sd_beta,rchi = fit_rotation(model,RADII,VEL,SIGMA_R,SIGMA_VEL)
    return beta,sd_beta,rchi

def plot_rotation(postiion_angle,inclination,fit_results,model=model_1\
    ,center=CENTRAL_SPECTRUM, labels=False,xlims=(5,35),ylims=(0,400)):
    
    RADII, SIGMA_R, VEL, SIGMA_VEL = M31.rotation_curve(M31.get_spectrum(center),
                                                        postiion_angle,inclination)
    if ylims[0]!=ylims[1]: plt.ylim(ylims)
    if xlims[0]!=xlims[1]: plt.xlim(xlims)
    plt.errorbar(RADII,VEL,xerr=SIGMA_R,yerr=SIGMA_VEL,fmt='bo')
    
    if labels:
        numbers = [spec.number for spec in M31.spectra \
                   if np.abs(spec.zeroth_moment()[0]) >= ZEROTH_MOMENT_LIM\
                       and spec.number!=74]
        for r,v,num in zip(RADII,VEL,numbers):
            plt.text(r+0.3,v+6,str(num))
    
        print(len(RADII),len(numbers))
    #fit_results = fit_rotation(model_0,RADII, SIGMA_R, VEL, SIGMA_VEL)
    betaparams,sd_beta,rchi = fit_results
    #beta,sd_beta,rchi = fit_rotation(fit1,RADII, SIGMA_R, VEL, SIGMA_VEL)
    #print(f"The parameter X is {betaparams[0]:.2f} \u00B1 {sd_beta[0]:.2f} UNIT.")
    fitted = lambda r: model(betaparams,r)
    t = np.linspace(0,max(RADII),num=1000)
    plt.plot(t, fitted(t), linewidth=5, color='red', label=f"Line of best fit, $\chi_R$ ={rchi:.2f}")
    
    plt.xlabel("Radius from center of M31 [kpc]")
    plt.ylabel("Tangential velocity [km/s]")
    plt.title("Rotation curve of M31")
    plt.legend()
    plt.show()
    
def fit_and_plot(postiion_angle,inclination,center=CENTRAL_SPECTRUM,model=model_2,\
                 labels=False,xlims=(0,40),ylims=(0,450)):
    fit_results = fit_rot_parametric(postiion_angle,inclination,model=model,center=center)
    plot_rotation(postiion_angle, inclination, fit_results,model=model,\
                  center=center,labels=labels,xlims=xlims,ylims=ylims)
    #fit_and_plot(POSITION_ANGLE, INCLINATION)
    return fit_results
    
def fit_angles(method="CG",model=model_1):
    fit_function = lambda params: fit_rot_parametric(params[0],params[1])[2]
    fit = scipy.optimize.minimize(fit_function,[-30,80]\
                    ,method = method,options={"disp":True},jac=False) #"maxiter":10,
    print(fit)
    print(method,fit.x, fit.fun)
            
    rot_fit_results = fit_rot_parametric(POSITION_ANGLE, INCLINATION,model=model)
    plot_rotation(POSITION_ANGLE, INCLINATION,rot_fit_results)
    print(rot_fit_results[2])
    df = pd.DataFrame(columns=['Method',"Chi","Status","Results"])
    df = df.append({"Method":method,"Chi":rot_fit_results[2],"Status":fit.status\
                            ,"Results":fit}, ignore_index=True)
    return df
    
    
def fit_anlges_all_methods():    
    methods = ["Nelder-Mead","Powell","BFGS","CG",\
               "L-BFGS-B","TNC","COBYLA","SLSQP","trust-constr"]
    
    model = model_0
    df = pd.DataFrame(columns=['Method',"Chi","Status","Results"])
    for method in methods:
        try:
            df = df.append(fit_angles(method,model), ignore_index=True)
        except ValueError as e:
            print(e)
    print(df)
    return df




#Main logic
if __name__ == "__main__":
    #load data
    M31 = Spectra(FOLDER) 
    
    #make contour plots
    #_,c0 = contour_plot(M31,0,labels=True) #contour plot of zero-th moment
    _,c1 = contour_plot(M31,1) #controur plot of first-moment
    print(f"The mass of Neutral-hydrogen in M31 "\
          f'is {scientific_formatting(M31.mass()[0],M31.mass()[1])} solar masses.')
        
     
    #sys.exit()
      
    
    #fit inclination and position anlges using all possible methods
    #fit_anlges_all_methods() 
    
    #fit rotation curve
    betaparams,sd_beta,chi = fit_and_plot(POSITION_ANGLE, INCLINATION, \
                                          model=model_0, labels=False)
    rot_results = results_0(betaparams,sd_beta)
    print(f'The total masss of M31 is {rot_results[0]} Solar masses.')
    #print(f'Meaning {rot_results[0]/M31.mass()[0]} of M31 is Neutral hydrogen.')
    print(f"The terminal rotation velocity is v={rot_results[1][0]:.0f} ±\
          {rot_results[1][1]:.0f} km/s.")
    
    sys.exit()
    
    #plot with labels
    _,c0 = contour_plot(M31,0,labels=True) 
    _,c1 = contour_plot(M31,1,labels=True)
    fit_and_plot(POSITION_ANGLE, INCLINATION,labels=True)
    
    #show mdoel going to zero
    fit_and_plot(POSITION_ANGLE, INCLINATION,xlims=(0,0))
    

    
    # nice controur
    # model fit
    # merge balays script angle error prop, ellipse
    
     
    error_in_v = 1/(4*np.sqrt(np.abs(betaparams[0])))*sd_beta[0]
    
    





