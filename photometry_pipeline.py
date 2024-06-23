# imports 
import numpy as np
import pandas as pd
from pathlib import Path

import ccdproc as ccdp
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus, ApertureStats

from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm
from photutils.background import MADStdBackgroundRMS, MMMBackground
from photutils.detection import IRAFStarFinder
from photutils.psf import (DAOGroup, IntegratedGaussianPRF,
                           IterativelySubtractedPSFPhotometry)
from photutils.psf import BasicPSFPhotometry
import warnings 

def find_stars(ccd_data):
    """Uses DAOStarFinder to locate each star in the provided ccd data

    Args:
        ccd_data (_type_): CCD Data of the image we are working with

    Returns:
        _type_: returns the coordinates of every star found in the image
    """    
    # record statistics of the current ccd data
    mean, median, std = sigma_clipped_stats(ccd_data, sigma=3.0)
    # The threshold is how we can help the starfinder ignore dim sources
    threshold = 10* std
    # Instantiate the DAO Starfinder
    daofind = DAOStarFinder(fwhm=3.0, threshold=threshold)

    # Every image in this dataset has a large instrumental artifact in the top left of the image that must be removed 
    # We create a small rectangular mask
    # If we do not do this the star finder will find many "sources" along the artifact 
    mask = np.zeros(ccd_data.shape, dtype=bool)
    mask[2800:3100, 700:710] = True
    # Note that we subtract the median value from the data to minimize background effects
    sources = daofind(ccd_data - median, mask = mask)
    return sources


def get_circular_aperture_flux(sources, ccd_data, median, object_index):
    """ Finds the flux of the object of interest using the circular annulus method
    This method simply draws a circle of specified radius around the source and adds up the flux


    Args:
        sources (_type_): The locations of stars determined by DAOStarFinder
        ccd_data (_type_): the actual image data
        median (_type_): the median flux value in the image
        object_index (_type_): the index of the object of interest in the list of sources

    Returns:
        float: the flux of the object 
    """    
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures_circular = CircularAperture(positions, r=50.0) 
    background_subtrcted =  ccd_data - median#subtract background from the data
    phot_table_circular = aperture_photometry(background_subtrcted, apertures_circular)

    object_flux = phot_table_circular[object_index]['aperture_sum']
    return object_flux


def get_annulus_aperture_flux(sources, ccd_data, object_index):
    """gets the annulus aperture flux of our star. In simple terms this works quite similarly to circular apertures.
    It differs in its background substitutuion, using larger apertures around the star to get a localized background flux estimate. 

    Args:
        sources (_type_): _description_
        ccd_data (_type_): _description_
        object_index (_type_): _description_

    Returns:
        _type_: _description_
    """    
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    # Create our aperture and annulus
    aperture = CircularAperture(positions, r=50)
    annulus_aperture = CircularAnnulus(positions, r_in=100, r_out=150)
    aperstats = ApertureStats(ccd_data, annulus_aperture)
    bkg_mean = aperstats.mean
    # Apply our apertures and subtract the local background around the star
    phot_table = aperture_photometry(ccd_data, aperture)
    total_bkg = bkg_mean * aperture.area
    phot_bkgsub = phot_table['aperture_sum'] - total_bkg
    phot_table['aperture_sum_bkgsub'] = phot_bkgsub
    object_flux = phot_table[object_index]['aperture_sum_bkgsub']

    return object_flux



def get_psf_flux(sources, ccd_data, object_index) -> float:
    """ Fit a point spread function on our object to estimate the total flux from it.

    Args:
        sources (_type_): The locations of all stars in the image found by DAOStarfinder
        ccd_data (_type_): the ccd data of the image
        object_index (_type_): the index of the object we are estimating the flux of

    Returns:
        float: the psf flux of the object
    """    
    sigma_psf = 4.0
    # Load in the background estimator and fitter
    daogroup = DAOGroup(2.0 * sigma_psf * gaussian_sigma_to_fwhm)
    mmm_bkg = MMMBackground()
    fitter = LevMarLSQFitter()
    # The specific psf model we will use. We pick a gaussian model here
    psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
    psf_model.x_0.fixed = True
    psf_model.y_0.fixed = True
    # load the positions we found with DAO 
    pos = Table(names=['x_0', 'y_0'], data=[sources['xcentroid'],
                                            sources['ycentroid']])
    # Fit our gaussian psfs on each source in the image
    photometry = BasicPSFPhotometry(group_maker=daogroup,
                                bkg_estimator=mmm_bkg,
                                psf_model=psf_model,
                                fitter=fitter,
                                fitshape=(11, 11))
    result_tab = photometry(image=ccd_data, init_guesses=pos)
    # Return the psf flux of the object in question
    return result_tab[object_index]['flux_0']


def main():
    warnings.filterwarnings('ignore')
    # load in the reference index table
    reference_table = pd.read_csv('/home/toomeh/uw/spring-23/astr-480/final-project/indexes.csv')
    # Load in the reduced images
    reduced_images_path = Path('/home/toomeh/uw/spring-23/astr-480/final-project/Data/reduced_images')
    reduced_images = ccdp.ImageFileCollection(reduced_images_path)
    # Create the path and column names for the csv where we will save the results from this analysis
    output_path = Path('/home/toomeh/uw/spring-23/astr-480/final-project/flux.csv')
    output_csv_columns = ('Object Name', 'image number','Circular Aperture Flux','Annulus Aperture Flux', 'PSF Flux' )

    # Check if the CSV file exists. Create it if it doesnt 
    if not output_path.exists():
        df = pd.DataFrame(columns=output_csv_columns)
        df.to_csv(output_path, index=False)

    # for ccd_data, file_name in reduced_images.ccds(imagetyp='Light Frame', filter = 'V', return_fname=True, ccd_kwargs=dict(unit='adu')):
    for ccd_data, file_name in reduced_images.ccds(imagetyp='Light Frame', filter = 'V', return_fname=True):
        if file_name not in reference_table['filename'].tolist():
            # if the file name is not one we care about, skip it 
            continue
        print(f'Currently finding the flux for the file: {file_name}')
        # get star locations
        star_locations = find_stars(ccd_data)
        # Get the object index 
        object_index = reference_table[reference_table['filename'] == file_name].iloc[0]['star_index']
        # get the median flux value to pass into the aperture function 
        _, median, _ = sigma_clipped_stats(ccd_data, sigma=3.0)
        # call the circular aperture function
        circular_flux = get_circular_aperture_flux(star_locations, ccd_data, median, object_index)
        # Call the annulus aperture function
        annulus_flux = get_annulus_aperture_flux(star_locations, ccd_data, object_index)
        # call the psf function
        psf_flux = get_psf_flux(star_locations, ccd_data, object_index)


        # Get the object name and image number from the filename 
        split_file_name = file_name.split('_')
        object_name = split_file_name[0] + split_file_name[1]
        # start indexing at zero
        image_number = split_file_name[4]

        # Append the current row to the output CSV
        df = pd.DataFrame([[object_name, image_number, circular_flux, annulus_flux, psf_flux]], columns=output_csv_columns)
        df.to_csv(output_path, mode='a', header=False, index=False)

        print(f'Added {file_name} fluxes to output csv')
    print(f'All fluxes estimated and logged in {output_path}')

if __name__ == '__main__':
    main()

