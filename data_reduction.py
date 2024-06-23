import numpy as np
from astropy.stats import mad_std
from astropy.nddata import CCDData
from astropy import units as u 
import ccdproc as ccdp
from pathlib import Path
import warnings 


def create_master_bias(image_file_collection_raw:ccdp.ImageFileCollection, reduced_data_path:Path) -> Path:
    """Combines all bias frames to create a master bias frame. 

    Args:
        image_file_collection_raw (ccdp.ImageFileCollection): collection of raw images
        reduced_data_path (Path): directory to save the master dark to

    Returns:
        Path: the path to the master bias that was just created 
    """    
    # Loop through each bias frame and write it to the reduced path
    n_biases = 0
    for ccd, file_name in image_file_collection_raw.ccds(imagetyp='BIAS',            # Just get the bias frames
                                 ccd_kwargs={'unit': 'adu'}, # CCDData requires a unit for the image if 
                                                             # it is not in the header
                                 return_fname=True           # Provide the file name too.
                                ):   
        # If our ccd had overscan to subtract or trim, this is where it would be done
        # ARCSAT has no overscan, so we just save the files
        ccd.write(reduced_data_path / file_name)
        n_biases += 1

    # Load in the reduced image file colleciton to loop over
    reduced_images = ccdp.ImageFileCollection(reduced_data_path)
    calibrated_biases = reduced_images.files_filtered(imagetyp='bias', include_path=True)

    # combine each of our bias frames into one master bias
    combined_bias = ccdp.combine(calibrated_biases,
                             method='average',
                             sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                             sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
                             mem_limit=350e6
                            )

    combined_bias.meta['combined'] = True

    combined_bias_output_path = reduced_data_path / 'combined_bias.fit'
    combined_bias.write(combined_bias_output_path)
    print(f'Successfully combined {n_biases} bias frames and wrote to: {combined_bias_output_path}')
    return combined_bias_output_path

def create_master_dark(image_file_collection_raw:ccdp.ImageFileCollection, 
                       reduced_data_path:Path, master_bias_path:Path) -> Path:
    """Creates the master dark image by subtracting the master bias from each dark, then combining. 

    Args:
        image_file_collection_raw (ccdp.ImageFileCollection): collection of raw images
        reduced_data_path (Path): directory to save the master dark to
        master_bias_path (Path): location of the master bias file

    Returns:
        Path: Path to the master dark that was just created
    """    

    master_bias_ccd_data = CCDData.read(master_bias_path)
    dark_exposure_time = 900 # seconds
    n_darks = 0
    # remove the bias from each dark frame 
    for ccd, file_name in image_file_collection_raw.ccds(imagetyp='DARK',            # Just get the dark frames
                                         ccd_kwargs={'unit': 'adu'},          
                                          return_fname=True           # Provide the file name too.
                                         ):
        # Subtract bias
        ccd = ccdp.subtract_bias(ccd, master_bias_ccd_data)
        # Save the result
        ccd.write( reduced_data_path / file_name)
        n_darks += 1

    # Load in the reduced darks
    reduced_images = ccdp.ImageFileCollection(reduced_data_path)
    calibrated_darks = reduced_images.files_filtered(imagetyp='dark', exptime=dark_exposure_time,
                                                     include_path=True)
    # Create our master dark and save 
    combined_dark = ccdp.combine(calibrated_darks,
                                 method='average',
                                 sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                                 sigma_clip_func=np.ma.median, signma_clip_dev_func=mad_std,
                                 mem_limit=350e6
                                )
    # update metadata
    combined_dark.meta['combined'] = True
    # save the master dark
    master_dark_file_path = reduced_data_path / f'combined_dark_{dark_exposure_time}.fit'
    combined_dark.write(master_dark_file_path)
    print(f'Combined {n_darks} dark frames and wrote to: {master_dark_file_path}')
    return master_dark_file_path


def inv_median(a):
    """Helper function to calculate the inverse median. Needed when combining the flats 

    Args:
        a (_type_): _description_

    Returns:
        _type_: _description_
    """    
    return 1 / np.median(a)

def create_master_flat(image_file_collection_raw:ccdp.ImageFileCollection, reduced_data_path:Path,
                        master_bias_path:Path, master_dark_path:Path) -> Path:
    """Subtract the bias and dark current from each flat frame then combine into one master flat. 
    Note that here only one filter was used for data collection. If data in multiple filters needs to 
    be reduced then there must be a master flat for each filter. 

    Args:
        image_file_collection_raw (ccdp.ImageFileCollection): collection of raw images
        reduced_data_path (Path): directory to save the master dark to
        master_bias_path (Path): location of the master bias file
        master_dark_path (Path): location of the master dark file

    Returns:
        Path: Path to the created master flat 
    """    

    # Load in the CCD data for our master bias and dark
    master_bias_CCDData = CCDData.read(master_bias_path)
    master_dark_CCDData = CCDData.read(master_dark_path)

    n_flats = 0
    # this subtracts the bias then the dark current from each flat field
    for ccd, file_name in image_file_collection_raw.ccds(imagetyp='FLAT',filter ='V', ccd_kwargs={'unit': 'adu'},            
                                   return_fname=True           
                                  ):
        #  subtract the bias
        ccd = ccdp.subtract_bias(ccd, master_bias_CCDData)
        # Subtract the dark current 
        ccd = ccdp.subtract_dark(ccd, master_dark_CCDData,
                                exposure_time='exptime', exposure_unit=u.second, scale=True)
        # Save the result
        ccd.write(reduced_data_path / file_name)
        n_flats += 1

    # Load in the updated reduced images
    reduced_images = ccdp.ImageFileCollection(reduced_data_path)
    
    flats_to_combine = reduced_images.files_filtered(imagetyp='flat', filter='V', include_path=True)
    combined_flat = ccdp.combine(flats_to_combine,
                                method='average', scale=inv_median,
                                sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                                sigma_clip_func=np.ma.median, signma_clip_dev_func=mad_std,
                                mem_limit=350e6
                                )

    combined_flat.meta['combined'] = True
    flat_file_name = 'combined_flat_V.fit'
    combined_flat_output_path = reduced_data_path / flat_file_name
    combined_flat.write(combined_flat_output_path)
    print(f'Combined {n_flats} flat images and wrote to {combined_flat_output_path}')
    return combined_flat_output_path
    



def reduce_science_images(image_file_collection_raw:ccdp.ImageFileCollection, reduced_data_path:Path,  
                          master_bias_path:Path, master_dark_path:Path, master_flat_path:Path) -> None:
    """ Removes all instrumental effects from each science image by subtracting the bias, dark, and flat. 

    Args:
        image_file_collection_raw (ccdp.ImageFileCollection): collection of raw images
        reduced_data_path (Path): directory to save the master dark to
        master_bias_path (Path): location of the master bias file
        master_dark_path (Path): location of the master dark file
        master_flat_path (Path): location of the master flat file
    """    

    # Load in each master file CCD Data
    master_bias_CCDData = CCDData.read(master_bias_path)
    master_dark_CCDData = CCDData.read(master_dark_path)
    master_flat_CCDData = CCDData.read(master_flat_path)

    n_science_images = 0 
    # Filter for science images and loop through each
    for ccd, file_name in image_file_collection_raw.ccds(imagetyp='Light Frame', filter = 'V', return_fname=True, ccd_kwargs=dict(unit='adu')):
  
        #First subtract the bias
        reduced = ccdp.subtract_bias(ccd, master_bias_CCDData)

        #now subtract the dark 
        reduced = ccdp.subtract_dark(reduced, master_dark_CCDData,
                                    exposure_time='exptime', exposure_unit=u.second, scale = True
                                    )
        #correct the flat
        reduced = ccdp.flat_correct(reduced, master_flat_CCDData)
        #write to reduced images directory
        reduced.write(reduced_data_path / file_name)
        n_science_images += 1 
    
    print(f'Reduced and saved {n_science_images} science images')


def main():
    warnings.filterwarnings('ignore')

    raw_data_path = Path('/home/toomeh/uw/spring-23/astr-480/final-project/Data')
    reduced_data_path = Path(raw_data_path / 'reduced_images')
    reduced_data_path.mkdir(exist_ok=True)
    image_file_collection_raw = ccdp.ImageFileCollection(raw_data_path)

    
    master_bias_path = create_master_bias(image_file_collection_raw, reduced_data_path)
    master_dark_path = create_master_dark(image_file_collection_raw, reduced_data_path, master_bias_path)
    master_flat_path = create_master_flat(image_file_collection_raw, reduced_data_path, master_bias_path, master_dark_path)
    reduce_science_images(image_file_collection_raw, reduced_data_path, master_bias_path, master_dark_path, master_flat_path)




if __name__ == '__main__':
    main()
    