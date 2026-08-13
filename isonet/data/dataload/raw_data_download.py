import urllib.request
import gzip
import shutil 

from isonet.config import ROOT


def load_data(filename):
    folder_zip = "/isonet/data/test_dataset_zip/"
    folder = "/isonet/data/test_dataset/"
    filepath_zip = ROOT + folder_zip + filename 
    filepath = ROOT + folder + filename[:-3]  # .gz제거한 경로

    url = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/" + filename


    urllib.request.urlretrieve(url, filepath_zip)


    with gzip.open(filepath_zip, 'rb') as f_in:
        with open(filepath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)