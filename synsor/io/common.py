#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# Standard library imports
import gzip
import shutil
from pathlib import Path

# External imports

# Internal imports

#------------------- Constants ------------------------------#

#------------------- Classes & Functions --------------------#

def toZFile(file):
    with open(file, 'rb') as f_input:
        with gzip.open(f"{file}.gz", 'wb') as f_output:
            shutil.copyfileobj(f_input, f_output)

            # Delete the original file after the gzip is done
            Path(file).unlink()

def isZFile(file):
    if (file.suffix == '.gz' or file.suffix == '.gzip'):
        return True

    return False

#------------------- Main -----------------------------------#

