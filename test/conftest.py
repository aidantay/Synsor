#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# Standard library imports

# External imports
import pytest
from pyspark.sql import SparkSession

# Internal imports

#------------------- Constants ------------------------------#

#------------------- Classes & Functions --------------------#

@pytest.fixture
def spark():
    with SparkSession.builder.getOrCreate() as ss:
        with ss.sparkContext as sc:
            yield sc

@pytest.fixture
def evenKmerDf(spark):
    df = io.kmer.read(TEST_DATA_2MER)
    return df

@pytest.fixture
def oddKmerDf(spark):
    df = io.kmer.read(TEST_DATA_3MER)
    return df

#------------------- Main -----------------------------------#
