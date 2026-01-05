#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# Standard library imports
from pathlib import Path

# External imports
import pytest

# Internal imports
from synsor import io
from synsor import schema

#------------------- Constants ------------------------------#

TEST_DATA_FASTA        = 'test/data/seqs.fna'
TEST_DATA_2MER_PARQUET = 'test/data/freqs_parquet_2mer'

#------------------- Classes & Functions --------------------#

def test_fastx_read():
    seqRecs = list(io.fastx.read(Path(TEST_DATA_FASTA)))
    assert len(seqRecs) == 25

def test_kmer_read_parquet(spark):
    kmerDf = io.kmer.read(Path(TEST_DATA_2MER_PARQUET))
    nSeqs  = kmerDf.select(schema.SEQID_COL_NAME).distinct().count()
    nKmers = kmerDf.select(schema.KMER_COL_NAME).distinct().count()
    assert nSeqs == 25
    assert nKmers == 16

#------------------- Main -----------------------------------#
