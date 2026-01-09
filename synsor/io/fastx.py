#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# Standard library imports
import gzip
import itertools
from pathlib import Path

# External imports
from Bio import SeqIO

# Internal imports
from .common import isZFile
from .common import toZFile

#------------------- Constants ------------------------------#

FASTA = 'fasta'
FASTQ = 'fastq'

#------------------- Classes & Functions --------------------#

def read(*files, **kwargs):
    seqRecs = (_readFile(f, **kwargs) for f in files)
    seqRecs = itertools.chain(*seqRecs)
    return seqRecs

def _readFile(file, **kwargs):
    if (isZFile(file)):
        fastxType = _getFormat(file.suffix)
        with gzip.open(file, 'rt') as filehandle:
            seqIter = SeqIO.parse(filehandle, fastxType, **kwargs)
            yield from seqIter

    else:
        fastxType = _getFormat(file.suffix)
        seqIter   = SeqIO.parse(file, fastxType, **kwargs)
        yield from seqIter

def _getFormat(suffix):
    if (suffix == '.fa' \
        or suffix == '.fna' \
        or suffix == '.fasta'):
        return FASTA

    elif (suffix == '.fq' \
          or suffix == '.fnq' \
          or suffix == '.fastq'):
        return FASTQ

    else:
        raise NotImplementedError("Unknown fastX file")

#------------------- Main -----------------------------------#

