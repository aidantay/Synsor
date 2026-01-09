#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# Standard library imports
import itertools
import functools

# External imports

# Internal imports
from . import seq

#------------------- Constants ------------------------------#

#------------------- Classes & Functions --------------------#

def getExpectedKmers(kmerLength):
    f = itertools.product(seq.NUCLEOTIDES, repeat=kmerLength)
    return (''.join(c) for c in f)

def getExpectedTotal(seq_or_kLen):
    ## Check whether we're dealing with a k-mer sequence or a k-mer length
    if (isinstance(seq_or_kLen, str)):
        seq = seq_or_kLen
        ## Check whether the k-mer sequence is paired with its reverse
        ## complement (i.e., FWD-REV)
        if ('-' in seq):
            kmer = seq.split('-')[0]
            kLen = len(kmer)
            ## Check whether the length of the k-mer sequence is even or odd
            if (kLen % 2 == 0):
                ## Even k-mer lengths can generate palindromes which
                ## changes the total number of possible k-mers.
                ## Based on formulas reported in:
                ## * Apostolou-Karampelis et al. (2019)
                x = (2 * kLen) - 1
                y = (kLen - 1)
                t = (2 ** x) + (2 ** y)
                return t

            else:
                t = (4 ** kLen) / 2
                return int(t)

        else:
            kLen = len(seq)
            return (4 ** kLen)

    else:
        kLen = seq_or_kLen
        return (4 ** kLen)

#------------------- Main -----------------------------------#

