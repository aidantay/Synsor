#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# Standard library imports

# External imports

# Internal imports

#------------------- Constants ------------------------------#

## Base infomation
NUCLEOTIDES = ['A', 'C', 'G', 'T']
REV_COMP    = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}

#------------------- Public Classes & Functions -------------#

def seqToRevComp(seq):
    bases  = reversed(list(seq))
    bases  = [REV_COMP[b] for b in bases]
    revSeq = ''.join(bases)
    return revSeq

#------------------- Private Classes & Functions ------------#

#------------------- Main -----------------------------------#

if (__name__ == "__main__"):
    main()

#------------------------------------------------------------------------------
