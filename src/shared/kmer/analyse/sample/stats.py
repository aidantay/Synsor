#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# Standard library imports
import traceback

# External imports
import numpy as np
import scipy.stats as sps
import pyspark.sql.functions as F
import pyspark.sql.types as T

# Internal imports
from .... import kmer
from ... import transform
from .common import *

#------------------- Constants ------------------------------#

#------------------- Public Classes & Functions -------------#

def getTestResults(kmerDfX, statTest, kmerDfY=None):
    def _prepareData(kmerDfX, kmerDfY):
        nParts  = kmerDfX.rdd.getNumPartitions()
        kmerDfX = transform.counts.toVector(kmerDfX)
        kmerDfX = kmerDfX.select(kmer.SEQID_COL_NAME, kmer.KMER_COL_NAME)
        kmerDfX = kmerDfX.coalesce(nParts)

        if (kmerDfY is not None):
            kmerDfY = transform.counts.toVector(kmerDfY)
            kmerDfY = kmerDfY.select(kmer.SEQID_COL_NAME, kmer.KMER_COL_NAME)
            kmerDfY = kmerDfY.coalesce(nParts)

        return (kmerDfX, kmerDfY)

    ## Prepare the data
    (kmerDfX, kmerDfY) = _prepareData(kmerDfX, kmerDfY)

    ## Find all pairs depending on the DataFrames available
    kmerStatDf = getPairs(kmerDfX, kmerDfY)

    ## Perform statistical test between each pair
    f = lambda x, y: list(statTest(x, y))
    f = F.udf(f, T.ArrayType(T.FloatType()))
    kmerStatDf = kmerStatDf.select(
        F.col('l.seqId').alias('seqId_1'),
        F.col('r.seqId').alias('seqId_2'),
        f('l.kmer', 'r.kmer').alias('test_result')
    )
    return kmerStatDf

#------------------- Private Classes & Functions ------------#

def _chi2_n_cramersV(x, y):
    try:
        ## Calculate Chi2
        (chi2, p) = _chi2(x, y)

        ## Calculate Cramer's V (with bias correction) for effect size
        z = np.array([x, y])
        vcorr = _cramersV(z, chi2)

    except Exception as e:
        ## Raise any errors that may occur so that we can debug later
        ## As placeholders, we'll set p-value == 1 and vcorr == 0
        # print("Unable to compute, result might be inaccurate")
        # print(e)
        # print(cDf)
        # traceback.print_exc()
        chi2  = 0
        p     = 1
        vcorr = 0

    return (float(chi2), float(p), float(vcorr))

def _chi2(x, y):
    ## The Chi-squared test of independence tests the hypothesis that
    ## the observed COUNTS of two (or more) samples are consistent with
    ## the expected COUNTS. Therefore, p-value < 0.05 == reject the hypothesis
    ## that the observed discrepency between samples is due to chance
    ## and hence the samples are different.

    ## Because of the high dimensionality, tiny differences
    ## can lead to statistically significant results (i.e., near-zero
    ## p-values) regardless of the statistical test used. Thus, relying on p-values
    ## for statistical significance may not correlate with the practical
    ## significance anymore.

    ## Instead, we must also quantify the practical significance of the results by
    ## looking at the effect size (e.g., difference between two means).

    try:
        ## Calculate Chi2 for p-value
        z = np.array([x, y])
        (chi2, p) = sps.chi2_contingency(z)

    except Exception as e:
        ## Raise any errors that may occur so that we can debug later
        ## As placeholders, we'll set p-value == 1 and vcorr == 0
        # print("Unable to compute, result might be inaccurate")
        # print(e)
        # print(cDf)
        # traceback.print_exc()
        chi2 = 0
        p = 1

    return (float(chi2), float(p))

def _cramersV(cDf, chi2):
    n    = np.sum(cDf)
    phi2 = chi2 / n
    k, r = cDf.shape

    phi2corr = max(0, phi2 - (((k - 1) * (r - 1)) / (n - 1)))
    kcorr    = k - (((k - 1) ** 2) / (n - 1))
    rcorr    = r - (((r - 1) ** 2) / (n - 1))
    vcorr    = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
    return vcorr

#------------------- Main -----------------------------------#

if (__name__ == "__main__"):
    main()

#------------------------------------------------------------------------------

# from statsmodels.stats import multitest
# ## Correct p-values due to multiple testing
# pvalues = hpvDf['p_value'].tolist()
# # results = multitest.multipletests(pvalues, alpha=0.01, method='fdr_bh')
# # pvalues = results[1]
# hpvDf['p_value'] = pvalues
# hpvDf = hpvDf.sort_values(by=['p_value'], ignore_index=True)
# return hpvDf
