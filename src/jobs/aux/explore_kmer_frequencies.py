#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# Standard library imports
import argparse
import sys
from pathlib import Path

# External imports
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from scipy.stats import ttest_ind

# Internal imports
from ...shared.kmer import analyse
from ...shared.kmer import transform
from ...shared import io
from ...shared import kmer

#------------------- Constants ------------------------------#

#------------------- Public Classes & Functions -------------#

def main(iDirs, clsFiles, aMode, oDir):
    if (aMode == 'table'):
        runDataTable(iDirs, clsFiles, oDir)

    elif (aMode == 'colnames'):
        runDataColumnNames(iDirs, oDir)

    elif (aMode == 'reduction'):
        runKmerReduction(iDirs, clsFiles, oDir)

def runDataTable(iDirs, clsFiles, oDir):
    print("Data table")

    ## Read Kmer frequencies
    kmerDf = io.kmer.read(*iDirs)
    nParts = kmerDf.rdd.getNumPartitions()

    ## Prepare data
    kmerDf = transform.counts.groupRevComp(kmerDf)
    kmerDf = kmerDf.withColumn(kmer.KMER_COL_NAME, F.split(kmer.KMER_COL_NAME, '-').getItem(0))
    kmerDf = transform.counts.toVector(kmerDf)
    kmerDf = kmerDf.coalesce(nParts)

    ## Read class information and join
    clsDf  = _readClassInfo(clsFiles)
    kmerDf = kmerDf.join(clsDf, on=kmer.SEQID_COL_NAME, how='left')
    kmerDf = kmerDf.repartitionByRange(nParts, kmer.KMER_COL_NAME)

    ## Try to load everything into memory for ML.
    ## SparkML just doesn't seem to be as good as sklearn
    (kmerId, kmerCount) = transform.table.toDriver(kmerDf)
    kmerCount = pd.DataFrame(kmerCount.toarray())
    kmerCount = kmerCount.loc[:, (kmerCount != 0).any(axis=0)]
    kmerDf = pd.concat([kmerId, kmerCount], axis=1)

    ## Write the table to disk
    print("Writing output")
    ss = SparkSession.getActiveSession()
    kmerDf = ss.createDataFrame(kmerDf)
    nFiles = [str(f) for d in iDirs for f in Path(d).glob("*.avro")]
    nParts = len(nFiles)
    kmerDf = kmerDf.coalesce(nParts)
    kmerDf.write.csv(oDir, mode='overwrite', sep='\t', header=True, compression='gzip')

def runDataColumnNames(iDirs, oDir):
    print("Data column names")

    ## Read Kmer frequencies
    kmerDf = io.kmer.read(*iDirs)

    ## We lose the Kmer string information after we vectorise the Kmers..
    ## Therefore, we must save the Kmer string information, if we want to
    ## do any analyses involving Kmer information.
    kmerDf = transform.counts.groupRevComp(kmerDf)
    kmerDf = kmerDf.withColumn(kmer.KMER_COL_NAME, F.split(kmer.KMER_COL_NAME, '-').getItem(0))
    kmerStrToIdx = transform.table.insertKmerIdxCol(kmerDf)
    kmerStrToIdx = kmerStrToIdx.select(kmer.KMER_COL_NAME, 'kmerIdx').distinct()

    ## Write the table to disk
    print("Writing output")
    nFiles  = [str(f) for d in iDirs for f in Path(d).glob("*.avro")]
    nParts  = len(nFiles)
    kmerStrToIdx = kmerStrToIdx.coalesce(nParts)
    kmerStrToIdx.write.csv(oDir, mode='overwrite', sep='\t', header=True)

def runKmerReduction(iDirs, clsFiles, oDir):
    def _getComponents(X):
        from sklearn.decomposition import PCA
        from sklearn.decomposition import TruncatedSVD

        algo   = PCA(n_components=3)
        X_algo = algo.fit_transform(X)

        nComps = len(algo.explained_variance_ratio_)
        cols = ['Component{}'.format(i) for i in range(1, nComps + 1)]
        X_algo = pd.DataFrame(X_algo, columns=cols)
        return (algo, X_algo)

    def _getExplainedVariance(algo):
        if (hasattr(algo, 'explained_variance_')):
            print("Explained Variance\t{}".format(algo.explained_variance_))

        if (hasattr(algo, 'explained_variance_ratio_')):
            print("Explained Variance Ratio\t{}".format(algo.explained_variance_ratio_))

        df = pd.DataFrame([algo.explained_variance_, algo.explained_variance_ratio_]).T
        df = df.reset_index()
        df['index'] = df['index'] + 1
        df.columns = ['Component', 'ExplainedVar', 'ExplainedVarRatio']
        return df

    def _getLoadings(algo):
        df = pd.DataFrame(algo.components_).T
        df.columns = ['Component{}'.format(i) for i in range(1, len(df.columns) + 1)]
        df['idx'] = np.arange(0, len(df))
        return df

    print("Running reduction")

    ## Read Kmer frequencies
    kmerDf = io.kmer.read(*iDirs)
    nParts = kmerDf.rdd.getNumPartitions()

    ## Prepare data
    kmerDf = transform.counts.groupRevComp(kmerDf)
    kmerDf = kmerDf.withColumn(kmer.KMER_COL_NAME, F.split(kmer.KMER_COL_NAME, '-').getItem(0))
    kmerDf = transform.counts.toProbabilities(kmerDf)
    kmerDf = transform.counts.toL2Normalised(kmerDf)
    kmerDf = transform.counts.toVector(kmerDf)
    kmerDf = kmerDf.coalesce(nParts)

    ## Read class information and join
    clsDf  = _readClassInfo(clsFiles)
    kmerDf = kmerDf.join(clsDf, on=kmer.SEQID_COL_NAME, how='left')
    kmerDf = kmerDf.repartitionByRange(nParts, kmer.KMER_COL_NAME)

    ## Try to load everything into memory for ML.
    ## SparkML just doesn't seem to be as good as sklearn
    (kmerId, kmerCount) = transform.table.toDriver(kmerDf)
    colIdxs = sorted(set(kmerCount.nonzero()[1]))
    kmerCount = kmerCount[:, colIdxs].toarray()

    ## Run PCA
    (algo, kmerComps) = _getComponents(kmerCount)
    kmerComps = pd.concat([kmerId, kmerComps], axis=1)

    ## Get the explained variance
    kmerVariance = _getExplainedVariance(algo)

    ## Find top features responsible for each component
    kmerLoading = _getLoadings(algo)
    kmerLoading['kmerIdx'] = kmerLoading['idx'].replace({i:v for i, v in enumerate(colIdxs)})
    kmerLoading = kmerLoading.drop(columns=['idx'])

    ## Write the table to disk
    print("Writing output")
    ss = SparkSession.getActiveSession()
    kmerComps = ss.createDataFrame(kmerComps)
    nFiles = [str(f) for d in iDirs for f in Path(d).glob("*.avro")]
    nParts = len(nFiles)
    kmerComps = kmerComps.coalesce(nParts)
    kmerComps.write.csv(oDir, mode='overwrite', sep='\t', header=True)

    f = '{}/pcaExplainedVariance.tsv'.format(oDir)
    kmerVariance.to_csv(f, sep='\t', index=False)

    f = '{}/pcaLoading.tsv'.format(oDir)
    kmerLoading.to_csv(f, sep='\t', index=False)

#------------------- Private Classes & Functions ------------#

def _readClassInfo(clsFiles):
    ## May not work because we changed the column names
    ## Might need to join the seqAcc and seqDesc columns to make seqId
    ## But Worry about this later...
    ss = SparkSession.getActiveSession()
    clsDf = ss.read.csv(clsFiles, sep='\t', header=True)
    clsDf = clsDf.withColumn(kmer.SEQID_COL_NAME, F.concat_ws(' ', 'seqAcc', 'seqDesc'))
    clsDf = (
        clsDf.select(
            F.col(kmer.SEQID_COL_NAME),
            F.col('class'))
        .distinct()
    )

    ## Encode the class value
    idxer = StringIndexer(
        inputCol='class', outputCol='classIdx',
        stringOrderType='alphabetAsc'
    )
    clsDf = idxer.fit(clsDf).transform(clsDf)
    return clsDf

#------------------- Main -----------------------------------#

if (__name__ == "__main__"):
    main()

#------------------------------------------------------------------------------
