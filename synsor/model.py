#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# Standard library imports
from pathlib import Path

# External imports
import numpy as np
import pyspark.sql.types as T
import pyspark.ml.functions as F
from tensorflow.keras import models

# Internal imports
from . import transform
from . import schema

#------------------- Constants ------------------------------#

#------------------- Classes & Functions --------------------#

def prepareData(kmerDf):
    nParts = kmerDf.rdd.getNumPartitions()
    kmerDf = transform.kmercounts.groupRevComp(kmerDf)
    kmerDf = transform.table.insertZeroCounts(kmerDf, possibleKmers=True)
    kmerDf = transform.kmercounts.toProbabilities(kmerDf)
    kmerDf = transform.kmercounts.toL2Normalised(kmerDf)
    kmerDf = transform.kmercounts.toVector(kmerDf)
    kmerDf = kmerDf.repartition(nParts)
    return kmerDf

def getPredictions(modelFile, kmerDf, iDim):
    def _getSchema():
        colNames = ['predicted_prob', 'predicted_classIdx', 'predicted_class', 'model']
        colTypes = [T.FloatType(), T.IntegerType(), T.StringType(), T.StringType()]
        cols     = [T.StructField(n, t) for n, t in zip(colNames, colTypes)]
        schema   = T.StructType(cols)
        return schema

    def _makeClf():
        clf = models.load_model(modelFile)

        def _makePrediction(a, b, c, d):
            prob   = clf.predict(d)
            clsIdx = np.where(prob > 0.5, 1, 0)
            clsStr = np.where(clsIdx == 1, 'Synthetic', 'Not synthetic')
            pred = {
                'predicted_prob' : prob.flatten(),
                'predicted_classIdx' : clsIdx.flatten(),
                'predicted_class' : clsStr.flatten(),
                'model' : np.full((prob.shape), mName).flatten()
            }
            return pred

        return _makePrediction

    f = F.predict_batch_udf(_makeClf,
        return_type=_getSchema(), batch_size=10,
        input_tensor_shapes=[[1], [1], iDim, iDim])

    mName = modelFile.name
    predDf = (
        kmerDf.withColumn('prediction', f('kmer'))
        .select(*schema.KMERDF_SEQINFO_COL_NAMES, 'prediction.*')
    )
    return predDf

#------------------- Main -----------------------------------#

