#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# Standard library imports
import itertools
from pathlib import Path

# External imports
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers

# Internal imports

#------------------- Constants ------------------------------#

#------------------- Public Classes & Functions -------------#

def main(kmerLength, oDir):
    print("Building models")

    ## Tried
    # reduction = [True, False]
    # neurons = (
    #     (4096, 1024, 256, 64, 16),
    #     (2048, 512, 128, 32, 8),
    #     (256, 64, 16, 4),
    #     (128, 32, 8)
    # )
    # learning_rates = [0.1, 0.001, 0.00001, 0.0000001, 0.0000001]
    # l2_penalties = [0.00, 0.1, 0.01, 0.001, 0.0001]

    ## Tried but poor-ish performance
    ## learing_rates = [0.1, 0.0000001, 0.0000001]
    ## l2_penalties  = [0.1]

    hSizes = (
        (512, 16),
    )
    learning_rates = [0.001, 0.0001, 0.00001]
    iDim = int((4 ** kmerLength) / 2)

    for hs, lr in itertools.product(hSizes, learning_rates):
        d = 'hs{}_lr{}'.format(
            0 if len(hs) == 0 else ':'.join([str(h) for h in hs]),
            "{:.12f}".format(float(lr)).rstrip('0').replace('.', ''),
        )
        print(d)

        ## Based on 7-mer signatures
        clf = _buildClassifier(iDim, hs)
        clf = _compileClassifier(clf, lr)
        clf.summary()

        p = Path(oDir)
        p.mkdir(parents=True, exist_ok=True)
        f = '{}/{}/{}'.format(oDir, d, 'KerasClassifier')
        models.save_model(clf, f)

#------------------- Private Classes & Functions ------------#

def _buildClassifier(iDim, hSizes):
    ## Input -> Hidden -> Output
    ## Ideally, there should be a layer in our model for each preprocessing step.
    ## Although this can be done using Lambda layers, saving and loading models
    ## containing Lambda layers encounters a few issues. So to avoid these issues,
    ## we should be replacing Lambda layers with custom layers. However,
    ## for what we want to do, creating custom layers seems far more complicated.
    clf = models.Sequential(name='synbiodetector')
    clf.add(layers.Input(shape=iDim, name='input'))
    clf.add(layers.Dropout(0.2))
    for i, h in enumerate(hSizes):
        clf.add(layers.Dense(h, name='h{}'.format(i+1), activation='relu'))
        clf.add(layers.Dropout(0.5))

    clf.add(layers.Dense(1, name='o', activation='sigmoid'))
    return clf

def _compileClassifier(clf, learning_rate=0.001):
    ## Compile model
    scoring = [
        'accuracy', metrics.Precision(),
        metrics.Recall(), metrics.MeanAbsoluteError()
    ]

    clf.compile(loss='binary_crossentropy', metrics=scoring,
        optimizer=optimizers.Adam(learning_rate=learning_rate))
    return clf

#------------------- Main -----------------------------------#

if (__name__ == "__main__"):
    main()

#------------------------------------------------------------------------------



