import numpy as np
from numpy.typing import NDArray

type StrAlignDict = dict[str, list[str]]
type IntAlignDict = dict[str, NDArray[np.int8]]
type FloatAlignDict = dict[str, NDArray[np.float32]]
type LabelDict = dict[str, int]

PADDING_TOKEN = 0
AMBIG_TOKEN = 99

LABEL_REAL = 0
LABEL_SIMULATED = 1

RANDOM_SEED = 42

CLASSIFIERS = (
    "LogisticRegressionClassifier",
    "DenseMsaClassifier",
    "DenseSiteClassifier",
    "AACnnClassifier",
    "DNACnnClassifier",
    "AttentionClassifier",
)
