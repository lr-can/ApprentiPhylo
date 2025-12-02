import numpy as np
from numpy.typing import NDArray

type StrAlignDict = dict[str, list[str]]
type IntAlignDict = dict[str, NDArray[np.int8]]
type FloatAlignDict = dict[str, NDArray[np.float32]]
type LabelDict = dict[str, int]

PADDING_TOKEN = 0
AMBIG_TOKEN = 99

# Label definitions (aligned with model output indices)
# Model outputs: [prob_class_0, prob_class_1]
# prob_real = probs[:, 1] → so LABEL_REAL must be 1
LABEL_SIMULATED = 0
LABEL_REAL = 1

RANDOM_SEED = 42

CLASSIFIERS = (
    "LogisticRegressionClassifier",
    "DenseMsaClassifier",
    "DenseSiteClassifier",
    "AACnnClassifier",
    "DNACnnClassifier",
    "AttentionClassifier",
)
