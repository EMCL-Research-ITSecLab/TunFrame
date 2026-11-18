"""Table 2 input contract; do not infer unreported features from collector metadata.

The paper does not release its extractor. In particular, word segmentation,
resource-record aggregation, injected volume and extra UDP data are undefined.
Input must therefore be supplied by an explicit, independently documented
extractor. This module deliberately does not claim a PCAP reproduction.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Table 2, in order. Row 11 has a duplicated/mistaken Resource_Record_Type
# heading; its stated meaning is consecutive consonants. Row 22 is RR type.
FEATURE_NAMES = (
    "Subdomain_Length", "Subdomain_Entropy", "Subdomain_Digit_Ratio",
    "Subdomain_Uppercase_Ratio", "Subdomain_Word_Count", "Max_Word_Length",
    "Jaccard_Index_Bi", "Jaccard_Index_Tri", "Consecutive_Digits_Ratio",
    "Letter_Digit_Alternation_Ratio", "Consecutive_Consonants_Ratio",
    "Label_Count_QNAME", "UDP_Payload_Length", "Resource_Record",
    "Response_IP_Count", "DNSTTL", "Payload_Length", "Response_Type",
    "Entropy_Resource_Record_Content", "Data_Injection_Volume",
    "Additional_Data_in_UDP", "Resource_Record_Type", "Response_Code",
)


def feature_matrix(records):
    """Reject missing, nonnumeric or nonfinite features instead of imputing."""
    frame = pd.DataFrame(records)
    missing = set(FEATURE_NAMES).difference(frame.columns)
    if missing:
        raise ValueError("Missing DBM-ENSec features: " + ", ".join(sorted(missing)))
    try:
        matrix = frame.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("DBM-ENSec requires numeric feature values") from exc
    if not np.isfinite(matrix).all():
        raise ValueError("DBM-ENSec features contain NaN or infinity")
    return matrix


def load_csv(path):
    frame = pd.read_csv(path)
    matrix = feature_matrix(frame)
    if "label" not in frame or not frame.label.isin([0, 1]).all():
        raise ValueError("CSV requires binary label: 0=benign, 1=tunnel")
    labels = frame.label.to_numpy(dtype=np.int64)
    if len(np.unique(labels)) != 2:
        raise ValueError("Training requires both benign and tunnel examples")
    groups = None
    if "group" in frame:
        if frame.group.isna().any():
            raise ValueError("group must be populated for every sample")
        groups = frame.group.astype(str).to_numpy()
    return matrix, labels, groups


def fit_scaler(matrix):
    return MinMaxScaler(feature_range=(0, 255), clip=True).fit(matrix)


def grayscale(matrix, scaler):
    """Train-fitted scaling, integer pixels, row-major 5x5, two zero pads.

    Rounding, clipping and train-only fitting are explicit implementation
    choices; the paper does not specify these boundary conventions. Divide
    pixels by 255 for Bernoulli mean-field inference in the DBM.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != 23 or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite N x 23 feature matrix")
    pixels = np.rint(scaler.transform(matrix)).astype(np.uint8)
    return np.pad(pixels, ((0, 0), (0, 2))).reshape(-1, 5, 5)
