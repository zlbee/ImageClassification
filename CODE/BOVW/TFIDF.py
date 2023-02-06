from scipy.sparse import csr_matrix
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer


def list2csr_matrix(hist_list):
    return csr_matrix(np.asarray(hist_list).astype(np.int64))


def calc_TFIDF(hist_csr_matrix):
    transformer = TfidfTransformer(use_idf=True, smooth_idf=False)
    tfidf = transformer.fit_transform(hist_csr_matrix)
    return tfidf

