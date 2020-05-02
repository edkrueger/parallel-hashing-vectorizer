from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.base import TransformerMixin
from joblib import Parallel, delayed, effective_n_jobs
from scipy.sparse import vstack
import numpy as np

class ParallelHashingVectorizer(HashingVectorizer):
    
    def __init__(self, n_jobs=1, **kwargs):
        super().__init__(**kwargs)
        self.n_jobs = n_jobs
    
    def transform(self, X, y=None, **fit_params):
        
        delayed_hashing_vectorizer = delayed(super().transform)
        
        X_parts = np.array_split(X, effective_n_jobs(self.n_jobs))
        
        X_parts_transformed = Parallel(n_jobs=effective_n_jobs(self.n_jobs))(delayed_hashing_vectorizer(X_part) for X_part in X_parts)
        
        X_transformed = vstack(X_parts_transformed)
        
        return X_transformed
