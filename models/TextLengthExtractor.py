import pandas as pd
#Customize transformer
from sklearn.base import BaseEstimator, TransformerMixin
# Create textLengthExtractor()
class TextLengthExtractor(BaseEstimator, TransformerMixin):
    
    def count_length(self, text):
        text_length = len(text)
        return text_length
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        text_length = pd.Series(X).apply(self.count_length)
        return pd.DataFrame(text_length)