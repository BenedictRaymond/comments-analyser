from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np

class SentimentAnalyzer:
    def __init__(self):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', MultinomialNB())
        ])
        
    def train(self, texts, labels):
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score
        }
    
    def predict(self, text):
        prediction = self.model.predict([text])[0]
        probabilities = self.model.predict_proba([text])[0]
        
        return {
            'sentiment': prediction,
            'confidence': float(max(probabilities))
        }
    
    def analyze_batch(self, texts):
        predictions = self.model.predict(texts)
        probabilities = self.model.predict_proba(texts)
        confidences = np.max(probabilities, axis=1)
        
        return [
            {
                'text': text,
                'sentiment': pred,
                'confidence': float(conf)
            }
            for text, pred, conf in zip(texts, predictions, confidences)
        ] 