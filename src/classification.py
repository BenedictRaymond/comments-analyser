from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class TextClassifier:
    def __init__(self):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', LogisticRegression(max_iter=1000))
        ])
        
    def train(self, texts, labels):
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'train_accuracy': self.model.score(X_train, y_train),
            'test_accuracy': self.model.score(X_test, y_test),
            'classification_report': report
        }
    
    def predict(self, text):
        prediction = self.model.predict([text])[0]
        probabilities = self.model.predict_proba([text])[0]
        
        return {
            'class': prediction,
            'confidence': float(max(probabilities)),
            'probabilities': dict(zip(self.model.classes_, probabilities))
        }
    
    def plot_confusion_matrix(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.model.classes_,
                   yticklabels=self.model.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self, top_n=10):
        feature_names = self.model.named_steps['tfidf'].get_feature_names_out()
        coefficients = self.model.named_steps['clf'].coef_
        
        feature_importance = {}
        for i, class_name in enumerate(self.model.classes_):
            top_indices = np.argsort(coefficients[i])[-top_n:]
            feature_importance[class_name] = [
                (feature_names[idx], coefficients[i][idx])
                for idx in top_indices
            ]
        
        return feature_importance 