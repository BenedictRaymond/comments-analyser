import nltk
from src.preprocessing import TextPreprocessor
from src.sentiment import SentimentAnalyzer
from src.classification import TextClassifier

def download_nltk_data():
    """Download required NLTK data."""
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def main():
    # Download required NLTK data
    download_nltk_data()
    
    # Initialize components
    preprocessor = TextPreprocessor()
    sentiment_analyzer = SentimentAnalyzer()
    text_classifier = TextClassifier()
    
    # Example texts for demonstration
    sample_texts = [
        "I absolutely love this product! It's amazing!",
        "This is terrible, I hate it.",
        "The weather is nice today.",
        "The movie was okay, but nothing special.",
        "This is a technical document about machine learning."
    ]
    
    # Example labels for sentiment analysis (1: positive, 0: negative)
    sentiment_labels = [1, 0, 1, 0, 1]
    
    # Example labels for text classification
    classification_labels = ['review', 'review', 'general', 'review', 'technical']
    
    # Preprocess texts
    print("Preprocessing texts...")
    processed_texts = [preprocessor.preprocess(text) for text in sample_texts]
    for original, processed in zip(sample_texts, processed_texts):
        print(f"\nOriginal: {original}")
        print(f"Processed: {processed}")
    
    # Train and test sentiment analysis
    print("\nTraining sentiment analysis model...")
    sentiment_results = sentiment_analyzer.train(processed_texts, sentiment_labels)
    print(f"Sentiment Analysis Results:")
    print(f"Training Accuracy: {sentiment_results['train_accuracy']:.2f}")
    print(f"Testing Accuracy: {sentiment_results['test_accuracy']:.2f}")
    
    # Test sentiment analysis on new text
    test_text = "This is a fantastic experience!"
    processed_test = preprocessor.preprocess(test_text)
    sentiment_result = sentiment_analyzer.predict(processed_test)
    print(f"\nSentiment Analysis for '{test_text}':")
    print(f"Sentiment: {'Positive' if sentiment_result['sentiment'] == 1 else 'Negative'}")
    print(f"Confidence: {sentiment_result['confidence']:.2f}")
    
    # Train and test text classification
    print("\nTraining text classification model...")
    classification_results = text_classifier.train(processed_texts, classification_labels)
    print(f"Text Classification Results:")
    print(f"Training Accuracy: {classification_results['train_accuracy']:.2f}")
    print(f"Testing Accuracy: {classification_results['test_accuracy']:.2f}")
    
    # Test text classification on new text
    test_text = "This is a scientific research paper about artificial intelligence."
    processed_test = preprocessor.preprocess(test_text)
    classification_result = text_classifier.predict(processed_test)
    print(f"\nText Classification for '{test_text}':")
    print(f"Class: {classification_result['class']}")
    print(f"Confidence: {classification_result['confidence']:.2f}")
    print("\nClass Probabilities:")
    for class_name, prob in classification_result['probabilities'].items():
        print(f"{class_name}: {prob:.2f}")

if __name__ == "__main__":
    main() 