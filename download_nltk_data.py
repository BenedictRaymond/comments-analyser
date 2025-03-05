import nltk

def download_nltk_data():
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')
    print("NLTK data download completed!")

if __name__ == "__main__":
    download_nltk_data() 