# Sentiment Analysis and Text Classification

A comprehensive Natural Language Processing (NLP) project that performs sentiment analysis and text classification using machine learning techniques.

## Features

- Text preprocessing (tokenization, stopwords removal, lemmatization)
- Sentiment analysis (positive/negative classification)
- Multi-class text classification
- Data visualization (confusion matrix)
- Feature importance analysis

## Project Structure

```
sentiment-analysis/
├── data/               # Sample data directory
├── src/               # Source code
│   ├── preprocessing.py  # Text preprocessing utilities
│   ├── sentiment.py      # Sentiment analysis implementation
│   └── classification.py # Text classification implementation
├── main.py            # Main script with example usage
├── download_nltk_data.py  # NLTK data download script
├── requirements.txt   # Project dependencies
├── LICENSE           # MIT License
└── README.md         # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/BenedictRaymond/Sentiment-analysis.git
cd Sentiment-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data:
```bash
python download_nltk_data.py
```

## Usage

Run the main script to see example usage:
```bash
python main.py
```

### Example Output

The script demonstrates:
- Text preprocessing
- Sentiment analysis on sample texts
- Text classification into categories
- Confidence scores for predictions

## Dependencies

- numpy: For numerical computations
- pandas: For data manipulation
- scikit-learn: For machine learning algorithms
- nltk: For natural language processing
- spacy: For advanced NLP tasks
- matplotlib & seaborn: For data visualization

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Benedict Raymond

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 