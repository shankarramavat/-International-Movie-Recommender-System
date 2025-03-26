import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string

# Ensure NLTK data path is set properly
nltk.data.path.append('/home/runner/nltk_data')

# Force download required NLTK data packages
nltk.download('punkt', quiet=False)
nltk.download('stopwords', quiet=False)
nltk.download('wordnet', quiet=False)

class DataProcessor:
    def __init__(self):
        """Initialize the data processor with NLP tools."""
        # Import NLTK components after ensuring downloads
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer, PorterStemmer

        # Create a simple tokenizer function that doesn't rely on word_tokenize
        self.simple_tokenize = lambda text: text.split() if text else []

        # Fallback stop words if NLTK's don't load
        default_stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 
                             'as', 'what', 'when', 'where', 'how', 'all', 'any', 
                             'both', 'each', 'few', 'more', 'most', 'some', 'such', 
                             'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
                             'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 
                             'should', 'now', 'to', 'of', 'in', 'for', 'on', 'by', 'with'}
        
        # Initialize stop words with fallback
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = default_stop_words

        # Add Hindi stopwords if available
        try:
            hindi_stop_words = set(stopwords.words('hindi'))
            self.stop_words.update(hindi_stop_words)
        except:
            pass
        
        # Add custom Bollywood-specific stopwords
        bollywood_stopwords = {
            'film', 'movie', 'story', 'character', 'bollywood', 
            'hindi', 'indian', 'india', 'cinema', 'actor', 'actress'
        }
        self.stop_words.update(bollywood_stopwords)
        
        # Initialize text processing tools with fallbacks
        try:
            self.lemmatizer = WordNetLemmatizer()
        except:
            # Simple identity lemmatizer as fallback
            self.lemmatizer = type('', (), {'lemmatize': lambda self, word: word})()
            
        try:
            self.stemmer = PorterStemmer()
        except:
            # Simple identity stemmer as fallback
            self.stemmer = type('', (), {'stem': lambda self, word: word})()
        
        # TF-IDF parameters
        self.min_df = 2
        self.max_df = 0.85
        self.max_features = 5000
        
    def preprocess_data(self, df):
        """Preprocess the movie dataframe."""
        # Create a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Clean up dataframe
        df_processed = df_processed.reset_index(drop=True)
        
        # Extract release year from release_date if available
        if 'release_date' in df_processed.columns:
            df_processed['release_year'] = pd.to_datetime(df_processed['release_date'], 
                                                          errors='coerce').dt.year
        elif 'year' in df_processed.columns:
            df_processed['release_year'] = df_processed['year']
        
        # Process text columns
        text_columns = ['overview', 'synopsis', 'plot', 'description']
        target_col = None
        
        # Find available text column to use
        for col in text_columns:
            if col in df_processed.columns and df_processed[col].notna().sum() > 0:
                target_col = col
                break
        
        if target_col is None and 'overview' in df_processed.columns:
            # If no text column with data is found, create an empty overview column
            df_processed['overview'] = ""
            target_col = 'overview'
        elif target_col is None:
            # If no text column exists, create one
            df_processed['overview'] = ""
            target_col = 'overview'
        
        # Preprocess text data
        df_processed['preprocessed_overview'] = df_processed[target_col].fillna("").apply(self.preprocess_text)
        
        # Process genres
        if 'genres' in df_processed.columns:
            # Check if already processed as list
            if df_processed['genres'].dtype != 'object' or isinstance(df_processed['genres'].iloc[0], list):
                pass  # already processed
            else:
                # Process genres from string format to list
                df_processed['genres'] = df_processed['genres'].apply(
                    lambda x: self.extract_genres(x) if isinstance(x, str) else []
                )
        else:
            df_processed['genres'] = [[] for _ in range(len(df_processed))]
        
        # Process cast if available
        if 'cast' in df_processed.columns:
            if df_processed['cast'].dtype != 'object' or isinstance(df_processed['cast'].iloc[0], list):
                pass  # already processed
            else:
                df_processed['cast'] = df_processed['cast'].apply(
                    lambda x: self.extract_names(x) if isinstance(x, str) else []
                )
        else:
            df_processed['cast'] = [[] for _ in range(len(df_processed))]
        
        # Process director if available
        if 'director' in df_processed.columns:
            if df_processed['director'].dtype != 'object':
                pass  # already processed
            else:
                df_processed['director'] = df_processed['director'].apply(
                    lambda x: x.strip() if isinstance(x, str) else ""
                )
        else:
            df_processed['director'] = ""
            
        return df_processed
    
    def preprocess_text(self, text):
        """Preprocess text data for NLP analysis."""
        if not isinstance(text, str) or not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Tokenize using our robust method
        try:
            # Try to import and use word_tokenize if available
            from nltk.tokenize import word_tokenize
            tokens = word_tokenize(text)
        except:
            # Fall back to simple tokenization if word_tokenize fails
            tokens = self.simple_tokenize(text)
        
        # Remove stopwords
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        # Lemmatization (with error handling)
        try:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        except:
            # If lemmatization fails, use tokens as is
            pass
        
        # Join tokens back to string
        processed_text = ' '.join(tokens)
        
        return processed_text
    
    def extract_genres(self, genre_text):
        """Extract genres from text representation."""
        if not genre_text or not isinstance(genre_text, str):
            return []
        
        # Handle common formats
        # Format 1: JSON-like - "[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]"
        if '{' in genre_text and '}' in genre_text and 'name' in genre_text:
            try:
                import json
                genre_list = json.loads(genre_text.replace("'", '"'))
                return [g['name'] for g in genre_list if 'name' in g]
            except:
                pass
                
        # Format 2: List-like - "['Action', 'Adventure', 'Fantasy']"
        if genre_text.startswith('[') and genre_text.endswith(']'):
            try:
                import ast
                return ast.literal_eval(genre_text)
            except:
                # Fallback to regex extraction
                genres = re.findall(r"'([^']*)'", genre_text)
                if genres:
                    return genres
        
        # Format 3: Comma-separated - "Action, Adventure, Fantasy"
        return [g.strip() for g in genre_text.split(',') if g.strip()]
    
    def extract_names(self, name_text):
        """Extract names from text representation."""
        if not name_text or not isinstance(name_text, str):
            return []
        
        # Handle common formats
        # Format 1: JSON-like with 'name' field
        if '{' in name_text and '}' in name_text and 'name' in name_text:
            try:
                import json
                name_list = json.loads(name_text.replace("'", '"'))
                return [n['name'] for n in name_list if 'name' in n]
            except:
                pass
                
        # Format 2: List-like - "['Name1', 'Name2', 'Name3']"
        if name_text.startswith('[') and name_text.endswith(']'):
            try:
                import ast
                return ast.literal_eval(name_text)
            except:
                # Fallback to regex extraction
                names = re.findall(r"'([^']*)'", name_text)
                if names:
                    return names
        
        # Format 3: Comma-separated - "Name1, Name2, Name3"
        return [n.strip() for n in name_text.split(',') if n.strip()]
    
    def vectorize_text(self, text_list):
        """Convert preprocessed text to TF-IDF vectors."""
        vectorizer = TfidfVectorizer(
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.max_features,
            stop_words='english',
            ngram_range=(1, 2)  # Use both unigrams and bigrams
        )
        
        # Fit and transform the text data
        tfidf_matrix = vectorizer.fit_transform(text_list)
        
        # Get feature names for later explanation
        feature_names = vectorizer.get_feature_names_out()
        
        return tfidf_matrix, feature_names
