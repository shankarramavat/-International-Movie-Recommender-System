import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, hstack
import heapq

class RecommendationEngine:
    def __init__(self, movies_df, tfidf_matrix, feature_names):
        """
        Initialize the recommendation engine with processed data.
        
        Args:
            movies_df (pd.DataFrame): Processed movie dataframe
            tfidf_matrix (scipy.sparse.csr_matrix): TF-IDF matrix of movie overviews
            feature_names (list): Feature names from the TF-IDF vectorizer
        """
        self.movies_df = movies_df
        self.tfidf_matrix = tfidf_matrix
        self.feature_names = feature_names
        
        # Pre-compute some metadata matrices for faster recommendations
        self._compute_metadata_similarity()
    
    def _compute_metadata_similarity(self):
        """Compute metadata similarity matrix based on genres, director, and cast."""
        # Genre similarity matrix (one-hot encoded)
        genre_matrix = self._create_genre_matrix()
        
        # Director one-hot encoding
        director_matrix = self._create_director_matrix()
        
        # Cast similarity matrix
        cast_matrix = self._create_cast_matrix()
        
        # Combine all features with different weights
        # Weights: genres (0.45), director (0.35), cast (0.20)
        if genre_matrix is not None and director_matrix is not None and cast_matrix is not None:
            self.metadata_matrix = hstack([
                genre_matrix * 0.45,
                director_matrix * 0.35,
                cast_matrix * 0.2
            ])
        elif genre_matrix is not None:
            # Fallback if only genres are available
            self.metadata_matrix = genre_matrix
        else:
            # Last resort - use a dummy matrix if no metadata is available
            self.metadata_matrix = csr_matrix((len(self.movies_df), 1))
    
    def _create_genre_matrix(self):
        """Create one-hot encoded matrix for genres."""
        # Get all unique genres
        all_genres = set()
        for genres in self.movies_df['genres']:
            if isinstance(genres, list):
                all_genres.update(genres)
        
        if not all_genres:
            return None
            
        # Create genre matrix
        genre_matrix = np.zeros((len(self.movies_df), len(all_genres)))
        genre_list = sorted(list(all_genres))
        
        # Fill the matrix
        for i, movie_genres in enumerate(self.movies_df['genres']):
            if isinstance(movie_genres, list):
                for genre in movie_genres:
                    if genre in genre_list:
                        genre_matrix[i, genre_list.index(genre)] = 1
        
        return csr_matrix(genre_matrix)
    
    def _create_director_matrix(self):
        """Create one-hot encoded matrix for directors."""
        if 'director' not in self.movies_df.columns:
            return None
            
        # Get all unique directors
        directors = self.movies_df['director'].dropna().unique()
        
        if len(directors) == 0:
            return None
            
        # Create director matrix
        director_matrix = np.zeros((len(self.movies_df), len(directors)))
        director_list = sorted(list(directors))
        
        # Fill the matrix
        for i, director in enumerate(self.movies_df['director']):
            if pd.notna(director) and director in director_list:
                director_matrix[i, director_list.index(director)] = 1
        
        return csr_matrix(director_matrix)
    
    def _create_cast_matrix(self):
        """Create weighted matrix for cast members."""
        if 'cast' not in self.movies_df.columns:
            return None
            
        # Get all unique cast members
        all_cast = set()
        for cast in self.movies_df['cast']:
            if isinstance(cast, list):
                all_cast.update(cast)
        
        if not all_cast:
            return None
            
        # Create cast matrix with position-based weighting
        # First position gets higher weight
        cast_matrix = np.zeros((len(self.movies_df), len(all_cast)))
        cast_list = sorted(list(all_cast))
        
        # Fill the matrix with weighted values
        for i, movie_cast in enumerate(self.movies_df['cast']):
            if isinstance(movie_cast, list):
                for j, actor in enumerate(movie_cast):
                    if actor in cast_list:
                        # Weight decreases with position - first actor gets weight 1.0, 
                        # subsequent actors get less weight
                        weight = 1.0 / (j + 1) if j < 3 else 0.2
                        cast_matrix[i, cast_list.index(actor)] = weight
        
        return csr_matrix(cast_matrix)
    
    def get_content_based_recommendations(self, movie_idx, n=5, content_type='plot', filters=None):
        """
        Get content-based recommendations for a movie.
        
        Args:
            movie_idx (int): Index of the target movie
            n (int): Number of recommendations to return
            content_type (str): Type of content to use ('plot' or 'metadata')
            filters (dict): Filters to apply to recommendations
            
        Returns:
            list: List of tuples (movie_idx, similarity_score)
        """
        # Choose the appropriate matrix based on content type
        if content_type == 'plot':
            similarity_matrix = self.tfidf_matrix
        else:  # metadata
            similarity_matrix = self.metadata_matrix
        
        # Compute similarity between the target movie and all other movies
        movie_vector = similarity_matrix[movie_idx]
        similarities = cosine_similarity(movie_vector, similarity_matrix).flatten()
        
        # Get indices of movies sorted by similarity (excluding the target movie)
        similar_indices = similarities.argsort()[::-1]
        similar_indices = similar_indices[similar_indices != movie_idx]
        
        # Apply filters if provided
        if filters:
            similar_indices = self._apply_filters(similar_indices, filters)
        
        # Get top N recommendations with their similarity scores
        top_n = [(idx, similarities[idx]) for idx in similar_indices[:n]]
        
        return top_n
    
    def get_hybrid_recommendations(self, movie_idx, n=5, weights=(0.6, 0.4), filters=None):
        """
        Get hybrid recommendations combining plot-based and metadata-based similarity.
        
        Args:
            movie_idx (int): Index of the target movie
            n (int): Number of recommendations to return
            weights (tuple): Weights for plot and metadata similarities
            filters (dict): Filters to apply to recommendations
            
        Returns:
            list: List of tuples (movie_idx, similarity_score)
        """
        plot_weight, metadata_weight = weights
        
        # Get plot-based similarity
        movie_vector_plot = self.tfidf_matrix[movie_idx]
        plot_similarities = cosine_similarity(movie_vector_plot, self.tfidf_matrix).flatten()
        
        # Get metadata-based similarity
        movie_vector_metadata = self.metadata_matrix[movie_idx]
        metadata_similarities = cosine_similarity(movie_vector_metadata, self.metadata_matrix).flatten()
        
        # Combine similarities with weights
        combined_similarities = (plot_weight * plot_similarities) + (metadata_weight * metadata_similarities)
        
        # Get indices of movies sorted by combined similarity (excluding the target movie)
        similar_indices = combined_similarities.argsort()[::-1]
        similar_indices = similar_indices[similar_indices != movie_idx]
        
        # Apply filters if provided
        if filters:
            similar_indices = self._apply_filters(similar_indices, filters)
        
        # Get top N recommendations with their similarity scores
        top_n = [(idx, combined_similarities[idx]) for idx in similar_indices[:n]]
        
        return top_n
    
    def _apply_filters(self, indices, filters):
        """
        Apply filters to recommendation indices.
        
        Args:
            indices (np.array): Array of movie indices
            filters (dict): Filters to apply
            
        Returns:
            np.array: Filtered indices
        """
        filtered_indices = []
        
        for idx in indices:
            movie = self.movies_df.iloc[idx]
            
            # Filter by year range
            if 'year_range' in filters and filters['year_range'] is not None:
                min_year, max_year = filters['year_range']
                if 'release_year' in movie and pd.notna(movie['release_year']):
                    if movie['release_year'] < min_year or movie['release_year'] > max_year:
                        continue
            
            # Filter by genres
            if 'genres' in filters and filters['genres'] and isinstance(movie.get('genres', []), list):
                # Check if there's overlap between selected genres and movie genres
                if not any(genre in movie['genres'] for genre in filters['genres']):
                    continue
            
            # Filter by industry
            if 'industries' in filters and filters['industries'] and 'industry' in movie:
                if movie['industry'] not in filters['industries']:
                    continue
            
            # Add more filters as needed
            
            # If passed all filters, add to results
            filtered_indices.append(idx)
            
            # Stop if we have enough results
            if len(filtered_indices) >= 100:  # Get more than needed to allow for ranking
                break
        
        return np.array(filtered_indices)
    
    def explain_similarity(self, movie1_idx, movie2_idx, top_n=5):
        """
        Explain the similarity between two movies by identifying common important terms.
        
        Args:
            movie1_idx (int): Index of the first movie
            movie2_idx (int): Index of the second movie
            top_n (int): Number of top features to return
            
        Returns:
            list: List of tuples (feature, importance)
        """
        # Get TF-IDF vectors for both movies
        movie1_vector = self.tfidf_matrix[movie1_idx].toarray().flatten()
        movie2_vector = self.tfidf_matrix[movie2_idx].toarray().flatten()
        
        # Compute feature importance as the product of the two vectors
        # (high values mean both movies have this feature strongly)
        feature_importance = movie1_vector * movie2_vector
        
        # Get indices of top features sorted by importance
        top_feature_indices = feature_importance.argsort()[::-1]
        
        # Filter zeros and get top N
        top_features = []
        for idx in top_feature_indices:
            if feature_importance[idx] > 0:
                feature = self.feature_names[idx]
                importance = feature_importance[idx]
                top_features.append((feature, importance))
                
                if len(top_features) >= top_n:
                    break
        
        return top_features
