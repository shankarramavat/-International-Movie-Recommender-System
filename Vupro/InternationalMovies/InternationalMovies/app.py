import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from data_processor import DataProcessor
from recommendation_engine import RecommendationEngine
from utils import fetch_poster, load_data

# Page configuration
st.set_page_config(
    page_title="Multi-Language Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# App title
st.title("🎬 International Movie Recommender System")
st.markdown("Discover movies from Bollywood, Regional Indian Cinema and Hollywood using advanced NLP and machine learning!")

# Initialize session state variables if they don't exist
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'movies_df' not in st.session_state:
    st.session_state.movies_df = None
if 'tfidf_matrix' not in st.session_state:
    st.session_state.tfidf_matrix = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None

# Data loading section with spinner
if not st.session_state.data_loaded:
    with st.spinner("Loading movie data and initializing recommendation engine..."):
        try:
            # Load data
            movies_df = load_data()
            
            # Initialize processor and engine
            processor = DataProcessor()
            movies_df = processor.preprocess_data(movies_df)
            
            # Process movie data for recommendations
            tfidf_matrix, feature_names = processor.vectorize_text(movies_df['preprocessed_overview'].tolist())
            
            # Initialize recommendation engine
            engine = RecommendationEngine(movies_df, tfidf_matrix, feature_names)
            
            # Save to session state
            st.session_state.processor = processor
            st.session_state.engine = engine
            st.session_state.movies_df = movies_df
            st.session_state.tfidf_matrix = tfidf_matrix
            st.session_state.feature_names = feature_names
            st.session_state.data_loaded = True
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.stop()

# Main app UI (only show after data is loaded)
if st.session_state.data_loaded:
    movies_df = st.session_state.movies_df
    engine = st.session_state.engine
    
    # Sidebar for filters and options
    with st.sidebar:
        st.header("Filters & Options")
        
        # Movie search/selection
        st.subheader("Find a Movie")
        selected_movie = st.selectbox(
            "Type or select a movie title:",
            options=movies_df['title'].tolist(),
            index=0,
            help="Select a movie to get recommendations similar to it"
        )
        
        # Recommendation count slider
        num_recommendations = st.slider(
            "Number of recommendations:", 
            min_value=1, 
            max_value=20, 
            value=5
        )
        
        # Filter options
        st.subheader("Filter Options")
        
        # Filter by decade/era
        available_years = sorted(movies_df['release_year'].dropna().unique())
        if len(available_years) > 0:
            min_year = int(min(available_years))
            max_year = int(max(available_years))
            
            year_range = st.slider(
                "Release Year Range:",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )
        
        # Genre selection (multi-select)
        all_genres = set()
        for genres in movies_df['genres'].dropna():
            if isinstance(genres, list):
                all_genres.update(genres)
        
        selected_genres = st.multiselect(
            "Select Genres:",
            options=sorted(list(all_genres)),
            default=[]
        )
        
        # Industry selection (multi-select) - New filter for different industries
        if 'industry' in movies_df.columns:
            all_industries = sorted(movies_df['industry'].dropna().unique())
            selected_industries = st.multiselect(
                "Select Movie Industries:",
                options=all_industries,
                default=[]
            )
        
        # Recommendation approach
        st.subheader("Recommendation Method")
        recommendation_method = st.radio(
            "Choose method:",
            ["Plot-based", "Genre-based", "Combined"],
            index=2
        )
        
        # Get recommendations button
        recommend_button = st.button("Get Recommendations", use_container_width=True)
    
    # Main content area
    # Display selected movie details
    if selected_movie:
        movie_idx = movies_df[movies_df['title'] == selected_movie].index[0]
        movie_info = movies_df.iloc[movie_idx]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            poster_path = movie_info.get('poster_path', '')
            # Get poster (either PIL Image or URL)
            poster = fetch_poster(poster_path)
            st.image(poster, width=250, caption=selected_movie)
        
        with col2:
            st.subheader(f"{selected_movie} ({movie_info.get('release_year', 'N/A')})")
            
            # Display genres as pills/tags
            genres = movie_info.get('genres', [])
            if isinstance(genres, list) and genres:
                st.write("**Genres:** " + ", ".join(genres))
            
            # Movie overview with expandable section if it's long
            overview = movie_info.get('overview', 'No overview available.')
            if len(overview) > 300:
                st.write("**Overview:**")
                st.expander("Show full overview", expanded=False).write(overview)
                st.write(overview[:300] + "...")
            else:
                st.write("**Overview:**")
                st.write(overview)
            
            # Cast and director info if available
            cast = movie_info.get('cast', [])
            if isinstance(cast, list) and cast:
                st.write("**Cast:** " + ", ".join(cast[:5]))
            
            director = movie_info.get('director', 'Unknown')
            if director:
                st.write(f"**Director:** {director}")
                
            # Display industry information if available
            industry = movie_info.get('industry', '')
            if industry:
                st.write(f"**Industry:** {industry}")
                
            # Display language if available
            language = movie_info.get('language', '')
            if language:
                language_names = {
                    'en': 'English',
                    'hi': 'Hindi',
                    'ta': 'Tamil',
                    'te': 'Telugu',
                    'ml': 'Malayalam',
                    'kn': 'Kannada', 
                    'bn': 'Bengali',
                    'mr': 'Marathi'
                }
                language_display = language_names.get(language, language)
                st.write(f"**Language:** {language_display}")
                
            # Display trailer link if available
            trailer_url = movie_info.get('trailer_url', '')
            if trailer_url:
                st.markdown(f"[🎬 **Watch Trailer**]({trailer_url})")
                
            # Display OTT/streaming platforms if available
            ott_providers = movie_info.get('ott_providers', {})
            if ott_providers:
                st.write("**Watch On:**")
                
                # First show subscription services
                if 'flatrate' in ott_providers:
                    st.write("📺 **Subscription:**")
                    cols = st.columns(min(3, len(ott_providers['flatrate'])))
                    for i, provider in enumerate(ott_providers['flatrate']):
                        cols[i % len(cols)].write(provider['name'])
                
                # Then show rental options
                if 'rent' in ott_providers:
                    st.write("🔄 **Rent:**")
                    cols = st.columns(min(3, len(ott_providers['rent'])))
                    for i, provider in enumerate(ott_providers['rent']):
                        cols[i % len(cols)].write(provider['name'])
                
                # Finally show purchase options
                if 'buy' in ott_providers:
                    st.write("💰 **Buy:**")
                    cols = st.columns(min(3, len(ott_providers['buy'])))
                    for i, provider in enumerate(ott_providers['buy']):
                        cols[i % len(cols)].write(provider['name'])
        
        # Get and display recommendations when button is clicked
        if recommend_button:
            with st.spinner("Finding movies you'll love..."):
                # Apply filters to recommendation criteria
                filters = {
                    'year_range': year_range if 'year_range' in locals() else None,
                    'genres': selected_genres if selected_genres else None,
                }
                
                # Add industry filter if selected
                if 'industry' in movies_df.columns and 'selected_industries' in locals() and selected_industries:
                    filters['industries'] = selected_industries
                
                # Get recommendations based on selected method
                if recommendation_method == "Plot-based":
                    recommendations = engine.get_content_based_recommendations(
                        movie_idx, 
                        n=num_recommendations,
                        content_type='plot',
                        filters=filters
                    )
                elif recommendation_method == "Genre-based":
                    recommendations = engine.get_content_based_recommendations(
                        movie_idx, 
                        n=num_recommendations,
                        content_type='metadata',
                        filters=filters
                    )
                else:  # Combined
                    recommendations = engine.get_hybrid_recommendations(
                        movie_idx, 
                        n=num_recommendations,
                        filters=filters
                    )
                
                if not recommendations:
                    st.warning("No recommendations found based on your filters. Try adjusting your criteria.")
                else:
                    st.subheader(f"Top {len(recommendations)} Recommendations for '{selected_movie}'")
                    
                    # Create 5 columns for recommendations
                    cols = st.columns(min(5, len(recommendations)))
                    
                    # Display each recommendation with poster and info
                    for i, (rec_idx, similarity) in enumerate(recommendations):
                        col_idx = i % len(cols)
                        with cols[col_idx]:
                            rec_info = movies_df.iloc[rec_idx]
                            title = rec_info['title']
                            
                            # Display poster
                            poster_path = rec_info.get('poster_path', '')
                            # Get poster (either PIL Image or URL)
                            poster = fetch_poster(poster_path)
                            st.image(poster, width=150, caption=title)
                            
                            # Movie info
                            year = rec_info.get('release_year', 'N/A')
                            st.write(f"**{title}** ({year})")
                            st.write(f"Similarity: {similarity:.1%}")
                            
                            # Key features
                            if isinstance(rec_info.get('genres', []), list):
                                genres_text = ", ".join(rec_info['genres'][:2])
                                st.write(f"Genres: {genres_text}")
                            
                            # Show industry if available
                            industry = rec_info.get('industry', '')
                            if industry:
                                st.write(f"Industry: {industry}")
                                
                            # Show trailer link if available
                            trailer_url = rec_info.get('trailer_url', '')
                            if trailer_url:
                                st.markdown(f"[🎬 Watch Trailer]({trailer_url})")
                    
                    # Additional technical details in expander
                    with st.expander("Technical Details", expanded=False):
                        st.write("### How these recommendations were generated")
                        st.write(f"Method: {recommendation_method}")
                        
                        if recommendation_method == "Plot-based":
                            st.write("Used TF-IDF vectorization of movie plot summaries to find semantically similar movies.")
                        elif recommendation_method == "Genre-based":
                            st.write("Used similarity matching based on movie metadata like genres, director, and cast.")
                        else:
                            st.write("Combined both plot-based and metadata-based similarities for a hybrid recommendation.")
                        
                        st.write("### Top common features with selected movie")
                        
                        # Show top matching features for first recommendation
                        if recommendations and recommendation_method != "Genre-based":
                            try:
                                first_rec_idx = recommendations[0][0]
                                common_features = engine.explain_similarity(movie_idx, first_rec_idx)
                                
                                if common_features:
                                    st.write(f"**{selected_movie}** and **{movies_df.iloc[first_rec_idx]['title']}** share these key elements:")
                                    for feature, importance in common_features[:5]:
                                        st.write(f"- '{feature}' (importance: {importance:.2f})")
                            except Exception as e:
                                st.write(f"Couldn't generate feature explanation: {str(e)}")
                        
    # Empty state
    else:
        st.info("Select a movie from the sidebar to get started!")

    # Footer with data source info
    st.markdown("---")
    st.markdown("**Data Sources:** TMDB and movie metadata collections. This app uses NLP and machine learning to analyze movie content and suggest similar titles.")
