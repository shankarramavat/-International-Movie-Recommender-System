# 🎬 Bollywood Movie Recommender

A sophisticated movie recommendation system for Bollywood films powered by Natural Language Processing (NLP) and machine learning techniques. This interactive web application provides personalized movie suggestions based on plot similarities, genre matching, and other metadata.

![Bollywood Movie Recommender](https://raw.githubusercontent.com/username/bollywood-movie-recommender/main/screenshot.png)
*Note: Replace the screenshot URL with your own after uploading to GitHub*

## ✨ Features

- **Content-based recommendation system** utilizing movie plots, genres, cast, and directors
- **Multiple recommendation methods**:
  - Plot-based (using NLP techniques)
  - Metadata-based (genres, directors, cast)
  - Hybrid approach (combination of both)
- **Interactive filtering** by release year and genres
- **Real movie posters** from TMDB (The Movie Database)
- **Explanation feature** showing why movies are recommended

## 🚀 Demo

Access the live demo [here](https://your-replit-link-here).

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **NLTK**: Natural Language Processing toolkit for text analysis
- **Scikit-learn**: Machine learning algorithms for recommendation engine
- **Pandas & NumPy**: Data manipulation and analysis
- **TMDB API**: For fetching real movie posters and data
- **Pillow**: Image processing

## 📋 Prerequisites

- Python 3.7+
- A TMDB API key (get one for free at [https://www.themoviedb.org/settings/api](https://www.themoviedb.org/settings/api))

## 🔧 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/bollywood-movie-recommender.git
   cd bollywood-movie-recommender
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

4. **Set up your TMDB API key**
   
   Option 1: Environment variable (recommended)
   ```bash
   # On Windows
   set TMDB_API_KEY=your_api_key_here
   
   # On macOS/Linux
   export TMDB_API_KEY=your_api_key_here
   ```
   
   Option 2: Create a .env file
   ```
   # Create a file named .env in the root directory
   TMDB_API_KEY=your_api_key_here
   ```
   
   *For this method, make sure to uncomment the dotenv loading code in utils.py*

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🖥️ Usage

1. Select a movie from the dropdown menu
2. Adjust filters if desired (release year range, specific genres)
3. Choose a recommendation method:
   - Plot-based: Recommendations based on movie plot similarity
   - Metadata-based: Recommendations based on genres, directors, and cast
   - Hybrid: Combined approach for more balanced recommendations
4. View recommended movies with their posters
5. See similarity explanations for why each movie was recommended

## 📊 How It Works

1. **Data Collection**: Movie data is fetched from TMDB API, focusing on Bollywood movies
2. **Text Processing**: Movie plots are processed using NLP techniques (tokenization, lemmatization, stopword removal)
3. **Feature Extraction**: TF-IDF vectorization converts text to numerical features
4. **Similarity Computation**: Cosine similarity measures how similar movies are to each other
5. **Recommendation Generation**: Top similar movies are recommended based on selected criteria

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [The Movie Database (TMDB)](https://www.themoviedb.org/) for providing the movie data and posters
- [Streamlit](https://streamlit.io/) for the web application framework
- [NLTK](https://www.nltk.org/) for natural language processing capabilities

---

⭐ If you found this project helpful, please give it a star on GitHub! ⭐