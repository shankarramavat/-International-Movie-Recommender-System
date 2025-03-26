import pandas as pd
import numpy as np
import requests
import os
from io import BytesIO
from PIL import Image
import random

def load_data():
    """
    Load movie data from either a predefined dataset or TMDB API.
    Returns a DataFrame with movie information.
    """
    print("Loading movie data...")
    
    # Try to load from disk if available
    if os.path.exists('movies_database.pkl'):
        try:
            import pickle
            with open('movies_database.pkl', 'rb') as f:
                df = pickle.load(f)
                print(f"Loaded {len(df)} movies from disk cache")
                return df
        except Exception as e:
            print(f"Error loading cached data: {e}")
    
    # Check if we have a TMDB API key
    tmdb_api_key = os.getenv('TMDB_API_KEY', 'ea568542a28df5689f148a9ec3908a53')
    if tmdb_api_key:
        try:
            print("TMDB API key found, fetching real movie data...")
            df = fetch_movies_from_tmdb(tmdb_api_key)
            if not df.empty:
                # Cache to disk for future use
                try:
                    import pickle
                    with open('movies_database.pkl', 'wb') as f:
                        pickle.dump(df, f)
                    print("Cached movie data to disk for future use")
                except Exception as e:
                    print(f"Error caching data: {e}")
                
                return df
        except Exception as e:
            print(f"Error fetching from TMDB: {e}")
    
    # Check if we have a local CSV file
    try:
        if os.path.exists('bollywood_movies.csv'):
            movies_df = pd.read_csv('bollywood_movies.csv')
            print(f"Loaded {len(movies_df)} movies from local CSV")
            return movies_df
    except Exception as e:
        print(f"Error loading local CSV: {str(e)}")
    
    # Fallback to sample dataset if everything else fails
    try:
        return create_sample_dataset()
    except Exception as e:
        print(f"Error creating sample dataset: {str(e)}")
        # Return empty dataframe with right columns as a last resort
        return pd.DataFrame(columns=[
            'title', 'overview', 'genres', 'release_year', 
            'director', 'cast', 'poster_path', 'language', 'industry'
        ])

def fetch_movies_from_tmdb(api_key):
    """
    Fetch movies from TMDB API including all Indian languages and Hollywood
    
    Args:
        api_key (str): TMDB API key
        
    Returns:
        pd.DataFrame: DataFrame with movie information
    """
    from time import sleep
    
    print("Fetching diverse movie collection from TMDB...")
    
    base_url = "https://api.themoviedb.org/3"
    
    # Will store all fetched movies
    movies_data = []
    
    # Movie sources to fetch (categories)
    categories = [
        # Indian movies in different languages
        {"name": "Hindi/Bollywood", "url_params": "with_original_language=hi&region=IN"},
        {"name": "Tamil", "url_params": "with_original_language=ta&region=IN"},
        {"name": "Telugu", "url_params": "with_original_language=te&region=IN"},
        {"name": "Malayalam", "url_params": "with_original_language=ml&region=IN"},
        {"name": "Kannada", "url_params": "with_original_language=kn&region=IN"},
        {"name": "Bengali", "url_params": "with_original_language=bn&region=IN"},
        {"name": "Marathi", "url_params": "with_original_language=mr&region=IN"},
        # Hollywood/English movies
        {"name": "Hollywood", "url_params": "with_original_language=en&sort_by=popularity.desc"}
    ]
    
    # Fetch 2 pages per category (around 40 movies per category)
    for category in categories:
        print(f"Fetching {category['name']} movies...")
        # We'll get several pages of results
        for page in range(1, 3):  # Get 2 pages per category
            # Search for popular movies of this category
            url = f"{base_url}/discover/movie?api_key={api_key}&{category['url_params']}&page={page}"
            response = requests.get(url)
            
            if response.status_code != 200:
                print(f"Error fetching {category['name']} movies: {response.status_code}")
                break
                
            results = response.json()
            
            for movie in results.get('results', []):
                movie_id = movie.get('id')
                if not movie_id:
                    continue
                    
                # Get detailed movie info including credits and videos (trailers)
                details_url = f"{base_url}/movie/{movie_id}?api_key={api_key}&append_to_response=credits,videos,watch/providers"
                details_response = requests.get(details_url)
                
                if details_response.status_code != 200:
                    continue
                    
                details = details_response.json()
                
                # Extract director from crew
                director = ""
                for crew_member in details.get('credits', {}).get('crew', []):
                    if crew_member.get('job') == 'Director':
                        director = crew_member.get('name')
                        break
                
                # Extract top cast members
                cast = []
                for cast_member in details.get('credits', {}).get('cast', [])[:5]:  # Top 5 cast members
                    if cast_member.get('name'):
                        cast.append(cast_member.get('name'))
                
                # Extract genres
                genres = []
                for genre in details.get('genres', []):
                    if genre.get('name'):
                        genres.append(genre.get('name'))
                
                # Extract language and country info
                language = details.get('original_language', '')
                production_countries = [country.get('name', '') for country in details.get('production_countries', [])]
                
                # Determine movie industry (Bollywood, Tollywood, Hollywood, etc.)
                industry = category['name']
                
                # Extract trailer URL
                trailer_url = ""
                if 'videos' in details and 'results' in details['videos']:
                    for video in details['videos']['results']:
                        # Look for official trailers on YouTube
                        if video.get('site') == 'YouTube' and video.get('type') == 'Trailer' and video.get('official'):
                            trailer_url = f"https://www.youtube.com/watch?v={video.get('key')}"
                            break
                    
                    # If no official trailer, look for any trailer
                    if not trailer_url:
                        for video in details['videos']['results']:
                            if video.get('site') == 'YouTube' and video.get('type') == 'Trailer':
                                trailer_url = f"https://www.youtube.com/watch?v={video.get('key')}"
                                break
                
                # Extract OTT/streaming providers
                ott_providers = {}
                providers_data = details.get('watch/providers', {}).get('results', {})
                
                # Check for providers in US, IN (India), and GB (UK) regions
                priority_regions = ['IN', 'US', 'GB']
                for region in priority_regions:
                    if region in providers_data:
                        region_providers = providers_data[region]
                        # Collect flatrate (subscription), buy, and rent options
                        for provider_type in ['flatrate', 'buy', 'rent']:
                            if provider_type in region_providers:
                                ott_providers[provider_type] = [
                                    {
                                        'name': provider.get('provider_name', ''),
                                        'logo': f"https://image.tmdb.org/t/p/original{provider.get('logo_path', '')}" if provider.get('logo_path') else ''
                                    }
                                    for provider in region_providers[provider_type][:3]  # Limit to top 3 providers
                                ]
                        # If we found providers for this region, no need to check others
                        if ott_providers:
                            break
                
                # Create movie dict with all information
                movie_data = {
                    'title': details.get('title', ''),
                    'overview': details.get('overview', ''),
                    'release_year': int(details.get('release_date', '').split('-')[0]) if details.get('release_date') else None,
                    'genres': genres,
                    'director': director,
                    'cast': cast,
                    'poster_path': details.get('poster_path', ''),
                    'language': language,
                    'industry': industry,
                    'production_countries': production_countries,
                    'trailer_url': trailer_url,
                    'ott_providers': ott_providers
                }
                
                movies_data.append(movie_data)
                
            # Be nice to the API with a small delay
            sleep(0.2)
    
    # Convert to DataFrame
    df = pd.DataFrame(movies_data)
    print(f"Fetched {len(df)} movies from TMDB across multiple languages and industries")
    
    return df

def create_sample_dataset():
    """Create a sample Bollywood movie dataset with essential information."""
    # This is a sample dataset with popular Bollywood movies
    # In a production environment, this would be replaced with real data
    
    data = {
        'title': [
            '3 Idiots', 'Dilwale Dulhania Le Jayenge', 'Lagaan', 'PK', 
            'Sholay', 'Bajrangi Bhaijaan', 'Dangal', 'Queen', 
            'Kabhi Khushi Kabhie Gham', 'Andhadhun', 'Gully Boy', 
            'Zindagi Na Milegi Dobara', 'Dil Chahta Hai', 'Barfi!',
            'Gangs of Wasseypur', 'Padmaavat', 'Devdas', 'Kuch Kuch Hota Hai',
            'Dil Se..', 'Jab We Met', 'Haider', 'Rang De Basanti',
            'Mughal-E-Azam', 'Mother India', 'The Lunchbox'
        ],
        'trailer_url': [
            'https://www.youtube.com/watch?v=xvszmNXdM4w',   # 3 Idiots
            'https://www.youtube.com/watch?v=cmax1C1p660',   # DDLJ
            'https://www.youtube.com/watch?v=oSIGQ0YkFxs',   # Lagaan
            'https://www.youtube.com/watch?v=82ZEDGPCkT8',   # PK
            'https://www.youtube.com/watch?v=XjiluhItzIA',   # Sholay
            'https://www.youtube.com/watch?v=vyX4toD395U',   # Bajrangi Bhaijaan
            'https://www.youtube.com/watch?v=x_7YlGv9u1g',   # Dangal
            'https://www.youtube.com/watch?v=KGC6vl3lzf0',   # Queen
            'https://www.youtube.com/watch?v=7uY1JbWZKPA',   # K3G
            'https://www.youtube.com/watch?v=2iVYI99VGaw',   # Andhadhun
            'https://www.youtube.com/watch?v=JfbxcD6biOk',   # Gully Boy
            'https://www.youtube.com/watch?v=FXHNMYwtxsE',   # ZNMD
            'https://www.youtube.com/watch?v=m13b25V0B10',   # Dil Chahta Hai
            'https://www.youtube.com/watch?v=itByMWpKpPo',   # Barfi!
            'https://www.youtube.com/watch?v=9Hu60mzCPPk',   # Gangs of Wasseypur
            'https://www.youtube.com/watch?v=X_5_BLt76c0',   # Padmaavat
            'https://www.youtube.com/watch?v=YF3XxgsVKhw',   # Devdas
            'https://www.youtube.com/watch?v=S9PzLvPiugg',   # KKHH
            'https://www.youtube.com/watch?v=YzFEHx_2XdA',   # Dil Se
            'https://www.youtube.com/watch?v=V1OXK1YTsvU',   # Jab We Met
            'https://www.youtube.com/watch?v=ZmN_VSo8DOo',   # Haider
            'https://www.youtube.com/watch?v=QHhnhqxB4E8',   # Rang De Basanti
            'https://www.youtube.com/watch?v=9xwgSVOv6Js',   # Mughal-E-Azam
            'https://www.youtube.com/watch?v=8JF4l-3sAPs',   # Mother India
            'https://www.youtube.com/watch?v=IUwsTHGY4Ms'    # The Lunchbox
        ],
        'overview': [
            'Two friends are searching for their long lost companion. They revisit their college days and recall the memories of their friend who inspired them to think differently, even as the rest of the world called them "idiots".',
            'A young man and woman fall in love against their parents\' wishes. He follows her to Europe to win her back, and they get a chance to spend time together when her strict father is injured during a trip.',
            'In 1893, a cricket-playing British officer in colonial India challenges the locals to a cricket match, offering to cancel their taxes for three years if they win. With only three months to learn, can the villagers defeat the British?',
            'An alien on Earth loses the only device he can use to communicate with his spaceship. His innocent nature and child-like questions force the country to evaluate their religious beliefs, superstitions and rituals.',
            'Two criminals in rural India are hired to capture a ruthless dacoit but instead set their sights on the bandit\'s beautiful companion.',
            'A little girl from Pakistan gets separated from her mother at an Indian train station, and an Indian man takes it upon himself to reunite her with her family despite the tensions between the two countries.',
            'Former wrestler Mahavir Singh Phogat and his two wrestler daughters struggle towards glory at the Commonwealth Games in the face of societal oppression.',
            'A Delhi girl from a traditional family sets out on a solo honeymoon after her marriage gets canceled.',
            'After a family tragedy, a man adopts his brother\'s children, but his Western-raised son finds it difficult to adapt to Indian culture.',
            'A series of mysterious events changes the life of a blind pianist, who now must report a crime that he never actually witnessed.',
            'A boy from the streets of Mumbai rises to become a successful rapper, highlighting the struggles of the underprivileged and their journey to success.',
            'Three friends decide to turn their fantasy vacation into reality after one of their friends gets engaged.',
            'Three inseparable childhood friends whose relationship is tested when they all fall in love.',
            'A mute man with a tragic childhood develops a relationship with an autistic woman.',
            'A clash between Sultan and Shahid Khan leads to the expulsion of Khan from Wasseypur, and ignites a deadly blood feud spanning three generations.',
            'Set in medieval Rajasthan, Queen Padmavati is married to a noble king and they live in a prosperous fortress with their subjects until an ambitious Sultan hears of Padmavati\'s beauty and forms an obsessive love for her.',
            'After his wealthy family prohibits him from marrying the woman he is in love with, Devdas turns to alcohol and a life of vice to alleviate the pain.',
            'During their years in college, Anjali fell in love with her best friend Rahul, but he had eyes only for Tina. Years later, Rahul and Anjali meet again, but will love bloom this time?',
            'A depressed man travels across India to meet a reporter who writes about sensitive issues of terrorism and ends up falling for her.',
            'A feisty Punjabi girl decides to give her ex-boyfriend another chance, only to find herself on a train journey across India with a free-spirited man who makes her question her choices.',
            'A modern-day adaptation of Shakespeare\'s Hamlet set in the volatile Kashmir conflict between India and Pakistan.',
            'After a group of friends witness the death of a friend due to corrupt officials, they decide to form a group to awaken the youth and bring about a change in the country.',
            'A historical epic about a courtesan who challenges the Prince of the Mughal Empire for her lover.',
            'The story of a poverty-stricken village woman who, in the absence of her husband, struggles to raise her sons and survive against many odds.',
            'A mistaken delivery in Mumbai\'s famously efficient lunchbox delivery system connects a young housewife to an old man in the dusk of his life as they build a fantasy world together through notes in the lunchbox.'
        ],
        'genres': [
            ['Comedy', 'Drama'], ['Romance', 'Drama'], ['Drama', 'Sport'], 
            ['Comedy', 'Drama', 'Sci-Fi'], ['Action', 'Adventure', 'Drama'], 
            ['Adventure', 'Drama', 'Comedy'], ['Biography', 'Drama', 'Sport'], 
            ['Adventure', 'Comedy', 'Drama'], ['Drama', 'Romance'], 
            ['Crime', 'Drama', 'Mystery'], ['Drama', 'Music'], 
            ['Drama', 'Comedy', 'Adventure'], ['Comedy', 'Drama', 'Romance'], 
            ['Comedy', 'Drama', 'Romance'], ['Crime', 'Drama', 'Thriller'], 
            ['Drama', 'History', 'Romance'], ['Drama', 'Musical', 'Romance'], 
            ['Drama', 'Romance'], ['Drama', 'Romance', 'Thriller'], 
            ['Comedy', 'Drama', 'Romance'], ['Action', 'Crime', 'Drama'], 
            ['Comedy', 'Drama', 'Romance'], ['Drama', 'History', 'Romance'], 
            ['Drama', 'Family'], ['Drama', 'Romance']
        ],
        'release_year': [
            2009, 1995, 2001, 2014, 1975, 2015, 2016, 2013, 
            2001, 2018, 2019, 2011, 2001, 2012, 2012, 2018, 
            2002, 1998, 1998, 2007, 2014, 2006, 1960, 1957, 2013
        ],
        'director': [
            'Rajkumar Hirani', 'Aditya Chopra', 'Ashutosh Gowariker', 'Rajkumar Hirani', 
            'Ramesh Sippy', 'Kabir Khan', 'Nitesh Tiwari', 'Vikas Bahl', 
            'Karan Johar', 'Sriram Raghavan', 'Zoya Akhtar', 'Zoya Akhtar', 
            'Farhan Akhtar', 'Anurag Basu', 'Anurag Kashyap', 'Sanjay Leela Bhansali', 
            'Sanjay Leela Bhansali', 'Karan Johar', 'Mani Ratnam', 'Imtiaz Ali', 
            'Vishal Bhardwaj', 'Rakeysh Omprakash Mehra', 'K. Asif', 'Mehboob Khan', 'Ritesh Batra'
        ],
        'cast': [
            ['Aamir Khan', 'Madhavan', 'Sharman Joshi', 'Kareena Kapoor'],
            ['Shah Rukh Khan', 'Kajol', 'Amrish Puri', 'Farida Jalal'],
            ['Aamir Khan', 'Gracy Singh', 'Rachel Shelley', 'Paul Blackthorne'],
            ['Aamir Khan', 'Anushka Sharma', 'Sushant Singh Rajput', 'Boman Irani'],
            ['Dharmendra', 'Amitabh Bachchan', 'Hema Malini', 'Sanjeev Kumar'],
            ['Salman Khan', 'Kareena Kapoor', 'Harshali Malhotra', 'Nawazuddin Siddiqui'],
            ['Aamir Khan', 'Fatima Sana Shaikh', 'Sanya Malhotra', 'Sakshi Tanwar'],
            ['Kangana Ranaut', 'Rajkummar Rao', 'Lisa Haydon', 'Mish Boyko'],
            ['Shah Rukh Khan', 'Kajol', 'Hrithik Roshan', 'Kareena Kapoor'],
            ['Ayushmann Khurrana', 'Tabu', 'Radhika Apte', 'Anil Dhawan'],
            ['Ranveer Singh', 'Alia Bhatt', 'Siddhant Chaturvedi', 'Kalki Koechlin'],
            ['Hrithik Roshan', 'Farhan Akhtar', 'Abhay Deol', 'Katrina Kaif'],
            ['Aamir Khan', 'Saif Ali Khan', 'Akshaye Khanna', 'Preity Zinta'],
            ['Ranbir Kapoor', 'Priyanka Chopra', 'Ileana D\'Cruz', 'Saurabh Shukla'],
            ['Manoj Bajpayee', 'Nawazuddin Siddiqui', 'Richa Chadha', 'Tigmanshu Dhulia'],
            ['Deepika Padukone', 'Ranveer Singh', 'Shahid Kapoor', 'Aditi Rao Hydari'],
            ['Shah Rukh Khan', 'Aishwarya Rai', 'Madhuri Dixit', 'Jackie Shroff'],
            ['Shah Rukh Khan', 'Kajol', 'Rani Mukerji', 'Salman Khan'],
            ['Shah Rukh Khan', 'Manisha Koirala', 'Preity Zinta', 'Mita Vasisht'],
            ['Shahid Kapoor', 'Kareena Kapoor', 'Tarun Arora', 'Dara Singh'],
            ['Shahid Kapoor', 'Tabu', 'Shraddha Kapoor', 'Kay Kay Menon'],
            ['Aamir Khan', 'Siddharth', 'Sharman Joshi', 'Kunal Kapoor'],
            ['Prithviraj Kapoor', 'Dilip Kumar', 'Madhubala', 'Durga Khote'],
            ['Nargis', 'Sunil Dutt', 'Rajendra Kumar', 'Raaj Kumar'],
            ['Irrfan Khan', 'Nimrat Kaur', 'Nawazuddin Siddiqui', 'Lillete Dubey']
        ],
        'poster_path': [
            '/66A9MqXOyVFCssoloscw79z8Tew.jpg',  # 3 Idiots
            '/kGRavMK9PgCSB0LS6krDo8XCl98.jpg',  # DDLJ
            '/y9TyrEPHjohqM9lDRJKcRqKGQdj.jpg',  # Lagaan
            '/imo7U2mLGqSqK0oQ7dMkZIgXJTP.jpg',  # PK
            '/5euApaOQGoko9njg9Z8uIvpK6Eg.jpg',  # Sholay 
            '/oA8iTmkMJJA10iBNkP9wCuFPfL2.jpg',  # Bajrangi Bhaijaan
            '/mdSrxMg4GhZJSY0kbfDlrSG6lU3.jpg',  # Dangal
            '/vTMWdYTA8OZZx6uEYXTbZt4Xwgf.jpg',  # Queen
            '/3o5xtT3dOlqAIf3S0CEpQQXUZkw.jpg',  # K3G
            '/hjkJkPL9z5TjU3GPnUUKWJoQJlT.jpg',  # Andhadhun
            '/q9xY6AJjXbAQ0zdYVztKbVZvBKb.jpg',  # Gully Boy
            '/v0mkvMPTeReA0JIT4KBQ9n0vwKU.jpg',  # ZNMD
            '/gOFIxG3KkuztFsQPC4qyYbXx5Ou.jpg',  # Dil Chahta Hai
            '/2NIvmlXD7jHl4bM4ijvtTP5XmrN.jpg',  # Barfi!
            '/uLUpWZzz6Ed0jJcYuYUhJnrjDHe.jpg',  # Gangs of Wasseypur
            '/xEPXbCCrn2FodoF3zXnWXJsRWxp.jpg',  # Padmaavat
            '/6X6r9g21F4qxjBE1k9Qe1gWWKE0.jpg',  # Devdas
            '/5XJcTKqvg0hPeD8qQxmSJ9D4wO6.jpg',  # KKHH
            '/gKcnlBTgRqSdImS3OV4BgwFxKWd.jpg',  # Dil Se
            '/eRCYY8uWxs7PIIKBmLvUIgGdXY7.jpg',  # Jab We Met
            '/qwVKbWhp8NhfWs2t8ipsmjbzlS7.jpg',  # Haider
            '/7VCPbN4dlwIxKeXsBDQqKLxKHfz.jpg',  # Rang De Basanti
            '/kZxzBKyQKDOM0l3OFsHuCLsr7iv.jpg',  # Mughal-E-Azam
            '/eDKZHXQVpJ9ZKlgOQjn6MSRvy9F.jpg',  # Mother India
            '/n52Vlz5ac6LsT1xvUzwY8m5vR1f.jpg'   # The Lunchbox
        ]
    }
    
    # Add OTT provider data for some popular movies
    ott_providers = [
        # 3 Idiots
        {
            'flatrate': [
                {'name': 'Netflix', 'logo': 'https://image.tmdb.org/t/p/original/t2yyOv40HZeVlLjYsCsPHnWLk4W.jpg'},
                {'name': 'Amazon Prime', 'logo': 'https://image.tmdb.org/t/p/original/68MNrwlkpF7WnmNPXLah69CR5cb.jpg'}
            ]
        },
        # DDLJ
        {
            'flatrate': [
                {'name': 'Amazon Prime', 'logo': 'https://image.tmdb.org/t/p/original/68MNrwlkpF7WnmNPXLah69CR5cb.jpg'}
            ],
            'rent': [
                {'name': 'Apple TV', 'logo': 'https://image.tmdb.org/t/p/original/peURlLlr8jggOwK53fJ5wdQl05y.jpg'}
            ]
        },
        # For the rest of the movies, create some dummy or empty OTT data
    ] + [{} for _ in range(len(data['title']) - 2)]  # Empty dictionaries for the rest
    
    # Add 'industry' field to sample data
    industries = ['Hindi/Bollywood'] * len(data['title'])
    
    # Add all fields to the data dictionary
    data['ott_providers'] = ott_providers
    data['industry'] = industries
    
    # Create the DataFrame
    df = pd.DataFrame(data)
    
    print(f"Created sample dataset with {len(df)} Bollywood movies")
    return df

def generate_placeholder_image(movie_name="No Poster", width=250, height=375):
    """
    Generate a colored placeholder image with the movie name
    
    Args:
        movie_name (str): Name to display on the placeholder
        width (int): Width of the image
        height (int): Height of the image
        
    Returns:
        PIL.Image: A generated placeholder image
    """
    from PIL import Image, ImageDraw, ImageFont
    import hashlib
    
    # Generate a color based on the movie name for consistency
    hash_value = hashlib.md5(movie_name.encode()).hexdigest()
    r = int(hash_value[:2], 16)
    g = int(hash_value[2:4], 16)
    b = int(hash_value[4:6], 16)
    
    # Create a blank image with the color
    img = Image.new('RGB', (width, height), color=(r, g, b))
    draw = ImageDraw.Draw(img)
    
    # Draw a border
    border_width = 4
    draw.rectangle(
        ((border_width, border_width), 
         (width - border_width, height - border_width)),
        outline=(255, 255, 255)
    )
    
    # Draw the movie name text
    font_size = 20
    text_y_position = height // 2
    
    # Split the movie name if it's too long
    words = movie_name.split()
    lines = []
    current_line = []
    
    for word in words:
        current_line.append(word)
        if len(' '.join(current_line)) > 15:  # Limit chars per line
            if len(current_line) > 1:
                lines.append(' '.join(current_line[:-1]))
                current_line = [current_line[-1]]
            else:
                lines.append(' '.join(current_line))
                current_line = []
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # Draw each line of text
    for i, line in enumerate(lines):
        line_y = text_y_position - ((len(lines) - 1) * font_size // 2) + (i * font_size)
        text_width = len(line) * font_size // 2
        text_x = (width - text_width) // 2
        draw.text((text_x, line_y), line, fill=(255, 255, 255))
    
    return img

def fetch_poster(poster_path):
    """
    Fetch movie poster from path or URL.
    
    Args:
        poster_path (str): Path or URL to poster image
        
    Returns:
        PIL.Image: A PIL Image object (either loaded from URL or generated)
    """
    # First, prioritize TMDB paths (starting with /)
    if poster_path and poster_path.startswith('/'):
        try:
            tmdb_api_key = os.getenv('TMDB_API_KEY', 'ea568542a28df5689f148a9ec3908a53')
            if tmdb_api_key:
                print(f"Fetching poster from TMDB: {poster_path}")
                base_url = "https://image.tmdb.org/t/p/w500"
                response = requests.get(f"{base_url}{poster_path}")
                if response.status_code == 200:
                    return Image.open(BytesIO(response.content))
                else:
                    print(f"Failed to fetch TMDB image: HTTP {response.status_code}")
        except Exception as e:
            print(f"Error loading TMDB image: {e}")
    
    # Check for local files (like sample dataset posters)
    if poster_path and os.path.exists(poster_path):
        try:
            return Image.open(poster_path)
        except Exception as e:
            print(f"Error opening local image: {e}")
    
    # If it's a full URL
    if poster_path and poster_path.startswith(('http://', 'https://')):
        try:
            response = requests.get(poster_path)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"Error loading image from URL: {e}")
    
    # For the sample dataset with _poster.jpg paths
    if poster_path and '_poster.jpg' in poster_path:
        try:
            # Try to extract movie name from the path
            movie_name = poster_path.split('_poster.jpg')[0].split('/')[-1]
            return generate_placeholder_image(movie_name)
        except Exception as e:
            print(f"Error generating placeholder: {e}")
    
    # Default placeholder if nothing worked
    movie_name = "Unknown Movie" if not poster_path else poster_path.split('/')[-1]
    return generate_placeholder_image(movie_name)

def calculate_similarity(movie1, movie2, features, weights=None):
    """
    Calculate similarity between two movies based on multiple features.
    
    Args:
        movie1 (dict/Series): First movie
        movie2 (dict/Series): Second movie
        features (list): List of features to compare
        weights (dict): Feature weights (default: equal weights)
        
    Returns:
        float: Similarity score (0-1)
    """
    if weights is None:
        weights = {feature: 1/len(features) for feature in features}
    
    similarity = 0
    
    for feature in features:
        if feature not in movie1 or feature not in movie2:
            continue
            
        if feature == 'genres':
            # Jaccard similarity for genre lists
            genres1 = set(movie1[feature]) if isinstance(movie1[feature], list) else set()
            genres2 = set(movie2[feature]) if isinstance(movie2[feature], list) else set()
            
            if not genres1 or not genres2:
                feature_sim = 0
            else:
                intersection = len(genres1.intersection(genres2))
                union = len(genres1.union(genres2))
                feature_sim = intersection / union if union > 0 else 0
        
        elif feature == 'director':
            # Binary similarity for director
            feature_sim = 1 if movie1[feature] == movie2[feature] else 0
        
        elif feature == 'cast':
            # Weighted overlap for cast lists
            cast1 = set(movie1[feature][:3]) if isinstance(movie1[feature], list) else set()
            cast2 = set(movie2[feature][:3]) if isinstance(movie2[feature], list) else set()
            
            if not cast1 or not cast2:
                feature_sim = 0
            else:
                intersection = len(cast1.intersection(cast2))
                feature_sim = intersection / 3  # Normalize by max possible intersection (3)
        
        elif feature == 'release_year':
            # Normalized difference for years
            year1 = movie1[feature] if pd.notna(movie1[feature]) else 0
            year2 = movie2[feature] if pd.notna(movie2[feature]) else 0
            
            if year1 == 0 or year2 == 0:
                feature_sim = 0
            else:
                diff = abs(year1 - year2)
                # Closer years are more similar, with a 10-year difference giving 0.5 similarity
                feature_sim = max(0, 1 - (diff / 20))
        
        else:
            # Default: treat as binary feature
            feature_sim = 1 if movie1[feature] == movie2[feature] else 0
        
        # Add weighted feature similarity
        similarity += weights[feature] * feature_sim
    
    return similarity
