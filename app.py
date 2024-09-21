import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import time
import json
import os
import logging
import nltk

# ---------------------------- Configuration ---------------------------- #

# Download NLTK data (only the first time)
nltk.download('vader_lexicon')

# Cache directories
CACHE_DIR = "cache"
HISTORICAL_DIR = "historical_data"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
if not os.path.exists(HISTORICAL_DIR):
    os.makedirs(HISTORICAL_DIR)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sentiment Analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()


# ---------------------------- Functions ---------------------------- #

def get_owned_games(api_key, steam_id):
    """
    Fetches the list of owned games for the specified Steam user.
    """
    OWNED_GAMES_URL = 'https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/'
    params = {
        'key': api_key,
        'steamid': steam_id,
        'include_appinfo': True,
        'include_played_free_games': True,
        'format': 'json'
    }

    try:
        response = requests.get(OWNED_GAMES_URL, params=params)
        response.raise_for_status()
        data = response.json()
        games = data.get('response', {}).get('games', [])
        logger.info(f"Retrieved {len(games)} games from Steam library.")
        return games
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred while fetching owned games: {http_err}")
        st.error(f"HTTP error occurred while fetching owned games: {http_err}")
        return []
    except Exception as err:
        logger.error(f"An error occurred while fetching owned games: {err}")
        st.error(f"An error occurred while fetching owned games: {err}")
        return []


def get_app_details(app_id, api_key=None, max_retries=5, request_delay=1.0):
    """
    Fetches detailed information for a specific app/game using the Steam Store API.
    Implements exponential backoff in case of rate limiting.
    """
    APP_DETAILS_URL = 'https://store.steampowered.com/api/appdetails'
    params = {
        'appids': app_id,
        'cc': 'us',  # Country code to get consistent data
        'l': 'en'  # Language
    }

    delay = request_delay
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(APP_DETAILS_URL, params=params)
            if response.status_code == 429:
                logger.warning(f"Rate limit hit. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                continue
            response.raise_for_status()
            data = response.json()
            app_data = data.get(str(app_id), {})
            if app_data.get('success'):
                return app_data.get('data', {})
            else:
                logger.error(f"Failed to retrieve details for App ID {app_id}.")
                return {}
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred for App ID {app_id}: {http_err}")
            return {}
        except Exception as err:
            logger.error(f"An error occurred for App ID {app_id}: {err}")
            return {}
    logger.error(f"Max retries exceeded for App ID {app_id}. Skipping...")
    return {}


def load_cache(cache_path):
    """
    Load cached app details from a JSON file.
    """
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_cache(cache, cache_path):
    """
    Save app details cache to a JSON file.
    """
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)


def get_app_details_with_cache(app_id, cache, api_key=None, max_retries=5, request_delay=1.0):
    """
    Fetch app details using cache to minimize API calls.
    """
    if str(app_id) in cache:
        return cache[str(app_id)]

    app_details = get_app_details(app_id, api_key, max_retries, request_delay)
    cache[str(app_id)] = app_details
    return app_details


def extract_game_info(game, app_details):
    """
    Extracts relevant information from the game and its app details.
    """
    info = {}
    info['App ID'] = game.get('appid', '')
    info['Name'] = game.get('name', '')
    info['Playtime (minutes)'] = game.get('playtime_forever', 0)
    info['Playtime (2 weeks)'] = game.get('playtime_2weeks', 0)
    info['Playtime Last Session (minutes)'] = game.get('playtime_windows_forever', 0)

    # From app details
    info['Release Date'] = app_details.get('release_date', {}).get('date', '')
    genres = app_details.get('genres', [])
    info['Genres'] = ', '.join([genre['description'] for genre in genres]) if genres else 'Unknown'
    developers = app_details.get('developers', [])
    info['Developers'] = ', '.join(developers) if developers else 'Unknown'
    publishers = app_details.get('publishers', [])
    info['Publishers'] = ', '.join(publishers) if publishers else 'Unknown'
    info['Short Description'] = app_details.get('short_description', '').replace('\n', ' ').replace('\r', ' ')

    # Additional Metrics
    platforms = app_details.get('platforms', {})
    info['Platforms'] = []
    if platforms.get('windows'):
        info['Platforms'].append('Windows')
    if platforms.get('mac'):
        info['Platforms'].append('macOS')
    if platforms.get('linux'):
        info['Platforms'].append('Linux')
    info['Platforms'] = ', '.join(info['Platforms']) if info['Platforms'] else 'Unknown'

    # Achievements (if available)
    achievements = app_details.get('achievements', {})
    if achievements:
        total_achievements = achievements.get('total', 0)
        locked_achievements = achievements.get('locked', 0)
        info['Achievements'] = f"{total_achievements} Total, {locked_achievements} Locked"
    else:
        info['Achievements'] = 'N/A'

    # Current Price
    price_overview = app_details.get('price_overview', {})
    if price_overview:
        price = price_overview.get('final', 0) / 100  # Price is in cents
        currency = price_overview.get('currency', 'USD')
        discount = price_overview.get('discount_percent', 0)
        info['Current Price'] = f"{price} {currency} ({discount}% off)" if discount > 0 else f"{price} {currency}"
    else:
        info['Current Price'] = 'Free' if app_details.get('is_free', False) else 'N/A'

    # Sentiment Analysis Placeholder
    info['Review Sentiment'] = analyze_reviews(info['Name'])

    return info


def analyze_reviews(game_name):
    """
    Placeholder function to analyze user reviews.
    Steam API does not provide direct access to user reviews. This requires web scraping or using third-party APIs.
    For this implementation, we'll mock the sentiment analysis.
    """
    # Mock data as Steam API access to reviews is limited
    # In a real-world scenario, consider using the Steam Reviews API or web scraping with respect to Steam's terms of service
    sample_reviews = [
        "Great game! Had a lot of fun.",
        "Not bad, but could use more content.",
        "Terrible experience. Lots of bugs.",
        "Loved the storyline and gameplay.",
        "Average game. Nothing special."
    ]
    sentiments = [sentiment_analyzer.polarity_scores(review)['compound'] for review in sample_reviews]
    average_sentiment = np.mean(sentiments)
    if average_sentiment >= 0.05:
        sentiment = "Positive"
    elif average_sentiment <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment


def calculate_metrics(df):
    """
    Calculate various metrics for the dashboard.
    """
    metrics = {}
    metrics['Total Games'] = df.shape[0]
    metrics['Total Playtime (Hours)'] = round(df['Playtime (minutes)'].sum() / 60, 2)
    metrics['Average Playtime per Game (Hours)'] = round((df['Playtime (minutes)'].sum() / 60) / df.shape[0], 2)
    # Most Played Genre
    genre_series = df['Genres'].str.split(', ').explode()
    metrics['Most Played Genre'] = genre_series.mode().values[0] if not genre_series.empty else 'N/A'
    # Top Developer
    developer_series = df['Developers'].str.split(', ').explode()
    metrics['Top Developer'] = developer_series.mode().values[0] if not developer_series.empty else 'N/A'
    # Top Publisher
    publisher_series = df['Publishers'].str.split(', ').explode()
    metrics['Top Publisher'] = publisher_series.mode().values[0] if not publisher_series.empty else 'N/A'
    # Platform Distribution
    platforms = df['Platforms'].str.split(', ').explode()
    platform_counts = platforms.value_counts().to_dict()
    metrics['Platform Distribution'] = platform_counts
    return metrics


def plot_playtime_distribution(df):
    """
    Plot the distribution of playtime across games.
    """
    df['Playtime (hours)'] = df['Playtime (minutes)'] / 60
    fig = px.histogram(df, x='Playtime (hours)', nbins=50, title='Playtime Distribution (Hours)',
                       labels={'Playtime (hours)': 'Playtime (Hours)'})
    return fig


def plot_genre_distribution(df):
    """
    Plot the distribution of genres.
    """
    genre_counts = df['Genres'].str.split(', ').explode().value_counts().reset_index()
    genre_counts.columns = ['Genre', 'Count']
    fig = px.bar(genre_counts, x='Genre', y='Count', title='Games per Genre',
                 labels={'Count': 'Number of Games', 'Genre': 'Genre'})
    return fig


def plot_playtime_over_time(df):
    """
    Plot total playtime over the years.
    """
    df_time = df.dropna(subset=['Release Date']).copy()
    df_time['Year'] = pd.to_datetime(df_time['Release Date'], errors='coerce').dt.year
    playtime_per_year = df_time.groupby('Year')['Playtime (minutes)'].sum().reset_index()
    playtime_per_year['Playtime (hours)'] = playtime_per_year['Playtime (minutes)'] / 60
    fig = px.line(playtime_per_year, x='Year', y='Playtime (hours)', title='Total Playtime Over Years',
                  labels={'Playtime (hours)': 'Total Playtime (Hours)', 'Year': 'Release Year'})
    return fig


def plot_platform_distribution(df):
    """
    Plot the distribution of platforms.
    """
    platform_counts = df['Platforms'].str.split(', ').explode().value_counts().reset_index()
    platform_counts.columns = ['Platform', 'Count']
    fig = px.pie(platform_counts, names='Platform', values='Count', title='Platform Distribution',
                 hole=0.3)
    return fig


def prepare_recommendation_model(df):
    """
    Prepare the TF-IDF matrix and cosine similarity for recommendations.
    """
    tfidf = TfidfVectorizer(stop_words='english')
    df['combined_features'] = df['Genres'] + ' ' + df['Developers'] + ' ' + df['Publishers'] + ' ' + df[
        'Short Description']
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['Name']).drop_duplicates()
    return cosine_sim, indices


def get_recommendations(df, game_name, cosine_sim, indices):
    """
    Get game recommendations based on cosine similarity.
    """
    if game_name not in indices:
        return pd.DataFrame()

    idx = indices[game_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get top 10 to filter later
    sim_scores = sim_scores[1:11]
    game_indices = [i[0] for i in sim_scores]
    recommended_games = df.iloc[game_indices][['Name', 'Genres', 'Playtime (minutes)', 'Review Sentiment']]

    # Filter recommendations with positive or neutral sentiment
    recommended_games = recommended_games[recommended_games['Review Sentiment'].isin(['Positive', 'Neutral'])]

    # Limit to top 5
    return recommended_games.head(5)


def track_playtime_history(df, steam_id):
    """
    Track historical playtime data by saving current playtime to a JSON file.
    """
    historical_path = os.path.join(HISTORICAL_DIR, f"{steam_id}_history.json")
    current_playtime = df[['Name', 'Playtime (minutes)']].set_index('Name').to_dict()['Playtime (minutes)']

    if os.path.exists(historical_path):
        with open(historical_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
    else:
        history = {}

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    history[timestamp] = current_playtime
    with open(historical_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

    return history


def plot_playtime_trends(history):
    """
    Plot playtime trends over time.
    """
    if not history:
        st.write("No historical data available.")
        return

    df_history = pd.DataFrame(history).T  # Transpose to have timestamps as rows
    df_history.index = pd.to_datetime(df_history.index)
    df_sum = df_history.sum(axis=1).reset_index()
    df_sum.columns = ['Timestamp', 'Total Playtime (minutes)']
    df_sum['Total Playtime (hours)'] = df_sum['Total Playtime (minutes)'] / 60
    fig = px.line(df_sum, x='Timestamp', y='Total Playtime (hours)', title='Total Playtime Over Time',
                  labels={'Timestamp': 'Timestamp', 'Total Playtime (hours)': 'Total Playtime (Hours)'})
    st.plotly_chart(fig, use_container_width=True)


def export_visualizations(fig, filename):
    """
    Export Plotly figures as PNG images.
    """
    fig.write_image(filename)
    st.success(f"Visualization exported as {filename}")


def export_data(df, filename):
    """
    Export DataFrame to CSV.
    """
    df.to_csv(filename, index=False)
    st.success(f"Data exported as {filename}")


# Function to fetch app details with progress bar and estimated time
def fetch_app_details_with_progress(owned_games, cache, api_key):
    """
    Fetch detailed information for each game in the user's library,
    showing progress and estimated time to completion.
    """
    games_info = []
    total_games = len(owned_games)
    start_time = time.time()
    progress = st.progress(0)
    status_text = st.empty()

    for index, game in enumerate(owned_games, start=1):
        app_id = game.get('appid')
        progress.progress(index / total_games)

        # Calculate elapsed time and estimated time remaining
        elapsed_time = time.time() - start_time
        avg_time_per_game = elapsed_time / index
        estimated_time_remaining = avg_time_per_game * (total_games - index)

        # Update status text with estimated time remaining
        minutes, seconds = divmod(int(estimated_time_remaining), 60)
        status_text.text(
            f"Fetching data for {index}/{total_games} games... Estimated time remaining: {minutes}m {seconds}s")

        # Fetch app details
        app_details = get_app_details_with_cache(app_id, cache, api_key)
        game_info = extract_game_info(game, app_details)
        games_info.append(game_info)
        time.sleep(1.0)  # Respect rate limits

    # Clear progress and status text after completion
    progress.empty()
    status_text.empty()
    return games_info


# ---------------------------- Main Application ---------------------------- #

def main():
    st.set_page_config(page_title="Steam Library Analysis Dashboard", layout="wide")
    st.title("🎮 Steam Game Library Analysis Dashboard")

    # User Inputs
    st.sidebar.header("User Input")
    api_key = st.sidebar.text_input("Enter your Steam API Key:", type="password")
    steam_id = st.sidebar.text_input("Enter your Steam ID64:")
    refresh_data = st.sidebar.button("Fetch/Refresh Data")
    upload_history = st.sidebar.file_uploader("Upload Previous Playtime CSV for Historical Analysis", type=["csv"])

    if refresh_data:
        if not api_key or not steam_id:
            st.sidebar.error("Please enter both Steam API Key and Steam ID64.")
        else:
            with st.spinner('Fetching your Steam game library...'):
                owned_games = get_owned_games(api_key, steam_id)
                if owned_games:
                    # Initialize cache for this session
                    cache_path = os.path.join(CACHE_DIR, f"{steam_id}_cache.json")
                    cache = load_cache(cache_path)
                    games_info = fetch_app_details_with_progress(owned_games, cache, api_key)

                    # Save cache
                    save_cache(cache, cache_path)

                    # Create DataFrame and store in session state
                    df = pd.DataFrame(games_info)
                    st.session_state['df'] = df

                    # Track playtime history
                    history = track_playtime_history(df, steam_id)
                    st.session_state['history'] = history
                    st.success("Data fetched and processed successfully!")

    # Load cached data if available
    if 'df' not in st.session_state and api_key and steam_id:
        cache_path = os.path.join(CACHE_DIR, f"{steam_id}_cache.json")
        if os.path.exists(cache_path):
            with st.spinner('Loading cached data...'):
                cache = load_cache(cache_path)
                owned_games = get_owned_games(api_key, steam_id)
                if owned_games:
                    games_info = fetch_app_details_with_progress(owned_games, cache, api_key)

                    # Save cache
                    save_cache(cache, cache_path)

                    # Create DataFrame and store in session state
                    df = pd.DataFrame(games_info)
                    st.session_state['df'] = df

                    # Track playtime history
                    history = track_playtime_history(df, steam_id)
                    st.session_state['history'] = history
                    st.success("Cached data loaded successfully!")

    # Handle uploaded historical data
    if upload_history:
        try:
            df_uploaded = pd.read_csv(upload_history)
            st.session_state['df_uploaded'] = df_uploaded
            st.success("Historical playtime data uploaded successfully!")
        except Exception as e:
            st.error(f"Error uploading file: {e}")

    # If data is available, proceed with analysis
    if 'df' in st.session_state:
        df = st.session_state['df']

        # Sidebar Filters
        st.sidebar.header("Filter Options")

        # Genre Filter
        genres = df['Genres'].str.split(', ').explode().unique()
        selected_genres = st.sidebar.multiselect("Select Genres", options=sorted(genres), default=sorted(genres))

        # Playtime Filter
        min_playtime = float(df['Playtime (minutes)'].min() / 60)
        max_playtime = float(df['Playtime (minutes)'].max() / 60)
        playtime_range = st.sidebar.slider(
            "Select Playtime Range (Hours)",
            min_value=0.0,
            max_value=round(max_playtime + 10, 1),
            value=(0.0, round(max_playtime + 10, 1))
        )

        # Release Year Filter
        df['Release Year'] = pd.to_datetime(df['Release Date'], errors='coerce').dt.year
        min_year = int(df['Release Year'].min())
        max_year = int(df['Release Year'].max())
        release_year_range = st.sidebar.slider(
            "Select Release Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )

        # Apply Filters
        filtered_df = df[
            df['Genres'].str.contains('|'.join(selected_genres)) &
            (df['Playtime (minutes)'] / 60 >= playtime_range[0]) &
            (df['Playtime (minutes)'] / 60 <= playtime_range[1]) &
            (df['Release Year'] >= release_year_range[0]) &
            (df['Release Year'] <= release_year_range[1])
            ]

        # Calculate Metrics
        metrics = calculate_metrics(filtered_df)

        # Display Metrics
        st.markdown("### 📊 Key Metrics")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Total Games", metrics['Total Games'])
        col2.metric("Total Playtime (Hours)", metrics['Total Playtime (Hours)'])
        col3.metric("Avg Playtime/Game (Hours)", metrics['Average Playtime per Game (Hours)'])
        col4.metric("Most Played Genre", metrics['Most Played Genre'])
        col5.metric("Top Developer", metrics['Top Developer'])
        col6.metric("Top Publisher", metrics['Top Publisher'])

        # Visualizations
        st.markdown("### 📈 Visualizations")
        col7, col8, col9, col10 = st.columns(4)

        with col7:
            st.plotly_chart(plot_genre_distribution(filtered_df), use_container_width=True)

        with col8:
            st.plotly_chart(plot_playtime_distribution(filtered_df), use_container_width=True)

        with col9:
            st.plotly_chart(plot_playtime_over_time(filtered_df), use_container_width=True)

        with col10:
            st.plotly_chart(plot_platform_distribution(filtered_df), use_container_width=True)

        # Playtime Trends Over Time
        st.markdown("### 📉 Playtime Trends Over Time")
        if 'history' in st.session_state:
            plot_playtime_trends(st.session_state['history'])
        else:
            st.write("No historical data available.")

        # Game Recommendations
        st.markdown("### 🎮 Game Recommendations")

        # Prepare Recommendation Model
        cosine_sim, indices = prepare_recommendation_model(filtered_df)

        # Recommendation Input
        selected_game = st.selectbox("Select a Game to Get Recommendations", options=filtered_df['Name'])

        if selected_game:
            recommendations = get_recommendations(filtered_df, selected_game, cosine_sim, indices)
            if not recommendations.empty:
                st.write("**Recommended Games:**")
                st.table(recommendations)
            else:
                st.write("No recommendations found.")

        # Games Table
        st.markdown("### 🕹️ Games Table")
        st.dataframe(filtered_df[['Name', 'Genres', 'Playtime (minutes)', 'Release Date', 'Developers', 'Publishers',
                                  'Platforms', 'Achievements', 'Review Sentiment']])

        # Search Functionality
        st.markdown("### 🔍 Search Games")
        search_query = st.text_input("Search for a Game by Name")
        if search_query:
            search_results = filtered_df[filtered_df['Name'].str.contains(search_query, case=False, na=False)]
            st.dataframe(search_results[
                             ['Name', 'Genres', 'Playtime (minutes)', 'Release Date', 'Developers', 'Publishers',
                              'Platforms', 'Achievements', 'Review Sentiment']])

        # Additional Metrics or Visualizations
        st.markdown("### 📊 Additional Metrics")
        col11, col12 = st.columns(2)

        with col11:
            # Achievement Analysis (if available)
            achievement_data = filtered_df['Achievements'].dropna()
            if not achievement_data.empty:
                achievement_counts = achievement_data.value_counts().head(10).reset_index()
                achievement_counts.columns = ['Achievement Status', 'Count']
                fig_achievements = px.bar(achievement_counts, x='Achievement Status', y='Count',
                                          title='Achievement Status Distribution',
                                          labels={'Count': 'Number of Games', 'Achievement Status': 'Status'})
                st.plotly_chart(fig_achievements, use_container_width=True)
            else:
                st.write("No achievement data available.")

        with col12:
            # Sentiment Analysis of Reviews
            sentiment_counts = df['Review Sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            fig_sentiment = px.pie(sentiment_counts, names='Sentiment', values='Count',
                                   title='User Review Sentiment Distribution',
                                   hole=0.3)
            st.plotly_chart(fig_sentiment, use_container_width=True)

        # Export Options
        st.markdown("### 📥 Export Insights")
        col13, col14 = st.columns(2)

        with col13:
            if st.button("Export Data as CSV"):
                export_data(filtered_df, "filtered_steam_library.csv")

        with col14:
            if st.button("Export Genre Distribution Chart"):
                fig = plot_genre_distribution(filtered_df)
                export_visualizations(fig, "genre_distribution.png")

        # About Section
        st.markdown("### 📝 About")
        st.info(
            "This enhanced dashboard provides an interactive analysis of your Steam game library, including key metrics, visualizations, and personalized game recommendations based on your playing habits. Additional features include sentiment analysis of user reviews, playtime trend tracking, and export options for your data and visualizations. You can manually refresh your data by entering your Steam API Key and Steam ID64 in the sidebar and clicking 'Fetch/Refresh Data'.")


# ---------------------------- Run the Application ---------------------------- #

if __name__ == "__main__":
    main()