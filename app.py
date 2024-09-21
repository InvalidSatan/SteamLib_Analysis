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

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define NLTK data directory for Heroku
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")

if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.data.path.append(nltk_data_dir)

# Download vader_lexicon if not already present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', download_dir=nltk_data_dir)

# Initialize Sentiment Analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Cache directories
CACHE_DIR = "cache"
HISTORICAL_DIR = "historical_data"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
if not os.path.exists(HISTORICAL_DIR):
    os.makedirs(HISTORICAL_DIR)


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


def get_app_details(app_id, max_retries=5, request_delay=1.0):
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


def get_app_details_with_cache(app_id, cache, max_retries=5, request_delay=1.0):
    """
    Fetch app details using cache to minimize API calls.
    """
    if str(app_id) in cache:
        return cache[str(app_id)]

    app_details = get_app_details(app_id, max_retries, request_delay)
    cache[str(app_id)] = app_details
    return app_details


def extract_game_info(game, app_details):
    """
    Extracts relevant information from the game and its app details, including tags.
    """
    info = {}
    info['App ID'] = game.get('appid', '')
    info['Name'] = game.get('name', '')
    info['Playtime (minutes)'] = game.get('playtime_forever', 0)
    info['Playtime (2 weeks)'] = game.get('playtime_2weeks', 0)
    info['Playtime Last Session (minutes)'] = game.get('playtime_windows_forever', 0)

    # From app details
    info['Release Date'] = app_details.get('release_date', {}).get('date', '')
    tags = app_details.get('categories', [])
    info['Tags'] = ', '.join([tag['description'] for tag in tags]) if tags else 'Unknown'

    # Sanitize Developer Names
    developers = app_details.get('developers', [])
    sanitized_developers = [dev.replace("Ltd.", "").replace("Inc.", "").replace("LLC", "").strip() for dev in
                            developers]
    info['Developers'] = ', '.join(sanitized_developers) if sanitized_developers else 'Unknown'

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
    Calculate enhanced metrics that provide deeper insights into the user's gaming habits.
    """
    metrics = {}
    metrics['Total Games'] = df.shape[0]
    metrics['Total Playtime (Hours)'] = round(df['Playtime (minutes)'].sum() / 60, 2)

    # Top Played Tags
    tag_series = df['Tags'].str.split(', ').explode()
    tag_playtimes = df.explode('Tags').groupby('Tags')['Playtime (minutes)'].sum().sort_values(ascending=False)
    metrics['Top Played Tags'] = tag_playtimes.head(5).index.tolist() if not tag_playtimes.empty else ['N/A']

    # Playtime Concentration (Top 5 Games)
    top_5_playtime = df.nlargest(5, 'Playtime (minutes)')['Playtime (minutes)'].sum()
    total_playtime = df['Playtime (minutes)'].sum()
    metrics['Top 5 Playtime Concentration (%)'] = round((top_5_playtime / total_playtime) * 100,
                                                        2) if total_playtime > 0 else 0.0

    # Playtime Diversity (Variety Index)
    tag_counts = tag_series.value_counts(normalize=True)
    variety_index = -np.sum(tag_counts * np.log(tag_counts))  # Entropy-based diversity index
    metrics['Variety Index'] = round(variety_index, 2) if not np.isnan(variety_index) else 0.0

    # Average Playtime per Game Session
    metrics['Avg Playtime per Game Session (Minutes)'] = round(df['Playtime (minutes)'].mean(), 2) if not df[
        'Playtime (minutes)'].empty else 0.0

    return metrics


def plot_playtime_distribution(df):
    """
    Plot the distribution of playtime across games.
    """
    df['Playtime (hours)'] = df['Playtime (minutes)'] / 60
    fig = px.histogram(df, x='Playtime (hours)', nbins=50, title='Playtime Distribution (Hours)',
                       labels={'Playtime (hours)': 'Playtime (Hours)'})
    fig.update_layout(template="plotly_dark")
    return fig


def plot_genre_distribution(df):
    """
    Plot the distribution of genres based on tags.
    """
    tag_counts = df['Tags'].str.split(', ').explode().value_counts().reset_index()
    tag_counts.columns = ['Tag', 'Count']
    fig = px.bar(tag_counts, x='Tag', y='Count', title='Games per Tag',
                 labels={'Count': 'Number of Games', 'Tag': 'Tag'})
    fig.update_layout(template="plotly_dark")
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
    fig.update_layout(template="plotly_dark")
    return fig


def plot_platform_distribution(df):
    """
    Plot the distribution of platforms.
    """
    platform_counts = df['Platforms'].str.split(', ').explode().value_counts().reset_index()
    platform_counts.columns = ['Platform', 'Count']
    fig = px.pie(platform_counts, names='Platform', values='Count', title='Platform Distribution',
                 hole=0.3)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(template="plotly_dark")
    return fig


def prepare_recommendation_model(df):
    """
    Prepare the TF-IDF matrix and cosine similarity for recommendations.
    """
    tfidf = TfidfVectorizer(stop_words='english')
    df['combined_features'] = df['Tags'] + ' ' + df['Developers'] + ' ' + df['Publishers'] + ' ' + df[
        'Short Description']
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['Name']).drop_duplicates()
    return cosine_sim, indices


def get_tag_based_recommendations(df, top_tags):
    """
    Generate recommendations based on the user's top-played tags and engagement patterns.
    """
    # Filter games by top tags, prioritizing those with positive sentiment
    recommended_games = df[df['Tags'].apply(lambda x: any(tag in x for tag in top_tags))]
    recommended_games = recommended_games[recommended_games['Review Sentiment'] == 'Positive']

    # Sort by playtime and engagement (recent playtime or session frequency could be included)
    recommended_games = recommended_games.sort_values(by=['Playtime (minutes)'], ascending=False)

    # Return the top 5 recommendations based on tag match and playtime
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
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


def export_visualizations(fig, filename):
    """
    Export Plotly figures as PNG images.
    """
    try:
        fig.write_image(filename)
        st.success(f"Visualization exported as {filename}")
    except Exception as e:
        st.error(f"Error exporting visualization: {e}")


def export_data(df, filename):
    """
    Export DataFrame to CSV.
    """
    try:
        df.to_csv(filename, index=False)
        st.success(f"Data exported as {filename}")
    except Exception as e:
        st.error(f"Error exporting data: {e}")


# Function to fetch app details with progress bar and estimated time
def fetch_app_details_with_progress(owned_games, cache):
    """
    Fetch detailed information for each game in the user's library,
    showing progress and estimated time to completion.
    """
    api_key = st.session_state.get('api_key')
    if not api_key:
        st.error("API Key not found in session state.")
        return []

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
        app_details = get_app_details_with_cache(app_id, cache)
        game_info = extract_game_info(game, app_details)
        games_info.append(game_info)
        time.sleep(1.0)  # Respect rate limits

    # Clear progress and status text after completion
    progress.empty()
    status_text.empty()
    return games_info


# ---------------------------- Main Application ---------------------------- #

def main():
    # Apply custom CSS for Warhammer 40k Nurgle theme
    st.markdown(
        """
        <style>
        body {
            background-color: #1a1a1a;
            color: #d4af37;
        }
        .css-18e3th9 {
            background-color: #262626;
        }
        .css-1aumxhk {
            background-color: #262626;
        }
        .stButton>button {
            background-color: #6b8e23;
            color: white;
        }
        .stTextInput>div>div>input {
            background-color: #333333;
            color: #d4af37;
        }
        .stDataFrame div {
            color: #d4af37;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Set page configuration
    st.set_page_config(page_title="Steam Library Analysis Dashboard", layout="wide", page_icon="🎮")

    # Title and Header Image
    st.markdown(
        """
        <h1 style='text-align: center; color: #d4af37;'>🎮 Steam Game Library Analysis Dashboard</h1>
        """,
        unsafe_allow_html=True
    )

    # Optional: Add a themed banner image
    st.image(
        "https://i.imgur.com/8YqUQmX.png",  # Replace with a Warhammer 40k Nurgle-themed image URL
        use_column_width=True
    )

    # User Inputs
    st.sidebar.header("User Input")
    api_key = st.sidebar.text_input("Enter your Steam API Key:", type="password")
    steam_id = st.sidebar.text_input("Enter your Steam ID64:")
    refresh_data = st.sidebar.button("Fetch/Refresh Data")
    upload_history = st.sidebar.file_uploader("Upload Previous Playtime CSV for Historical Analysis", type=["csv"])

    # Store API key in session state for use in functions
    if api_key:
        st.session_state['api_key'] = api_key

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
                    games_info = fetch_app_details_with_progress(owned_games, cache)

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
                    games_info = fetch_app_details_with_progress(owned_games, cache)

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

        # Tag Filter
        tags = df['Tags'].str.split(', ').explode().unique()
        selected_tags = st.sidebar.multiselect("Select Tags", options=sorted(tags), default=sorted(tags))

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
        min_year = int(df['Release Year'].min()) if not pd.isna(df['Release Year'].min()) else 2000
        max_year = int(df['Release Year'].max()) if not pd.isna(df['Release Year'].max()) else 2025
        release_year_range = st.sidebar.slider(
            "Select Release Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )

        # Apply Filters
        filtered_df = df[
            df['Tags'].str.contains('|'.join(selected_tags)) &
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
        col3.metric("Top 5 Playtime (%)", f"{metrics['Top 5 Playtime Concentration (%)']}%")
        col4.metric("Variety Index", metrics['Variety Index'])
        col5.metric("Avg Playtime/Session (Min)", metrics['Avg Playtime per Game Session (Minutes)'])
        col6.metric("Top Played Tags", ', '.join(metrics['Top Played Tags']))

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

        # Recommendation based on Top Played Tags
        top_tags = metrics['Top Played Tags']
        recommendations = get_tag_based_recommendations(filtered_df, top_tags)

        if not recommendations.empty:
            st.table(recommendations[['Name', 'Tags', 'Playtime (minutes)', 'Review Sentiment']])
        else:
            st.write("No recommendations found based on your top tags.")

        # Games Table
        st.markdown("### 🕹️ Games Table")
        st.dataframe(filtered_df[
                         ['Name', 'Tags', 'Playtime (minutes)', 'Release Date', 'Developers', 'Publishers', 'Platforms',
                          'Achievements', 'Review Sentiment']])

        # Search Functionality
        st.markdown("### 🔍 Search Games")
        search_query = st.text_input("Search for a Game by Name")
        if search_query:
            search_results = filtered_df[filtered_df['Name'].str.contains(search_query, case=False, na=False)]
            st.dataframe(search_results[
                             ['Name', 'Tags', 'Playtime (minutes)', 'Release Date', 'Developers', 'Publishers',
                              'Platforms', 'Achievements', 'Review Sentiment']])

        # Additional Metrics or Visualizations
        st.markdown("### 📊 Additional Metrics")
        col11, col12 = st.columns(2)

        with col11:
            # Achievement Analysis (if available)
            achievement_data = filtered_df['Achievements'].dropna()
            if not achievement_data.empty and achievement_data.iloc[0] != 'N/A':
                achievement_counts = achievement_data.value_counts().head(10).reset_index()
                achievement_counts.columns = ['Achievement Status', 'Count']
                fig_achievements = px.bar(achievement_counts, x='Achievement Status', y='Count',
                                          title='Achievement Status Distribution',
                                          labels={'Count': 'Number of Games', 'Achievement Status': 'Status'})
                fig_achievements.update_layout(template="plotly_dark")
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
            fig_sentiment.update_traces(textposition='inside', textinfo='percent+label')
            fig_sentiment.update_layout(template="plotly_dark")
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