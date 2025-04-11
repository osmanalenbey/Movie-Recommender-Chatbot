import streamlit as st
import openai
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

from difflib import get_close_matches

import random

import requests
from PIL import Image, ImageEnhance
import io

import shutil
import os
# Get the user's home directory dynamically
home_dir = os.path.expanduser("~")  # This will return C:\Users\YourUsername on Windows

# Define source and destination paths
source = os.path.join(os.getcwd(), "kaggle.json")  # Adjust this based on your actual file location
destination_dir = os.path.join(home_dir, ".kaggle")
destination = os.path.join(destination_dir, "kaggle.json")

# Create the .kaggle directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Move the file
shutil.copy(source, destination)

import kaggle



# Streamlit Page Config
st.set_page_config(page_title="Movie Recommender Chatbot", page_icon="🎬")
st.title("Movie Recommender Chatbot")




def create_movie_dataframe(dataset='artificial'):
    if dataset=='artificial':
        # Sample data for the movies.csv file
        data = {
            "movie_id": range(1, 11),
            "title": ["Movie A", "Movie B", "Movie C", "Movie D", "Movie E", "Movie F", "Movie G", "Movie H", "Movie I", "Movie J"],
            "genres": ["Action", "Comedy", "Drama", "Sci-Fi", "Horror", "Thriller", "Romance", "Documentary", "Animation", "Fantasy"],
            "ratings": [7.8, 6.5, 8.2, 7.0, 5.4, 6.9, 7.5, 8.0, 6.3, 7.1]
        }

        # Create a DataFrame
        df = pd.DataFrame(data)

        print("Artificial movie data generated successfully!")

    elif dataset=='imdb_top_1000':

        # Define dataset name and output directory
        dataset_name = "harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows"
        output_dir = "imdb_top_1000"

        # Download and extract
        kaggle.api.dataset_download_files(dataset_name, path=output_dir, unzip=True)

        print("Download complete. Files extracted to:", output_dir)

        df = pd.read_csv("imdb_top_1000/imdb_top_1000.csv")
        df.rename(columns={"IMDB_Rating": "ratings", "Series_Title": "title", "Genre": "genres"}, inplace=True)

    return df

# movies = create_movie_dataframe(dataset='artificial')
movies = create_movie_dataframe(dataset='imdb_top_1000')

# --- Build enhanced feature matrix ---

movies['genres'] = movies['genres'].str.split(', ')
movies['Director'] = movies['Director'].fillna('')
movies['Star1'] = movies['Star1'].fillna('')
movies['Released_Year'] = movies.get('Released_Year', pd.Series([2000]*len(movies)))  # Dummy fallback

# Ensure the columns are numeric and drop rows with invalid data
movies['ratings'] = pd.to_numeric(movies['ratings'], errors='coerce')
movies['Released_Year'] = pd.to_numeric(movies['Released_Year'], errors='coerce')
movies.dropna(subset=['ratings', 'Released_Year'], inplace=True)



# Encode genres
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(movies['genres'])

# Encode director
director_vec = CountVectorizer()
director_encoded = director_vec.fit_transform(movies['Director'])

# Encode top star
star_vec = CountVectorizer()
star_encoded = star_vec.fit_transform(movies['Star1'])

# Normalize rating and year
scaler = MinMaxScaler()
scaled_numeric = scaler.fit_transform(movies[['ratings', 'Released_Year']])

final_feature_matrix = hstack([genre_encoded, director_encoded, star_encoded, scaled_numeric])
final_feature_matrix = final_feature_matrix.tocsr()  # <-- Add this line



# Movie Recommendation Function
def recommend_movies(movie_title, num_recommendations=5):
    idx = movies[movies['title'].str.lower() == movie_title.lower()].index
    if len(idx) == 0:
        return ["Movie not found in catalog."]
    
    idx = idx[0]
    sim_scores = cosine_similarity(final_feature_matrix[idx], final_feature_matrix).flatten()
    sim_indices = sim_scores.argsort()[::-1][1:num_recommendations+1]
    recommended = movies.iloc[sim_indices]
    return recommended[['title', 'ratings']].values.tolist()

mood_to_genre = {
    "tired": ["Comedy", "Family", "Animation"],
    "overwhelmed": ["Drama", "Romance"],
    "motivated": ["Biography", "Sport"],
    "happy": ["Comedy", "Romance"],
    "sad": ["Feel-Good", "Drama", "Family"]
}

def recommend_for_mood(mood, num_recommendations=5):
    genres = mood_to_genre.get(mood.lower(), [])
    filtered = movies[movies['genres'].apply(lambda g: any(genre in g for genre in genres))]
    if filtered.empty:
        return ["No suitable movies found."]
    top = filtered.sort_values(by='ratings', ascending=False).head(num_recommendations)
    return top[['title', 'ratings']].values.tolist()

def find_closest_title(user_input, titles, cutoff=0.7):
    user_input = user_input.lower()
    title_matches = get_close_matches(user_input, [t.lower() for t in titles], n=1, cutoff=cutoff)
    if title_matches:
        return title_matches[0]
    return None

# Define LangChain Tool
def get_movie_recommendation(movie_name):
    return recommend_movies(movie_name, num_recommendations=5)


tool_movie_similarity = Tool(
    name="Movie Similarity Recommender",
    func=recommend_movies,
    description="Use this to recommend similar movies when the user provides a movie name. Input should be a single movie title."
)

tool_mood_based = Tool(
    name="Mood Recommender",
    func=recommend_for_mood,
    description="Use this to recommend movies based on the user's current mood. Input should be a mood like 'tired', 'happy', 'sad', etc."
)


# Initialize LangChain Chatbot
llm = ChatOpenAI(max_tokens=100, temperature=0.8) # temperature 0.7-1.0 is creative and engaging (good for conversational chat) without going total silly randomness

system_message = (
    "You are a friendly, intelligent movie-savvy chatbot that engages in natural conversations about movies and general topics. "
    "Your goal is to keep the conversation smooth, helpful, and enjoyable.\n\n"

    "You have access to two tools:\n"
    "1. Movie Similarity Recommender — for suggesting similar movies based on a title.\n"
    "2. Mood Recommender — for suggesting movies based on the user's emotional state or mood (e.g., tired, happy, overwhelmed).\n\n"

    "Use these tools **only** when the user clearly or implicitly asks for movie recommendations.\n"
    "- If a user explicitly asks for a recommendation (e.g., 'Can you recommend a movie?'), use the appropriate tool.\n"
    "- If the user casually mentions a movie (e.g., 'What about Shrek?') or describes a mood (e.g., 'I’m tired'), decide whether a recommendation is implied. "
    "If you're unsure, ask a clarifying question first.\n"
    "- Avoid using tools for generic uses of words like 'suggest' or 'recommend' unless it's obviously about movies. For example, if the user asks, 'Can you suggest a food recipe?', respond normally and **do not** call any tools.\n\n"

    "You may comment on movies, actors, or genres to keep the conversation interesting. Do not jump directly into recommendations without proper context. "
    "Maintain a friendly, conversational tone and avoid overwhelming the user with lists unless asked for."
)

agent = initialize_agent(
    tools=[tool_movie_similarity, tool_mood_based],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"system_message": system_message}
)


# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
prompt = st.chat_input("Ask for a movie recommendation")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)


    client = openai.OpenAI()  # This uses your environment variable OPENAI_API_KEY

    def detect_mood(prompt):
        system = "You are a mood classifier. Based on the user's message, output the user's current emotional state as a single word like 'happy', 'sad', 'tired', 'romantic', 'anxious', etc. Do not explain."
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        mood = response.choices[0].message.content.strip().lower()
        return mood


    def is_recommendation_request(prompt):
        intent_prompt = f"""You are an assistant that classifies user messages.

                            Determine whether the following message is asking for a movie recommendation:

                            "{prompt}"

                            Respond with only "yes" or "no"."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if needed
            messages=[
                {"role": "system", "content": "You are a helpful intent classifier."},
                {"role": "user", "content": intent_prompt}
            ],
            temperature=0
        )

        reply = response.choices[0].message.content.strip().lower()
        return reply == "yes"
          
    def format_recommendations(movie_list, label, is_mood=False, fuzzy_matched=False):
        if isinstance(movie_list, str) or "not found" in str(movie_list[0]).lower():
            return f"🎬 Sorry, I couldn't find anything for **{label}**. Want to try another movie or tell me what mood you're in?"

        movie_list = movie_list[:5]
        movies = [f"*{title}*" for title, _ in movie_list]

        # Intro templates
        mood_intros = [
            f"If you're in a **{label}** mood, you might enjoy ",
            f"Feeling **{label}**? Then you might like ",
            f"To match that **{label}** vibe, how about ",
            f"When you're feeling **{label}**, these might be great picks: "
        ]

        title_intros = [
            f"Since you mentioned **{label.title()}**, I'd recommend ",
            f"If you liked **{label.title()}**, you may also enjoy ",
            f"Based on **{label.title()}**, here are a few you might love: ",
            f"Thinking of **{label.title()}**? These could be similar in spirit: "
        ]

        intro = random.choice(mood_intros if is_mood else title_intros)

        # Body
        if len(movies) == 1:
            body = f"{movies[0]}."
        elif len(movies) == 2:
            body = f"{movies[0]} or {movies[1]}."
        else:
            body = ", ".join(movies[:-1]) + f", and {movies[-1]}."

        # Outro options
        outros = [
            " Let me know if you'd like more suggestions!",
            " I can suggest more if you're interested.",
            " Feel free to ask for another kind of vibe.",
            " Let me know what you think!",
            ""
        ]
        outro = random.choice(outros)
        
        return intro + body + outro

    def display_posters(best_movie, full_catalog_df):
        st.markdown(f"**{best_movie}**")

        # Find the poster row from the full DataFrame
        print(f"full_catalog_df['title'].str.strip().str.lower() is {full_catalog_df['title'].str.strip().str.lower()}")
        print(f"best_movie.strip().lower() is {best_movie.strip().lower()}")
        matched = full_catalog_df[full_catalog_df['title'].str.strip().str.lower() == best_movie.strip().lower()]        

        if not matched.empty:
            poster_url = matched.iloc[0]['Poster_Link']
            try:
                img_data = requests.get(poster_url, timeout=4).content
                img = Image.open(io.BytesIO(img_data))
                img = ImageEnhance.Color(img).enhance(1.5)
                st.image(img, width=160)
            except:
                st.text("📷 Poster unavailable.")
        else:
            st.text("📷 Poster not found in catalog.")
                
        # for title, _ in movie_list:
        #     st.markdown(f"**{title}**")
    
        #     # Find the poster row from the full DataFrame
        #     matched = full_catalog_df[full_catalog_df['title'].str.strip().str.lower() == title.strip().lower()]
    
        #     if not matched.empty:
        #         poster_url = matched.iloc[0]['Poster_Link']
        #         try:
        #             img_data = requests.get(poster_url, timeout=4).content
        #             img = Image.open(io.BytesIO(img_data))
        #             img = ImageEnhance.Color(img).enhance(1.5)
        #             st.image(img, width=160)
        #         except:
        #             st.text("📷 Poster unavailable.")
        #     else:
        #         st.text("📷 Poster not found in catalog.")
        
    if is_recommendation_request(prompt):
        # Try fuzzy match from entire title list
        matched_title = find_closest_title(prompt, movies['title'])

        if matched_title:
            recommendations = recommend_movies(matched_title)
            response = format_recommendations(recommendations, matched_title, fuzzy_matched=True)
            display_posters(recommendations[:1], movies)

        else:
            # Use LLM to detect mood
            mood = detect_mood(prompt)
            if mood in mood_to_genre:
                recommendations = recommend_for_mood(mood)
                response = format_recommendations(recommendations, mood, is_mood=True)
                display_posters(recommendations[:1], movies)
            else:
                response = f"I'm not sure what mood you're in — could you tell me more about how you're feeling or what you're in the mood for?"
    else:
        response = llm.invoke(prompt)




    if hasattr(response, "content"):
        response = response.content

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Restart Button
if st.button("Restart Chat"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()
