import streamlit as st
import openai
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType

#import kaggle


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
        import kaggle

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

# Vectorize Genres
tfidf = TfidfVectorizer(stop_words='english')
genre_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# Movie Recommendation Function
def recommend_movies(movie_title, num_recommendations=3):
    idx = movies[movies['title'].str.lower() == movie_title.lower()].index
    if len(idx) == 0:
        return ["Movie not found in catalog."]
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]
    recommended_movies = movies.iloc[movie_indices]
    return recommended_movies[["title", "ratings"]].values.tolist()

# Define LangChain Tool
def get_movie_recommendation(movie_name):
    return recommend_movies(movie_name, num_recommendations=5)

recommendation_tool = Tool(
    name="Movie Recommender",
    func=get_movie_recommendation,
    description="Get movie recommendations based on a movie name."
)

# Initialize LangChain Chatbot
llm = ChatOpenAI(max_tokens=100, temperature=0.8) # temperature 0.7-1.0 is creative and engaging (good for conversational chat) without going total silly randomness

system_message = (
    f"You are a friendly and engaging chatbot that can chat about movies and general topics. "
    f"You should respond naturally to greetings, small talk, and general movie discussions. "
    f"ONLY provide movie recommendations when explicitly asked (e.g., 'Can you recommend a movie?'). "
    f"Do NOT assume every user message is about recommendations. If unclear, ask for clarification."
)


agent = initialize_agent(
    tools=[recommendation_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"system_message": system_message}  # Adding the system message
)

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback_shown" not in st.session_state:
    st.session_state.feedback_shown = False

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

    # Check if the user is explicitly asking for a recommendation
    recommendation_keywords = ["recommend", "suggest", "give me a movie", "movie like"]
    is_recommendation_request = any(keyword in prompt.lower() for keyword in recommendation_keywords)

    if is_recommendation_request:
        response = agent.run(prompt)  # Use the recommendation agent
    else:
        response = llm.invoke(prompt)  # Use OpenAI chat model for normal conversation

    if hasattr(response, "content"):
        response = response.content

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Feedback Button
if not st.session_state.feedback_shown and len(st.session_state.messages) > 1:
    if st.button("Get Feedback"):
        st.session_state.feedback_shown = True
        conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
        feedback_response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Give a score (1-10) and feedback on the quality of the recommendations."},
                {"role": "user", "content": f"Evaluate this interaction: {conversation_history}"}
            ]
        )
        st.write(feedback_response["choices"][0]["message"]["content"])

# Restart Button
if st.button("Restart Chat"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()