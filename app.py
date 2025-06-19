import streamlit as st
import praw
import joblib
import matplotlib.pyplot as plt

# Load your model and vectorizer
model = joblib.load("sentiment_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Reddit API setup
reddit = praw.Reddit(
    client_id='Erkqst5twPmRMxKZ8IF2jQ',
    client_secret='AgQ8DdJHtJP_FI4F38NQoQm0kUV-gg',
    user_agent='sentiment-analysis-script'
)

# Function to get sentiment data using your model
def get_sentiments(subreddits):
    sentiment_data = []
    overall = {"Positive": 0, "Neutral": 0, "Negative": 0}
    line_data = []
    all_sentiments = []

    for sub in subreddits:
        subreddit = reddit.subreddit(sub)
        comments = []
        try:
            for submission in subreddit.hot(limit=3):
                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list()[:50]:
                    comments.append(comment.body)
        except Exception as e:
            st.warning(f"Failed to fetch from r/{sub}: {e}")
            continue

        pos = neu = neg = 0
        labeled_comments = []

        for c in comments:
            try:
                features = vectorizer.transform([c])
                pred = model.predict(features)[0]
            except Exception as e:
                continue  # Skip if model/vectorizer throws an error

            if pred == 1:
                sentiment = "Positive"; pos += 1; overall["Positive"] += 1
            elif pred == 0:
                sentiment = "Neutral"; neu += 1; overall["Neutral"] += 1
            else:
                sentiment = "Negative"; neg += 1; overall["Negative"] += 1

            labeled_comments.append({
                "text": c.strip()[:150],
                "sentiment": sentiment
            })
            all_sentiments.append(sentiment)

        sentiment_data.append({
            "subreddit": sub,
            "positive": pos,
            "neutral": neu,
            "negative": neg,
            "comments": labeled_comments
        })

        line_data.append((sub, pos, neu, neg))

    return sentiment_data, overall, line_data, all_sentiments

# Plotting functions
def plot_sentiment_path(all_sentiments):
    y_map = {"Positive": 2, "Neutral": 1, "Negative": 0}
    color_map = {"Positive": "green", "Neutral": "gray", "Negative": "red"}

    y_values = [y_map[s] for s in all_sentiments]
    colors = [color_map[s] for s in all_sentiments]
    x_values = list(range(1, len(all_sentiments) + 1))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_values, y_values, color='black', linewidth=1.5, zorder=1)
    for x, y, color in zip(x_values, y_values, colors):
        ax.scatter(x, y, color=color, s=100, edgecolor='black', zorder=2)

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Negative", "Neutral", "Positive"])
    ax.set_xlabel("Sentiment Index")
    ax.set_title("Sentiment Flow Across All Comments")
    ax.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig)

def plot_overall_sentiment(overall):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(overall.keys(), overall.values(), color=['green', 'gray', 'red'])
    ax.set_title("Overall Sentiment Distribution")
    st.pyplot(fig)

# Streamlit UI
st.set_page_config(page_title="Reddit Sentiment Analyzer", layout="wide")
st.title("🔍 Reddit Sentiment Analyzer")

sub_input = st.text_input("Enter subreddit names (comma-separated)", "UTAustin,LonghornNation")

if st.button("Analyze"):
    subreddits = [s.strip() for s in sub_input.split(",") if s.strip()]
    sentiment_data, overall, line_data, all_sentiments = get_sentiments(subreddits)

    col1, col2 = st.columns([1.2, 1.8])

    with col1:
        for sub in sentiment_data:
            st.subheader(f"r/{sub['subreddit']}")
            for c in sub["comments"]:
                color = {"Positive": "green", "Neutral": "gray", "Negative": "red"}[c["sentiment"]]
                st.markdown(f"""
                    <div style='background:#f9f9f9;padding:10px;border-radius:6px;margin-bottom:8px'>
                        {c["text"]}<br>
                        <strong style='color:{color}'>Sentiment: {c["sentiment"]}</strong>
                    </div>
                """, unsafe_allow_html=True)

    with col2:
        st.subheader("📈 Sentiment Path")
        plot_sentiment_path(all_sentiments)

        st.subheader("📊 Overall Sentiment Bar Chart")
        plot_overall_sentiment(overall)