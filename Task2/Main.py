import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "wordcloud"])

from wordcloud import WordCloud

# Download resources
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Load data
df = pd.read_csv(r"C:\Users\VIP.DESKTOP-F6DB1GS\Desktop\BrainwaveMatrixSolutions\Task2\Fake_iPhone16_Tweets_10000.csv")

# Generate random dates for demonstration
df["Date"] = pd.to_datetime(
    np.random.choice(
        pd.date_range("2024-09-01", "2025-06-01", freq="D"),
        size=len(df)
    )
)

# Preprocess text
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = re.sub(r"http\S+|@\S+|#\S+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = text.lower()
    return " ".join([w for w in text.split() if w not in stop_words])

df["Clean_Tweet"] = df["Tweet"].apply(clean_text)

# Sentiment analysis
sia = SentimentIntensityAnalyzer()
df["Sentiment_Score"] = df["Clean_Tweet"].apply(lambda x: sia.polarity_scores(x)["compound"])
df["Sentiment"] = df["Sentiment_Score"].apply(
    lambda s: "Positive" if s >= 0.05 else ("Negative" if s <= -0.05 else "Neutral")
)

# Visualization 1: Sentiment Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Sentiment", hue="Sentiment", palette="Set2", legend=False)
plt.title("Sentiment Distribution")
plt.xlabel("")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Visualization 2: Weekly Sentiment Trend
df.set_index("Date", inplace=True)
weekly = df.resample("W")["Sentiment_Score"].mean()
plt.figure(figsize=(8, 4))
weekly.plot(marker='o')
plt.axhline(0, linestyle='--', color='gray')
plt.grid(True)
plt.title("Weekly Average Sentiment Trend")
plt.xlabel("Week")
plt.ylabel("Avg Sentiment Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Prepare data for word cloud
from collections import defaultdict
word_stats = defaultdict(lambda: {"count": 0, "total": 0})
for _, row in df.iterrows():
    for w in row["Clean_Tweet"].split():
        word_stats[w]["count"] += 1
        word_stats[w]["total"] += row["Sentiment_Score"]

# Build frequency and sentiment maps
freq = {w: stats["count"] for w, stats in word_stats.items() if stats["count"] > 1}
avg_sent = {w: stats["total"]/stats["count"] for w, stats in word_stats.items() if stats["count"] > 1}

# Color function
def color_func(word, **kwargs):
    s = avg_sent.get(word, 0)
    if s >= 0.05:
        return "green"
    elif s <= -0.05:
        return "red"
    else:
        return "gray"

# Visualization 3: Word Cloud
wc = WordCloud(width=800, height=400, background_color="white")
wc.generate_from_frequencies(freq)
plt.figure(figsize=(10, 5))
plt.imshow(wc.recolor(color_func=color_func), interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud Colored by Sentiment", fontsize=16)
plt.tight_layout()
plt.show()
