import re
from nltk.sentiment import SentimentIntensityAnalyzer

# initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

DISASTER_KEYWORDS = [
    "flood","flooding",
    "heavy rain","heavy rainfall",
    "cyclone","storm",
    "overflow","waterlogging",
    "dam break","landslide",
    "evacuation","rescue","damage"
]

def keyword_boost(text):
    score = 0
    text = text.lower()

    for word in DISASTER_KEYWORDS:
        if word in text:
            score += 0.12

    return min(score,0.6)


def sentiment_score(text):
    s = sia.polarity_scores(text)["compound"]

    # convert sentiment to disaster risk
    if s < -0.5:
        return 0.7
    elif s < -0.2:
        return 0.5
    elif s < 0:
        return 0.3
    else:
        return 0.1


def predict_text_risk(text):

    base = sentiment_score(text)
    boost = keyword_boost(text)

    risk = min(1.0, base + boost)

    return float(risk)