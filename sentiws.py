import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from collections import defaultdict


# Funktion zum Laden der SentiWS-Daten, einschließlich aller Flexionen
def load_sentiws(sentiws_pos,sentiws_neg):
    sentiws = defaultdict(float)
    def split_flex(flexions):
        if flexions:
            return flexions.split(",")
        else:
            pass

    with open(sentiws_neg, 'r', encoding='utf-8') as f:
        for line in f:
            # Zerlege jede Zeile in Basiswort und Rest (POS, Polarität und Flexionen)
            parts = line.strip().split('|')
            base_word = parts[0]
            # Zerlege Rest in POS, Polarität und Flexionen
            pol_flex = parts[1].strip().split("\t")
            polarity = float(pol_flex[1])
            # Flexionen durch Komma getrennt

            # Füge Basiswort und alle Flexionen mit derselben Polarität ins Lexikon ein
            sentiws[base_word] = polarity
            if len(pol_flex) == 3:
                flexions = split_flex(pol_flex[2])
                for flexion in flexions:
                    sentiws[flexion] = polarity
    with open(sentiws_pos, 'r', encoding='utf-8') as f:
        for line in f:
            # Zerlege jede Zeile in Basiswort und Rest (POS, Polarität und Flexionen)
            parts = line.strip().split('|')
            base_word = parts[0]
            # Zerlege Rest in POS, Polarität und Flexionen
            pol_flex = parts[1].strip().split("\t")
            polarity = float(pol_flex[1])
            # Flexionen durch Komma getrennt

            # Füge Basiswort und alle Flexionen mit derselben Polarität ins Lexikon ein
            sentiws[base_word] = polarity
            if len(pol_flex) == 3:
                flexions = split_flex(pol_flex[2])
                for flexion in flexions:
                    sentiws[flexion] = polarity


    return sentiws


# Funktion zur Bestimmung des Sentiments eines Satzes
def get_sentiment(sentence, sentiws_lexicon):
    tokens = word_tokenize(sentence.lower())  # Tokenisiere den Satz und wandle ihn in Kleinbuchstaben um
    sentiment_score = 0
    for token in tokens:
        if token in sentiws_lexicon:  # Wenn das Token (Basiswort oder Flexion) im SentiWS-Lexikon ist
            sentiment_score += sentiws_lexicon[token]

    # Klassifiziere das Gesamtsentiment des Satzes
    if sentiment_score > 0:
        return "positiv"
    elif sentiment_score < 0:
        return "negativ"
    else:
        return "neutral"


# Hauptfunktion zur Verarbeitung einer Liste von Sätzen
def analyze_sentiments(sentences, sentiws_pos,sentiws_neg):
    sentiws_lexicon = load_sentiws(sentiws_pos,sentiws_neg)  # Lade das SentiWS-Lexikon
    sentiments = []

    for sentence in sentences:
        sentiment = get_sentiment(sentence, sentiws_lexicon)
        sentiments.append(sentiment)

    return sentiments

def summarize_sent(df,column_name):
    pos = 0
    neg = 0
    neutral = 0
    for row in df.itertuples():
        column_value = getattr(row, column_name)
        if "negativ" in column_value:
            c = column_value.count("negativ")
            neg += c
        if "positiv" in column_value:
            c = column_value.count("positiv")
            pos += c
        if "neutral" in column_value:
            c = row.m_senti.count("neutral")
            neutral += c
    return {"positiv": pos, "neutral":neutral, "negativ": neg}

def sentis(results):

    # Konvertiere die Eingabe in einen DataFrame
    r = pd.DataFrame(results)
    # Definiere eine Funktion zur Sentiment-Bewertung
    def predict_sentiment(text):
        if text:  # Check if text is not empty
            return analyze_sentiments(text,"SentiWS_v2.0_Positive.txt","SentiWS_v2.0_Negative.txt")
        return "X"  # Return "X" if text is empty
    # Wende die Funktion auf die entsprechenden Spalten an
    r['m_senti'] = r['m_sent'].apply(predict_sentiment)
    r['fm_senti'] = r['fm_sent'].apply(predict_sentiment)
    return r

def num_weight(df,col):
    def weigh(cell):
        if cell == "negativ":
            return -1
        if cell == "positiv":
            return 1
        else:
            return 1
    return df[col].apply(weigh)


if __name__ == "__main__":
    pass

