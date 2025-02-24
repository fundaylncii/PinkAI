
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re

nltk.download("stopwords")
sw = stopwords.words("turkish")

def clean_text(text):
    """
    Yorum metinlerini temizler ve ön işler:
    - Küçük harfe çevirme
    - Noktalama işaretlerini kaldırma
    - Sayıları kaldırma
    - Stopwords temizleme
    - Nadir kelimeleri kaldırma
    - En çok kullanılan ilk 5 kelimenin analiz edilmesi
    """
    # Küçük harfe çevir
    text = text.str.lower()
    # Noktalama ve özel karakterleri temizle
    text = text.str.replace(r"[^\w\s]", "", regex=True)
    # Sayıları kaldır
    text = text.str.replace(r"\d+", "", regex=True)
    # Stopwords kaldır
    text = text.apply(lambda x: " ".join(word for word in x.split() if word not in sw))

    # Nadir kelimeleri kaldırma
    word_freq = text.apply(lambda x: pd.Series(x.split())).stack().value_counts()
    rare_words = word_freq[word_freq < 5].index
    text = text.apply(lambda x: " ".join(word for word in x.split() if word not in rare_words))

    # En çok kullanılan kelimeleri göster
    top_words = word_freq.head(5)
    print("En çok kullanılan kelimeler:")
    print(top_words)

    return text

def generate_ngrams(texts, n=2):
    """
    Verilen metinlerden n-gram özellikleri oluşturur.
    """
    if not texts or all(t.strip() == "" for t in texts):
        return []  # Boş veya sadece stopwords içeren metinler için boş liste dön
    
    vectorizer = CountVectorizer(ngram_range=(n, n))
    try:
        ngram_matrix = vectorizer.fit_transform(texts)
        return vectorizer.get_feature_names_out()
    except ValueError:
        return []  # Eğer hata oluşursa (boş kelime listesi) yine boş dön

def preprocess_with_ngrams(texts, tokenizer):
    """
    BERT tokenizasyonu uygular ve n-gram özelliklerini ekler.
    """
    encodings = tokenizer(list(texts), max_length=128, truncation=True, padding="max_length")

    # N-gram özelliklerini çıkar
    bigrams = generate_ngrams(texts, n=2)
    trigrams = generate_ngrams(texts, n=3)

    encodings["bigram_features"] = bigrams
    encodings["trigram_features"] = trigrams

    return encodings

def feature_engineering(rate):
    """
    Puanı 4 ve üzeri olan yorumlara 1 (pozitif), diğerlerine 0 (negatif) atar.
    """
    return 1 if rate >= 4 else 0
