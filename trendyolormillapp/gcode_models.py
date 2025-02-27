
## !pip install dicttoxml optuna datasets nbimporter
import data_fetching
import preprocessing
import training
import evaluation
import pandas as pd

"""
importlib.reload(data_fetching)
importlib.reload(preprocessing)
importlib.reload(training)
importlib.reload(evaluation)
"""

pd.set_option("display.width", None)
pd.set_option("display.max_columns", 500)
pd.set_option('display.max_colwidth', None)

# Ürün yorumlarını çek
product_url = input("Ürün URL'sini girin: ")
df = data_fetching.get_reviews(product_url)

# Feature Engineering uygula
df["sent_score"] = df["rate"].apply(preprocessing.feature_engineering)
df["comment_cleaned"] = preprocessing.clean_text(df["comment"])

## Boş veya tamamen stopwords olan yorumları temizle
df = df[df["comment_cleaned"].notna()]  # NaN olanları temizle
df = df[df["comment_cleaned"].str.strip() != ""]  # Boş string olanları temizle
df = df[df["comment_cleaned"].str.split().str.len() > 1]  # Tek kelimelik yorumları kaldır

# N-gram özelliklerini ekle (eğer yorum boş değilse)
if not df.empty:
    df["bigrams"] = df["comment_cleaned"].apply(lambda x: preprocessing.generate_ngrams([x], n=2) if len(x.split()) > 1 else [])
    df["trigrams"] = df["comment_cleaned"].apply(lambda x: preprocessing.generate_ngrams([x], n=3) if len(x.split()) > 2 else [])

# Boş bir DataFrame kaldıysa hata vermemesi için uyarı ekleyelim
if df.empty:
    raise ValueError("Tüm yorumlar boş veya anlamsız. Model eğitimi için yeterli veri yok.")

## Modeli eğit ve kaydet
model_path = "/content/drive/MyDrive/NLP_Models/best_model3"

model, tokenizer, best_params = training.model_tuning(
    modelname= model_path,
    texts=df["comment_cleaned"].tolist(),
    scores=df["sent_score"].tolist(),
    savemodel=True,
    savemodeltext="best_model4",
    trials = 10
)


# Modeli yükle ve değerlendirme yap
model_path = "/content/drive/MyDrive/NLP_Models/best_model3"
model, tokenizer = evaluation.load_model(model_path)

df = evaluation.evaluate_model(df, model, tokenizer)

# Sonuçları göster
print(df[["comment", "sent_score", "sentiment"]].head(2))

## 3 den küçük puan veren ancak pozitif duygu içeren 10 yorum
df[(df["sent_score"] == 0) & (df["sentiment"] == "positive")].head(3)

## 4 den büyük puan veren ancak negatif duygu içeren 10 yorum
df[(df["sent_score"] == 1) & (df["sentiment"] == "negative")].head(3)

## yorum duygu dağılımı
df["sentiment"].value_counts()


"""
text ="yorumlara bakarak m beden aldım baya büyük 62 kiloyum s daha rahat olacakmış m nin altı baya büyük ona göre alabilirsiniz"
evaluation.get_sentiment(text,model, tokenizer)

"""


"""
df.to_excel("/content/yorum.xlsx")

"""


"""
# FARKLI ÜRÜNDE DENEME
product_url = input("Ürün URL'sini girin: ")
df_other = data_fetching.get_reviews(product_url)

df_other = evaluation.evaluate_model(df_other, model, tokenizer)

# Sonuçları göster
print(df_other[["comment", "sentiment"]].head(2))

"""


# Olumlu ve olumsuz yorumları ayıralım
positive_comments = df[df["sentiment"] == "positive"]["comment_cleaned"].dropna()
negative_comments = df[df["sentiment"] == "negative"]["comment_cleaned"].dropna()

# Kelime frekanslarını çıkaralım
positive_words = " ".join(positive_comments).split()
negative_words = " ".join(negative_comments).split()

# En sık geçen kelimeleri belirleyelim
positive_word_counts = Counter(positive_words).most_common(10)
negative_word_counts = Counter(negative_words).most_common(10)


import matplotlib.pyplot as plt

# Olumlu kelimeleri görselleştirme
positive_words_df = pd.DataFrame(positive_word_counts, columns=["Kelime", "Frekans"])
negative_words_df = pd.DataFrame(negative_word_counts, columns=["Kelime", "Frekans"])

# Olumlu kelimeler grafiği
plt.figure(figsize=(10, 5))
plt.barh(positive_words_df["Kelime"], positive_words_df["Frekans"])
plt.xlabel("Frekans")
plt.ylabel("Kelime")
plt.title("Olumlu Yorumlarda En Sık Geçen Kelimeler")
plt.gca().invert_yaxis()
plt.show()

# Olumsuz kelimeler grafiği
plt.figure(figsize=(10, 5))
plt.barh(negative_words_df["Kelime"], negative_words_df["Frekans"], color="red")
plt.xlabel("Frekans")
plt.ylabel("Kelime")
plt.title("Olumsuz Yorumlarda En Sık Geçen Kelimeler")
plt.gca().invert_yaxis()
plt.show()


## iade nedenlerinin belirlenmesi

# "iade" kelimesi geçen yorumları filtreleyelim
iade_yorumlari = df[df["comment_cleaned"].str.contains("iade", na=False)]

# "iade" içeren yorumları alalım
iade_texts = iade_yorumlari["comment_cleaned"].dropna()


# Kelime frekanslarını analiz etmek için CountVectorizer kullanalım
from sklearn.feature_extraction.text import CountVectorizer
import nltk

# İndir nltk stopwords
nltk.download('stopwords')

# Türkçe stop kelimeleri yükleyin
from nltk.corpus import stopwords
turkish_stop_words = stopwords.words('turkish')

vectorizer = CountVectorizer(stop_words=turkish_stop_words, ngram_range=(1,2), max_features=20)
X = vectorizer.fit_transform(iade_texts)


# En sık geçen kelimeleri alalım
word_freq = dict(zip(vectorizer.get_feature_names_out(), X.toarray().sum(axis=0)))


# Kelime frekanslarını sıralayalım
sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)


# Sonuçları gösterelim
iade_nedenleri_df = pd.DataFrame(sorted_word_freq, columns=["Kelime", "Frekans"])
iade_nedenleri_df[:3]
