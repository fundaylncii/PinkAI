
## githup dosyalarını clonelama
!git clone https://github.com/fundaylncii/PinkAI.git
%cd PinkAI

import sys
sys.path.append("/content/PinkAI/trendyolormillapp")

!pip install dicttoxml optuna datasets nbimporter cloudscraper
import data_fetching
import preprocessing
import training
import evaluation
import pandas as pd

## githup eklenen dosyaları güncelleme 
%cd /content/PinkAI
!git pull origin main

## dosyaları reload etme
import importlib
importlib.reload(data_fetching)
importlib.reload(preprocessing)
importlib.reload(training)
importlib.reload(evaluation)

## python table ayarları
pd.set_option("display.width", None)
pd.set_option("display.max_columns", 500)
pd.set_option('display.max_colwidth', None)

# Ürün yorumlarını çek
product_url = input("Ürün URL'sini girin: ")
df = data_fetching.get_reviews(product_url,channel_no=2)

# Feature Engineering uygula
df["sent_score"] = df["rate"].apply(preprocessing.feature_engineering)
df["comment_cleaned"] = preprocessing.clean_text(df["comment"])

## Boş veya tamamen stopwords olan yorumları temizle
df = df[df["comment_cleaned"].notna()]  
df = df[df["comment_cleaned"].str.strip() != ""]  
df = df[df["comment_cleaned"].str.split().str.len() > 1]  

df.head(2)


## huggingface token girişi
from huggingface_hub import notebook_login
notebook_login()

"""
## başlangıç modelimiz de roberta base alınacak veri setimiz ile eğitip kendi modelimizi oluşturacağız
model_name = "xlm-roberta-base"
model, tokenizer, best_params = training.model_tuning(
    modelname=model_name,
    texts=df["comment_cleaned"].tolist(),
    scores=df["sent_score"].tolist(),
    savemodel=True,
    savemodeltext="best_model",
    trials=5
)

## Model Dosya içeriğinin kontrol edilmesi
import os

model_dir = "/content/best_model"
print(os.listdir(model_dir))

## huggingface authentication
from huggingface_hub import notebook_login
notebook_login()

## model huggingface ilk yükleme
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import HfApi

hf_model_name = "fundaylnci/TurkReviewSentiment-RoBERTa"

model = AutoModelForSequenceClassification.from_pretrained("/content/best_model", use_safetensors=True)
tokenizer = AutoTokenizer.from_pretrained("/content/best_model")

model.push_to_hub(hf_model_name)
tokenizer.push_to_hub(hf_model_name)

print(f"Model başarıyla Hugging Face'e yüklendi: https://huggingface.co/{hf_model_name}")

"""

## oluşturduğumuz modeli yeni veriler fine-tunning etme
model_name = "fundaylnci/TurkReviewSentiment-RoBERTa"
model, tokenizer, best_params = training.model_tuning(
    modelname=model_name,
    texts=df["comment_cleaned"].tolist(),
    scores=df["sent_score"].tolist(),
    trials=5,
    hugpush=True
)



## hugging face de modlein versiyonunu görüntüleme
from huggingface_hub import HfApi

api = HfApi()
repo_id = "fundaylnci/TurkReviewSentiment-RoBERTa"

# Mevcut tüm versiyonları listele
versions = api.list_repo_refs(repo_id)
for version in versions.branches:
    print(version)

## 2edae8274bb40d48e074120755c50770b2130e2a
## 67380dbb9948feabf264082471646c2e8df4b3b3

## huggingface geçmiş işlem kayıtlarını görüntüleme
from huggingface_hub import HfApi

api = HfApi()
repo_id = "fundaylnci/TurkReviewSentiment-RoBERTa"  # Kendi model adını kullan

# Modelin commit geçmişini al
commits = api.list_repo_commits(repo_id)

print("Modelde yapılan güncellemeler:")
for commit in commits:
    print(f"Tarih: {commit.created_at} | Hash: {commit.commit_id} | Mesaj: {commit.title}")


## Yeni eğitilen model ile tahminleme
model_path = "fundaylnci/TurkReviewSentiment-RoBERTa"
model, tokenizer = evaluation.load_model(model_path)

## Dataset Tahminleme
df = evaluation.evaluate_model(df, model, tokenizer)

## yorum duygu dağılımı
df["sent_score"].value_counts()

## Örnek Text ile deneme
text ="İç gösteriyor maalesef iade"
evaluation.get_sentiment(text,model, tokenizer)

## Model negatif değerleri bile pozitif algılıyor veri seti dengelimi ? Hayır
print(df["sent_score"].value_counts())  # Pozitif ve negatif yorum sayısını kontrol et

## modelin ngramları nasıl öğrendiğini analiz etme
from transformers import pipeline

classifier = pipeline("text-classification", model="fundaylnci/TurkReviewSentiment-RoBERTa")

print(classifier("berbat bir ürün"))
print(classifier("çok kötü bir ürün, sakın almayın!"))
print(classifier("bu ürüne bayıldım, harika!"))

## yorumların score ları 1 e daha veri setinde olumlu - olumsuz veriler dengesiz data set arttırılarak model yeniden eğitilmeli.
