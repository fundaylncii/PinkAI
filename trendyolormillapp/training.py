
import os
import torch
import optuna
import shutil
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset
from google.colab import files 
import preprocessing
import torch.nn as nn
from transformers import AutoConfig
from huggingface_hub import notebook_login

def model_tuning(modelname, texts, scores, savemodel=False, savemodeltext=None, downloadmodel=False, trials=5, hugpush=False):
    """
    Modeli verilen verilerle fine-tune eder, en iyi hiperparametreleri belirler ve kaydeder.
    """
    os.environ["WANDB_DISABLED"] = "true"
    
    model = AutoModelForSequenceClassification.from_pretrained(modelname, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(modelname)

    encodings = preprocessing.preprocess_with_ngrams(texts, tokenizer)
    labels = torch.tensor(scores, dtype=torch.long)

    class CustomDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
            self.bigrams = encodings["bigram_features"]
            self.trigrams = encodings["trigram_features"]
    
        def __len__(self):
            return len(self.labels)
    
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items() if key not in ["bigram_features", "trigram_features"]}
            item["bigrams"] = torch.tensor(self.bigrams[idx])  # Bigramları ekle
            item["trigrams"] = torch.tensor(self.trigrams[idx])  # Trigramları ekle
            item["labels"] = torch.tensor(self.labels[idx])
            return item


    train_dataset = CustomDataset(encodings, labels)

    def objective(trial):
        """
        Optuna ile en iyi hiperparametreleri bulmak için kullanılan fonksiyon.
        """
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
        )

        trainer.train()
        return trainer.state.log_history[-1]["train_loss"]

    # **Optuna ile en iyi hiperparametreleri bulma**
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trials)
    best_params = study.best_params
    print("Best hyperparameters:", best_params)

    # **En iyi hiperparametrelerle modeli yeniden eğitme**
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=best_params["learning_rate"],
        per_device_train_batch_size=best_params["batch_size"],
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
    )

    trainer.train()

    # **Modeli kaydetme**
    if savemodel:
        if not savemodeltext:
            raise ValueError("savemodeltext must be provided if savemodel is True.")
        save_path = f"/content/{savemodeltext}"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    # **Modeli indirme**
    if downloadmodel:
        if not savemodeltext:
            raise ValueError("savemodeltext must be provided if downloadmodel is True.")
        shutil.make_archive(savemodeltext, 'zip', save_path)
        save_zip = f"/content/{savemodeltext}.zip"
        files.download(save_zip)

    # Hugging Face'e modeli yükleme (Eğer giriş başarılıysa)
    if hugpush:
        try:
            notebook_login()
            user_info = whoami()  # Kullanıcı giriş kontrolü
            if user_info:
                model.push_to_hub(hf_model_name)
                tokenizer.push_to_hub(hf_model_name)
                print(f"Model başarıyla Hugging Face'e yüklendi: https://huggingface.co/{hf_model_name}")
            else:
                print("Hugging Face girişi başarısız, modeli yükleyemiyoruz.")
        except Exception as e:
            print(f"Hugging Face'e yükleme başarısız: {e}")


    return model, tokenizer, best_params
