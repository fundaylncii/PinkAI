
import os
import torch
import optuna
import shutil
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset
from google.colab import files 
import preprocessing
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

def model_tuning(modelname, texts, scores, savemodel=False, savemodeltext=None, downloadmodel=False, trials=7):
    """
    Modeli verilen verilerle fine-tune eder, en iyi hiperparametreleri belirler ve kaydeder.
    """
    os.environ["WANDB_DISABLED"] = "true"
    
    class CustomBertForSequenceClassification(BertPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.bert = BertModel(config)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            
            # Bigram ve trigram eklemek için giriş katmanını genişletiyoruz
            self.classifier = nn.Linear(config.hidden_size + 2, config.num_labels)
    
            self.init_weights()
    
        def forward(self, input_ids, attention_mask=None, bigrams=None, trigrams=None, labels=None):
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            pooled_output = outputs[1]
            
            # Bigram ve trigramları birleştir
            bigram_trigram_features = torch.cat([bigrams.unsqueeze(1), trigrams.unsqueeze(1)], dim=1)
            pooled_output = torch.cat([pooled_output, bigram_trigram_features], dim=1)
    
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
    
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
            return (loss, logits) if loss is not None else logits
 
    ## model = AutoModelForSequenceClassification.from_pretrained(modelname, num_labels=2)
    model = CustomBertForSequenceClassification.from_pretrained(modelname, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(modelname)

    # **Preprocessing ve n-gramları ekleme**
    encodings = preprocessing.preprocess_with_ngrams(texts, tokenizer)
    labels = list(scores)

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

    return model, tokenizer, best_params
