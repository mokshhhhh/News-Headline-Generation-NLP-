📰 T5-Based News Headline Generator

A deep learning project that generates short, meaningful headlines from long news articles using Google’s T5 (Text-to-Text Transfer Transformer) model.
This project fine-tunes T5-Small on a merged and cleaned version of two popular news-summary datasets to produce high-quality one-line headlines.

🚀 Project Overview

This project focuses on abstractive headline generation, where the model learns to generate completely new sentences rather than extracting text from the article.

✔️ Key Features

Fine-tunes T5-Small, a transformer encoder-decoder model.

Uses news_summary.csv + news_summary_more.csv merged into a unified dataset.

Cleans and preprocesses text (lowercasing, URL removal, special-character filtering).

Implements ROUGE and SacreBLEU metrics for evaluation.

Trains using HuggingFace Seq2SeqTrainer with beam search during inference.

Saves fine-tuned model weights directly to Google Drive.

🧠 Model Architecture & Training Pipeline
🔹 Model Used: T5-Small

60M parameters

Fully text-to-text architecture (everything → text)

Perfect for summarization, translation, paraphrasing, and headline generation.

🔹 Why T5?

Handles prefix-based tasks (e.g., "summarize: <text>")

Lightweight → Faster training

High quality generation for short summaries like headlines

📦 Dataset Processing
1️⃣ Loading Two Datasets

The project merges:

news_summary.csv

news_summary_more.csv

Each dataset has different column formats, so they are:

Aligned into a uniform structure → article + summary

Cleaned using regex-based rules:

lowercasing

removing URLs

removing special characters

Filtered for minimum article/summary length

Duplicates removed

2️⃣ Conversion to HuggingFace Dataset

The cleaned dataframe is converted to a Dataset object and split into:

95% training

5% test

🔧 Training Setup

Training relies on:

HuggingFace Transformers

Seq2SeqTrainer

DataCollatorForSeq2Seq

ROUGE + SacreBLEU metrics

Training Hyperparameters
Parameter	Value
Epochs	3
Batch Size	8
Weight Decay	0.01
Warmup Steps	500
FP16	Auto-enabled on GPU
Evaluation	Every epoch
Saving	Every epoch
Best Model Metrics	eval_loss



🛠️ Installation & Setup
1️⃣ Install Dependencies
pip install transformers datasets sentencepiece accelerate -U
pip install evaluate rouge_score sacrebleu

2️⃣ Ensure Files Are in Colab Root

Upload these datasets:

news_summary.csv

news_summary_more.csv

3️⃣ Mount Google Drive (if training in Colab)
from google.colab import drive
drive.mount('/content/drive')

▶️ How to Run the Project
1. Preprocess Data

The preprocessing pipeline loads → cleans → merges → filters → splits the dataset.

2. Tokenize the Dataset

T5 requires a prefix:

summarize: <article text>

3. Train the Model

Using HuggingFace Seq2SeqTrainer.

4. Run Inference

Use the helper function:

headline = generate_headline(article_text, model, tokenizer)
print(headline)


Beam search (num_beams=4) ensures higher quality generations.

📤 Output Demo 

👉 I will update this section later with generated outputs and comparisons.

📚 Tech Stack

Python

PyTorch

HuggingFace Transformers

Datasets Library

Google Colab (GPU)

🧑‍💻 Author

Moksh Dhiman
