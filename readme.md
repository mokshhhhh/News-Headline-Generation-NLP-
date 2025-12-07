# 📰 T5-Based News Headline Generator

A deep learning project that generates short, meaningful headlines from long news articles using **Google’s T5 (Text-to-Text Transfer Transformer)** model.  
This project fine-tunes **T5-Small** on a merged and cleaned version of two popular news-summary datasets to produce high-quality one-line headlines.

---

## 🚀 Project Overview

This project focuses on **abstractive headline generation**, where the model learns to generate completely new sentences rather than just extracting parts of the article.

---

## ✔️ Key Features

<ul>
  <li>Fine-tunes <b>T5-Small</b>, a transformer encoder–decoder model.</li>
  <li>Uses <code>news_summary.csv</code> and <code>news_summary_more.csv</code> merged into a unified dataset.</li>
  <li>Cleans and preprocesses text:
    <ul>
      <li>Lowercasing</li>
      <li>URL removal</li>
      <li>Special-character filtering</li>
    </ul>
  </li>
  <li>Implements <b>ROUGE</b> and <b>SacreBLEU</b> metrics for evaluation.</li>
  <li>Trains using HuggingFace <code>Seq2SeqTrainer</code> with beam search during inference.</li>
  <li>Saves fine-tuned model weights directly to <b>Google Drive</b>.</li>
</ul>

---

## 🧠 Model Architecture & Training Pipeline

### 🔹 Model Used: T5-Small

- ~60M parameters  
- Fully **text-to-text** architecture (everything → text)  
- Suitable for:
  - Summarization  
  - Translation  
  - Paraphrasing  
  - Headline generation  

### 🔹 Why T5?

- Handles **prefix-based** tasks (e.g., `summarize: <text>`).
- Lightweight → **Faster training** compared to larger variants.
- Produces **high-quality short summaries**, ideal for news headlines.

---

## 📦 Dataset Processing

### 1️⃣ Loading Two Datasets

The project merges:

- `news_summary.csv`
- `news_summary_more.csv`

Each dataset has different column formats, so they are:

<ul>
  <li>Aligned into a uniform structure → <b>article</b> + <b>summary</b></li>
  <li>Cleaned using regex-based rules:
    <ul>
      <li>Lowercasing</li>
      <li>Removing URLs</li>
      <li>Removing special characters</li>
    </ul>
  </li>
  <li>Filtered for minimum article/summary length</li>
  <li>Duplicates removed</li>
</ul>

### 2️⃣ Conversion to HuggingFace Dataset

The cleaned dataframe is converted to a HuggingFace `Dataset` object and split into:

- **95%** → training set  
- **5%** → test set  

---

## 🔧 Training Setup

Training relies on:

- **HuggingFace Transformers**
- **Seq2SeqTrainer**
- **DataCollatorForSeq2Seq**
- **ROUGE** + **SacreBLEU** metrics

### 🧪 Training Hyperparameters

| Parameter          | Value              |
|--------------------|--------------------|
| Epochs             | 3                  |
| Batch Size         | 8                  |
| Weight Decay       | 0.01               |
| Warmup Steps       | 500                |
| FP16               | Auto-enabled on GPU|
| Evaluation         | Every epoch        |
| Saving             | Every epoch        |
| Best Model Metric  | `eval_loss`        |

---

## 🛠️ Installation & Setup

### 1️⃣ Install Dependencies

```bash
pip install transformers datasets sentencepiece accelerate -U
pip install evaluate rouge_score sacrebleu
```

### 2️⃣ Ensure Files Are in Colab Root

Upload the following datasets to your Colab working directory:

<ul>
  <li><code>news_summary.csv</code></li>
  <li><code>news_summary_more.csv</code></li>
</ul>

---

### 3️⃣ Mount Google Drive (if training inside Google Colab)

```python
from google.colab import drive
drive.mount('/content/drive')

```

<h3>🔹 2️⃣ Ensure Files Are in Colab Root</h3>

<p>Upload the following datasets to your Colab working directory:</p>

<ul>
  <li><code>news_summary.csv</code></li>
  <li><code>news_summary_more.csv</code></li>
</ul>

<hr/>

<h3>🔹 3️⃣ Mount Google Drive (if training inside Google Colab)</h3>

```python
from google.colab import drive
drive.mount('/content/drive')

```

<h2>▶️ <b>How to Run the Project</b></h2> <br/> <h3><b>1️⃣ Preprocess the Data</b></h3> <p>The preprocessing pipeline performs:</p> <ul> <li>Loading both dataset files</li> <li>Cleaning text (lowercasing, removing URLs, removing unwanted characters)</li> <li>Merging both datasets into a unified structure</li> <li>Filtering out short, empty, or duplicate samples</li> <li>Splitting into <b>train</b> (95%) and <b>test</b> (5%) sets</li> </ul> <br/> <h3><b>2️⃣ Tokenize the Dataset</b></h3> <p>T5 uses a prefix-based input format:</p>
summarize: <article text>

<p>Tokenization handles:</p> <ul> <li>Truncating long articles</li> <li>Padding to fixed sequence lengths</li> <li>Preparing target summaries with label masking</li> </ul> <br/> <h3><b>3️⃣ Train the Model</b></h3> <p>Training is handled by HuggingFace’s <code>Seq2SeqTrainer</code>:</p> <ul> <li>Evaluates ROUGE & SacreBLEU metrics</li> <li>FP16 training enabled when GPU supports it</li> <li>Automatically saves best model checkpoints</li> <li>Saves the final trained model to Google Drive</li> </ul> <br/> <h3><b>4️⃣ Run Inference</b></h3> <p>Use the helper function to generate headlines:</p>

headline = generate_headline(article_text, model, tokenizer)
print(headline)


<p>This uses <b>beam search (num_beams = 4)</b> to generate higher-quality results.</p> <hr/> <h2>📤 <b>Output Demo</b></h2> <p>👉 This section will be updated soon with model-generated headlines, comparisons, and screenshots.</p> <hr/> <h2>📚 <b>Tech Stack</b></h2> <ul> <li><b>Python</b></li> <li><b>PyTorch</b></li> <li><b>HuggingFace Transformers</b></li> <li><b>Datasets Library</b></li> <li><b>Google Colab (GPU)</b></li> </ul> <hr/> <h2>🧑‍💻 <b>Author</b></h2> <p><b>Moksh Dhiman</b></p>
