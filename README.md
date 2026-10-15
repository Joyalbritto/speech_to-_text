# Comprehensive Text Preprocessing and NLP Pipeline

A Python-based project implementing foundational Natural Language Processing (NLP), text mining, regex pattern extraction, and speech-to-text integration pipelines using NLTK, spaCy, and scikit-learn.

## 📋 Table of Contents
1. [Prerequisites & Libraries](#prerequisites--libraries)
2. [Tokenization & Sentence Splitting (NLTK)](#1-tokenization--sentence-splitting-nltk)
3. [spaCy Linguistic Pipeline (POS & NER)](#2-spacy-linguistic-pipeline-pos--ner)
4. [Stop Word and Punctuation Filtering](#3-stop-word-and-punctuation-filtering)
5. [Regex Pattern Extraction](#4-regex-pattern-extraction)
6. [N-Grams Generation](#5-n-grams-generation)
7. [Document-Term Matrix Vectorization](#6-document-term-matrix-vectorization)
8. [Rule-Based Sentiment Analysis](#7-rule-based-sentiment-analysis)
9. [Stemming vs. Lemmatization](#8-stemming-vs-lemmatization)
10. [Text Statistical Analysis via Pandas](#9-text-statistical-analysis-via-pandas)
11. [Speech-to-Text & Audio NLP Pipeline](#10-speech-to-text--audio-nlp-pipeline)

---

## 🛠 Prerequisites & Libraries
Ensure the following packages are installed in your environment:
* `nltk`
* `spacy`
* `scikit-learn`
* `pandas`
* `SpeechRecognition`
* `pyaudio` / `ffmpeg`

Run the following command to download necessary NLTK and spaCy model resources:
```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords')"
