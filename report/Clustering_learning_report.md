# 📚 Text Clustering Learning Report: From TF-IDF to Keyphrase Extraction

## 👩‍🎓 Overview

This report documents my complete learning process, attempts, failures, insights, and results from working on a text clustering task. The goal was to extract meaningful clusters and keywords from a collection of political speech data (Trump rally transcripts).

## 🎯 Objective

* Cluster a list of political event transcripts into meaningful topics.
* Extract keywords from each cluster to understand its content.

## 🧱 Pipeline Structure

1. **Preprocessing**: Clean and prepare the text.
2. **Feature Extraction**: Convert text into numerical form (TF-IDF).
3. **Dimensionality Reduction**: Reduce dimensions (optional for clustering, required for plotting).
4. **Clustering**: Group similar texts.
5. **Keyword Extraction**: Get key terms per cluster.
6. **Named Entity Filtering**: Remove place names or overused political words.
7. **Visualization**: UMAP for visual layout of clusters.

---

## ⚙️ Attempt 1: Basic TF-IDF + HDBSCAN

### Code Summary

```python
vectorizer = TfidfVectorizer(max_features=1000)
tfidf_vectors = vectorizer.fit_transform(texts).toarray()
```

* Used TF-IDF to vectorize text.
* Used HDBSCAN for clustering.

### Result

* Clusters formed, but keywords were repetitive: mostly 'trump', 'rally', 'president', city names like 'ohio', 'charlotte', etc.

### Issue

* TF-IDF favored high-frequency words, including locations and common political terms.
* Clusters lacked conceptual themes.

---

## ⚙️ Attempt 2: Remove Common Political Words

### What I Tried

Added a custom stopword list to exclude terms like 'trump', 'rally', 'president', 'campaign', etc.

```python
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
custom_stopwords = set(['trump', 'president', 'rally', 'campaign', 'remarks'])
all_stopwords = ENGLISH_STOP_WORDS.union(custom_stopwords)
vectorizer = TfidfVectorizer(max_features=1000, stop_words=all_stopwords)
```

### Result

* Slight improvement.
* Still lots of geographic terms like 'nevada', 'toledo', 'charlotte', etc.

### Issue

* Even after removing political terms, geo-entities still dominated due to frequency.
* TF-IDF couldn't understand meaning or concept.

---

## ⚙️ Attempt 3: Named Entity Filtering with spaCy

### What I Tried

Used spaCy to extract and count named entities (GPE, ORG, PERSON), then filtered frequent ones.

```python
nlp = spacy.load("en_core_web_sm")
all_ents = []
for doc in nlp.pipe(texts, batch_size=50):
    all_ents.extend([ent.text.lower() for ent in doc.ents if ent.label_ in ['GPE', 'ORG', 'PERSON']])
ent_freq = Counter(all_ents)
common_ents = {ent for ent, count in ent_freq.items() if count > threshold}
```

### Result

* Reduced some place names from entity outputs.
* But **this didn't affect TF-IDF keywords**, which are generated independently.

### Reflection

* This helped me realize entity filtering is not connected to TF-IDF keywords.
* TF-IDF still surfaced irrelevant location terms due to document frequency.

---

## ⚙️ Attempt 4: DEC Embeddings + HDBSCAN

### What I Tried

Switched from raw TF-IDF vectors to learned embeddings using a Deep Embedded Clustering (DEC) model.

```python
class Autoencoder(Model):
    def __init__(self, input_dim, latent_dim):
        ...
    def call(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

autoencoder = Autoencoder(input_dim, latent_dim)
autoencoder.fit(tfidf_vectors, tfidf_vectors, epochs=30)
embeddings = autoencoder.encoder(tfidf_vectors).numpy()
```

### Result

* UMAP visualization showed tighter clusters.
* But keyword extraction still relied on TF-IDF from original text.

### Issue

* Improved clustering, but keyword problem remained.
* TF-IDF is too simplistic and doesn't capture meaning.

---

## 🔄 What Didn’t Work and Why

| Attempt | What Was Tried   | Why It Failed or Didn't Help                                         |
| ------- | ---------------- | -------------------------------------------------------------------- |
| 1       | TF-IDF only      | Too reliant on frequency, surface-level words                        |
| 2       | Custom stopwords | Still kept geo terms because they’re frequent and domain-specific    |
| 3       | Entity filtering | Doesn’t affect TF-IDF keywords (only affects separate entity counts) |
| 4       | DEC embeddings   | Better clustering but same keyword limitations                       |

---

## 🔍 Why TF-IDF Fails for Our Case

* It treats all words as independent tokens.
* It has **no understanding of context, semantics, or topic structure**.
* High-frequency terms across clusters dominate.
* Location names dominate in rally texts because they’re specific to each event.

---

## 🌱 What to Try Next: Semantic Keyphrase Extraction

### Better Methods (Non-TF-IDF):

* **YAKE**: Statistical, unsupervised keyphrase extraction.
* **RAKE**: Uses co-occurrence and phrase boundaries.
* **TextRank**: Graph-based, like PageRank but for text.
* **TopicRank, MultipartiteRank**: Rank phrases based on topical structure.
* **KeyBERT**: Uses BERT embeddings to find most representative phrases.

### Where They Fit in the Pipeline:

* Replace or augment TF-IDF in the **feature extraction** step.
* Or used *after* clustering to summarize the conceptual meaning of clusters.

---

## 🧭 Conclusion and Next Steps

* The clustering process is sound, but **keyword extraction** is the weak link.
* TF-IDF is not semantically rich enough for meaningful topic labeling.
* Need to replace TF-IDF with **semantic keyphrase extractors**.

### Next:

* Try **YAKE**, **KeyBERT**, or **TextRank with dependency parsing**.
* Test whether new keyphrases better capture the "concept" of each cluster.
* Consider training embeddings directly from text using **Sentence-BERT** for clustering.

---

## 🧾 Appendix: Useful Code Blocks

### spaCy Named Entity Filtering

```python
nlp = spacy.load("en_core_web_sm")
all_ents = []
for doc in nlp.pipe(texts, batch_size=50):
    all_ents.extend([ent.text.lower() for ent in doc.ents if ent.label_ in ['GPE', 'ORG', 'PERSON']])
```

### Custom Stopwords in TF-IDF

```python
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
custom_stopwords = set(['trump', 'rally', 'president'])
all_stopwords = ENGLISH_STOP_WORDS.union(custom_stopwords)
tfidf = TfidfVectorizer(stop_words=all_stopwords)
```

---

*This report is a snapshot of an active learning process.*
**Failures are not signs of incompetence—they’re road signs toward mastery.**
