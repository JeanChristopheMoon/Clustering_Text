1. Introduction
This project aims to cluster political speech texts by meaningful topics, avoiding clustering bias driven by location names or boilerplate phrases. The dataset contains political speech excerpts, often dominated by location mentions (e.g., city names, states), which distorts unsupervised clustering results.

2. Problem Definition
Challenge: Text clusters are primarily driven by location names, overwhelming thematic content.

Goal: Obtain topic-driven clusters free from location dominance, for better semantic grouping of speeches.

Subproblems:

Boilerplate phrase detection and removal

Deduplication of near-duplicate texts

Named entity recognition and filtering

Embedding and clustering of textual data

Keyword extraction to interpret clusters

Visualization of clusters for interpretability

3. Methodology
3.1 Data Loading and Deduplication
Initially, political speech texts were loaded from TSV files. Deduplication was performed by:

Vectorizing texts with TF-IDF (1 to 3-grams).

Computing cosine similarity matrix.

Removing texts with similarity above 0.9 to clean duplicates.

Code snippet:

python
Copy
Edit
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load TSV
df = pd.read_csv("rawtext.tsv", sep='\t')
texts = df.iloc[:, 0].dropna().astype(str).tolist()

# Deduplication
vectorizer = TfidfVectorizer(ngram_range=(1,3))
tfidf_matrix = vectorizer.fit_transform(texts)
similarity_matrix = cosine_similarity(tfidf_matrix)

threshold = 0.9
duplicates = []
for i in range(len(similarity_matrix)):
    for j in range(i+1, len(similarity_matrix)):
        if similarity_matrix[i][j] > threshold:
            duplicates.append(j)

unique_texts = [text for i, text in enumerate(texts) if i not in duplicates]
3.2 Boilerplate Phrase Detection
To remove repeated boilerplate phrases that skew clustering (e.g., "President Trump delivers remarks"), frequent n-grams were extracted via CountVectorizer with min_df set to appear in a certain fraction of documents.

Code snippet:

python
Copy
Edit
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

texts_for_ngram = unique_texts

vectorizer = CountVectorizer(ngram_range=(3,5), stop_words='english', min_df=0.1)
X = vectorizer.fit_transform(texts_for_ngram)
ngrams = vectorizer.get_feature_names_out()
freqs = np.asarray(X.sum(axis=0)).flatten()

sorted_ngrams = sorted(zip(ngrams, freqs), key=lambda x: x[1], reverse=True)

print("Top frequent n-grams (possible boilerplate):")
for phrase, freq in sorted_ngrams[:20]:
    print(f"{phrase} — appears in {freq} documents")
These boilerplate phrases were manually or semi-automatically filtered out from texts.

3.3 Named Entity Extraction and Filtering
Using spaCy's NER model, entities of types GPE (locations), ORG, PERSON were extracted to identify and later filter common entities that dominated text clusters.

A threshold was applied: entities occurring in more than 30% of texts were considered "common" and filtered out in later steps.

Code snippet:

python
Copy
Edit
import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")
all_ents = []

for doc in nlp.pipe(unique_texts, batch_size=50):
    all_ents.extend([ent.text.lower() for ent in doc.ents if ent.label_ in ['GPE', 'ORG', 'PERSON']])

ent_freq = Counter(all_ents)
threshold = 0.3 * len(unique_texts)
common_ents = {ent for ent, count in ent_freq.items() if count > threshold}
print(f"Common entities to filter out ({len(common_ents)}): {common_ents}")
3.4 Embedding and Clustering
Sentence-BERT (all-MiniLM-L6-v2) was used to embed texts into semantic vectors.

Embeddings were normalized.

Hierarchical Agglomerative Clustering was applied with a distance threshold to form clusters.

Code snippet:

python
Copy
Edit
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(unique_texts, show_progress_bar=True)
normalized_embeddings = normalize(embeddings)

clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5)
labels = clusterer.fit_predict(normalized_embeddings)
3.5 Keyword Extraction
Two approaches were tried for cluster keyword extraction:

TF-IDF based: extract top keywords per cluster based on term frequency.

KeyBERT (BERT-based keyword extraction): used a pre-trained BERT model to extract semantically meaningful keywords per cluster.

KeyBERT Code snippet:

python
Copy
Edit
from keybert import KeyBERT

kw_model = KeyBERT(model)
cluster_keywords = {}
for label in set(labels):
    cluster_texts = [unique_texts[i] for i in range(len(unique_texts)) if labels[i] == label]
    joined_text = " ".join(cluster_texts)
    keywords = kw_model.extract_keywords(joined_text, top_n=5, stop_words='english')
    cluster_keywords[label] = [kw for kw, score in keywords]
    print(f"Cluster {label}: {cluster_keywords[label]}")
3.6 Cluster Visualization
UMAP was applied to reduce embedding dimensions to 2D, and clusters were visualized with matplotlib and seaborn.

Code snippet:

python
Copy
Edit
import umap
import matplotlib.pyplot as plt
import seaborn as sns

reducer = umap.UMAP(n_neighbors=10, n_components=2, metric='cosine')
embedding_2d = reducer.fit_transform(normalized_embeddings)

plt.figure(figsize=(12, 7))
palette = sns.color_palette("hls", len(set(labels)))

for label in set(labels):
    mask = (labels == label)
    plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], label=f"Cluster {label}", alpha=0.75, s=100)

plt.title("Clusters of Texts using SBERT + KeyBERT + Hierarchical Clustering")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
4. Results & Observations
Initial clustering showed location names dominating keywords and clusters (e.g., 'Arizona', 'Pennsylvania', 'Carolina').

Entity filtering reduced some location dominance but was insufficient to remove all location bias.

Clusters often mixed locations with event or person names, making interpretation noisy.

Some clusters had empty or uninformative keywords due to lack of clear thematic signal.

Boilerplate phrase filtering helped but manual pattern specification was tedious and incomplete.

5. Critical Scientific Analysis of Weak Points
Entity Filtering Alone is Insufficient: Location names are embedded implicitly through context, not just explicitly.

Embedding Bias: SBERT embeddings encode location-related context strongly, biasing clustering.

Unsupervised Clustering Limits: Without guided supervision, clusters capture dominant lexical patterns (locations) instead of topics.

Boilerplate Phrase Detection Requires Automation: Manual patterns are incomplete, need data-driven extraction.

Keyword Extraction is Descriptive Not Prescriptive: Keywords explain clusters but do not influence clustering.

6. Future Directions & Recommendations
6.1 Enhanced Preprocessing
Entity Masking: Replace all location, person, org mentions with generic tokens <LOCATION>, <PERSON>, etc., before embedding.

Data Augmentation: Use LLMs to rewrite texts removing location mentions or summarizing topics.

6.2 Advanced Topic Modeling
Use BERTopic or Top2Vec that integrate embedding + clustering + topic extraction, with built-in techniques to downweight location bias.

6.3 LLM Integration
Use large language models for topic-focused summarization and embedding.

Generate embeddings that emphasize thematic content over named entities.

6.4 Supervised or Weakly Supervised Learning
Collect topic labels on a subset for training classifiers or fine-tuning embeddings.

Use contrastive learning to separate topics ignoring locations.

7. Conclusion
This project demonstrated the complexity of unsupervised text clustering in politically rich datasets with dominant location signals. While initial cleaning, entity filtering, embedding, and clustering workflows provided a starting point, the strongest challenges remain in removing or mitigating location dominance at the embedding and clustering stages. Advanced NLP techniques, LLM-powered semantic filtering, and topic modeling frameworks are necessary next steps.
