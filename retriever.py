# CSC-128 Module 6
# Retriever for grounded policy bot

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from knowledge import DOCUMENTS


# Common words that do not help much when searching
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "i", "me", "my", "you", "your", "to", "of",
    "for", "and", "or", "in", "on", "at", "do",
    "does", "can", "how", "what"
}

# I started with a lower threshold so relevant student questions
# can still match even if they are worded a little differently.
DEFAULT_THRESHOLD = 0.22
DEFAULT_TOP_K = 3


def stem(word):
    """
    Simple stemming so similar versions of a word can match.
    This does not need to be a full Porter stemmer.
    """
    suffixes = ["ing", "ed", "es", "s"]

    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]

    return word


def analyze(text):
    """
    Normalize the text, split it into words, stem the words,
    and add bigrams to help retrieval.
    """
    text = text.lower()

    # Keep letters and numbers and replace punctuation with spaces.
    words = re.findall(r"[a-z0-9]+", text)

    tokens = []

    for word in words:
        if word not in STOP_WORDS:
            tokens.append(stem(word))

    # Add bigrams such as "library_hours" and "research_help".
    bigrams = []

    for i in range(len(tokens) - 1):
        bigrams.append(tokens[i] + "_" + tokens[i + 1])

    return tokens + bigrams


class Retriever:
    def __init__(self, documents=None, threshold=DEFAULT_THRESHOLD):
        """
        Build the TF-IDF index using the knowledge base.
        """
        self.documents = documents if documents is not None else DOCUMENTS
        self.threshold = threshold

        self.vectorizer = TfidfVectorizer(
            analyzer=analyze,
            lowercase=False
        )

        document_text = [doc["text"] for doc in self.documents]

        self.document_matrix = self.vectorizer.fit_transform(document_text)


    def search(self, question, top_k=DEFAULT_TOP_K):
        """
        Find the chunks that are most similar to the student's question.
        Results below the threshold are left out.
        """
        question_vector = self.vectorizer.transform([question])

        scores = cosine_similarity(
            question_vector,
            self.document_matrix
        )[0]

        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True
        )

        hits = []

        for index, score in ranked:
            if score >= self.threshold:
                hits.append((self.documents[index], float(score)))

            if len(hits) >= top_k:
                break

        return hits


    def build_context(self, hits):
        """
        Format retrieved chunks so the model gets both the
        information and its source.
        """
        context_parts = []

        for document, score in hits:
            chunk = (
                f"[{document['id']}]\n"
                f"Source: {document['source']}\n"
                f"{document['text']}"
            )

            context_parts.append(chunk)

        return "\n\n".join(context_parts)