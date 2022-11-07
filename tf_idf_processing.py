import requests
import numpy as np
import spacy
import base64

from config import *

os_client = get_opensearch_client()
lang_models = {
    "en" : spacy.load("en_core_web_sm"),
    "es" : spacy.load("es_core_news_sm"),
    # "de" : spacy.load("de_core_news_sm"),
}
if len(lang_models) != len(SUPPORTED_LANGS):
    # Check that all langs are supported
    raise Exception("LANG_DETECTOR_URL must contain all supported languages")

def count_token_appearences(tokens : list) -> dict:
    token_count = {} # {token : count}

    # Count tokens
    for token in tokens:
        # Initialize token-count
        if token not in token_count:
            token_count[token] = 0

        token_count[token] += 1

    return token_count

def get_wordvec_docs(tokens : list, lang : str) -> dict:
    docs = {} # {token : {vec : wordvec, idf : idf-score}}

    # Get wordvecs for each token by using scroll search in batches of a 1,000
    batch = os_client.search(index="word-vectors-" + lang, body={"query": {"ids": {"values": tokens}}, "size" : 1000}, scroll="1m")
    scroll_id = batch['_scroll_id']

    # Scroll through Cursor
    while True:
        if not batch["hits"]["hits"]:
            break # Finished scrolling

        # Process batch
        for hit in batch["hits"]["hits"]:
            docs[hit['_id']] = {
                "idf" : hit['_source']["idf"],
                "vec" : hit['_source']["bvec"]
            }

        # Get next batch
        batch = os_client.scroll(scroll_id=scroll_id, scroll="1m")
        scroll_id = batch['_scroll_id']

    # Decode vectors: From Base64 (utf-8) to numpy array (np.float32)
    for token in docs:
        vector = base64.b64decode(docs[token]['vec'].encode("utf-8"))
        vector = np.frombuffer(vector, dtype=np.float32)
        docs[token]['vec'] = vector

    return docs

class TfIdfProcessor:
    def __init__(self, _id : str, start : bool = True):
        self.post_id = _id            

        self.doc_vectors = {} # {lang : vector}
        self.known_total_ratio = {} # {lang : ratio}

        if start: # do Work
            self.get_post_text()
            self.tokenize()
            self.vectorize_all()

    def get_post_text(self) -> None:
        # Sent request to LangDetector
        result = requests.get(LANG_DETECTOR_URL, params={"id" : self.post_id})
        if result.status_code != 200:
            # TODO: log error
            print(result.text)
            raise Exception("LANG_DETECTOR_URL returned status code {}".format(result.status_code))

        # Parse response
        self.lang_text = {} # {en : sen1 + sen2, ...}
        sentences = result.json()['text']
        for sent_text, sent_langs in sentences:
            # add all sentences with the same lang together
            for lang in sent_langs:
                if lang not in SUPPORTED_LANGS:
                    continue # Skip unsupported langs

                if lang not in self.lang_text:
                    # Add new lang
                    self.lang_text[lang] = ""

                # Add Sentence to lang
                self.lang_text[lang] += sent_text + " "

    def tokenize(self) -> None:
        # Tokenize text for each lang with the correct spacy lang-model
        self.lang_tokens = {}
        for lang in self.lang_text:
            self.lang_tokens[lang] = [token.text for token in lang_models[lang](self.lang_text[lang])]

    def get_vector(self, lang : str) -> np.ndarray:
        token_counts = count_token_appearences(self.lang_tokens[lang]) # {token : count}
        wordvec_docs = get_wordvec_docs(list(token_counts.keys()), lang) # {token : {vec : wordvec, idf : idf-score}}

        # Find known token count
        known_token_count = 0
        for token in token_counts:
            if token in wordvec_docs:
                known_token_count += token_counts[token]
        
        # If we don't have any (or to less) known tokens, return an empty vector
        self.known_total_ratio[lang] = known_token_count / sum(token_counts.values())
        if known_token_count <= 5 or self.known_total_ratio[lang] <= 0.35:
            return np.array([])

        # Finally calculate doc-vector
        doc_vector = np.zeros(300, dtype=np.float32)
        for token in token_counts:
            if token not in wordvec_docs:
                continue # Skip tokens that are not in the word-vector-db
            
            # Add token-vector to doc-vector with BM25-weighting
            # term-frequency = token-appearences / known_token_count
            term_freq = token_counts[token] / known_token_count
            doc_vector += wordvec_docs[token]['vec'] * wordvec_docs[token]['idf'] * term_freq

        return doc_vector

    def vectorize_all(self) -> None:
        self.doc_vectors = {} # {en : vector, ...}
        for lang in self.lang_tokens:
            vector = self.get_vector(lang)
            if vector.size == 300:
                self.doc_vectors[lang] = vector

    def get_lang_vectors(self) -> dict:
        return self.doc_vectors

