import os
import time
from pymongo import MongoClient, UpdateOne
import spacy
import concurrent.futures
import requests
import numpy as np
from random import randint, choice

mongo_client = MongoClient(os.environ["MONGO_URI"])

BATCH_SIZE = os.environ.get("BATCH_SIZE", 10)
LANG_DETECTOR_URI = os.environ["LANG_DETECTOR_URI"]
VECTORIZER_HEARTBEAT_URL = os.environ.get("VECTORIZER_HEARTBEAT_URL", None)

LANGUAGE = os.environ["LANGUAGE"]
SPACY_MODEL_NAME = ({ "en" : "en_core_web_sm", "es" : "es_core_news_sm", "de" : "de_core_news_sm"})[LANGUAGE]
nlp = spacy.load(SPACY_MODEL_NAME)

JOB_NAME = "jobs.vectorizer_" + LANGUAGE

total_posts_found = {} # {target : total_posts_found}

def get_text_lang(target : str, _id : int):
    """Get the language and text from the lang-detector"""
    resp = requests.get(LANG_DETECTOR_URI + "/" + target + "/" + str(_id) + "?filter=" + LANGUAGE)
    if resp.status_code == 200:
        text = resp.json()["text"] # [ sentence1, sentence2, ... ]
        return ' '.join(text)

    raise Exception("Could not get text from lang-detector: " + str(resp.status_code) + " " + resp.text)

def calc_tf_of_text(text : str) -> tuple:
    """Calculate the tf of a text"""
    tokens = nlp(text.lower())
    tf = {}

    if len(tokens) == 0:
        return tf, 0

    # Count occurences of each token
    for token in tokens:
        if token.text in tf:
            tf[token.text] += 1
        else:
            tf[token.text] = 1

    # Calculate the tf
    for token in tf:
        tf[token] /= len(tokens)

    return tf, len(tokens)

def get_wordvecs(tokens : list) -> dict:
    """Get the word-vectors for the tokens"""
    # Prepare the cursor
    cursor = mongo_client["fasttext"]["word-vectors-" + LANGUAGE].find({"_id": {"$in": tokens}}, {"_id": 1, "vector": 1, "idf" : 1})
    
    # Retrieve the word-vectors
    wordvecs = {} # {token : {vector : wordvec, idf : idf-score}}
    for doc in cursor:
        wordvecs[doc["_id"]] = {
            "idf" : doc["idf"],
            "vector" : np.frombuffer(doc["vector"], dtype=np.float32)
        }

    return wordvecs

def process_post(target : str, _id : int) -> UpdateOne:
    """Process a post and return an UpdateOne for MongoDB"""
    # Get all the required information
    text = get_text_lang(target, _id)
    tf_dict, token_count = calc_tf_of_text(text)
    wordvecs = get_wordvecs(list(tf_dict.keys()))

    # Calculate the tf-idf
    known_tokens = 0
    doc_vector = np.zeros(300, dtype=np.float32)
    for token in tf_dict:
        if token not in wordvecs:
            continue

        doc_vector += tf_dict[token] * wordvecs[token]["idf"] * wordvecs[token]["vector"]
        known_tokens += 1

    # Return the UpdateOne
    return UpdateOne(
        {"_id": _id}, 
        {"$set": {
            "doc_vectors": {LANGUAGE: doc_vector.tobytes() if known_tokens > 0 else None}, 
            "known_tokens_ratio" : (known_tokens / token_count if token_count > 0 else 0), 
            JOB_NAME : True
        }}
    )

def get_batch(target : str) -> list:
    """Get a batch of posts to process"""
    # Prepare the cursor
    cursor = mongo_client["hive"][target].find({JOB_NAME : {"$ne" : True}}, {"_id": 1})
    if cursor.count() == 0:
        return []    

    # randomly select BATCH_SIZE posts
    total_posts_found[target] = cursor.count()
    if total_posts_found[target] > (BATCH_SIZE * 4):
        cursor.skip(randint(0, total_posts_found[target] - BATCH_SIZE))

    return list(cursor.limit(BATCH_SIZE))
    
def process_batch() -> tuple:
    target = choice(["replies", "comments"])
    batch = get_batch(target)
    if len(batch) == 0:
        return 0, target

    # Process the batch
    bulk_updates = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        results = executor.map(process_post, [target] * len(batch), [doc["_id"] for doc in batch])
        bulk_updates = list(results)

    # Update the database
    if len(bulk_updates) > 0:
        mongo_client["hive"][target].bulk_write(bulk_updates, ordered=False)

    return len(bulk_updates), target

def send_heartbeat(elapsed_ms : int):
    """Send a heartbeat to the monitoring"""
    params = {'msg': 'OK', 'ping' : elapsed_ms}

    if VECTORIZER_HEARTBEAT_URL is not None:
        try:
            requests.get(VECTORIZER_HEARTBEAT_URL, params=params)
        except Exception as e:
            print("CANNOT SEND HEARTBEAT: ")
            print(e)

def main():
    while True:
        start = time.time()
        processed, target = process_batch()
        elapsed = time.time() - start
        send_heartbeat(elapsed * 1000)

        if processed == 0:
            time.sleep(5)
            continue

        print(f"Processed {processed}/{total_posts_found[target]} {target} in {elapsed:.2f}s")


if __name__ == '__main__':
    main()