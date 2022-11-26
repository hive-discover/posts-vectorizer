import os
import math
from tqdm import tqdm
from pymongo import MongoClient, UpdateOne
import spacy

LANGUAGE = os.environ["LANGUAGE"]

MONGO_DB = "fasttext"
MONGO_COLLECTION = "word-vectors-" + LANGUAGE

BATCH_SIZE = 500

# Establish & Test MongoDB Connection
mongo_client = MongoClient(os.environ["MONGO_URI"])
print("Connected to MongoDB")
print("MongoDB Version: {}".format(mongo_client.server_info()["version"]))

# Load spacy model
print("Loading spacy model for {} in Memory".format(LANGUAGE))
SPACY_MODEL_NAME = ({ "en" : "en_core_web_sm", "es" : "es_core_news_sm", "de" : "de_core_news_sm"})[LANGUAGE]
spacy.cli.download(LANGUAGE)
nlp = spacy.load(SPACY_MODEL_NAME)
print("Loaded spacy model for {}".format(LANGUAGE))

# Get all Vocabs from MongoDB in this format: {word : occurences}
print("Getting all Vocabs")
vocabs = {word["_id"] : 0 for word in mongo_client[MONGO_DB][MONGO_COLLECTION].find({}, {"_id": 1})}
print("Got all Vocabs: " + str(len(vocabs)))

LANG_QUERY = {"language" : {"$elemMatch" : {"lang" : LANGUAGE, "word_count" : {"$gte" : 15 }}}}

# Get all comment-ids from MongoDB where the comment contains at least 15 words of the language
print("Getting all comment-ids")
comment_cursor = mongo_client["hive"]["comments"].find(LANG_QUERY, {"_id": 1})
comment_ids = [comment["_id"] for comment in comment_cursor]
print("Got all comment-ids: " + str(len(comment_ids)))

# Get all reply-ids from MongoDB where the reply contains at least 15 words of the language
print("Getting all reply-ids")
reply_cursor = mongo_client["hive"]["replies"].find(LANG_QUERY, {"_id": 1})
reply_ids = [reply["_id"] for reply in reply_cursor]
print("Got all reply-ids: " + str(len(reply_ids)))

def process_text(text : str):
    """Process the text with spacy and update the occurences of the words in the vocabs dict"""
    doc = nlp(text.lower())
    for token in doc:
        if token.text in vocabs:
            vocabs[token.text] += 1

for collection_name, ids in [("comments", comment_ids), ("replies", reply_ids)]:
    progress = tqdm(total=len(ids), desc="Processing " + collection_name)

    counter = 0
    while counter < len(ids):
        batch_ids = ids[counter:counter + BATCH_SIZE]
        counter += len(batch_ids)        

        # Get all texts from MongoDB
        doc_cursor = mongo_client["hive"][collection_name].find({"_id" : {"$in" : batch_ids}}, {"title": 1, "body" : 1})
        texts = [doc["title"] + " \n " + doc["body"] for doc in doc_cursor]

        # Process all texts with spacy
        for text in texts:
            process_text(text)
        progress.update(len(batch_ids))

    progress.close()

# Calculate the idf-scores
# Replace vocabs with zero occurences with the lowest idf-score
print("Calculating idf-scores")
TOTAL_POSTS = len(comment_ids) + len(reply_ids)
vocabs = {word : (math.log10(TOTAL_POSTS / occurences)  if occurences > 0 else -1) for word, occurences in vocabs.items()}
min_idf = min(vocabs.values())
vocabs = [(word, idf if idf > 0 else min_idf) for word, idf in vocabs.items()]
print("Calculated idf-scores")

# Update the MongoDB with the idf-scores
print("Updating MongoDB in batches of " + str(BATCH_SIZE))
for batch_idx in tqdm(range(0, len(vocabs), BATCH_SIZE)):
    batch_vocabs = vocabs[batch_idx:batch_idx + BATCH_SIZE]
    update_requests = [UpdateOne({"_id" : word}, {"$set" : {"idf" : idf}}) for word, idf in batch_vocabs]
    mongo_client[MONGO_DB][MONGO_COLLECTION].bulk_write(update_requests)

print("Updated MongoDB")
print("Done")