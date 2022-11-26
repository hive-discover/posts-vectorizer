import os
import fasttext.util
from tqdm import tqdm
import numpy as np
from pymongo import MongoClient, UpdateOne

LANGUAGE = os.environ["LANGUAGE"]
MODEL_PATH = f'cc.{LANGUAGE}.300.bin'

MONGO_DB = "fasttext"
MONGO_COLLECTION = "word-vectors-" + LANGUAGE

BULK_UPDATE_SIZE = 1000

# Establish & Test MongoDB Connection
mongo_client = MongoClient(os.environ["MONGO_URI"])
print("Connected to MongoDB")
print("MongoDB Version: {}".format(mongo_client.server_info()["version"]))

# Download Model
if not os.path.exists(MODEL_PATH):
    print("Downloading model for {}".format(LANGUAGE))
    fasttext.util.download_model(LANGUAGE, if_exists='ignore')
    print("Downloaded model for {}".format(LANGUAGE))

# Load Model to Memory
print("Loading model for {} in Memory".format(LANGUAGE))
model = fasttext.load_model(MODEL_PATH)
print("Loaded model for {} in Memory".format(LANGUAGE))

# Enter all Vocabs
print("Entering all Vocabs")
bulk_updates = []
for vocab in tqdm(model.words):
    # skip vocabs with more than 100 chars, they are garbage
    if len(vocab) >= 100:
        continue

    # create update operation
    bulk_updates.append(UpdateOne(
        {"_id": vocab},
        {"$set" : {"vector" : np.array(model[vocab]).astype(np.float32).tobytes()}},
        upsert=True
    ))

    # execute bulk update every BULK_UPDATE_SIZE vocabs
    if len(bulk_updates) >= BULK_UPDATE_SIZE:
        mongo_client[MONGO_DB][MONGO_COLLECTION].bulk_write(bulk_updates, ordered=False)
        bulk_updates = []

# execute remaining bulk updates
if len(bulk_updates) > 0:
    mongo_client[MONGO_DB][MONGO_COLLECTION].bulk_write(bulk_updates, ordered=False)

print("Entered all Vocabs")
print("Done")