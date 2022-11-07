import os
import fasttext.util
from tqdm import tqdm
import numpy as np
import base64

from config import *
os_client = get_opensearch_client()

LANGUAGE = "de"
MODEL_PATH = f'cc.{LANGUAGE}.300.bin'
INDEX_NAME = "word-vectors-" + LANGUAGE
INDEX_DEFINITION = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 1,
    },
    "mappings": {
        "properties": {
            # "word": {
            #     "type": "keyword",
            # }
            #{bvec : binary (utf-8 encoded 300 float32))}
            # "vec" : {
            #     "type" : "knn_vector",
            #     "dimension" : 300
            # }
        }
    }
}

# Download Model
if not os.path.exists(MODEL_PATH):
    fasttext.util.download_model(LANGUAGE, if_exists='ignore')
    print("Downloaded model for {}".format(LANGUAGE))

# Load Model to Memory
model = fasttext.load_model(MODEL_PATH)
print("Loaded model for {} in Memory".format(LANGUAGE))

# Delete Index if exists
try:
    os_client.indices.delete(INDEX_NAME)
    print("Deleted Index")
except:
    pass

# Create Index
os_client.indices.create(INDEX_NAME, INDEX_DEFINITION)
print("Created Index")

# Now Enter all Vocabs
bulk = []
for vocab in tqdm(model.words):
    if len(vocab) >= 100:
        continue # skip vocabs with more than 100 chars, they are garbage

    bin_vector = np.array(model[vocab]).astype(np.float32) 
    bin_vector = base64.b64encode(bin_vector).decode("utf-8")

    bulk.append({"create" : {"_index": INDEX_NAME, "_id": vocab}})
    bulk.append({"bvec": bin_vector})

    # Enter every 1000 vocabs
    if len(bulk) >= 2000:
        os_client.bulk(body=bulk, refresh="wait_for")
        bulk = []

# Enter rest
if len(bulk) > 0:
    os_client.bulk(body=bulk, refresh="wait_for")

print("Finished")