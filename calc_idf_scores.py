from tqdm import tqdm
import spacy
import math
import requests
from threading import Thread

from config import *

os_client = get_opensearch_client()

LANGUAGE = "es"
SPACY_LANG_MODEL = "es_core_news_sm"
LANG_DETECTOR_API = "https://lang-detector.hive-discover.tech"
INDEX_NAME = "word-vectors-" + LANGUAGE

spacy.cli.download(SPACY_LANG_MODEL)
spacy_nlp = spacy.load(SPACY_LANG_MODEL)

# get all vocabs from our DB
all_vocabs = {}
batch = os_client.search(index=INDEX_NAME, body={"query": {"match_all": {}}, "_source": {"includes": ["nothing"]}, "size" : 1000}, scroll="10m")
pbar = tqdm(total=batch["hits"]["total"]["value"], desc="Getting All Vocabs")
while True:
    if not batch['hits']['hits']:
        break # end of the scroll

    # Process the batch
    for hit in batch['hits']['hits']:
        all_vocabs[hit['_id']] = 1
    pbar.update(len(batch['hits']['hits']))

    # Get the next batch
    scroll_id = batch['_scroll_id']
    batch = os_client.scroll(scroll_id=scroll_id, scroll="10m")

pbar.close()
print(f"Got {len(all_vocabs)} vocabs")

def filter_langs(author, permlink) -> str:
    payload = {"author": author, "permlink": permlink, "filter" : LANGUAGE}
    result = requests.get(LANG_DETECTOR_API, params=payload)
    if result.status_code == 200:
        text = result.json()['text']
        return ' '.join(text)

    return None

def process_doc(author, permlink) -> None:
    # Filter text for lang and Tokenize the doc
    text = filter_langs(author, permlink)
    if text is None:
        return
    unique_tokens = set([token.text for token in spacy_nlp(text)])

    # Add 1 to the count of each unique token
    for token in unique_tokens:
        if token in all_vocabs:
            all_vocabs[token] += 1

# scroll through all english posts
search_body = {
    "size" : 25,
    "query" : {
        "nested" : {
            "path" : "language",
            "query" : {
                "bool" : {
                    "must" : [
                        {"term" : {"language.lang" : LANGUAGE}},
                    ]
                }
            }
        }
    },
    "_source": {"includes": ["author", "permlink"]}
}
batch = os_client.search(index="hive-posts", body=search_body, scroll="1d")
TOTAL_POSTS_COUNT = batch["hits"]["total"]["value"]
pbar = tqdm(total=TOTAL_POSTS_COUNT, desc="Processing all Posts")
while True:
    if not batch['hits']['hits']:
        break # end of the scroll

    # Define threads to work on these batches in parallel
    threads = []
    for hit in batch['hits']['hits']:
        threads.append(Thread(target=process_doc, args=(hit['_source']['author'], hit['_source']['permlink'])))
    
    # Start and join them
    _ = [t.start() for t in threads]
    _ = [t.join() for t in threads]

    pbar.update(len(batch['hits']['hits']))

    # Get the next batch
    scroll_id = batch['_scroll_id']
    batch = os_client.scroll(scroll_id=scroll_id, scroll="100m")

pbar.close()
print("Finished processing all posts")

# Calc idf
# idf_score (t) = log(n / df(t))
# t == token (=idf_matrix-key) | n == count of all used posts (=text_count) | df(t) == counter of how documents contain token t (=idf_matrix-value)
for token, count in all_vocabs.items():
    all_vocabs[token] = math.log10(TOTAL_POSTS_COUNT / count)

# Enter all idf scores into DB
bulk = []
for token, idf_score in tqdm(all_vocabs.items(), desc="Entering All idf Scores"):
    bulk.append({"update" : {"_id" : token}})
    bulk.append({"doc" : {"idf" : idf_score}})

    # Perform bulk update
    if len(bulk) >= 1000:
        counter = 0
        while counter < 5:
            try:
                os_client.bulk(index=INDEX_NAME, body=bulk, timeout="1m")
                bulk = []
                break
            except Exception as e:
                print(f"Failed to bulk update {counter} times: {e}")
                counter += 1               

        


if len(bulk) > 0:
    os_client.bulk(index=INDEX_NAME, body=bulk)
print("Finished entering all idf scores")

print("Done.")