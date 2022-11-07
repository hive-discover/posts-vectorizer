from math import ceil
import requests
from multiprocessing.pool import ThreadPool
import time
from datetime import datetime
import weaviate
import uuid

import numpy as np
from config import *
from tf_idf_processing import TfIdfProcessor

weaviate_client = weaviate.Client(os.environ.get("WEAVIATE_HOST", None), additional_headers={"Authorization" : os.environ.get("WEAVIATE_AUTH", None)})

BATCH_SIZE = 25
CONCURRENT_PROCESSES = 3
TOTAL_POSTS_FOUND = 0
WORKER_FIND_QUERY = {
        "size" : str(BATCH_SIZE),
        "query" : {
            "bool" : {
                "must" : [
                    {"exists" : {"field" : "text_title"}},
                    {"exists" : {"field" : "text_body"}},
                    {"nested" : {
                        "path" : "jobs",
                        "query" : {
                            "bool" : {
                                # Find items which got not processed yet
                                "must_not" : [
                                    {"term" : {"jobs.vectorizer" : True}},                          
                                ],       
                                # Lang has to be calculated
                                "must" : [
                                    {"term" : {"jobs.lang_detected" : True}},                          
                                ],                
                            }
                        }
                    }}
                ]
            }
        },
       "_source" : {
           "includes" : ["timestamp", "author", "permlink", "parent_permlink", "tags"]
       }
    }

os_client = get_opensearch_client()

def get_batch() -> list:
    global TOTAL_POSTS_FOUND
    results = os_client.search(index="hive-posts", body=WORKER_FIND_QUERY, timeout="60s")
    TOTAL_POSTS_FOUND = results['hits']['total']["value"]
    return results['hits']['hits']

def proces_one(post_item : tuple) -> list:
    ''''Process one post and return the bulk-update list'''
    # Calculate doc-vectors for each lang in post
    post_id, post_index, post_dict = post_item
    tfidf_processor = TfIdfProcessor(post_id, start=True)
    lang_doc_vectors = tfidf_processor.get_lang_vectors() # {lang: vector}
    known_total_ratio = tfidf_processor.known_total_ratio

    # Create update document for OpenSearch
    doc = {"jobs" : {"vectorizer" : True}, "known_total_ratio" : known_total_ratio, "doc_vector" : {}}
    for lang in SUPPORTED_LANGS:
        if lang in lang_doc_vectors:
            doc["doc_vector"][lang] = lang_doc_vectors[lang].tolist()
        else: 
            doc["doc_vector"][lang] = None # Empty lang ==> null   

    # Add this post to weaviate (in a batch)
    weave_doc = {
        "author" : post_dict["author"],
        "permlink" : post_dict["permlink"],
        "os_id" : post_id,
        "parent_permlink" : post_dict["parent_permlink"],
        "tags" : post_dict["tags"],
        "timestamp" : post_dict["timestamp"] + "Z"    
    }

    if "en" in lang_doc_vectors and lang_doc_vectors["en"] is not None:
        weave_doc["known_total_ratio"] = known_total_ratio["en"]
        weave_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"{weave_doc['author']}/{weave_doc['permlink']}")
        weaviate_client.batch.add_data_object(weave_doc, "HivePostsEnVectors", weave_uuid, vector=lang_doc_vectors["en"].tolist())

    return [
        {"update" : {"_id" : post_id, "_index" : post_index}},
        {"doc" : doc}
    ]

def process_batch() -> int:
    # Get work and remap it to list of just ids
    data_batch = get_batch()
    batch_of_work = [(post['_id'], post['_index'], post['_source']) for post in data_batch]

    # Process batch and get update_bulk
    update_bulk = []
    with ThreadPool(CONCURRENT_PROCESSES) as pool:
        # Get thread results, flat it and add to update_bulk
        task_results = pool.map(proces_one, batch_of_work)
        task_results = [j for i in task_results for j in i]
        update_bulk += task_results

    # Send update_bulk to OS and flush Weaviate Batch
    if len(update_bulk) > 0:
        res = os_client.bulk(body=update_bulk, refresh="wait_for", timeout="60s")
        weaviate_client.batch.flush()
        return len(update_bulk)
        
    return 0

def send_heartbeat(elapsed_time : int = 0) -> None:
    params = {'msg': 'OK', 'ping' : elapsed_time}

    if HEARTBEAT_URL is not None:
        try:
            requests.get(HEARTBEAT_URL, params=params)
        except Exception as e:
            print("CANNOT SEND HEARTBEAT: ")
            print(e)

def run():

    while True:
        # Do Work and measure time
        start_time = time.time()
        counter = process_batch()
        elapsed_time = time.time() - start_time

        # Send heartbeat and print stats
        send_heartbeat(elapsed_time * 1000)  
        if counter > 0:
            print("Processed {}/{} posts in {:.2f} seconds".format(counter, TOTAL_POSTS_FOUND, elapsed_time))
        else:
            # Sleep only when we have no posts left to process
            time.sleep(8)

if __name__ == '__main__':
    run()

# docker run --env-file V:\Projekte\HiveDiscover\Python\docker_variables.env registry.hive-discover.tech/vectorizer:0.1.7