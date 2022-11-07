import os

LANG_DETECTOR_URL = os.environ.get('LANG_DETECTOR_URL', None)
SUPPORTED_LANGS = ["en", "es"]

HEARTBEAT_URL = os.environ.get("VECTORIZER_HEARTBEAT_URL", None)
if not HEARTBEAT_URL:
    print("[WARNING] HEARTBEAT_URL is not set in environment variables")

from opensearchpy import OpenSearch
OPENSEARCH_HOSTS = os.environ.get("OPENSEARCH_HOSTS", "").split(",")

print("Opensearch Connectionmade with:")
print(f"\tHosts: {OPENSEARCH_HOSTS}")

def get_opensearch_client() -> OpenSearch:
    return OpenSearch(
        OPENSEARCH_HOSTS,
        http_compression=True,
    )