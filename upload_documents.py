from clip_processor import *
from elasticsearch import Elasticsearch
import os
from dotenv import load_dotenv

load_dotenv()

def list_metadata_files(path):
    files = []
    for file in os.listdir(path):
        if file.endswith(".json"):
            files.append(file)

    return files


def upload_document(es: Elasticsearch, doc, index):
    res = es.index(document=doc, index=index)
    print(res)
    if res["result"] == "created":
        print(f"Uploaded file {doc['image_filename']}")
    else:
        print(f"Failed to upload file {doc['image_filename']}")



def index_logic():
    es_host = os.getenv("ES_HOST")
    es_username = os.getenv("ES_USERNAME")
    es_password = os.getenv("ES_PASSWORD")
    index = os.getenv("ES_INDEX")

    if not es_host:
        raise ValueError("ES_HOST not set")
    if not es_username:
        raise ValueError("ES_USERNAME not set")
    if not es_password:
        raise ValueError("ES_PASSWORD not set")
    if not index:
        raise ValueError("ES_INDEX not set")

    es = Elasticsearch(
        hosts=[es_host],
        basic_auth=(es_username, es_password),
        verify_certs=False,
        ssl_show_warn=False
    )

    metadata_files = list_metadata_files('images_metadata/')

    for doc in metadata_files:
        data = add_embeddings(doc)
        upload_document(es, data, index)


index_logic()