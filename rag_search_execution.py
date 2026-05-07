from elasticsearch_dsl import Search, Q
from elasticsearch import Elasticsearch
from clip_processor import create_text_embedding
import os


index = os.getenv('ES_INDEX')


def _build_geo_filter_for_parks(parks_coordinates, distance):
    if not parks_coordinates:
        return Q('match_all')

    if len(parks_coordinates) == 1:
        lat, lon = parks_coordinates[0]
        return Q('geo_distance', distance=distance, geolocation={'lat': lat, 'lon': lon})

    geo_queries = [
        Q('geo_distance', distance=distance, geolocation={'lat': lat, 'lon': lon})
        for lat, lon in parks_coordinates
    ]
    return Q('bool', should=geo_queries, minimum_should_match=1)


def rrf_search(es_client, index_name, park_coordinates, distance, text_query, k=5,
               num_candidates=30):
    """
    Execute a combined Elasticsearch bool search for multiple parks.

    Args:
        es_client: Elasticsearch client instance
        index_name (str): Name of the Elasticsearch index
        park_coordinates (List[Tuple[float, float]]): Coordinates for park geo filters
        distance (int/str): Distance for geo filtering
        text_query (str): Text to search in description fields
        k (int): Number of top results for KNN search (unused when RRF is disabled)
        num_candidates (int): Number of candidates for KNN search (unused when RRF is disabled)

    Returns:
        List[dict]: Elasticsearch hit documents
    """

    text_query = (text_query or '').strip()
    geo_filter = _build_geo_filter_for_parks(park_coordinates, distance)

    if text_query:
        embedding = create_text_embedding(text_query).tolist()
        text_queries = [
            Q('match', generated_description={'query': text_query, 'boost': 2}),
            Q('match', description={'query': text_query})
        ]
        standard_query = Q('bool', filter=[geo_filter], should=text_queries, minimum_should_match=1)
    else:
        standard_query = Q('bool', filter=[geo_filter])

    s = Search(using=es_client, index=index_name)
    s = s.source(["image_filename", "generated_description", "geolocation"])

    if text_query:
        s = s.query(standard_query)
    else:
        s = s.query(standard_query)

    s = s[:5]
    results = s.execute()["hits"]["hits"]
    return results


def execute_rrf_search_dsl(es_client, search_obj):
    """
    Execute the RRF search using elasticsearch_dsl.

    Args:
        es_client: Elasticsearch client instance
        search_obj: Search object created with create_rrf_search_from_index

    Returns:
        Response: elasticsearch_dsl Response object
    """
    # Execute using the elasticsearch_dsl client integration
    response = search_obj.using(es_client).execute()
    return response

