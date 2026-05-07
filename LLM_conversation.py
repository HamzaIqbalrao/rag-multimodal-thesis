from ollama import chat
from ollama import ChatResponse
import json
from typing import Dict, List, Any, Optional
from elasticsearch import Elasticsearch
from rag_search_execution import *
import re


index = os.getenv('ES_INDEX')

national_parks = {
    "mt_rainier_national_park": {
        "coordinates": (46.8523, -121.7603),
        "state": "Washington"
    },
    "crater_lake_national_park": {
        "coordinates": (42.8684, -122.1685),
        "state": "Oregon"
    },
    "death_valley_national_park": {
        "coordinates": (36.5054, -117.0794),
        "state": "California"
    },
    "grand_canyon_national_park": {
        "coordinates": (36.0544, -112.1401),
        "state": "Arizona"
    },
    "arches_national_park": {
        "coordinates": (38.7331, -109.5925),
        "state": "Utah"
    },
    "grand_teton_national_park": {
        "coordinates": (43.7904, -110.6818),
        "state": "Wyoming"
    },
    "yellowstone_national_park": {
        "coordinates": (44.4280, -110.5885),
        "state": "Wyoming"
    },
    "katmai_national_park": {
        "coordinates": (58.5969, -155.0063),
        "state": "Alaska"
    },
    "great_smoky_mountains_national_park": {
        "coordinates": (35.6118, -83.4895),
        "state": "Tennessee"
    },
    "antietam_national_battlefield": {
        "coordinates": (39.4751, -77.7411),
        "state": "Maryland"
    },
    "canyonlands_national_park": {
        "coordinates": (38.2619, -109.8782),
        "state": "Utah"
    }
}


def format_parks_for_prompt(parks_dict):
    """Helper function to format parks data for the prompt"""
    parks_list = []
    for park_id, info in parks_dict.items():
        park_name = park_id.replace('_', ' ').title()
        parks_list.append(
            f"- {park_name}: {info['state']} (Lat: {info['coordinates'][0]}, Lon: {info['coordinates'][1]})")
    return "\n".join(parks_list)


def extract_search_parameters(query: str) -> Optional[Dict[str, Any]]:
    """Extract search parameters from user query using LLM"""
    model_name = "cogito:3b"

    parks_info = format_parks_for_prompt(national_parks)

    content = f"""You are going to extract data from a user query for a national parks search system. 

Available National Parks:
{parks_info}

Extract the following information and format as JSON:
- context_search: the main activity or interest (e.g., "hike", "walk dog", "camping")
- distance_km: estimated search radius in kilometers (default: 100 if not specified)
- location_type: specific state, city, or region mentioned
- reference_location: if a city is mentioned, include it (e.g., "Boston", "Denver")
- relevant_parks: list of park IDs that might be relevant based on location (use the exact park IDs from the list above)

Examples:
User query: "Where can I hike in Utah?"
Response: {{"context_search": "hike", "distance_km": 100, "location_type": "Utah", "reference_location": null, "relevant_parks": ["arches_national_park", "canyonlands_national_park"]}}

Only respond with valid JSON. No additional text. If a city is mentioned use the State that city is on as the location_type.

User query: {query}
"""

    try:
        response: ChatResponse = chat(
            model=model_name,
            messages=[{'role': 'user', 'content': content}],
            options={'temperature': 0.3, 'think': False}  # Enable reasoning mode to help with extraction
        )

        if response and response['message']['content']:
            response_text = response['message']['content']

            # Strip <think>...</think> blocks produced by qwen3's reasoning mode
            response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

            # Strip markdown code fences in case the model wraps JSON in ```json ... ```
            response_text = re.sub(r'```json|```', '', response_text).strip()

            if not response_text:
                print("Error: LLM returned empty content after stripping think blocks")
                return None

            extracted_data = json.loads(response_text)
            return extracted_data

    except (json.JSONDecodeError, Exception) as e:
        print(f"Error extracting search parameters: {e}")
        return None

    return None


def find_nearest_park(geolocation: Dict[str, Any], parks: Dict[str, Dict[str, Any]]) -> Optional[str]:
    if not geolocation or 'lat' not in geolocation or 'lon' not in geolocation:
        return None

    best_park = None
    best_distance = float('inf')
    lat1 = geolocation['lat']
    lon1 = geolocation['lon']

    for park_id, info in parks.items():
        lat2, lon2 = info['coordinates']
        dist = (lat1 - lat2) ** 2 + (lon1 - lon2) ** 2
        if dist < best_distance:
            best_distance = dist
            best_park = park_id

    return best_park


def search_parks_elasticsearch(search_params: Dict[str, Any], es_host, es_username, es_password) -> List[Dict[str, Any]]:
    """Execute a combined Elasticsearch search for the requested parks."""
    index_name = os.getenv('ES_INDEX')

    relevant_parks = search_params.get('relevant_parks', []) or list(national_parks.keys())
    relevant_parks = [park_id for park_id in relevant_parks if park_id in national_parks]
    if not relevant_parks:
        relevant_parks = list(national_parks.keys())

    park_coordinates = [national_parks[park_id]['coordinates'] for park_id in relevant_parks]
    search_text = search_params.get('context_search', '').strip() or search_params.get('location_type', '').strip() or ''
    search_distance = f"{search_params.get('distance_km', 100)}km"

    print(f"Searching {len(relevant_parks)} parks for: '{search_text}' within {search_distance}")

    es = Elasticsearch(
        hosts=[es_host],
        basic_auth=(es_username, es_password),
        verify_certs=False,
        ssl_show_warn=False
    )
    search_results = rrf_search(
        es_client=es,
        index_name=index_name,
        park_coordinates=park_coordinates,
        distance=search_distance,
        text_query=search_text
    )

    all_results = []
    for result in search_results:
        source = dict(result['_source']) if '_source' in result else {}
        nearest_park = find_nearest_park(source.get('geolocation', {}), national_parks)
        park_id = nearest_park or relevant_parks[0]
        park_info = national_parks.get(park_id, {})

        all_results.append({
            'image_filename': source.get('image_filename'),
            'generated_description': source.get('generated_description'),
            'park_id': park_id,
            'park_state': park_info.get('state'),
            'park_coordinates': park_info.get('coordinates'),
            '_score': result['_score'] if '_score' in result else None
        })

    return all_results


def generate_response(original_query: str, search_results: List[Dict[str, Any]], search_params: Dict[str, Any]) -> str:
    """Generate final response using LLM with search results"""
    model_name = "cogito:3b"

    # Format search results for the prompt
    if not search_results:
        results_text = "No results found for your query."
    else:
        results_text = "Search Results:\n"
        for result in search_results[:5]:
            score = result['_score']
            # Adjust these field names based on your Elasticsearch document structure
            title = result['image_filename']
            content_snippet = result['generated_description'][:150] + "..."

            results_text += f"   Title: {title}\n"
            results_text += f"   Content: {content_snippet}\n"
            results_text += f"   Relevance Score: {score}\n"

    content = f"""You are a helpful assistant for national parks activities. Based on the search results below, provide a comprehensive and helpful response to the user's original query.

Original User Query: {original_query}

Search Parameters Used:
- Activity/Interest: {search_params.get('context_search', 'N/A')}
- Search Distance: {search_params.get('distance_km', 'N/A')} km
- Location: {search_params.get('location_type', 'N/A')}
{f"- Reference Location: {search_params.get('reference_location')}" if search_params.get('reference_location') else ""}

{results_text}

Instructions:
- Provide a natural, conversational response
- Recommend specific activities and locations based on the search results only
- Include practical information when available
- Do not suggest alternatives if no results were found
- Be enthusiastic and helpful about national parks experiences
- Keep the response focused and not too lengthy
- Structure your response separating your suggestions per national park
- Do not include anything about national parks that are not in the results

Response:"""

    try:
        response: ChatResponse = chat(
            model=model_name,
            messages=[{'role': 'user', 'content': content}],
            options={'temperature': 0.3,
                     'num_predict' : 300}  # Slightly higher temperature for more natural responses
        )

        if response and response['message']['content']:
            return response['message']['content']
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I apologize, but I encountered an error while generating a response to your query."

    return "I wasn't able to generate a proper response. Please try rephrasing your question."


def process_parks_query(user_query: str, es_host, es_username, es_password) -> str:
    """Main function to process a user query end-to-end"""
    print(f"Processing query: {user_query}")

    # Step 1: Extract search parameters
    search_params = extract_search_parameters(user_query)
    if not search_params:
        print("Search parameter extraction failed; falling back to a direct query search.")
        search_params = {
            'context_search': user_query,
            'distance_km': 100,
            'location_type': None,
            'reference_location': None,
            'relevant_parks': []
        }

    print(f"Extracted parameters: {search_params}")

    # Step 2: Execute searches across relevant parks
    search_results = search_parks_elasticsearch(search_params, es_host, es_username, es_password)

    # Step 3: Generate final response
    final_response = generate_response(user_query, search_results, search_params)

    return final_response, search_results

