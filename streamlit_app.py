import streamlit as st
import os
from PIL import Image
from typing import List, Dict, Any
from LLM_conversation import process_parks_query
from dotenv import load_dotenv




# Page configuration
st.set_page_config(
    page_title="National Parks Activity Finder",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .park-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .park-title {
        font-size: 18px;
        font-weight: bold;
        color: #2e7d32;
        margin-bottom: 10px;
    }
    .park-description {
        font-size: 14px;
        color: #555;
        line-height: 1.4;
    }
    .response-box {
        background-color: #f0f8f0;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)


def load_image_safe(image_path: str) -> Image.Image:
    """Safely load an image with fallback to placeholder"""
    try:
        if os.path.exists(image_path):
            return Image.open(image_path)
        else:
            # Create a placeholder image if file doesn't exist
            placeholder = Image.new('RGB', (300, 200), color='lightgray')
            return placeholder
    except Exception as e:
        st.error(f"Error loading image {image_path}: {e}")
        placeholder = Image.new('RGB', (300, 200), color='lightgray')
        return placeholder


def format_park_name(park_id: str) -> str:
    """Convert park_id to readable format"""
    return park_id.replace('_', ' ').title()


def display_search_results(search_results: List[Dict[str, Any]], images_dir:str):
    """Display search results in a grid layout with 3 images per row"""
    if not search_results:
        st.info("No images found for your search.")
        return

    st.subheader("🏞️ Recommended Locations")


    # Group results by park_id to avoid duplicates
    park_results = {}
    for result in search_results:
        park_id = result['park_id']
        if park_id not in park_results:
            park_results[park_id] = result

    # Display in rows of 3
    for i in range(0, len(search_results), 3):
        cols = st.columns(3)

        for j, col in enumerate(cols):
            if i + j < len(search_results):
                result = search_results[i + j]

                with col:
                    # Get data from result
                    park_id = result['park_id']
                    image_filename = result['image_filename']
                    description = result['generated_description']
                    title = result['image_filename']

                    # Format park name
                    park_name = format_park_name(park_id)

                    # Image path
                    image_path = os.path.join(images_dir, image_filename)

                    # Load and display image
                    image = load_image_safe(image_path)
                    st.image(image_path, use_container_width=True, caption=title)

                    # Display park information
                    st.markdown(f"""
                    <div class="park-card">
                        <div class="park-title">{park_name}</div>
                        <div class="park-description">{description}</div>
                    </div>
                    """, unsafe_allow_html=True)


def main():
    load_dotenv()
    es_host = os.getenv('ES_HOST')
    es_username = os.getenv('ES_USERNAME')
    es_password = os.getenv('ES_PASSWORD')
    index = os.getenv('ES_INDEX')
    images_dir = os.getenv('IMAGES_DIR', 'images_metadata')
    # App header
    st.title("🏔️ National Parks Activity Finder")
    st.markdown("Discover amazing activities and locations in America's National Parks!")

    # Initialize session state
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'llm_response' not in st.session_state:
        st.session_state.llm_response = ""

    # Search section
    st.markdown("### 🔍 Search for Activities")

    # Search input
    query = st.text_input(
        label="Search Query",
        placeholder="Ask your question here:",
        key="search_input",
        label_visibility="collapsed"
    )

    # Search button
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        search_button = st.button("🔍 Search", type="primary")
    with col2:
        clear_button = st.button("🗑️ Clear")

    # Clear functionality
    if clear_button:
        st.session_state.search_results = []
        st.session_state.llm_response = ""
        st.rerun()

    # Process search
    if search_button and query.strip():
        with st.spinner("Searching national parks..."):
            try:

                response, search_results = process_parks_query(query, es_host, es_username, es_password)

                st.session_state.llm_response = response
                st.session_state.search_results = search_results

                st.success("Search completed!")

            except Exception as e:
                st.error(f"An error occurred during search: {str(e)}")

    elif search_button and not query.strip():
        st.warning("Please enter a search query.")

    # Display LLM response
    if st.session_state.llm_response:
        st.markdown("### 🤖 AI Assistant Response")
        st.markdown(f"""
        <div style="background-color: #f8f9fa; color: #212529; padding: 20px; border-radius: 10px; border: 1px solid #dee2e6; margin: 10px 0;">
        {st.session_state.llm_response}
        </div>
        """, unsafe_allow_html=True)

    # Display search results
    if st.session_state.search_results:
        display_search_results(st.session_state.search_results, images_dir)

    # Sidebar with app information
    with st.sidebar:
        st.markdown("### About")
        st.markdown("""
        This app helps you discover activities in America's National Parks using AI-powered search.

        **How to use:**
        1. Enter your question in the search box
        2. Click Search to get recommendations
        3. Browse the suggested locations and activities

        **Example queries:**
        - "Where can I hike in Utah?"
        - "Dog-friendly trails near Boston"
        - "Best camping spots in Wyoming"
        - "Photography locations in national parks"
        """)

        st.markdown("### 📊 Statistics")
        if st.session_state.search_results:
            st.metric("Results Found", len(st.session_state.search_results))
            unique_parks = len(set(r['park_id'] for r in st.session_state.search_results))
            st.metric("Parks Covered", unique_parks)


if __name__ == "__main__":
    main()