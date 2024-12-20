import streamlit as st
from vertexai.preview import reasoning_engines
from src.config import AGENT_NAME
from pydantic import BaseModel
from typing import List
import requests
from src.config import API_URL, GCS_BUCKET
from google.cloud import storage
from io import BytesIO
from PIL import Image

storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET)

def get_blob(gcs_url: str) -> (str, str):
    """
    Extracts the bucket name and blob path from a Google Cloud Storage URI.
    """
    if not gcs_url.startswith("gs://"):
        raise ValueError("URL must start with gs://")

    no_gs = gcs_url[len("gs://"):]
    parts = no_gs.split("/", 1)
    bucket = parts[0]
    blob = parts[1] if len(parts) > 1 else ""
    return bucket, blob

class House(BaseModel):
    id: str
    title: str
    description: str
    image_urls: List[str]
    is_accessible: bool

# Create or load the reasoning engine instance.
remote_agent = reasoning_engines.ReasoningEngine(AGENT_NAME)

st.set_page_config(page_title="Accessibility Analysis Assistant", page_icon="🤖")
st.title("Accessibility Analysis Assistant")

tab1, tab2 = st.tabs(["Property Search", "Chat"])

with tab1:
    st.write("## Property Accessibility Check")
    property_id = st.text_input("Enter Property ID:", "")

    if st.button("Fetch Property Details"):
        if property_id:
            # Fetch property info from the API
            url = f"{API_URL}/houses/{property_id}"
            resp = requests.get(url)
            if resp.status_code == 200:
                house_data = resp.json()
                # Parse into House model
                try:
                    house = House(**house_data)
                    st.write(f"**Title:** {house.title}")
                    st.write(f"**Accessibility:** {'Yes' if house.is_accessible else 'No'}")
                    st.write(f"**Description:** {house.description}")

                    # Display images if available
                    if house.image_urls:
                        st.write("**Images:**")
                        # Create one column per image to place them side-by-side
                        columns = st.columns(len(house.image_urls))
                        for col, img_url in zip(columns, house.image_urls):
                            _, blob_name = get_blob(img_url)
                            blob = bucket.blob(blob_name)
                            image_data = blob.download_as_bytes()
                            image = Image.open(BytesIO(image_data))

                            # Resize image to 1/4 original size
                            new_width = max(1, image.width // 4)
                            new_height = max(1, image.height // 4)
                            resized_image = image.resize((new_width, new_height))

                            col.image(resized_image)
                    else:
                        st.write("No images available.")

                except Exception as e:
                    st.error(f"Error parsing house data: {e}")
            else:
                st.error("Failed to fetch property details. Check the property ID or try again later.")

with tab2:
    # Displaying the conversation UI in the Chat tab
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    user_input = st.chat_input("Ask about accessibility, e.g., 'Check accessibility of property <id>'")
    if user_input:
        # Add user's message to session state and display it
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Send the entire conversation to the reasoning engine
        conversation = st.session_state["messages"]
        assistant_response = remote_agent.query(conversation=conversation)

        # Add assistant's response to session state and display it
        st.session_state["messages"].append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)


