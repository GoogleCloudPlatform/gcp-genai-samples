# Agentic Accessibility Checker Project

This project showcases an end-to-end system that leverages Google's cutting-edge AI tools to create an "agentic" solution for real estate accessibility analysis.

It includes synthetic data generation, a cloud-based API, a sophisticated LangGraph agent powered by Gemini and deployed using Vertex AI Reasoning Engine, and a user-friendly streamlit interface. The project's core is its demonstration of the power of the Vertex AI Reasoning Engine and its seamless integration with LangGraph, highlighting how these technologies can build powerful, autonomous agents.

## Key Features

*   **Synthetic Data Generation:**
    *   Utilizes Gemini and Imagen3's generative capabilities to produce realistic, multi-modal real estate data, including property titles, descriptions, and room images.
    *   Generates data for both accessible (wheelchair-friendly) and standard properties, allowing the agent to handle a varied environment.
    *   Focuses on capturing realistic nuances in the descriptions that the agent can then use to understand accessibility.
    *   Stores images in Google Cloud Storage (GCS) and metadata in Firestore, creating a structured dataset for agent interactions.
*   **Cloud Run API - A Simulated Environment for the Agent:**
    *   Deploys a FastAPI-based API on Cloud Run, acting as a "real world" system for the agent to interact with.
    *   Provides endpoints for the agent to retrieve property details, enhancing its capability to gather information and make decisions.
*   **LangGraph Agent Powered by Gemini and the Reasoning Engine:**
    *   Employs Gemini-1.5 Pro as the agent's "brain", facilitating reasoning and complex decision-making capabilities.
    *   Implements a ReAct-style agent, enabling the agent to take actions and leverage tools based on the current context.
    *   Uses tools for fetching data from the API, analyzing text and images, and even drafting emails—creating a fully agentic workflow.
    *   Demonstrates the power of the Vertex AI Reasoning Engine for managed and scalable agent deployments.
*   **Streamlit Application - Agent Interaction Layer:**
    *   Offers a user-friendly interface built with Streamlit for seamless interaction with the agent.
    *   Provides both a property viewing and a chat interface, where users can engage with the agent for accessibility inquiries.
    *   Users can experience the agent's reasoning capabilities firsthand, showcasing how it makes decisions and interacts with the system.
*   **Vertex AI Reasoning Engine - Seamless LangGraph Deployment:**
    *   Demonstrates the capability of the Vertex AI Reasoning Engine for deploying complex LangGraph agents in a scalable and managed way.
    *   The agent is available as a cloud resource that can be used by external client applications and other Vertex AI services.
*   **Gemini Multi-Modal Capabilities - Enhanced Agent Perception:**
    *   Leverages Gemini's ability to process both text and images, providing a robust perception layer for the agent.
    *   The agent can analyze both property descriptions and images for accessibility features and issues, showcasing its ability to leverage multi-modal data.

## Getting Started

Follow these steps to set up and run the project:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
This command will install all necessary libraries specified in `requirements.txt`

### 2. Generate Synthetic Data
```bash
python -m src.main
```
This command will run the `src/main.py` script, which uses `src/data_generation/house_dataset_generator.py` to create your multi-modal real estate dataset.

### 3. Deploy the API
```bash
./deploy_api.sh
```
This script builds and deploys the FastAPI application (`src/api/app.py`) to Google Cloud Run. The API provides the agent with a means to retrieve property details.

### 4. Build and Deploy the Agent using the Reasoning Engine
```bash
python -m src.agent.agent
```
This command runs `src/agent/agent.py`, which creates and deploys the agent as a Vertex AI Reasoning Engine. This demonstrates how easily complex agents can be managed using the Reasoning Engine.

**Important:** After deployment, you need to edit `src/config.py` and set the correct `AGENT_NAME` with the name of the deployed Reasoning Engine. The engine name will be printed to the console upon successful execution of the previous step. The `AGENT_NAME` should be in the format: `projects/<PROJECT_ID>/locations/<LOCATION>/reasoningEngines/<ENGINE_ID>`.

### 5. Deploy the Streamlit Application
```bash
./deploy_streamlit.sh
```
This script builds and deploys the Streamlit application (`src/frontend/streamlit_app.py`) to Google Cloud Run, allowing users to interact with the deployed agent.

## Configuration

Key configuration settings can be found in `src/config.py`:

*   `PROJECT_ID`: Your Google Cloud project ID.
*   `LOCATION`: Your Google Cloud project's region (e.g. us-central1).
*   `GCS_BUCKET`: The name of your Google Cloud Storage bucket.
*   `FIRESTORE_COLLECTION`: The Firestore collection name for storing the property data.
*   `API_URL`: The URL for your deployed Cloud Run API.
*   `STREAMLIT_URL`: The URL for your deployed Streamlit application.
*   `STAGING_BUCKET`: The Google Cloud Storage bucket used for staging agent resources.
*   `AGENT_NAME`: The resource name of the deployed Vertex AI Reasoning Engine. This is crucial for connecting the agent to other services.

Adjust these values to match your specific Google Cloud environment.

## Project Structure

```
project/
    requirements.txt        # main requirements file
    requirements_api.txt    # requirements for the API service
    Dockerfile.streamlit    # dockerfile for the streamlit service
    deploy_api.sh           # script to deploy the API
    Dockerfile.api          # dockerfile for the API service
    README.md               # this file
    .dockerignore           # files to ignore during docker builds
    requirements_streamlit.txt # requirements for the streamlit service
    deploy_streamlit.sh     # script to deploy streamlit service
    src/
        config.py           # Configuration settings
        retry.py            # Retry logic for API calls
        main.py             # Main script to generate the dataset
        frontend/
            streamlit_app.py # Streamlit application
        agent/
            agent.py        # LangGraph agent using Vertex AI and Gemini
        api/
            app.py          # FastAPI application
        data_generation/
            house_dataset_generator.py # Generator for synthetic data
```

## Notes

*   Ensure the Google Cloud SDK (gcloud) is installed, configured, and authenticated to your Google Cloud project.
*   Make sure the required Google Cloud APIs are enabled, including Cloud Run, Cloud Storage, Firestore, and Vertex AI.
*   The Agent Engine resource name is in the format `projects/<PROJECT_ID>/locations/<LOCATION>/reasoningEngines/<ID>`. This is required to configure the agent properly.

## Contributing

This project is intended for demonstration and learning. Feel free to fork, contribute, and submit pull requests to make it even better!
This project demonstrates an "agentic" approach to accessibility analysis and how modern AI tools can be used to build practical applications.