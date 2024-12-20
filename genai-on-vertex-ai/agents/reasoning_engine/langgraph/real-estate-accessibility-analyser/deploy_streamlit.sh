#!/usr/bin/env bash
set -euo pipefail

# Set your GCP project and location
PROJECT_ID="<your-project-id>"
LOCATION="<your-location>"

# 1. Build and deploy the Streamlit image/service
# Temporarily rename Dockerfile.streamlit to Dockerfile
mv Dockerfile.streamlit Dockerfile

# Build and push the image
gcloud builds submit --tag gcr.io/${PROJECT_ID}/streamlit-image:latest .

# Restore the original Dockerfile name
mv Dockerfile Dockerfile.streamlit

# Deploy the Streamlit service to Cloud Run
gcloud run deploy real-estate-streamlit \
  --image gcr.io/${PROJECT_ID}/streamlit-image:latest \
  --platform managed \
  --region ${LOCATION} \
  --allow-unauthenticated