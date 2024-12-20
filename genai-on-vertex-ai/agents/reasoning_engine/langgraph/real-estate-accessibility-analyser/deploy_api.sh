#!/usr/bin/env bash
set -euo pipefail

# Set your GCP project and location
PROJECT_ID="<your-project-id>"
LOCATION="<your-location>"

# 2. Build and deploy the API image/service
# Temporarily rename Dockerfile.api to Dockerfile
mv Dockerfile.api Dockerfile

# Build and push the image
gcloud builds submit --tag gcr.io/${PROJECT_ID}/api-image:latest .

# Restore the original Dockerfile name
mv Dockerfile Dockerfile.api

# Deploy the API service to Cloud Run
gcloud run deploy real-estate-api \
  --image gcr.io/${PROJECT_ID}/api-image:latest \
  --platform managed \
  --region ${LOCATION} \
  --allow-unauthenticated
