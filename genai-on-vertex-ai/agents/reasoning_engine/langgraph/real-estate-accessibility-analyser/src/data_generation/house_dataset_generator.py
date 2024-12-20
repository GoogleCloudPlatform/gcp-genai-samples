import os
import json
import time
import logging
from datetime import datetime
from typing import Optional, Dict, List
import shutil

import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting
from vertexai.preview.vision_models import ImageGenerationModel
from google.cloud import storage, firestore

from src.retry import RetryError, retry_with_exponential_backoff
from src.config import PROJECT_ID, LOCATION, GCS_BUCKET, FIRESTORE_COLLECTION

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class HouseDatasetGenerator:
    HOUSE_SCHEMA = {
        "type": "OBJECT",
        "properties": {
            "title": {"type": "STRING"},
            "description": {"type": "STRING"},
        },
        "required": ["title", "description"]
    }

    ROOM_PROMPTS_SCHEMA = {
        "type": "OBJECT",
        "properties": {
            "prompts": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            }
        },
        "required": ["prompts"]
    }

    def __init__(self, base_output_dir: str = "house_dataset", max_retries: int = 3, retry_delay: int = 2):
        """
        Initialize the HouseDatasetGenerator.

        Args:
            base_output_dir (str): Directory where generated data will be stored.
            max_retries (int): Max number of retries for generation API calls.
            retry_delay (int): Delay between retries in seconds.
        """
        self.project_id = PROJECT_ID
        self.location = LOCATION
        self.base_output_dir = base_output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        vertexai.init(project=self.project_id, location=self.location)
        self.generation_config = {
            "max_output_tokens": 8192,
            "temperature": 0.1,
            "top_p": 0.95,
            "response_mime_type": "application/json",
        }

        self.safety_settings = [
            SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=SafetySetting.HarmBlockThreshold.OFF),
            SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=SafetySetting.HarmBlockThreshold.OFF),
            SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=SafetySetting.HarmBlockThreshold.OFF),
            SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=SafetySetting.HarmBlockThreshold.OFF),
        ]

        self.text_model = GenerativeModel("gemini-1.5-flash-002")
        self.image_model = ImageGenerationModel.from_pretrained("imagen-3.0-fast-generate-001")

        os.makedirs(base_output_dir, exist_ok=True)
        self.storage_client = storage.Client(project=self.project_id)
        self.db = firestore.Client(project=self.project_id)
        self.bucket = self._ensure_bucket_exists(GCS_BUCKET)

    def _ensure_bucket_exists(self, bucket_name: str) -> storage.Bucket:
        """
        Ensure the specified GCS bucket exists, create it if not.
        """
        bucket = self.storage_client.bucket(bucket_name)
        if not bucket.exists():
            logger.info(f"Creating bucket: {bucket_name}")
            bucket.create(location=self.location)
        return bucket

    def _get_house_description_prompt(self, is_accessible: bool) -> str:
        """
        Create a prompt to generate a house title and description.

        If is_accessible=True, the house is wheelchair accessible and should
        mention accessibility features. Otherwise, mention that the house may
        have obstacles like stairs or clutter that impede accessibility.
        """
        if is_accessible:
            return """
            Generate a title and a description for a wheelchair accessible vacation rental house.
            The description should highlight accessibility features such as ramps, wide doorways,
            and grab bars, ensuring the property is suitable for wheelchair users.
            Return a JSON object with 'title' and 'description' fields.
            """
        else:
            return """
            Generate a title and a description for a standard (not wheelchair accessible) vacation rental house.
            The description should mention obstacles such as stairs, narrow doorways, or cluttered areas
            that might make it challenging for wheelchair users. Never mention too obviously that the house is
            not wheelchair accessible. It should be understood from context.
            Return a JSON object with 'title' and 'description' fields.
            """

    def _get_room_prompts_prompt(self, house_desc: Dict, is_accessible: bool) -> str:
        """
        Create a prompt to generate image prompts for the rooms:
        - Living room
        - Kitchen
        - Main bathroom
        - Master bedroom

        If is_accessible=True, each room prompt should highlight accessible features.
        If not, each room prompt should highlight barriers to accessibility.
        Return a JSON with 'prompts' as a list of image prompt strings.
        """
        if is_accessible:
            room_condition = "Highlight accessibility features (ramps, wide doorways, grab bars)"
        else:
            room_condition = "Mention obstacles such as stairs, narrow doorways, or clutter"

        return f"""
        The house description:
        {json.dumps(house_desc, indent=2)}

        Generate detailed image prompts for the following rooms:
        - Living room
        - Kitchen
        - Main bathroom
        - Master bedroom

        {room_condition} in each prompt.

        Focus on describing the room's style, lighting, materials, and layout.
        Return the results as a JSON object with a 'prompts' field containing a list of 4 prompt strings.
        """

    @retry_with_exponential_backoff
    def _generate_house_description_impl(self, is_accessible: bool) -> Dict:
        """
        Implementation of generating house description.
        """
        config = self.generation_config.copy()
        config["response_schema"] = self.HOUSE_SCHEMA
        prompt = self._get_house_description_prompt(is_accessible)

        logger.info("Generating house description...")
        response = self.text_model.generate_content(
            prompt,
            generation_config=config,
            safety_settings=self.safety_settings
        )
        return json.loads(response.text)

    def generate_house_description(self, is_accessible: bool) -> Optional[Dict]:
        """
        Generate house description data.

        Args:
            is_accessible (bool): Whether the house is wheelchair accessible.

        Returns:
            Optional[Dict]: {'title': str, 'description': str} or None on failure.
        """
        try:
            result = self._generate_house_description_impl(is_accessible)
            # Add is_accessible field programmatically
            result["is_accessible"] = is_accessible
            return result
        except RetryError as e:
            logger.error(f"Failed to generate house description: {e}")
            return None

    @retry_with_exponential_backoff
    def _generate_room_prompts_impl(self, house_desc: Dict, is_accessible: bool) -> List[str]:
        """
        Implementation of generating room prompts.
        """
        config = self.generation_config.copy()
        config["response_schema"] = self.ROOM_PROMPTS_SCHEMA
        prompt = self._get_room_prompts_prompt(house_desc, is_accessible)

        logger.info("Generating room prompts...")
        response = self.text_model.generate_content(
            prompt,
            generation_config=config,
            safety_settings=self.safety_settings
        )
        result = json.loads(response.text)
        return result["prompts"]

    def generate_room_prompts(self, house_desc: Dict, is_accessible: bool) -> Optional[List[str]]:
        """
        Generate image prompts for the specified rooms.

        Args:
            house_desc (Dict): The house description dictionary.
            is_accessible (bool): Whether the house is accessible.

        Returns:
            Optional[List[str]]: A list of prompts or None on failure.
        """
        try:
            return self._generate_room_prompts_impl(house_desc, is_accessible)
        except RetryError as e:
            logger.error(f"Failed to generate room prompts: {e}")
            return None

    def generate_and_save_images(self, prompts: List[str], house_dir: str) -> List[str]:
        """
        Generate images for each prompt and save them to GCS.

        Args:
            prompts (List[str]): The image generation prompts.
            house_dir (str): The local directory for saving images.

        Returns:
            List[str]: A list of GCS URLs for the generated images.
        """
        if not prompts:
            logger.warning("No prompts for image generation.")
            return []

        images_dir = os.path.join(house_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        image_urls = []

        for idx, prompt in enumerate(prompts):
            img_name = f"room_{idx + 1}.png"
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"Generating image {idx+1}, attempt {attempt+1}...")
                    images = self.image_model.generate_images(
                        prompt=prompt,
                        number_of_images=1,
                        aspect_ratio="4:3",
                        safety_filter_level="block_some",
                    )
                    image_path = os.path.join(images_dir, img_name)
                    images[0].save(image_path)

                    # Upload image
                    blob = self.bucket.blob(f"{house_dir}/{img_name}")
                    blob.upload_from_filename(image_path)
                    gcs_url = f"gs://{GCS_BUCKET}/{house_dir}/{img_name}"
                    image_urls.append(gcs_url)

                    # Save prompt
                    prompt_path = os.path.join(images_dir, f"room_{idx + 1}_prompt.txt")
                    with open(prompt_path, "w") as f:
                        f.write(prompt)

                    prompt_blob = self.bucket.blob(f"{house_dir}/room_{idx + 1}_prompt.txt")
                    prompt_blob.upload_from_filename(prompt_path)
                    break
                except Exception as e:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.error(f"Error generating image {idx+1}, attempt {attempt+1}: {e}")
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
            else:
                logger.error(f"Failed to generate image {idx + 1} after retries.")
        return image_urls

    def save_to_firestore(self, description: Dict, image_urls: List[str]):
        """
        Save the generated house data to Firestore.

        Args:
            description (Dict): The house description dictionary containing title, description, and is_accessible.
            image_urls (List[str]): The GCS URLs of the generated room images.
        """
        logger.info("Saving house data to Firestore...")
        doc_ref = self.db.collection(FIRESTORE_COLLECTION).document()
        house_data = {
            "title": description["title"],
            "description": description["description"],
            "is_accessible": description["is_accessible"],
            "image_urls": image_urls
        }
        doc_ref.set(house_data)

    def generate_dataset(self, num_accessible: int = 2, num_standard: int = 3):
        """
        Generate a dataset of houses. Accessible houses highlight accessibility features,
        while standard houses highlight obstacles.

        Args:
            num_accessible (int): Number of accessible houses to generate.
            num_standard (int): Number of standard (not accessible) houses to generate.
        """
        # Accessible houses
        for i in range(num_accessible):
            house_dir_name = f"accessible_house_{i + 1}"
            full_house_dir = os.path.join(self.base_output_dir, house_dir_name)
            os.makedirs(full_house_dir, exist_ok=True)

            logger.info(f"Generating accessible house {i+1}...")
            description = self.generate_house_description(True)
            if description is None:
                logger.error("Skipping due to description failure.")
                continue

            desc_path = os.path.join(full_house_dir, "description.json")
            with open(desc_path, "w") as f:
                json.dump(description, f, indent=2)

            logger.info("Generating images...")
            prompts = self.generate_room_prompts(description, True)
            if prompts:
                image_urls = self.generate_and_save_images(prompts, full_house_dir)
                self.save_to_firestore(description, image_urls)
            else:
                logger.error("Skipping image generation due to prompt failure.")

        # Standard (not accessible) houses
        for i in range(num_standard):
            house_dir_name = f"standard_house_{i + 1}"
            full_house_dir = os.path.join(self.base_output_dir, house_dir_name)
            os.makedirs(full_house_dir, exist_ok=True)

            logger.info(f"Generating standard (not accessible) house {i+1}...")
            description = self.generate_house_description(False)
            if description is None:
                logger.error("Skipping due to description failure.")
                continue

            desc_path = os.path.join(full_house_dir, "description.json")
            with open(desc_path, "w") as f:
                json.dump(description, f, indent=2)

            logger.info("Generating images...")
            prompts = self.generate_room_prompts(description, False)
            if prompts:
                image_urls = self.generate_and_save_images(prompts, full_house_dir)
                self.save_to_firestore(description, image_urls)
            else:
                logger.error("Skipping image generation due to prompt failure.")

        # Delete self.base_output_dir and all its files
        logger.info(f"Deleting output directory: {self.base_output_dir}")
        shutil.rmtree(self.base_output_dir)