import logging
from src.data_generation.house_dataset_generator import HouseDatasetGenerator

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    generator = HouseDatasetGenerator()
    generator.generate_dataset(num_accessible=5, num_standard=5)
    logging.info("Dataset generation completed.")