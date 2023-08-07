from typing import Any
import cv2
import numpy as np
import openpyxl
import os
import requests


class ImageSample:
    # class to download the sample images from the XLSX file
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.image_dict = {}
        os.makedirs("data/initial_images", exist_ok=True)
        os.makedirs("data/final_images", exist_ok=True)

    def read_xlsx_file(self):
        try:
            workbook = openpyxl.load_workbook(self.file_path)
            sheet = workbook.active

            for row in sheet.iter_rows(min_row=2, values_only=True):
                key = row[0]
                initial_image_url = row[3]
                final_image_url = row[4]

                if not key or not initial_image_url or not final_image_url:
                    continue

                self.image_dict[key] = {
                    'initial_image': self.download_image(
                        initial_image_url, "data/initial_images/"),
                    'final_image': self.download_image(
                        final_image_url, "data/final_images/")
                }

        except Exception as e:
            print(f"Error while reading the XLSX file: {e}")

    def download_image(self, url: str, image_directory: str) -> Any:
        image_path = image_directory + os.path.basename(url)
        if os.path.exists(image_path):
            return self.read_image(image_path)
        try:
            response = requests.get(url)
            if response.status_code == 200:

                with open(image_path, 'wb') as f:
                    f.write(response.content)

                print(f"Downloaded: {url} -> {image_path}")
                return response.content
            else:
                print(
                    f"Failed to download: {url})")
                return None

        except Exception as e:
            print(f"Error while downloading {url}: {e}")
            return None

    def read_image(self, image_path: str) -> np.ndarray:
        try:
            image = cv2.imread(image_path)
            return image
        except Exception as e:
            print(f"Error while reading the image: {e}")
            return None

    def get_image_dict(self):
        return self.image_dict
