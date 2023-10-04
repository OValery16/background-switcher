import cv2
import numpy as np
import torch
from helper import get_model_weight, norm_img, load_jit_model, resize_max_size, resize_min_size
from typing import Tuple


class InpaintModel:
    name = "big-lama"
    pad_mod = 8
    max_size = 512

    def init_model(self) -> None:
        filename = get_model_weight(self.name)
        self.model = load_jit_model(filename).eval()

    def prepare_data(
            self, original_image: np.ndarray,
            image_without_background: np.ndarray,
            output_size: tuple = (512, 512)
            ) -> Tuple[np.ndarray, np.ndarray, tuple, tuple]:
        """
        Build input for the network:
        - Build binary mask from alpha channel
        - Adjust input size to the final target size (pad/crop)
        - Dilate mask
        - Limit input size to the maximum authorized size
        """

        binary_mask = self.get_binary_mask(image_without_background)

        (original_image,
         binary_mask, original_size) = self.adjust_to_target_size(
            original_image, output_size, binary_mask)

        binary_mask = self.dilate_mask(binary_mask)

        (original_image,
         binary_mask, resized_image_size) = self.adjust_network_input_size(
            original_image, binary_mask)

        return original_image, binary_mask, resized_image_size, original_size

    def get_binary_mask(self,
                        image_without_background: np.ndarray) -> np.ndarray:
        alpha = image_without_background[:, :, 3]
        binary_mask = np.zeros(alpha.shape[:2])

        binary_mask[alpha > 0] = 255
        return binary_mask

    def adjust_network_input_size(self, original_image: np.ndarray,
                                  binary_mask: np.ndarray) -> Tuple[
                                      np.ndarray, np.ndarray, tuple]:
        # limit inputs size to max_size (512x512 by default)
        original_image, resized_image_size = resize_max_size(
            original_image, self.max_size)
        binary_mask, _ = resize_max_size(binary_mask, self.max_size)

        # pad image to be divisible by 8
        height, width = resized_image_size
        padding = ((self.pad_mod - height % self.pad_mod) % self.pad_mod,
                   (self.pad_mod - width % self.pad_mod) % self.pad_mod)

        original_image = cv2.copyMakeBorder(
            original_image, 0, padding[0], 0, padding[1], cv2.BORDER_CONSTANT,
            value=0)
        binary_mask = cv2.copyMakeBorder(
            binary_mask, 0, padding[0], 0, padding[1], cv2.BORDER_CONSTANT,
            value=255)
        return original_image, binary_mask, resized_image_size

    def dilate_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        kernel = np.ones((7, 7), np.uint8)
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=20)
        return binary_mask

    def adjust_to_target_size(self, original_image: np.ndarray,
                              output_size: tuple,
                              binary_mask: np.ndarray) -> Tuple[
                                  np.ndarray, np.ndarray, tuple]:
        """
        Modify the mask/image (padding/cropping) to adjust the final inpainted
        image size to the target size.
        """

        # resize inputs to min output size
        binary_mask, _= resize_min_size(binary_mask, min(output_size))
        original_image, _= resize_min_size(original_image, min(output_size))

        original_image, binary_mask, original_size = self.pad_to_target_size(
            original_image, output_size, binary_mask)

        original_image, binary_mask, original_size = self.crop_to_target_size(
            original_image, output_size, binary_mask, original_size)
        return original_image, binary_mask, original_size

    def crop_to_target_size(
            self, original_image: np.ndarray, output_size: tuple,
            binary_mask: np.ndarray, original_size: tuple) -> Tuple[
                np.ndarray, np.ndarray, tuple]:
        cropping_needed = (max(0, original_image.shape[0]-output_size[0]),
                           max(0, original_image.shape[1]-output_size[1]))
        crop = (
            (cropping_needed[0] // 2,
             original_size[0]-(cropping_needed[0] - cropping_needed[0] // 2)),
            (cropping_needed[1] // 2,
             original_size[1]-(cropping_needed[1] - cropping_needed[1] // 2)))
        original_image = original_image[crop[0][0]:crop[0][1],
                                        crop[1][0]:crop[1][1]]
        binary_mask = binary_mask[crop[0][0]:crop[0][1],
                                  crop[1][0]:crop[1][1]]
        original_size = original_image.shape[:2]
        return original_image, binary_mask, original_size

    def pad_to_target_size(self, original_image: np.ndarray,
                           output_size: tuple,
                           binary_mask: np.ndarray) -> Tuple[
                               np.ndarray, np.ndarray, tuple]:
        padding_needed = (max(0, output_size[0] - original_image.shape[0]),
                          max(0, output_size[1] - original_image.shape[1]))
        padding = (
            (padding_needed[0] // 2,
             padding_needed[0] - padding_needed[0] // 2),
            (padding_needed[1] // 2,
             padding_needed[1] - padding_needed[1] // 2))
        original_image = cv2.copyMakeBorder(
            original_image, padding[0][0], padding[0][1], padding[1][0],
            padding[1][1], cv2.BORDER_CONSTANT, value=0)
        binary_mask = cv2.copyMakeBorder(
            binary_mask, padding[0][0], padding[0][1], padding[1][0],
            padding[1][1], cv2.BORDER_CONSTANT, value=255)
        original_size = original_image.shape[:2]
        return original_image, binary_mask, original_size

    def inpaint(self, original_image: np.ndarray,
                image_without_background: np.ndarray,
                output_size: tuple = (512, 512)) -> np.ndarray:

        (original_image,
         binary_mask, resized_image_size, original_size) = self.prepare_data(
            original_image, image_without_background, output_size)

        result = self.forward(original_image, binary_mask)

        result = result[:resized_image_size[0], :resized_image_size[1], :]

        if resized_image_size != original_size:
            result = cv2.resize(result, original_size[::-1],
                                interpolation=cv2.INTER_CUBIC)

        return result

    def forward(self, image, mask):

        image = norm_img(image)
        mask = norm_img(mask)

        mask = (mask > 0) * 1

        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        with torch.no_grad():
            output = self.model(image, mask)

        inpainted_image = output[0].permute(1, 2, 0).detach().cpu().numpy()
        inpainted_image = np.clip(
            inpainted_image * 255, 0, 255).astype("uint8")
        return inpainted_image
