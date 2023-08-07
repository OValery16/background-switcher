import PIL
import numpy as np
import streamlit as st
from PIL.Image import Image
from image_inpainting import InpaintModel
from rembg import new_session, remove
from typing import Tuple


@st.cache_resource
def initialize_models(model_name: str = "u2net"):
    session = new_session(model_name)
    model = InpaintModel()
    model.init_model()
    return model, session


def main():
    st.header("Background switcher")
    progress = 0
    model, session = initialize_models()

    with st.sidebar:
        st.title('Select two images and switch their background')
        left_upload_column, right_upload_column = st.columns(2)
        with left_upload_column:
            image1_path = st.file_uploader(
                "Upload your image 1", type=["jpg", "png", "jpeg"])

        with right_upload_column:
            image2_path = st.file_uploader(
                "Upload your image 2", type=["jpg", "png", "jpeg"])

        progress_text = "Operation in progress. Please wait."
        progress_bar = st.progress(progress, text="")

        st.write(
            'Made with ❤️ by [Olivier VALERY](https://github.com/OValery16)')

    st.write("""
    The goal of this challenge is to build a web app that enables
    users to seamlessly switch the backgrounds of two input images.
    The app is able to handle images of different sizes and aspect
    ratios. All the computations are performed on the CPU. We proceed
    as follows:
    - We use a pre-trained model to remove the background of each image.
    - We use a pre-trained model to inpaint the missing part of the
    background for each image.
    - We paste the inpainted background of each image on the other image.
    """)

    if image1_path is not None and image2_path is not None:

        image1, image2 = load_images(image1_path, image2_path)

        display_images(image1, image2)

        image1, image2, foreground1, foreground2 = remove_backgrounds(
            session, image1_path, image2_path, progress_text, progress_bar)

        full_backgroung1, full_backgroung2 = inpaint_images(
            model, progress_text, progress_bar, image1, image2, foreground1,
            foreground2)

        foreground1_background2, foreground2_background1 = change_backgrounds(
            progress_text, progress_bar, foreground1, foreground2,
            full_backgroung1, full_backgroung2)

        display_images(foreground1_background2, foreground2_background1)


def display_images(image1: Image, image2: Image):
    left_column, right_column = st.columns(2)
    left_column.image(image1, caption='Image 1')
    right_column.image(image2, caption='Image 2')


def change_backgrounds(
        progress_text: str, progress_bar: st.progress, foreground1: Image,
        foreground2: Image, full_backgroung1: np.ndarray,
        full_backgroung2: np.ndarray, paste_position=(0, 0)) -> Tuple[
            np.ndarray, np.ndarray]:

    foreground1_background2 = change_background(full_backgroung2, foreground1,
                                                paste_position)
    progress_bar.progress(95, text=progress_text)
    foreground2_background1 = change_background(full_backgroung1, foreground2,
                                                paste_position)
    progress_bar.progress(100, text="")
    return foreground1_background2, foreground2_background1


def inpaint_images(model: InpaintModel, progress_text: str,
                   progress_bar: st.progress, image1: Image, image2: Image,
                   foreground1: Image, foreground2: Image) -> Tuple[
                    np.ndarray, np.ndarray]:
    full_backgroung1 = model.inpaint(
        image1, foreground1, output_size=foreground2.shape[:2])
    progress_bar.progress(55, text=progress_text)
    full_backgroung2 = model.inpaint(
        image2, foreground2, output_size=foreground1.shape[:2])
    progress_bar.progress(90, text=progress_text)
    return full_backgroung1, full_backgroung2


def remove_backgrounds(session: object, image1_path: str, image2_path: str,
                       progress_text: str, progress_bar: st.progress) -> Tuple[
                        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image1 = np.array(PIL.Image.open(image1_path))
    image2 = np.array(PIL.Image.open(image2_path))

    foreground1 = remove(image1, session=session)
    progress_bar.progress(10, text=progress_text)
    foreground2 = remove(image2, session=session)
    progress_bar.progress(20, text=progress_text)
    return image1, image2, foreground1, foreground2


def load_images(image1_path: str, image2_path: str) -> Tuple[Image, Image]:
    image1 = PIL.Image.open(image1_path)
    image2 = PIL.Image.open(image2_path)
    return image1, image2


def change_background(backgroung: np.ndarray, foreground: Image,
                      paste_position: Tuple[int, int]) -> Image:
    new_image = PIL.Image.fromarray(backgroung)
    foreground = PIL.Image.fromarray(foreground)
    new_image.paste(foreground, paste_position, foreground)
    return new_image


if __name__ == '__main__':
    main()
