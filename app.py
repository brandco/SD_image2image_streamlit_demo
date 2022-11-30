# a demo of how to get a camera feed into a streamlit app

import streamlit as st
import io
import os
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

os.environ["STABILITY_HOST"] = "grpc.stability.ai:443"
STABILITY_KEY = st.secrets.STABILITY_KEY

sampler_dict = {
    "k_dpm_2_ancestral": generation.SAMPLER_K_DPM_2_ANCESTRAL,
    "k_dpm_2": generation.SAMPLER_K_DPM_2,
    "ddim": generation.SAMPLER_DDIM,
    "k_euler": generation.SAMPLER_K_EULER,
    "k_euler_ancestral": generation.SAMPLER_K_EULER_ANCESTRAL,
    "k_heun": generation.SAMPLER_K_HEUN,
    "k_dpmpp_2s_ancestral": generation.SAMPLER_K_LMS,
}

engine_list = [
    "stable-diffusion-v1",
    "stable-diffusion-v1-5",
    "stable-diffusion-512-v2-0",
    "stable-diffusion-768-v2-0",
    "stable-inpainting-v1-0",
    "stable-inpainting-512-v2-0",
]


def crop_to_mul_of_64(image: Image) -> Image:
    """Crops the image to dimensions that are multiple of 64
    This is required for the api, implemented to make the minimal changes to the image size
    """
    h_m = image.height // 64
    w_m = image.width // 64
    final_height = h_m * 64
    final_width = w_m * 64
    h_diff = image.height - final_height
    w_diff = image.width - final_width
    # crop image on all sides to be a multiple of 64
    upper_left = w_diff // 2
    upper_right = h_diff // 2
    lower_right = image.width - w_diff // 2
    lower_left = image.height - h_diff // 2

    image = image.crop(
        (
            upper_left,
            upper_right,
            lower_right,
            lower_left,
        )
    )

    image = image.crop((0, 0, final_width, final_height))
    return image

# Set up our connection to the API.
@st.cache(allow_output_mutation=True)
def get_client(engine="stable-diffusion-v1-5"):
    return client.StabilityInference(
        key=STABILITY_KEY,  # API Key reference.
        verbose=False,  # Print debug messages.
        engine=engine,  
    )

st.title("Stability AI Demo")
st.write("This is a demo of how to use the Stability AI API to generate images from a camera feed or an uploaded image")

# picture input
with st.expander("Camera"):
    camera_picture = st.camera_input("Camera", key="camera")

uploaded_picture = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])


# get the picture from either source and show it on the screen
picture = None

if uploaded_picture:
    picture = uploaded_picture

if camera_picture:
    picture = camera_picture

if picture:
    image = Image.open(io.BytesIO(picture.getvalue()))
    image = crop_to_mul_of_64(image)
    picture = st.image(image, caption="Input Image")

if picture:
    in_btn = st.download_button(
        label="Download input image",
        data=image.tobytes(),
        file_name="input_image.png",
        mime="image/png",
    )

prompt = st.text_input("Prompt: ", "a picture of a dog")

# sidebar for the parameters to the generate api call
with st.sidebar:
    with st.form("parameters"):
        st.title("Parameters")
        start_schedule = st.slider(
            """Start Schedule:
        Set the strength of our prompt in relation to our initial image.""",
            0.0,
            1.0,
            0.6,
        )
        steps = st.slider("Steps", 1, 1000, 50)
        cfg_scale = st.slider(
            """CFG Scale:
        Influences how strongly your generation is guided to match your prompt.""",
            0.0,
            10.0,
            7.0,
        )
        sampler = st.selectbox(
            label="sampler", options=sampler_dict.keys(), index=0, key="sampler"
        )
        engine = st.selectbox(
            label="engine", options=engine_list, index=2, key="engine"
        )
        submit_button = st.form_submit_button(label="Set Parameters")

stability_api = get_client(engine)

if st.button("Generate"):

    img = image
    with st.spinner(text="Generating..."):
        generated = stability_api.generate(
            prompt=prompt,
            init_image=img,
            start_schedule=start_schedule,  # Set the strength of our prompt in relation to our initial image.
            seed=None,  # If attempting to transform an image that was previously generated with our API,
            # initial images benefit from having their own distinct seed rather than using the seed of the original image generation.
            steps=steps,  # Step Count defaults to 50 if not specified here.
            cfg_scale=cfg_scale,  # Influences how strongly your generation is guided to match your prompt.
            # Setting this value higher increases the strength in which it tries to match your prompt.
            # Defaults to 7.0 if not specified.
            width=512,  # Generation width, defaults to 512 if not included.
            height=512,  # Generation height, defaults to 512 if not included.
            sampler=sampler_dict[sampler],
        )

    for resp in generated:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                st.warning(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again."
                )
            if artifact.type == generation.ARTIFACT_IMAGE:
                global img2
                img2 = Image.open(io.BytesIO(artifact.binary))
                gen_image = st.image(img2)
                gen_img_name = (
                    str(artifact.seed) + "_" + prompt.replace(" ", "_") + ".png"
                )  # Save our generated image with its seed number as the filename and the img2img suffix so that we know this is our transformed image.

    # download button
    gen_button = st.download_button(
        label="Download generated image",
        data=img2.tobytes(),
        file_name=gen_img_name,
        mime="image/png",
    )
