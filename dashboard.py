import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from typing import Union
from fpdf import FPDF
from tempfile import NamedTemporaryFile
from training.model import VAE
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

st.set_page_config(layout="wide")

@st.cache_resource()
def load_model(filename: str) -> VAE:
    model = VAE(input_size=28, output_size=10 + 26, num_filters=32, num_latent_var=64).to(device)
    model.load_state_dict(torch.load(filename, map_location=device))
    model.eval()
    return model

def generate_images(model: VAE, y: torch.Tensor, z_means: torch.Tensor, log_z_vars: torch.Tensor, randomness: float) -> list[np.ndarray]:
    z_evals = model.sample(z_means, log_z_vars + 2 * np.log(randomness + 1e-8)).tile(len(y), 1)
    y_evals = (
        torch.nn.functional.one_hot(y.reshape(-1, 1).tile(1, z_means.shape[0]).flatten(), num_classes=model.output_size)
        .float()
        .to(device)
    )
    return [
        im
        for im in model.decode(z_evals, y_evals)
        .sigmoid()
        .reshape(len(y), z_means.shape[0], model.input_size, model.input_size)
        .mean(dim=1)
        .detach()
        .numpy()
    ]

def class_num_to_label(y: int) -> str:
    if y < 10:
        return str(y)
    return chr(y - 10 + ord("A"))

def label_to_class_num(label: str) -> Union[int, None]:
    if label.isnumeric():
        return int(label)
    if label < "A" or label > "Z":
        return None
    return ord(label) - ord("A") + 10

def clip_image(image: Image) -> Image:
    img_array = np.array(image)
    if (img_array == 0).all():
        return image
    img_array = img_array[np.any(img_array > 1e-3, 1), :]
    img_array = img_array[:, np.any(img_array > 1e-3, 0)]
    return Image.fromarray(img_array)

def recenter_image(image: Image) -> Image:
    image = clip_image(image)
    image = ImageOps.pad(image, (20, 20))
    return ImageOps.expand(image, (4, 4))

st.title("Handwriting Reconstruction")

stroke_width = st.slider("Stroke Width:", min_value=1, max_value=25, value=20)

# Initialize session state
if 'canvas_results_digits' not in st.session_state:
    st.session_state.canvas_results_digits = [None] * 10
if 'canvas_results_alphabets' not in st.session_state:
    st.session_state.canvas_results_alphabets = [None] * 26
if 'percentages_digits' not in st.session_state:
    st.session_state.percentages_digits = [""] * 10
if 'percentages_alphabets' not in st.session_state:
    st.session_state.percentages_alphabets = [""] * 26

# Digits Section
st.header("Draw Digits")
digit_containers = []
cols_digits = st.columns(10)
for i in range(10):
    with cols_digits[i]:
        canvas_result = st_canvas(
            stroke_width=stroke_width,
            stroke_color="white",
            background_color="black",
            height=150,
            width=150,
            key=f"digit_{i}",
            drawing_mode="freedraw"
        )
        if canvas_result.image_data is not None:
            st.session_state.canvas_results_digits[i] = canvas_result.image_data
        digit_containers.append(st.empty().text(st.session_state.percentages_digits[i]))

# Alphabets Section
st.header("Draw Alphabets")
alphabet_containers = []
cols_alphabets = st.columns(13)
for i in range(26):
    with cols_alphabets[i % 13]:
        canvas_result = st_canvas(
            stroke_width=stroke_width,
            stroke_color="white",
            background_color="black",
            height=150,
            width=150,
            key=f"alphabet_{i}",
            drawing_mode="freedraw"
        )
        if canvas_result.image_data is not None:
            st.session_state.canvas_results_alphabets[i] = canvas_result.image_data
        alphabet_containers.append(st.empty().text(st.session_state.percentages_alphabets[i]))

# Load model once
model = load_model("model.pt")

# Process digit images
if any(img is not None for img in st.session_state.canvas_results_digits):
    z_means_digits = []
    log_z_vars_digits = []
    for i, img_data in enumerate(st.session_state.canvas_results_digits):
        if img_data is not None:
            image = Image.fromarray(img_data)
            image = image.convert("L")
            image = recenter_image(image)
            im = np.expand_dims(np.array(image) / 255, axis=0)
            x = torch.Tensor([im])
            y_pred, z_mean, log_z_var = model.encode(x)
            y_prob = y_pred.softmax(dim=1)[0, :].detach()
            imax = y_prob.argmax().numpy()
            label = "{} ({:.2f}%)".format(class_num_to_label(imax), y_prob[imax] * 100)
            st.session_state.percentages_digits[i] = label
            z_means_digits.append(z_mean[0, :].detach())
            log_z_vars_digits.append(log_z_var[0, :].detach())

    for i, container in enumerate(digit_containers):
        container.text(st.session_state.percentages_digits[i])

# Process alphabet images
if any(img is not None for img in st.session_state.canvas_results_alphabets):
    z_means_alphabets = []
    log_z_vars_alphabets = []
    for i, img_data in enumerate(st.session_state.canvas_results_alphabets):
        if img_data is not None:
            image = Image.fromarray(img_data)
            image = image.convert("L")
            image = recenter_image(image)
            im = np.expand_dims(np.array(image) / 255, axis=0)
            x = torch.Tensor([im])
            y_pred, z_mean, log_z_var = model.encode(x)
            y_prob = y_pred.softmax(dim=1)[0, :].detach()
            imax = y_prob.argmax().numpy()
            label = "{} ({:.2f}%)".format(class_num_to_label(imax), y_prob[imax] * 100)
            st.session_state.percentages_alphabets[i] = label
            z_means_alphabets.append(z_mean[0, :].detach())
            log_z_vars_alphabets.append(log_z_var[0, :].detach())

    for i, container in enumerate(alphabet_containers):
        container.text(st.session_state.percentages_alphabets[i])

# Generate images if any valid latent variables are available
if z_means_digits or z_means_alphabets:
    z_means = torch.stack(z_means_digits + z_means_alphabets)
    log_z_vars = torch.stack(log_z_vars_digits + log_z_vars_alphabets)

    st.header("Write text to construct your handwriting")
    text = st.text_area(
        "Enter text to generate:",
        value="",
        help="Alphabetic characters will be converted to uppercase. Other characters will be ignored.",
    )

    with st.expander("Options"):
        randomness = st.slider("Randomness:", min_value=0.0, value=1.0)
        if st.checkbox("Average Latent Variables"):
            z_mean = z_means.mean(dim=0).unsqueeze(0)
            log_z_var = log_z_vars.mean(dim=0).unsqueeze(0)
        else:
            z_mean = z_means
            log_z_var = log_z_vars

    images = []
    for t in text:
        index = label_to_class_num(t.upper())
        if index is not None:
            images.append(generate_images(model, torch.tensor([index]), z_mean, log_z_var, randomness)[0])
        else:
            images.append(np.zeros((28, 28)))

    if images:
        combined_image = np.concatenate(images, axis=-1)
        st.image(combined_image)

        # Save to PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        temp_files = []
        img_count = 0
        for img in images:
            with NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                temp_files.append(tmp.name)
                Image.fromarray((img * 255).astype(np.uint8)).save(tmp.name)
                pdf.image(tmp.name, x=10 + (img_count % 10) * 28, y=10 + (img_count // 10) * 28, w=28, h=28)
                if img_count % 10 == 9:
                    pdf.add_page()
                img_count += 1
        for file in temp_files:
            os.remove(file)

        st.download_button(
            label="Save as PDF",
            data=pdf.output(dest='S').encode('latin1'),
            file_name="handwriting.pdf",
            mime="application/pdf",
        )
else:
    st.warning("Please draw on the canvas to generate handwriting.")