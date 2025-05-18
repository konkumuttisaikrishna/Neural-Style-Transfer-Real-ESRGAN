import streamlit as st
from style_transfer import run_style_transfer_main
import os
import glob
from PIL import Image
import subprocess

st.title("Neural Style Transfer with Real-ESRGAN Upscaling")

# File upload
content_file = st.file_uploader("Upload Content Image", type=["jpg", "png"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "png"])

# File save locations
content_path = "uploads/content.jpg"
style_path = "uploads/style.jpg"
output_path = "output/styled_img.jpeg"

if content_file and style_file:
    with open(content_path, "wb") as f:
        f.write(content_file.read())
    with open(style_path, "wb") as f:
        f.write(style_file.read())

    st.image(content_path, caption="Content Image", use_column_width=True)
    st.image(style_path, caption="Style Image", use_column_width=True)

    # Run Style Transfer
    if st.button("Run Style Transfer"):
        output_image = run_style_transfer_main(content_path, style_path, output_path)
        st.image(output_image, caption="Stylized Image", use_column_width=True)

    # Real-ESRGAN Upscaling
    if os.path.exists(output_path):
        if st.button("Upscale with Real-ESRGAN"):
            # Run Real-ESRGAN inference
            command = f"python inference_realesrgan.py -n RealESRGAN_x4plus -i {output_path} -o output --outscale 4"
            subprocess.run(command, shell=True)

            # Try to locate the actual output file created by Real-ESRGAN
            output_dir = "output"
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            pattern = os.path.join(output_dir, f"{base_name}*")

            matching_files = glob.glob(pattern)
            if matching_files:
                upscaled_path = matching_files[0]  # Just take the first match
                st.image(upscaled_path, caption="Upscaled Image", use_container_width=True)
            else:
                st.error("Upscaled image not found. Please check if Real-ESRGAN ran correctly.")
    else:
        st.info("Please upload both content and style images.")
