# importing the libraries
import streamlit as st
import PIL.Image as Image
import numpy as np
import pandas as pd
import tensorflow as tf
import pathlib

_ROOT = pathlib.Path(__file__).parent

# pd.options.display.float_format = "{:,.4f}".format
my_devices = tf.config.experimental.list_physical_devices(device_type="CPU")
tf.config.experimental.set_visible_devices(devices=my_devices, device_type="CPU")
import pdemo as D

############################################################################
# Load model and multilabel binarizer
plat = D.neural.model.PredictorForStreamlit()

# Designing the interface
st.sidebar.title("Stylistic Search")
st.sidebar.write("Finiding a PDF with similarly styled images")
# st.write("The training data was 12 different pdfs sourced from Cardinal Health's website")
# st.write("The goal was to provide a stylistic search to determine which pdf has images that look similar.")
# For newline
st.write("\n")
st.title("How data was gathered")
st.video('./cardinal_data_capture.mp4')
image = D.datasets.test_images.thumbs_up_glove
show = st.image(image, width=None)

st.sidebar.title("Upload Image")

# Disabling warning
st.set_option("deprecation.showfileUploaderEncoding", False)
# Choose your own image
uploaded_file = st.sidebar.file_uploader(" ", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    u_img = Image.open(uploaded_file)
    show.image(u_img, "Uploaded Image", width=256)

# For newline
st.sidebar.write("\n")

if st.sidebar.button("Click Here to Classify"):

    if uploaded_file is None:

        st.sidebar.write("Please upload an Image to Classify")

    else:

        with st.spinner("Classifying ..."):

            pred_df = plat.predict(u_img)
            st.success("Done!")

        st.sidebar.header("Algorithm Output: ")
        pred_df.sort_values(plat.prediction_col, ascending=False, inplace=True)
        tmp = pred_df.copy()
        tmp[plat.label_col] = tmp[plat.label_col].map(lambda x: x + "_pdf")
        st.write("Output from convolutional neural network:")
        st.write(
            tmp.set_index(plat.label_col).rename(
                {plat.prediction_col: "Similarity Score"}, axis=1
            )
        )

        plot = plat.plot_packaged_result(pred_df)
        st.sidebar.pyplot(plot)
        amt = pred_df[plat.prediction_col].sum() / pred_df.shape[0]
        st.sidebar.write(f"AoE Score: {amt*100:.3f} %")
        st.sidebar.write("Challenge: what image can get the highest AoE Score?")
        st.sidebar.write("AoE := Area of Effect (sum of all similarity scores)")

# url = "https://www.streamlit.io/"

# if st.button("Open browser"):
#    webbrowser.open_new_tab(url)

# st.title("PDFs")
st.sidebar.title("Links to PDFs")
tag_ref = D.datasets.pdfs.get_tag_lookup()
for tag, url_list in tag_ref.items():
    if url_list:
        for iii, url in enumerate(url_list):
            if iii == 0:
                link = f"[PDF for: {tag}]({url})"
                st.sidebar.markdown(link, unsafe_allow_html=True)
            else:
                link = f"[PDF for: {tag} source {iii+1:2}]({url})"
                st.sidebar.markdown(link, unsafe_allow_html=True)

# for pdf_tag, obj in D.datasets.pdfs.__dict__.items():
#     # img_rel_path = obj.img_location.relative_to(_ROOT)
#     # pdf_rel_path = obj.pdf_location.relative_to(_ROOT)
#     st.markdown(f"[![{pdf_tag}]({obj.img_location})]({obj.pdf_location}\n")
#     # [![Foo](http://www.google.com.au/images/nav_logo7.png)](http://google.com.au/)
