import streamlit as st


# this caches the output to store the output and not call this function again
# and again preventing time wastage. `allow_output_mutation = True` tells the
# function to not hash the output of this function and we can get away with it
# because no arguments are passed through this.
# https://docs.streamlit.io/en/stable/api.html#streamlit.cache
@st.cache(allow_output_mutation=True, show_spinner=False)
def get_models():
    from model import ICGEN
    return {
        'ICGEN' : ICGEN()
    }

# load all the models before the app starts
with st.spinner('Loading Model...'):
    MODELS = get_models()

# description
st.markdown("# :hibiscus: Image Compositon GAN :herb:!")
st.write('''
Image Composition is an crucial task in Computer Vision to generate high quality real-life-like images. Usually, when an extracted portion of an image is pasted \
on to another image, it does not look natural. This is because the properties of both the images are different. Image Composition can help with blending these images \
seamlessly. It reconstructs the foreground image to make it compatible and consistent with the background image. Some of the reconstruction may be changes in \
luminosity, color, and smoothness.
''')

# instruction
st.markdown(":high_brightness: Please upload a composite image and its corresponding mask. :high_brightness:")
model = MODELS['ICGEN']

# composite image
comp = st.file_uploader("Composite Image", type=['png', 'jpg'])

# display composite image
if comp:
    st.image(comp)

# mask
mask = st.file_uploader("Mask", type=['png', 'jpg'])

# display mask
if mask:
    st.image(mask)

# harmonize
if st.button("Harmonize"):
    with st.spinner('Harmonizing...'):
        # preprocess images
        inputs = model.preprocess(comp, mask)
        # harmonize image
        output = model.harmonize(inputs)
        # upsample image
        hr_image = model.upsample(output)
        # display output
        st.image(hr_image)
