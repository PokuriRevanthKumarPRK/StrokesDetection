import librosa
import streamlit as st

from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

model_path2 = hf_hub_download(repo_id="PokuriRevanthKumar123/Strokes",
                             filename="spectrogram_model.keras")
model_path = hf_hub_download(repo_id="PokuriRevanthKumar123/Strokes",
                             filename="face_strokes_pred.keras")

st.title("Strokes Detection Using Audio and Image")
st.write("This website is created by Pokuri Revanth Kumar")
image_model = tf.keras.models.load_model(model_path)
spectrogram_model = tf.keras.models.load_model(model_path2)

audio_val = st.audio_input("Say a sentence")

if audio_val:
    with open("users_audio.wav","wb")as f:
        f.write(audio_val.read())

if audio_val:
    st.audio(audio_val)
    try:
        y_2 ,sr_2 = librosa.load("users_audio.wav", sr= 22050)
    except EOFError:
        print("Error")
    D = librosa.stft(y_2)  
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    spectrogram_image = Image.fromarray(S_db)
    spectrogram_image = spectrogram_image.resize((224,224))
    spectrogram_array = np.array(spectrogram_image)



    spectrogram_array= spectrogram_array/255.0
    spectrogram_array = np.stack([spectrogram_array]*3, axis=-1)
    spectrogram_array = np.expand_dims(spectrogram_array, axis=0)
    


    plt.figure(figsize=(6,4))
    librosa.display.specshow(S_db, sr=sr_2, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    st.pyplot(plt)
    plt.close()

face_img = st.camera_input("Show your Face (RECOMMENDED: HAVE A WHITE BACKGROUND )")

if face_img:
    st.image(face_img)

if face_img is not None:
    img = Image.open(face_img)
    img_array = np.array(img.resize((224,224)))
    img_array= img_array/255.0
    img_array = np.expand_dims(img_array, axis=0)


if audio_val and face_img:
    prediction1 = spectrogram_model.predict(spectrogram_array)
    prediction2 = image_model.predict(img_array)
    st.write(f"Spectrogram prediction: {prediction1}, Face prediction: {prediction2}")
    detection = (prediction1+prediction2)/2*100
    if detection>0.5:
        st.write("Final Prediction: Have Stroke")
    else:
        st.write("Final Prediction: No Stroke")
elif audio_val:
    prediction1 = spectrogram_model.predict(spectrogram_array)
    st.write(f"Spectrogram prediction: {prediction1}")
elif face_img:
    prediction2 = image_model.predict(img_array)
    st.write(f"Face prediction: {prediction2}")
else:
    st.write("Please provide audio and/or face input to make predictions.")



