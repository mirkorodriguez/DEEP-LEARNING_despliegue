# Developed by Mirko J. Rodríguez mirko.rodriguezm@gmail.com

# ------------------------
# Loading model from disk
# ------------------------
import keras

def loadModelH5():

    MODEL_H5_FILE = "flowers_model_tl_full_tf2.h5"
    MODEL_H5_PATH = "../../models/"

    # Cargar el modelo DL desde disco
    loaded_model = keras.models.load_model(MODEL_H5_PATH + MODEL_H5_FILE)
    print(MODEL_H5_FILE, " Loading from disk >> ", loaded_model)

    return loaded_model
