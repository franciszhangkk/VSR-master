from keras.models import load_model
import os

vsr_model = model = load_model('model.h5')

def read_input_names():
    names = []
    for root, _, files in os.walk('Data'):
        for filename in files:
            names.append(os.path.join(root, filename))
    print("Loaded ", len(names))
    return names


