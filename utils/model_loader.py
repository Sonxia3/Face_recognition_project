from model.inception_resnet_v1 import InceptionResNetV1

def load_fr_model(weights_path="models/pesos.h5", input_shape=(160, 160, 3), embedding_size=128):
    model = InceptionResNetV1(input_shape=input_shape, classes=embedding_size)
    model.load_weights(weights_path)
    return model
