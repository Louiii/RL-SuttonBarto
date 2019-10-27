# import tensorflow as tf
from tensorflow.keras.layers import Dense#, Input
from tensorflow.keras.models import Sequential#, Model
from tensorflow.keras.models import model_from_json
# from time import time
# from tensorflow.python.keras.callbacks import TensorBoard


class DenseANN():
    """ Simple fully-connected ANN in Keras.
    params:
        input_dim    - int, specify the number of inputs in a feature vector
        hidden_dims  - list of ints, for the number of neurons in each layer
        output_dim   - int, specify the number of outputs of the network
    """
    def __init__(self, input_dim=2, hidden_dims=[10], output_dim=1, loadfile=None):
        self.dir = "models/"
        if loadfile is None:
            self.init(input_dim, hidden_dims, output_dim)
        else:
            self.load(loadfile)

    def init(self, input_dim, hidden_dims, output_dim):
        self.model = Sequential()
        self.model.add(Dense(units=hidden_dims[0], activation='relu', input_dim=input_dim))
        if len(hidden_dims)>1:
            for i in range(1, len(hidden_dims)):
                self.model.add(Dense(units=hidden_dims[i], activation='relu'))
        self.model.add(Dense(units=output_dim, activation='linear'))
        self.compile()

    def compile(self):
        self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

    def train(self, x_train, y_train, epochs, batch_size):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, x_test, batch_size):
        classes = self.model.predict(x_test, batch_size=batch_size)
        return classes

    def performance(self, x_test, y_test, batch_size):
        loss_and_metrics = self.model.evaluate(x_test, y_test, batch_size=batch_size)
        return loss_and_metrics

    def save(self, name):
        model_json = self.model.to_json()
        with open(self.dir+name+".json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(self.dir+name+".h5")

    def load(self, name):
        json_file = open(self.dir+name+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(self.dir+name+".h5")
        self.compile()

# class NeuralODE():
#     def __init__(self):
#



# x_train, x_test, y_train, y_test = makeData()# Make a dataset: X=2d positions , Y=value
# ann = DenseANN(2, [10, 6], 1)
# ann.train(x_train, y_train, 500, 1000) # dim(x_train) must be n x 2, dim(y_train) must be n x 1
# ann.save('test')
# ann = DenseANN(loadfile='test')
# y_pred = ann.predict(x_test, 10000)