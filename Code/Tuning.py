import os
import pandas as pd
import numpy as np
from tensorflow.keras import models, layers, callbacks, optimizers
import keras_tuner as kt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Read in the data
dftrain = pd.read_csv('../Files/cleantrain.csv', usecols=["id", "keyword","location", "text", "target"])
dftest = pd.read_csv('../Files/cleantest.csv', usecols=["id", "keyword","location", "text"])

#Generate a list of unique words their frequency followed by a unique id
unique_words = {}
for tweet in [*dftrain["text"], *dftest["text"]]:
    for word in tweet.split():
        if word.lower() not in unique_words.keys():
           unique_words[word.lower()] = [1, len(unique_words)]
        else:
            unique_words[word.lower()][0] += 1

#Generate an encoded list of lists where each list is a tweet represented by the id of each word
traindata = []
for  tweet in [*dftrain["text"]]:
    list = []
    for word in tweet.split():
        list.append(unique_words[word.lower()][1])
    traindata.append(list)

ytrain = dftrain["target"]

#Define the function for one hot encoding
def vectorize_sequences(sequences, dimension=len(unique_words)):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

#Convert the list of word ids into a onehot encoded vector
xtrain = vectorize_sequences(traindata)

min_accuracy = (100,)

#Define the splits of data for training vs validation
splits = 2500
validation = xtrain[:splits]
validationlabels = ytrain[:splits]
partial_train = xtrain[len(xtrain) - splits:]
partial_trainlabels = ytrain[len(ytrain) - splits:]


#Adapted from: www.tensorflow.org/tutorials/keras/keras_tuner
#Define the model builder which contains the values for the parameters which will be iterated over
def model_builder(hp):
    model = models.Sequential()

    #Tune the number of units in the first Dense layer
    #Choose an optimal value between 32-512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(layers.Dense(units=hp_units, activation='relu', input_shape=(len(unique_words),)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    #Tune the learning rate for the optimizer
    #Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=optimizers.Adam(learning_rate=hp_learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    return model

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=50,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

#Seach for the best hyperparameters
stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(xtrain, ytrain, epochs=50, validation_split=0.2, callbacks=[stop_early])
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

#Get the best epoch from the best hyperparameters
model = tuner.hypermodel.build(best_hps)
history = model.fit(xtrain, ytrain, epochs=50, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1

print('Best epoch: %d' % (best_epoch,))