import os
import nltk, csv
import pandas as pd
import numpy as np
from tensorflow.keras import models, layers
from keras.utils.vis_utils import plot_model as plot
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Read in the data
dftrain = pd.read_csv('../Files/cleantrain.csv', usecols=["id", "keyword","location", "text", "target"])
dftest = pd.read_csv('../Files/cleantest.csv', usecols=["id", "keyword","location", "text"])

#Generate a list of unique words and their frequency followed by a unique id
unique_words = {}
for tweet in [*dftrain["text"], *dftest["text"]]:
    for word in tweet.split():
        if word.lower() not in unique_words.keys():
           unique_words[word.lower()] = [1, len(unique_words)]
        else:
            unique_words[word.lower()][0] += 1


"""
print(len(unique_words))

#Check how many words appear more than 5 times
count = 0
for word in unique_words:
    if unique_words[word][0] > 5:
     count += 1
print(count)


#Sort words by frequency
unique_words = {k: v for k, v in sorted(unique_words.items(), key=lambda item: item[1], reverse=True)}

# Write output to a text file
textfile = open("../Files/unique_words.txt", "wb")
for word in unique_words.keys():
    textfile.write((word + ': ' + str(unique_words[word]) + '\n').encode('ascii', 'ignore'))
textfile.close()
"""

#Generate an encoded list of lists where each list is a tweet represented by the id of each word
traindata = []
for  tweet in [*dftrain["text"]]:
    list = []
    for word in tweet.split():
        list.append(unique_words[word.lower()][1])
    traindata.append(list)

#Adapted from: inside-machinelearning.com/en/a-simple-and-efficient-model-for-binary-classification-in-nlp/
#Define the function for one hot encoding
def vectorize_sequences(sequences, dimension=len(unique_words)):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

#Convert the list of word ids into a onehot encoded vector
xtrain = vectorize_sequences(traindata)

#Define the model
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(len(unique_words),)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])

#plot(model, to_file="model.png", show_shapes=True, show_layer_activations=True, show_layer_names=True)

"""
#Define the training and validation splits from the trainingdata
splits = 2500
validation = xtrain[:splits]
validationlabels = dftrain["target"][:splits]
partial_train = xtrain[len(xtrain) - splits:]
partial_trainlabels = dftrain["target"][len(dftrain["target"]) - splits:]

#Train the model and store the data
history = model.fit(partial_train,
                    partial_trainlabels,
                    epochs=10,
                    batch_size=512,
                    validation_data=(validation, validationlabels))

loss_values = history.history['loss']
val_loss_values = history.history['val_loss']

epochs = range(1, len(loss_values) + 1)

#Plot the loss and accuracy values
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()

acc_values = history.history['accuracy']
val_acc_values = history.history['val_accuracy']

plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

"""

#Train the model
model.fit(xtrain, dftrain["target"], epochs=4, batch_size=512)

#Prepare the testing data
testdata = []
for  tweet in [*dftest["text"]]:
    list = []
    for word in tweet.split():
        list.append(unique_words[word.lower()][1])
    testdata.append(list)

xtest = vectorize_sequences(testdata)

#Predict the data
results = model.predict(xtest)

#Write the data to a file
file = open("../Files/results.csv", "w", newline='')
writer = csv.writer(file)
writer.writerow(['id', 'target'])
for i in range(0, len(results)):
    writer.writerow([dftest["id"][i], round(results[i][0])]) 
file.close()
