import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


import keras
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers import Dense, Activation, Dropout, Flatten, Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Activation
from keras.layers import Conv2D, AveragePooling2D
from keras.models import Model, Sequential, load_model

from math import sqrt


class Attraction:
    def __init__(self, rawDataPath, target = 'Attractive'):
        self.rawData = pd.read_csv(rawDataPath)

        self.target = target # Attractive or Babyface
        self.cfd_instances = []
        self.df = pd.DataFrame()
        self.features = []

        self.test_size = 0.5
        self.train_size = 0.2
        self.random_state = 17

        self.test_x = 0
        self.test_y = 0
        self.train_x = 0
        self.train_y = 0
        self.val_x = 0
        self.val_y = 0

        self.attractiveness_model = Model()

        pd.options.display.max_columns = 1000
        pd.options.display.max_rows = 1000
        pd.options.display.max_seq_items = 1000

    def getFileNames(self):
        files = []
        file_count = 0
        path = os.path.join("CFD_Version_3", "Images", self.target)
        for r, d, f in os.walk(path):
            for file in f: #BF-001 has several images
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    files.append(file)
        return files

    def findEmotion(self, file):
        #file = CFD-WM-040-023-HO.jpg
        file_name = file.split(".")[0] #[1] is jpg
        emotion = file_name.split("-")[4]
        return emotion

    def findRace(self, file):
        #file = CFD-WM-040-023-HO.jpg
        file_name = file.split(".")[0] #[1] is jpg
        race = file_name.split("-")[1][0]
        return race

    def findGender(self, file):
        #file = CFD-WM-040-023-HO.jpg
        file_name = file.split(".")[0] #[1] is jpg
        gender = file_name.split("-")[1][1]
        return gender

    def retrievePixels(self, path):
        img = image.load_img(path, grayscale=False, target_size=(224, 224))
        x = image.img_to_array(img).reshape(1, -1)[0]
        return x

    def makeDF(self):
        self.rawData["files"] = self.rawData.Target.apply(att.getFileNames)

        for index, instance in self.rawData.iterrows():
            folder = instance.Target
            score = instance[target]
            for file in instance.files:
                tmp_instance = []
                #tmp_instance.append((target, file, score))
                tmp_instance.append(folder)
                tmp_instance.append(file)
                tmp_instance.append(score)
                self.cfd_instances.append(tmp_instance)

        self.df['emotion'] = self.df.file.apply(findEmotion)
        self.df['race'] = self.df.file.apply(findRace)
        self.df['gender'] = self.df.file.apply(findGender)

        self.df = pd.DataFrame(self.cfd_instances, columns = ["folder", "file", "score"])
        self.df['file'] = os.path.join("CFD_Version_3", "Images", self.df["folder"], self.df['file'])

        self.df['pixels'] = self.df['file'].progress_apply(retrievePixels)

    def analyse(self):
        for index, instance in self.df[(self.df.race == 'W') #A: Asian, B: Black, L: Latino, W: White]
                        & (self.df.gender == 'F') # F: Female / M: Male
                        & (self.df.emotion == 'HO') #HO: Happy Open Mouth, HC: Happy Closed Mouth, N: Neutral
                        ].sort_values(by=['score'], ascending = False).head(3).iterrows():
        #for index, instance in df.sort_values(by=['score'], ascending = False).head(3).iterrows():
            img = instance.pixels
            img = img.reshape(224, 224, 3)
            img = img / 255
            
            plt.imshow(img)
            plt.show()
            print(instance.file)
            print("Attractiveness score: ",instance.score)
            
            print("-------------------")

    def preprocessing(self):
        pixels = self.df['pixels'].values
        for i in range(0, pixels.shape[0]):
            self.features.append(pixels[i])

        self.features = np.array(self.features)
        self.features = self.features.reshape(self.features.shape[0], 224, 224, 3)

        self.features = self.features / 255

    def splitDF(self):
        #we have a few instances. that's why, do not use test and val sets differently
        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(self.features, 
            self.df.score.values, test_size=self.train_size, random_state=self.random_state)

        # self.test_x, self.val_x, self.test_y, self.val_y = train_test_split(self.val_x, 
        #     self.val_y, test_size=self.test_size, random_state=self.random_state)

        print("train set: ", self.train_x.shape[0])
        print("validation set: ", self.val_x.shape[0])
        #print("test set: ", test_x.shape[0])

    def modelling(self):
        base_model = Sequential()
        base_model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
        base_model.add(Convolution2D(64, (3, 3), activation='relu'))
        base_model.add(ZeroPadding2D((1,1)))
        base_model.add(Convolution2D(64, (3, 3), activation='relu'))
        base_model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        base_model.add(ZeroPadding2D((1,1)))
        base_model.add(Convolution2D(128, (3, 3), activation='relu'))
        base_model.add(ZeroPadding2D((1,1)))
        base_model.add(Convolution2D(128, (3, 3), activation='relu'))
        base_model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        base_model.add(ZeroPadding2D((1,1)))
        base_model.add(Convolution2D(256, (3, 3), activation='relu'))
        base_model.add(ZeroPadding2D((1,1)))
        base_model.add(Convolution2D(256, (3, 3), activation='relu'))
        base_model.add(ZeroPadding2D((1,1)))
        base_model.add(Convolution2D(256, (3, 3), activation='relu'))
        base_model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        base_model.add(ZeroPadding2D((1,1)))
        base_model.add(Convolution2D(512, (3, 3), activation='relu'))
        base_model.add(ZeroPadding2D((1,1)))
        base_model.add(Convolution2D(512, (3, 3), activation='relu'))
        base_model.add(ZeroPadding2D((1,1)))
        base_model.add(Convolution2D(512, (3, 3), activation='relu'))
        base_model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        base_model.add(ZeroPadding2D((1,1)))
        base_model.add(Convolution2D(512, (3, 3), activation='relu'))
        base_model.add(ZeroPadding2D((1,1)))
        base_model.add(Convolution2D(512, (3, 3), activation='relu'))
        base_model.add(ZeroPadding2D((1,1)))
        base_model.add(Convolution2D(512, (3, 3), activation='relu'))
        base_model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        base_model.add(Convolution2D(4096, (7, 7), activation='relu'))
        base_model.add(Dropout(0.5))
        base_model.add(Convolution2D(4096, (1, 1), activation='relu'))
        base_model.add(Dropout(0.5))
        base_model.add(Convolution2D(2622, (1, 1)))
        base_model.add(Flatten())
        base_model.add(Activation('softmax'))

        #pre-trained weights of vgg-face model. 
        #you can find it here: https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
        #related blog post: https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/
        base_model.load_weights('vgg_face_weights.h5')

        num_of_classes = 1 #this is a regression problem

        #freeze all layers of VGG-Face except last 7 one
        for layer in base_model.layers[:-7]:
            layer.trainable = False

        base_model_output = Sequential()
        base_model_output = Flatten()(base_model.layers[-4].output)
        base_model_output = Dense(num_of_classes)(base_model_output)

        self.attractiveness_model = Model(inputs=base_model.input, outputs=base_model_output)

    def train(self):
        # sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9)
        #attractiveness_model.compile(loss='mean_squared_error', optimizer=sgd)

        self.attractiveness_model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam())

        checkpointer = ModelCheckpoint(
            filepath='%s.hdf5' % (target), 
            monitor = "val_loss", 
            verbose=1, 
            save_best_only=True, 
            mode = 'auto'
        )

        earlyStop = EarlyStopping(monitor='val_loss', patience=50)

        score = attractiveness_model.fit(
            train_x, train_y, 
            epochs=5000, 
            validation_data=(val_x, val_y), 
            callbacks=[checkpointer, earlyStop]
        )

        best_iteration = np.argmin(score.history['val_loss'])+1

        val_scores = score.history['val_loss'][0:best_iteration]
        train_scores = score.history['loss'][0:best_iteration]

        plt.plot(val_scores, label='val_loss')
        plt.plot(train_scores, label='train_loss')
        plt.legend(loc='upper right')
        plt.show()

        self.attractiveness_model = load_model(f"{self.target}.hdf5")
        self.attractiveness_model.save_weights(f"{self.target}.h5")

    def performance(self):
        #actuals = test_y
        #predictions = self.attractiveness_model.predict(self.test_x)
        predictions = self.attractiveness_model.predict(self.val_x)
        actuals = self.val_y

        perf = pd.DataFrame(actuals, columns = ["actuals"])
        perf["predictions"] = predictions

        print("pearson correlation: ",perf[['actuals', 'predictions']].corr(method ='pearson').values[0,1])
        print("mae: ", mean_absolute_error(actuals, predictions))
        print("rmse: ", sqrt(mean_squared_error(actuals, predictions)))

        min_limit = df.score.min(); max_limit = df.score.max()
        best_predictions = []

        #for i in np.arange(1, 7, 0.01):
        for i in np.arange(int(min_limit), int(max_limit)+1, 0.01):
            best_predictions.append(round(i, 2))

        plt.scatter(best_predictions, best_predictions, s=1, color = 'black', alpha=0.5)
        plt.scatter(predictions, actuals, s=20, alpha=0.1)


if __name__ == "__main__":
    rawPath = os.path.join("CFD_Version_3", "metadata.csv")
    att = Attraction(rawPath, 'Attractive')
    print("Raw Data Shape\t", att.rawData.shape)

    att.makeDF()
    print("Dataframe Shape Pre-Emotion\t", att.df.shape)
    print("Dataframe Shape Post-Emotion\t", att.df.shape)

    att.analyse()

    att.preprocessing()
    print("Features\t", att.features.shape)

    att.splitDF()

    att.modelling()

    att.train()

    att.performance()
