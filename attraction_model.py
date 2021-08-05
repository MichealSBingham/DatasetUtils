import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()
from keras.preprocessing import image
from sklearn.model_selection import train_test_split

''' 
feeding data from database

Sorry this is messy-- I'm trying to get this to run and understand components of the code right now but this would be cleaner if separated 
by classes-- Atm having a problem with the keras module, but I think it may be my version of python bc keras is not fully compatible with 3.8'

This includes data processing, building the model, and training, testing, validating 

Also need to get the database through a request: https://www.chicagofaces.org/download/
''' 

class Attraction: # TODO: OOP.
    def __init__(self):
        self.target = 'Attractive'
        self.pd.options.display.max_columns = 1000
        self.pd.options.display.max_rows = 1000
        self.pd.options.display.max_seq_items = 1000
        # self.cfd_df_raw = pd.read_csv("CFD_Version_203/metadata.csv").head()
        self.cfd_df_raw = pd.read_csv(os.path.join("CFD_Version_203", "metadata.csv")).head()
        self.df['exact_file'] = os.path.join("CFD_Version_203", "CFD_203_Images", df["folder"], df['file'])
        self.df['pixels'] = df['exact_file'].apply(retrievePixels)
    def getFileNames(self):
        files = []
        file_count = 0
        # path = "CFD_Version_203/CFD_203_Images/%s/" % (self.target)
        path = os.path.join("CFD_Version_203", "CFD_203_Images", self.target)
        for r, d, f in os.walk(path):
            for file in f: #BF-001 has several images
                # if ('.jpg' in file) or ('.jpeg' in file) or ('.png' in file):
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    files.append(file)
        return files
    def retrievePixels(self, path):
        img = image.load_img(path, grayscale=False, target_size=(224, 224))
        x = image.img_to_array(img).reshape(1, -1)[0]
        return x


target = 'Attractive'
pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000
pd.options.display.max_seq_items = 1000
# cfd_df_raw = pd.read_csv("CFD_Version_203/metadata.csv")
cfd_df_raw = pd.read_csv(os.path.join("CFD_Version_203", "metadata.csv"))
cfd_df_raw.head()

def getFileNames(target):
    files = []
    file_count = 0
    #     path = "CFD_Version_203/CFD_203_Images/%s/" % (target)
    path = os.path.join("CFD_Version_203", "CFD_203_Images", "%s") % (target)
    for r, d, f in os.walk(path):
        for file in f: #BF-001 has several images
            # if ('.jpg' in file) or ('.jpeg' in file) or ('.png' in file):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                files.append(file)
    return files


cfd_df_raw["files"] = cfd_df_raw.Target.apply(getFileNames)
cfd_df_raw[['Target', 'files']].head()
cfd_instances = []
for index, instance in cfd_df_raw.iterrows():
    folder = instance.Target
    score = instance[target]
    for file in instance.files:
        tmp_instance = []
        #tmp_instance.append((target, file, score))
        tmp_instance.append(folder)
        tmp_instance.append(file)
        tmp_instance.append(score)
        cfd_instances.append(tmp_instance)
df = pd.DataFrame(cfd_instances, columns = ["folder", "file", "score"])
df[['file', 'score']].head()


def retrievePixels(path):
    img = image.load_img(path, grayscale=False, target_size=(224, 224))
    x = image.img_to_array(img).reshape(1, -1)[0]
    return x

# df['exact_file'] = "CFD_Version_203/CFD_203_Images/"+df["folder"]+"/"+df['file']
df['exact_file'] = os.path.join("CFD_Version_203", "CFD_203_Images", df["folder"], df['file'])
df['pixels'] = df['exact_file'].apply(retrievePixels)


def findEmotion(file):
    #file = CFD-WM-040-023-HO.jpg
    file_name = file.split(".")[0] #[1] is jpg
    emotion = file_name.split("-")[4]
    return emotion

def findRace(file):
    #file = CFD-WM-040-023-HO.jpg
    file_name = file.split(".")[0] #[1] is jpg
    race = file_name.split("-")[1][0]
    return race

def findGender(file):
    #file = CFD-WM-040-023-HO.jpg
    file_name = file.split(".")[0] #[1] is jpg
    gender = file_name.split("-")[1][1]
    return gender

df['emotion'] = df.file.apply(findEmotion)
df['race'] = df.file.apply(findRace)
df['gender'] = df.file.apply(findGender)


#include neutral, happen open mouth and happy close mouth
df = df[(df.emotion == 'N') | (df.emotion == 'HO') | (df.emotion == 'HC')]

# df['file'] = "CFD_Version_203/CFD_203_Images/"+df["folder"]+"/"+df['file']
df['file'] = os.path.join("CFD_Version_203", "CFD_203_Images", df["folder"], df['file'])



def retrievePixels(path):
    img = image.load_img(path, grayscale=False, target_size=(224, 224))
    x = image.img_to_array(img).reshape(1, -1)[0]
    return x


for index, instance in df[(df.race == 'W') #A: Asian, B: Black, L: Latino, W: White]
                          & (df.gender == 'F') # F: Female / M: Male
                          & (df.emotion == 'HO') #HO: Happy Open Mouth, HC: Happy Closed Mouth, N: Neutral
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
    
    
    
features = []
pixels = df['pixels'].values
for i in range(0, pixels.shape[0]):
    features.append(pixels[i])
features = np.array(features)
features = features.reshape(features.shape[0], 224, 224, 3)
features = features / 255
features.shape


#make train test validation batches by splitting dataset
train_x, val_x, train_y, val_y = train_test_split(features, df.score.values, test_size=0.2, random_state=17)
print("train set: ", train_x.shape[0])
print("validation set: ", val_x.shape[0])

#MODEL BUILDING


import keras
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers import Dense, Activation, Dropout, Flatten, Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Activation
from keras.layers import Conv2D, AveragePooling2D
from keras.models import Model, Sequential
#VGG-Face model

class BaseModel:
    def __init__(self):
        self.base_model = Sequential()
        self.base_model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
        self.base_model.add(Convolution2D(64, (3, 3), activation='relu'))
        self.base_model.add(ZeroPadding2D((1,1)))
        self.base_model.add(Convolution2D(64, (3, 3), activation='relu'))
        self.base_model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        self.base_model.add(ZeroPadding2D((1,1)))
        self.base_model.add(Convolution2D(128, (3, 3), activation='relu'))
        self.base_model.add(ZeroPadding2D((1,1)))
        self.base_model.add(Convolution2D(128, (3, 3), activation='relu'))
        self.base_model.add(MaxPooling2D((2,2), strides=(2,2)))
    
        self.base_model.add(ZeroPadding2D((1,1)))
        self.base_model.add(Convolution2D(256, (3, 3), activation='relu'))
        self.base_model.add(ZeroPadding2D((1,1)))
        self.base_model.add(Convolution2D(256, (3, 3), activation='relu'))
        self.base_model.add(ZeroPadding2D((1,1)))
        self.base_model.add(Convolution2D(256, (3, 3), activation='relu'))
        self.base_model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        self.base_model.add(ZeroPadding2D((1,1)))
        self.base_model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.base_model.add(ZeroPadding2D((1,1)))
        self.base_model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.base_model.add(ZeroPadding2D((1,1)))
        self.base_model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.base_model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        self.base_model.add(ZeroPadding2D((1,1)))
        self.base_model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.base_model.add(ZeroPadding2D((1,1)))
        self.base_model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.base_model.add(ZeroPadding2D((1,1)))
        self.base_model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.base_model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        self.base_model.add(Convolution2D(4096, (7, 7), activation='relu'))
        self.base_model.add(Dropout(0.5))
        self.base_model.add(Convolution2D(4096, (1, 1), activation='relu'))
        self.base_model.add(Dropout(0.5))
        self.base_model.add(Convolution2D(2622, (1, 1)))
        self.base_model.add(Flatten())
        self.base_model.add(Activation('softmax'))
    
    #pre-trained weights are from here:https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
        self.base_model.load_weights('vgg_face_weights.h5')

#TRANSFER LEARNING

num_of_classes = 1 #this is a regression problem


#freeze all layers of VGG-Face except last 7 one
for layer in base_model.layers[:-7]:
    layer.trainable = False

base_model_output = Sequential()
base_model_output = Flatten()(base_model.layers[-4].output)
base_model_output = Dense(num_of_classes)(base_model_output)

attractiveness_model = Model(inputs=base_model.input, outputs=base_model_output)

#TRAINING

#sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9)
#attractiveness_model.compile(loss='mean_squared_error', optimizer=sgd)
###thjis part moved

attractiveness_model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam())
checkpointer = ModelCheckpoint(
    filepath='%s.hdf5' % (target)
    , monitor = "val_loss"
    , verbose=1
    , save_best_only=True
    , mode = 'auto'
)

earlyStop = EarlyStopping(monitor='val_loss', patience=50)
score = attractiveness_model.fit(
    train_x, train_y
    , epochs=5000
    , validation_data=(val_x, val_y)
    , callbacks=[checkpointer, earlyStop]
)

#stops early if loss isnt significantly decreasing for awhile

best_iteration = np.argmin(score.history['val_loss'])+1

val_scores = score.history['val_loss'][0:best_iteration]
train_scores = score.history['loss'][0:best_iteration]

plt.plot(val_scores, label='val_loss')
plt.plot(train_scores, label='train_loss')
plt.legend(loc='upper right')
plt.show()

#restore the best weights
from keras.models import load_model
attractiveness_model = load_model("%s.hdf5" % (target))
attractiveness_model.save_weights('%s.h5' % (target))


from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

#actuals = test_y
#predictions = attractiveness_model.predict(test_x)
predictions = attractiveness_model.predict(val_x)
actuals = val_y

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



