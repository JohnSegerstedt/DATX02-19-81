from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten
from keras.callbacks import EarlyStopping, TensorBoard
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K
import keras
import pandas
import numpy
import os
import random

num_folds = 6 #Number of cross validation folds

file = "../reaperCSVs/vision data/" #File to parse and use for training
#file = "../reaperCSVs/cluster data/cluster_data10080.csv"
targetsFile = "clustering.csv" #File containing results from clustering, to use as targets
join_column = '0Replay_id' #Column to use as identifier when joining the files. Joining is done before dropping
drop_columns = ['Unnamed: 0'] #Drop these columns from the original csv-file because they're irrelevant
drop_columns_2 = ['0Frame_id']
target_column = 'Cluster' #Name of the column containing the training targets (labels)

batch_size = 50
epochs = 200
learning_rate = 0.0005
stopping_delta = 0.02
stopping_patience = 2


#num_columns = None
input_shape = None
num_classes = None

distribution = None


def get_class_distribution(df):
    classes = df[target_column].unique()
    classes.sort()
    dist = numpy.zeros(len(classes))
    total = len(df[target_column].values)
    for i in range(0, len(classes)):
        amount = len(df[df[target_column] == classes[i]])
        dist[i] = amount / total
    return dist

def load_data(file, targetsFile): #Should return: data array, target array
    data = pandas.read_csv(file)
    data = pandas.merge(data, pandas.read_csv(targetsFile), on=join_column, how='inner')
    data = data.drop(join_column, axis=1)
    data = data.drop(drop_columns, axis=1)
    targets = data.filter([target_column], axis=1)
    data = data.drop(target_column, axis=1)
    data = data.values

    global num_classes
    num_classes = int(targets[target_column].max())
    targets[target_column] = targets[target_column] - 1
    targets = targets.values
    targets = keras.utils.to_categorical(targets, num_classes)

    global input_shape
    input_shape = [len(data[0])]
    return data, targets

def load_data_over_time(path, targetsFile): #Should return: data array, target array
    if not os.path.isdir(path):
        exit(0)
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.lower().endswith(".csv")]
    if files.__len__() == 0:
        exit(0)

    print("Found", files.__len__(), "files.")
    data = pandas.read_csv(os.path.join(path, files[0]))
    data = data.drop(drop_columns, axis=1)
    for x in range(1, files.__len__()):
        df = pandas.read_csv(os.path.join(path, files[x]))
        df = df.drop(drop_columns, axis=1)
        df = df.drop(drop_columns_2, axis=1)
        data = pandas.merge(data, df, on=join_column, how='inner', suffixes=['_' + str(x - 1), '_' + str(x)])

    data = pandas.merge(data, pandas.read_csv(targetsFile), on=join_column, how='inner')
    global distribution
    distribution = get_class_distribution(data)
    global num_classes
    num_classes = int(data[target_column].max() + 1)
    data = data.drop(join_column, axis=1)
    #print(list(data.columns.values))
    #exit(0)
    targets = data.filter([target_column], axis=1)
    data = data.drop(target_column, axis=1)

    data = data.drop(data.filter(regex='P1').columns, axis=1)

    data = data.values
    data = MinMaxScaler().fit_transform(data)

    targets[target_column] = targets[target_column]
    targets = targets.values
    targets = keras.utils.to_categorical(targets, num_classes)

    global input_shape
    input_shape = [len(data[0])]
    return data, targets

def load_data_conv(path, targetsFile): #Should return: data array, target array
    if not os.path.isdir(path):
        exit(0)
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.lower().endswith(".csv")]
    if files.__len__() == 0:
        exit(0)

    print("Found", files.__len__(), "files.")
    data = pandas.read_csv(os.path.join(path, files[0]))
    data = data.drop(drop_columns, axis=1)

    for x in range(1, files.__len__()):
        df = pandas.read_csv(os.path.join(path, files[x]))
        df = df.drop(drop_columns, axis=1)
        data = data.append(df, ignore_index=True)

    data = data[data['0Frame_id'] <= 5040]

    data = data.drop(data.filter(regex='P2').columns, axis=1)

    cols = data.drop(join_column, axis=1).columns
    data[cols] = MinMaxScaler().fit_transform(data[cols])
    data = pandas.merge(data, pandas.read_csv(targetsFile), on=join_column, how='inner')
    global distribution
    distribution = get_class_distribution(data)

    global num_classes
    num_classes = int(data[target_column].max() + 1)
    data[target_column] = data[target_column]
    #print(list(data.columns.values))
    #exit(0)
    #list_of_df = [g for _, g in data.groupby(join_column)]

    groups = data.groupby(join_column)
    data_list = None
    targets_list = numpy.zeros(len(groups))
    count = 0
    for key, df_t in groups:
        df_t = df_t.sort_values(by=['0Frame_id'])
        if numpy.shape(df_t.values)[0] == 6:
            print("Dropped a replay due to missing data.")
            continue
        targets_list[count] = df_t[target_column].iloc[0]
        df_t = df_t.drop(target_column, axis=1)
        df_t = df_t.drop(join_column, axis=1)
        df_t = df_t.drop(drop_columns_2, axis=1)

        if data_list is None:
            data_list = numpy.zeros((len(groups), len(df_t.values), len(df_t.values[0])))

        #print(list(df_t.columns))
        #exit(0)
        data_list[count] = df_t.values
        count += 1

    #print(data_list.shape)
    #exit(0)
    img_rows = data_list.shape[1]
    img_cols = data_list.shape[2]

    global input_shape
    if K.image_data_format() == 'channels_first':
        data_list = data_list.reshape((data_list.shape[0], 1, img_rows, img_cols))
        input_shape = (1, img_rows, img_cols)
    else:
        data_list = data_list.reshape((data_list.shape[0], img_rows, img_cols, 1))
        input_shape = (img_rows, img_cols, 1)


    data = data_list
    targets = targets_list
    #targets = numpy.random.randint(num_classes, size=len(data))
    targets = keras.utils.to_categorical(targets, num_classes)

    return data, targets

def load_data_dummy_targets(): #Should return: data array, target array
    data = pandas.read_csv(file)
    data = data.drop(join_column, axis=1)
    data = data.drop(drop_columns, axis=1)
    data = data.values
    targets = numpy.zeros([len(data), num_classes])
    #for i in range(0, len(targets)):
     #   targets[i][random.randint(0, num_classes - 1)] = 1

    targets = numpy.random.randint(2, size=len(data))
    targets = keras.utils.to_categorical(targets, num_classes)

    global input_shape
    input_shape = [len(data[0])]
    return data, targets

def create_model():
    model = Sequential()

    model.add(Dense(90, activation='relu', kernel_initializer='random_normal', input_shape=input_shape))
    model.add(Dense(60, activation='relu', kernel_initializer='random_normal'))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer='random_normal'))
    #keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    adam = keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model

def create_model_conv():
    model = Sequential()

    model.add(Conv2D(filters=10,
                     kernel_size=(1, 4),
                     strides=(1, 2),
                     padding="same",
                     activation='relu',
                     input_shape=input_shape))
    #
    # model.add(BatchNormalization(axis=3))
    # model.add(MaxPooling2D(pool_size=(1, 2),
    #                        strides=[1, 2],
    #                        padding="valid"))
    # model.add(Conv2D(filters=20,
    #                  kernel_size=(1, 4),
    #                  strides=2,
    #                  padding="same",
    #                  activation='relu'))
    #
    model.add(BatchNormalization(axis=3))
    model.add(Flatten())

    model.add(Dense(num_classes, activation='softmax', kernel_initializer='random_normal'))
    #keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    adam = keras.optimizers.Adam(lr=0.0005)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model

def train_and_evaluate_model(model, data_train, labels_train, data_test, labels_test):
    model.fit(data_train, labels_train,
              epochs=epochs,
              batch_size=batch_size,
              verbose=1)
    score = model.evaluate(data_test, labels_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__ == "__main__":
    data, labels = load_data_over_time(file, targetsFile)
    #data, labels = load_data_conv(file, targetsFile)
    print(num_classes, "classes found.")
    kfold = KFold(num_folds, shuffle=True)
    estimator = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=1)
    #estimator = KerasClassifier(build_fn=create_model_conv, epochs=epochs, batch_size=batch_size, verbose=1)
    early_stopping = EarlyStopping(monitor='loss',
                                   min_delta=stopping_delta,
                                   patience=stopping_patience,
                                   verbose=0,
                                   mode='auto')
    tensorboard = TensorBoard()

    results = cross_val_score(estimator, data, labels, cv=kfold, fit_params={'callbacks': [early_stopping]})
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100), num_classes, "classes with distribution: ", distribution)
    #for train_index, test_index in skf.split(data, labels):
     #   print("Running Fold", train_index, "-", test_index, "Number of folds: ", num_folds)
      #  model = None # Clearing the NN.
       # model = create_model()
        #train_and_evaluate_model(model, data[train_index], labels[train_index], data[test_index], labels[test_index])

