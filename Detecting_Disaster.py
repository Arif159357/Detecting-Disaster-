import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from PIL import ImageFile
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import keras.backend as K
from keras.models import model_from_json

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

ImageFile.LOAD_TRUNCATED_IMAGES = True
# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'train3'
validation_data_dir = 'val'
test_data_dir = 'test'
testBD_data_dir = 'testBD'
#nb_train_samples = 14034
#nb_validation_samples = 3000
nb_train_samples = 5753
nb_validation_samples = 125
nb_test_samples = 132
nb_testBD_samples = 493
epochs = 50
batch_size = 11
batch_size_test = 12
batch_siza_val = 5
batch_size_testBD = 29

datagen = ImageDataGenerator(rescale=1. / 255)

# build the VGG16 network
print("Begin Reading model done")
vgg19_model = applications.VGG19(include_top=False, weights='imagenet')
vgg16_model = applications.VGG16(include_top=False, weights='imagenet')
inceptionV3_model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet')
incepResnetV2_model = applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet')

print("Reading model done")
generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)
bottleneck_features_vgg19train = vgg19_model.predict_generator(
    generator, nb_train_samples // batch_size)
np.save(open('bottleneck_features_vgg19train.npy', 'wb'),
        bottleneck_features_vgg19train)
print("Reading training data done")

generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)
bottleneck_features_vgg16train = vgg16_model.predict_generator(
    generator, nb_train_samples // batch_size)
np.save(open('bottleneck_features_vgg16train.npy', 'wb'),
        bottleneck_features_vgg16train)
print("Reading training data done")

generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)
bottleneck_features_inceptrain = inceptionV3_model.predict_generator(
    generator, nb_train_samples // batch_size)
np.save(open('bottleneck_features_inceptrain.npy', 'wb'),
        bottleneck_features_inceptrain)
print("Reading training data done")

generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)
bottleneck_features_incepRtrain = incepResnetV2_model.predict_generator(
    generator, nb_train_samples // batch_size)
np.save(open('bottleneck_features_incepRtrain.npy', 'wb'),
        bottleneck_features_incepRtrain)
print("Reading training data done")

val = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_siza_val,
    class_mode=None,
    shuffle=False)
bottleneck_features_vgg19validation = vgg19_model.predict_generator(
    val, nb_validation_samples // batch_siza_val)
np.save(open('bottleneck_features_vgg19validation.npy', 'wb'),
        bottleneck_features_vgg19validation)

val = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_siza_val,
    class_mode=None,
    shuffle=False)
bottleneck_features_vgg16validation = vgg16_model.predict_generator(
    val, nb_validation_samples // batch_siza_val)
np.save(open('bottleneck_features_vgg16validation.npy', 'wb'),
        bottleneck_features_vgg16validation)

val = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_siza_val,
    class_mode=None,
    shuffle=False)
bottleneck_features_incepvalidation = inceptionV3_model.predict_generator(
    val, nb_validation_samples // batch_siza_val)
np.save(open('bottleneck_features_incepvalidation.npy', 'wb'),
        bottleneck_features_incepvalidation)

val = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_siza_val,
    class_mode=None,
    shuffle=False)
bottleneck_features_incepRvalidation = incepResnetV2_model.predict_generator(
    val, nb_validation_samples // batch_siza_val)
np.save(open('bottleneck_features_incepRvalidation.npy', 'wb'),
        bottleneck_features_incepRvalidation)

print("Reading Validation data done")

predict = datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size_test,
    class_mode=None,
    shuffle=False)
bottleneck_features_inceppredict = inceptionV3_model.predict_generator(
    predict, nb_test_samples // batch_size_test)
np.save(open('bottleneck_features_inceppredict.npy', 'wb'),
        bottleneck_features_inceppredict)

predict = datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size_test,
    class_mode=None,
    shuffle=False)
bottleneck_features_vgg19predict = vgg19_model.predict_generator(
    predict, nb_test_samples // batch_size_test)
np.save(open('bottleneck_features_vgg19predict.npy', 'wb'),
        bottleneck_features_vgg19predict)

predict = datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size_test,
    class_mode=None,
    shuffle=False)
bottleneck_features_vgg16predict = vgg16_model.predict_generator(
    predict, nb_test_samples // batch_size_test)
np.save(open('bottleneck_features_vgg16predict.npy', 'wb'),
        bottleneck_features_vgg16predict)

predict = datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size_test,
    class_mode=None,
    shuffle=False)
bottleneck_features_incepRpredict = incepResnetV2_model.predict_generator(
    predict, nb_test_samples // batch_size_test)
np.save(open('bottleneck_features_incepRpredict.npy', 'wb'),
        bottleneck_features_incepRpredict)

predictBD = datagen.flow_from_directory(
    testBD_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size_testBD,
    class_mode=None,
    shuffle=False)
bottleneck_features_vgg19BDpredict = vgg19_model.predict_generator(
    predictBD, nb_testBD_samples // batch_size_testBD )
np.save(open('bottleneck_features_vgg19predictBD.npy', 'wb'),
        bottleneck_features_vgg19BDpredict)


predictBD = datagen.flow_from_directory(
    testBD_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size_testBD,
    class_mode=None,
    shuffle=False)
bottleneck_features_vgg16BDpredict = vgg16_model.predict_generator(
    predictBD, nb_testBD_samples // batch_size_testBD )
np.save(open('bottleneck_features_vgg16predictBD.npy', 'wb'),
        bottleneck_features_vgg16BDpredict)


predictBD = datagen.flow_from_directory(
    testBD_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size_testBD,
    class_mode=None,
    shuffle=False)
bottleneck_features_incepBDpredict = inceptionV3_model.predict_generator(
    predictBD, nb_testBD_samples // batch_size_testBD )
np.save(open('bottleneck_features_incepBDpredictBD.npy', 'wb'),
        bottleneck_features_incepBDpredict)


predictBD = datagen.flow_from_directory(
    testBD_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size_testBD,
    class_mode=None,
    shuffle=False)
bottleneck_features_incepRBDpredict = incepResnetV2_model.predict_generator(
    predictBD, nb_testBD_samples // batch_size_testBD )
np.save(open('bottleneck_features_incepRBDpredictBD.npy', 'wb'),
        bottleneck_features_incepRBDpredict)
print("Reading test data done")

train_data = np.load(open('bottleneck_features_vgg19train.npy','rb'))
train_labels = np.array(
    [0] * 331 + [1] * 363 + [2] * 219 + [3] * 1397 + [4] * 495 + [5] * 2948)

validation_data = np.load(open('bottleneck_features_vgg19validation.npy','rb'))
validation_labels = np.array(
    [0] * 18 + [1] * 22 + [2] * 21 + [3] * 20 + [4] * 20 + [5] * 24)

test_data = np.load(open('bottleneck_features_vgg19predict.npy','rb'))
test_labels = np.array(
    [0] * 22 + [1] * 22 + [2] * 22 + [3] * 22 + [4] * 22 + [5] * 22)

testBD_data = np.load(open('bottleneck_features_vgg19predictBD.npy','rb'))
testBD_labels = np.array(
    [0] * 82 + [1] * 82 + [2] * 65 + [3] * 73 + [4] * 81 + [5] * 110)



predict_data = np.load(open('bottleneck_features_predictBD.npy','rb'))

X_data = np.concatenate([train_data,validation_data,test_data,testBD_data])
Y_labels = np.concatenate([train_labels,validation_labels,test_labels,testBD_labels])
#X_data = np.concatenate([train_data,validation_data,test_data])
#Y_labels = np.concatenate([train_labels,validation_labels,test_labels])
print("Begin Model completion")
vgg19_model = Sequential()
vgg19_model.add(Flatten(input_shape=train_data.shape[1:]))
vgg19_model.add(Dense(512, activation='relu'))
vgg19_model.add(Dense(256, activation='relu'))
vgg19_model.add(Dropout(0.5))
vgg19_model.add(Dense(6, activation='sigmoid'))

vgg16_model = Sequential()
vgg16_model.add(Flatten(input_shape=train_data.shape[1:]))
vgg16_model.add(Dense(512, activation='relu'))
vgg16_model.add(Dense(256, activation='relu'))
vgg16_model.add(Dropout(0.5))
vgg16_model.add(Dense(6, activation='sigmoid'))

inceptionV3_model = Sequential()
inceptionV3_model.add(Flatten(input_shape=train_data.shape[1:]))
inceptionV3_model.add(Dense(512, activation='relu'))
inceptionV3_model.add(Dense(256, activation='relu'))
inceptionV3_model.add(Dropout(0.5))
inceptionV3_model.add(Dense(6, activation='sigmoid'))

incepResnetV2_model = Sequential()
incepResnetV2_model.add(Flatten(input_shape=train_data.shape[1:]))
incepResnetV2_model.add(Dense(512, activation='relu'))
incepResnetV2_model.add(Dense(256, activation='relu'))
incepResnetV2_model.add(Dropout(0.5))
incepResnetV2_model.add(Dense(6, activation='sigmoid'))

print("Model completion done")
#Savng the model and pretrained models to disk so that we can use the saved model(fresh model)
#every time during the crossvalidation train
# serialize model to JSON
Vgg19_json = vgg19_model.to_json()
with open("vgg19_model.json", "w") as json_file:
    json_file.write(Vgg19_json)
# serialize weights to HDF5
vgg19_model.save_weights("vgg19_model.h5")
print("Saved model to disk")

Vgg16_json = vgg16_model.to_json()
with open("vgg16_model.json", "w") as json_file:
    json_file.write(Vgg16_json)
# serialize weights to HDF5
vgg16_model.save_weights("vgg16_model.h5")
print("Saved model to disk")

InceptionV3_json = inceptionV3_model.to_json()
with open("inceptionV3_model.json", "w") as json_file:
    json_file.write(InceptionV3_json)
# serialize weights to HDF5
inceptionV3_model.save_weights("inceptionV3_model.h5")
print("Saved model to disk")

IncepResentV2_json = incepResnetV2_model.to_json()
with open("incepResentV2__model.json", "w") as json_file:
    json_file.write(IncepResentV2_json)
# serialize weights to HDF5
incepResnetV2_model.save_weights("incepResentV2_model.h5")
print("Saved model to disk")
#def precision_m(y_true, y_pred):
#    """Precision metric.
#    Only computes a batch-wise average of precision.
#    Computes the precision, a metric for multi-label classification of
#    how many selected items are relevant.
#    """
#    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#    precision = true_positives / (predicted_positives + K.epsilon())
#    return precision
#
#
#def recall_m(y_true, y_pred):
#    """Recall metric.
#    Only computes a batch-wise average of recall.
#    Computes the recall, a metric for multi-label classification of
#    how many relevant items are selected.
#    """
#    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#    recall = true_positives / (possible_positives + K.epsilon())
#    return recall
#
#def f1_m(y_true, y_pred):
#    precision = precision_m(y_true, y_pred)
#    recall = recall_m(y_true, y_pred)
#    return 2*((precision*recall)/(precision+recall+K.epsilon()))
#model.compile(optimizer='sgd',
#              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

###K folde Cross validation
    
# define 10-fold cross validation test harness
print("Starting kFolde cross validation")
#five fold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
vgg19cvscores = []
vgg16cvscores = []
incepcvscores = []
incepRcvscores = []
precision = []
recall = []
k = 1
for trainidx, testidx in kfold.split(X_data, Y_labels):
    #trainids are the indices for train fold
    #testidx are the indices for test fold  
   print("VGG19 start") 
   model1 = None
   # Load fresh model from disk
   # load json and create model
   json_file = open('vgg19_model.json', 'r')
   loaded_model_json = json_file.read()
   json_file.close()
   model1 = model_from_json(loaded_model_json)
   # load weights into new model
   model1.load_weights("vgg19_model.h5")
   print("Loaded model from disk") 
   # Compile model  for training 
   #now you also can include precision and recall into your performance of crossvalidation below
   model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	# Fit the model for training
   model1.fit(X_data[trainidx], Y_labels[trainidx], epochs=30, batch_size=11, verbose=0)
   
   test = model1.predict(X_data[testidx])

   test_label = []

   for i in test:
       test_label.append(int(np.where(i == np.amax(i))[0]))

   cm = confusion_matrix(Y_labels[testidx],test_label)
   print(cm)
   precision = []
   recall = []
   for i in range(len(cm)):
    for j in range(len(cm)):
        if i == j:
            TP = cm[i][j]
            TR = sum(cm[i][:])
            TC = sum([i[j] for i in cm])
            Pre = TP/TR
            re= TP/TC
            precision.append(Pre)
            recall.append(re)
	# evaluate the model wih test fold
   scores = model1.evaluate(X_data[testidx], Y_labels[testidx], verbose=0)
   #print the accuracy for this fold
   print("precision of each class",precision)
   print("recall of each class",recall)
   print("%s: %.2f%%" % (model1.metrics_names[1], scores[1]*100))
   vgg19cvscores.append(scores[1] * 100)
   
   np.savetxt('Vgg19.flod'+str(k)+'.txt',test_label,'%1i')
   np.savetxt('vgg19'+str(k)+'.txt',Y_labels[testidx],'%1i')
    
   print("VGG19 end and Vgg16 start")
   model2 = None
   # Load fresh model from disk
   # load json and create model
   json_file = open('vgg16_model.json', 'r')
   loaded_model_json = json_file.read()
   json_file.close()
   model2 = model_from_json(loaded_model_json)
   # load weights into new model
   model2.load_weights("vgg16_model.h5")
   print("Loaded model from disk") 
   # Compile model  for training 
   #now you also can include precision and recall into your performance of crossvalidation below
   model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	# Fit the model for training
   model2.fit(X_data[trainidx], Y_labels[trainidx], epochs=30, batch_size=11, verbose=0)
   
   test = model2.predict(X_data[testidx])

   test_label = []

   for i in test:
       test_label.append(int(np.where(i == np.amax(i))[0]))

   cm = confusion_matrix(Y_labels[testidx],test_label)
   print(cm)
   precision = []
   recall = []
   for i in range(len(cm)):
    for j in range(len(cm)):
        if i == j:
            TP = cm[i][j]
            TR = sum(cm[i][:])
            TC = sum([i[j] for i in cm])
            Pre = TP/TR
            re= TP/TC
            precision.append(Pre)
            recall.append(re)
	# evaluate the model wih test fold
   scores = model2.evaluate(X_data[testidx], Y_labels[testidx], verbose=0)
   #print the accuracy for this fold
   print("precision of each class",precision)
   print("recall of each class",recall)
   print("%s: %.2f%%" % (model2.metrics_names[1], scores[1]*100))
   vgg16cvscores.append(scores[1] * 100)
   
   np.savetxt('Vgg16.flod'+str(k)+'.txt',test_label,'%1i')
   np.savetxt('vgg16'+str(k)+'.txt',Y_labels[testidx],'%1i')
   print("VGG16 ends and inception starts")
   model3 = None
   # Load fresh model from disk
   # load json and create model
   json_file = open('inceptionV3_model.json', 'r')
   loaded_model_json = json_file.read()
   json_file.close()
   model3 = model_from_json(loaded_model_json)
   # load weights into new model
   model3.load_weights("inceptionV3_model.h5")
   print("Loaded model from disk") 
   # Compile model  for training 
   #now you also can include precision and recall into your performance of crossvalidation below
   model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	# Fit the model for training
   model3.fit(X_data[trainidx], Y_labels[trainidx], epochs=30, batch_size=11, verbose=0)
   
   test = model3.predict(X_data[testidx])

   test_label = []

   for i in test:
       test_label.append(int(np.where(i == np.amax(i))[0]))

   cm = confusion_matrix(Y_labels[testidx],test_label)
   print(cm)
   precision = []
   recall = []
   for i in range(len(cm)):
    for j in range(len(cm)):
        if i == j:
            TP = cm[i][j]
            TR = sum(cm[i][:])
            TC = sum([i[j] for i in cm])
            Pre = TP/TR
            re= TP/TC
            precision.append(Pre)
            recall.append(re)
	# evaluate the model wih test fold
   scores = model3.evaluate(X_data[testidx], Y_labels[testidx], verbose=0)
   #print the accuracy for this fold
   print("precision of each class",precision)
   print("recall of each class",recall)
   print("%s: %.2f%%" % (model3.metrics_names[1], scores[1]*100))
   incepcvscores.append(scores[1] * 100)
   
   np.savetxt('inceptionv3.flod'+str(i)+'.txt',test_label,'%1i')
   np.savetxt('inceptionv3'+str(i)+'.txt',Y_labels[testidx],'%1i')
   print("inceptionV3 ends and inceptionResnet starts")
   model4 = None
   # Load fresh model from disk
   # load json and create model
   json_file = open('incepResentV2__model.json', 'r')
   loaded_model_json = json_file.read()
   json_file.close()
   model4 = model_from_json(loaded_model_json)
   # load weights into new model
   model4.load_weights("incepResentV2_model.h5")
   print("Loaded model from disk") 
   # Compile model  for training 
   #now you also can include precision and recall into your performance of crossvalidation below
   model4.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	# Fit the model for training
   model4.fit(X_data[trainidx], Y_labels[trainidx], epochs=30, batch_size=11, verbose=0)
   
   test = model4.predict(X_data[testidx])

   test_label = []

   for i in test:
       test_label.append(int(np.where(i == np.amax(i))[0]))

   cm = confusion_matrix(Y_labels[testidx],test_label)
   print(cm)
   precision = []
   recall = []
   for i in range(len(cm)):
    for j in range(len(cm)):
        if i == j:
            TP = cm[i][j]
            TR = sum(cm[i][:])
            TC = sum([i[j] for i in cm])
            Pre = TP/TR
            re= TP/TC
            precision.append(Pre)
            recall.append(re)
	# evaluate the model wih test fold
   scores = model4.evaluate(X_data[testidx], Y_labels[testidx], verbose=0)
   #print the accuracy for this fold
   print("precision of each class",precision)
   print("recall of each class",recall)
   print("%s: %.2f%%" % (model4.metrics_names[1], scores[1]*100))
   incepRcvscores.append(scores[1] * 100)
  
   np.savetxt('inceptionRv2.fold'+str(k)+'.txt',test_label,'%1i')
   np.savetxt('inceptionRv2'+str(k)+'.txt',Y_labels[testidx],'%1i')
   print("inception resent ends")
   k = k + 1
   

print("Average and std of accuracy for five fold cross validation");
print("%.2f%% (+/- %.2f%%)" % (np.mean(vgg19cvscores), np.std(vgg19cvscores)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(vgg16cvscores), np.std(vgg16cvscores)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(incepcvscores), np.std(incepcvscores)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(incepRcvscores), np.std(incepRcvscores)))
print("End kFolde cross validation")
#######End K fold cross validation


model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_data, train_labels,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(validation_data, validation_labels))
model.save_weights(top_model_weights_path)
model.save('vgg_model3.h5')

scores = model.evaluate(testBD_data, testBD_labels, verbose=0)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
cvscores.append(scores[1] * 100)
print("Average and std of accuracy")
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
test = model.predict(predict_data)

test_label = []

for i in test:
    test_label.append(int(np.where(i == np.amax(i))[0]))


cm = confusion_matrix(testBD_labels,test_label)
print(cm)

np.savetxt('BDincepV3.txt',test_label,'%1i')
np.savetxt('BDInceprestV2s.txt',test_labels,'%1i')
