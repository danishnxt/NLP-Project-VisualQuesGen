import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger

from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout, Reshape
from keras.layers import GRU, Input, Concatenate, RepeatVector, Embedding, Lambda, Conv2D, Conv2DTranspose, Conv1D
from keras.layers import MaxPooling1D, Flatten, TimeDistributed, SeparableConv2D
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau, Callback
from keras.applications import vgg16 as vgg16_model
from keras import objectives
import keras.backend as K
import tensorflow as tf

import cv2
import os
import pickle
from datetime import datetime

dataset_dir = './Dataset/'
images_dir = 'C:/VQGP'
model_name = 'ctcnet-' + str(datetime.now().date())
if not os.path.exists('./'+model_name):
    os.mkdir(model_name)

np.random.seed(42)

with open(dataset_dir+'fcn_index2word.npy', 'rb') as f:
    index2word = np.load(f)


num_epochs = 100
batch_size = 32

input_shape = (224, 224, 3)
max_len = 20
vocab_size = len(index2word)
print(vocab_size)
print(len(os.listdir(images_dir)))

###########################################################################################

def get_sample(sample):
    im = cv2.cvtColor(cv2.imread(images_dir+sample['image_id']+'.jpg'), cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (input_shape[0], input_shape[1]))
    im = im / 255.0

    wOutput = np.zeros((max_len,vocab_size+1))
    row_index = np.arange(0, max_len)
    col_index = sample['question']
    wOutput[row_index, col_index] = 1
    
    WLogits = np.array(sample['question'])
    
    dummy_out = np.zeros((1,))
    
    return im, wOutput, WLogits, dummy_out
    
def data_generator(df, batch_size):
    indexes = np.arange(0, len(df), batch_size)
    
    # Last value removed to prevent creation of a batch with size < batch_size 
    # incase the dataset can not be divided into the correct number of batches
    # given the predefined batch_size
    # This approach effectively reduces the number of batches by 1
    
    if len(df) % batch_size != 0:
        indexes = indexes[:-1] 
    
    while True: # 1 iteration represents 1 epoch
        np.random.shuffle(indexes) # indexes shuffled for each epoch
        for index in indexes: # 1 iteration represents 1 batch
            batch_examples = df.iloc[index : index+batch_size].reset_index()
                        
            Xim = np.zeros((32, *input_shape))
            Xseq = np.zeros((32, max_len, vocab_size+1))
            Xlogit = np.zeros((32, max_len))
            Ys = np.zeros((32, 1))
            for index, row in batch_examples.iterrows(): # 1 iteration represents 1 entry in a batch
                
                xim, xseq, xlogit, y = get_sample(row)
                Xim[index, :, :, :] = xim
                Xseq[index, :, :] = xseq
                Xlogit[index, :] = xlogit
                Ys[index, :] = y # ONE MODIFICATION MADE HERE 

            yield [Xim,Xseq,Xlogit], Ys

###########################################################################################


dataset = pd.read_csv(dataset_dir+'fcn_data.csv').astype('object')
dataset['question'] = dataset['question'].apply(lambda x: eval(x))
print('Dataset size: ', dataset.shape)

train_set, val_set = train_test_split(dataset, test_size=0.2,random_state = 42)
train_set, test_set = train_test_split(train_set, test_size=0.2, random_state = 37)
print('Training samples:', len(train_set), 'Validation samples:', len(val_set), 'Testing samples:', len(test_set))

train_gen = data_generator(train_set, batch_size)
val_gen = data_generator(val_set, batch_size)
test_gen = data_generator(test_set, batch_size)

vggWeight = './Weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
vgg16 = vgg16_model.VGG16(include_top=True, weights=None, input_shape=input_shape, classes=1000)
vgg16.load_weights(vggWeight)

vgg16.layers.pop()
vgg16.layers.pop()
vgg16.layers.pop()
vgg16.layers.pop()
for l in vgg16.layers:
    l.trainable = False

image_embedding_model = Model(vgg16.input, vgg16.layers[-1].output, name='VGG16')


# In[40]:


feature_transformer = Sequential([
    Reshape((49,512)),
    Conv1D(512, kernel_size=(3,), strides=(2,), activation='relu', padding='valid'),
    Conv1D(512, kernel_size=(5,), strides=(1,), activation='relu', padding='valid'),
#     Conv1D(vocab_size, kernel_size=(1,), strides=(1,), activation=None, padding='valid')
    Dense(vocab_size+1, activation=None)
], name='feature_transformer')


# def feature_transformer(x):
#     y = Reshape((49,512))(x)
#     y = Conv1D(512, kernel_size=(3,), strides=(2,), activation='relu', padding='valid')(y)
#     y = Conv1D(512, kernel_size=(5,), strides=(1,), activation='relu', padding='valid')(y)
#     y = Dense(vocab_size, activation=None)(y)
#     return y

def transpose_image(i):
    return K.permute_dimensions(i, (0,2,1,3))

# Input image
input_im = Input(shape=(input_shape), name='input_im') 
input_im_trans = Lambda(transpose_image, name='input_im_transpose')(input_im)

# Branch 1
image_features = image_embedding_model(input_im)
out_seq = feature_transformer(image_features)

# Branch 2 (Transpose)
image_features_trans = image_embedding_model(input_im_trans)
out_seq_trans = feature_transformer(image_features_trans)

# vector of length 'batch_size' containing same value everywhere = 'max_len'


def custom_loss(arg_list):
    seq_lens = np.ones((batch_size,1))*max_len

    y_true_val, label_logits, y_pred1, y_pred2 = arg_list
    y_pred_prob = K.softmax(y_pred1, axis=1)

    cat_cross_ent = objectives.categorical_crossentropy(y_true_val, y_pred_prob)
    print (cat_cross_ent)
    
    y_pred_trans_prob = K.softmax(y_pred2, axis=1)
    
    cat_cross_ent_trans = objectives.categorical_crossentropy(y_true_val, y_pred_trans_prob)
    print (cat_cross_ent_trans)
    
    # PROB PRESENT IN THE TWO VARIABLES PRESENT BELOW

    ctc_orig = K.ctc_batch_cost(y_true=label_logits, y_pred=y_pred1, 
                                input_length=seq_lens, label_length=seq_lens)
    
    ctc_trans = K.ctc_batch_cost(y_true=label_logits, y_pred=y_pred2, 
                                 input_length=seq_lens, label_length=seq_lens)
   
    return seq_lens

    #return cat_cross_ent + cat_cross_ent_trans + ctc_orig + ctc_trans
    #return ctc_orig + ctc_trans

label_seq = Input(shape=(max_len,vocab_size+1), name='label_seq') # input directly 
label_logits = Input(shape=(max_len,), name='label_logits') # input directly

total_loss = Lambda(custom_loss, name='ctc_loss')([label_seq, label_logits, out_seq, out_seq_trans])
model = Model(inputs=[input_im,label_seq,label_logits], outputs=total_loss)


# model = Model(inputs=[input_im,label_seq,label_logits], outputs=out_seq)

adam = Adam()
# model.add_loss(K.sum(total_loss, axis=None))

model.compile(loss=lambda yt,yp: yp, 
              optimizer=adam) # the loss is whatever you get back from the model itself

# loss is thus built into the model
# Add loss to the metrics
# model.metrics_names.append('pred')
# model.metrics_tensors.append(model.layers[-2].output)

model.summary()


class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir):  # add other arguments to __init__ if you need
        super(LRTensorBoard, self).__init__(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super(LRTensorBoard, self).on_epoch_end(epoch, logs)

class SaveToDrive(Callback):
    def __init__(self, path):
        self.path = path
        
    def on_epoch_end(self, epoch):
        for fname in os.listdir(model_name):
            shutil.copy(os.path.join(model_name, fname), self.path)


checkpoint = ModelCheckpoint(model_name+'/'+model_name+'.h5', 
                             monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
tensorboard = LRTensorBoard(log_dir='./'+model_name)
# earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
cvslogger = CSVLogger(model_name+'/logs.csv', separator=',', append=True)
reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.000001, min_delta=0.03)
save2drive = SaveToDrive("drive/Rabeez-Danish/RunSave")

# callbacks = [checkpoint, tensorboard, earlystop, cvslogger, reducelr]
# callbacks = [checkpoint, tensorboard, cvslogger, reducelr, save2drive]
callbacks = [checkpoint, tensorboard, cvslogger, reducelr]
# callbacks = [checkpoint, cvslogger, reducelr]

hist = model.fit_generator(train_gen, epochs=num_epochs, steps_per_epoch=len(train_set)//batch_size,  validation_data=val_gen, validation_steps=len(val_set)//batch_size, callbacks=callbacks, verbose=1)


###################### FIX BEFORE, BEFORE WE CAN MOVE ON :) 


model.load_weights('../External Runs/20Ep-29Oct/gru_init_state-2018-10-26-20-3.50.h5')

wordlist = word2Index.tolist()
sample = test_gen.__next__()

from math import ceil 

vector = []
for num in range(4):
    current = []

    sampleXimg = sample[0][0][num,:,:,:]
    sampleXwords = sample[0][1][num,:,:]
    sampleY = sample[1][num,:]

    plt.imshow(sampleXimg)
    plt.axis('off')
#     plt.show()

    start_codon = wordlist.index('<start>')
    end_codon = wordlist.index('<end>')

    input_words = np.repeat(end_codon, max_len)
    input_words[0] = start_codon

    finalProb = []
    
    sentence = ''
    for i in range(max_len):
        stuff = [np.array([sampleXimg]), np.array([input_words.reshape(1,max_len)])]
        pred = model.predict(stuff).astype('float64')

        #     pred = np.exp(np.log(pred) / 0.3)
        #     pred /= np.sum(pred)
        #     pred = np.random.multinomial(1, pred.reshape((10326,)), 1)

        current.append(pred)
        
#         pred_1d = pred.reshape(vocab_size,)
#         overflow = pred_1d.sum() - 1
#         shiftby = overflow / len(pred_1d)
#         pred_1d -= shiftby

        pred_index = pred.argmax()
#         pred_index = np.random.choice(np.array(wordlist), p=pred_1d)
        next_word = wordlist[pred_index]
#         print(next_word)
        sentence += next_word + ' '

        if next_word == '<end>' or i == max_len-1:
            break

        input_words[i+1] = pred_index
    plt.show()
    print(sentence)

    vector.append(np.array(current))


# In[ ]:


model.save('model.h5')

