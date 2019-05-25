# test
import os
from util import *
import tensorflow as tf
import pandas as pd
import model
import numpy as np
import os
from DataLoader import *


# hyper parameter
T = 5
outclass = 21
learning_rate = 8e-6
epoch = 0       # the epoch number of the model to load
save_dir = './dummyval_info/'
data_dir = './data/'
label_dir = './labels/001L0.json'
model_dir = './ckptdummy0'
model_dir = os.path.join(model_dir,'lstm_pm_epoch{}.ckpt'.format(epoch))
batch_size = 1

 
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


# **************************************** test all images ****************************************


# print('********* test data *********')


#placeholder for the image
image = tf.placeholder(tf.float32,shape=[None,368,368,T*3],name='temporal_info')
    
# the output prediction should come out as 46*46*21
# label_map = tf.placeholder(tf.float32,shape=[None,T,46,46,outclass])

# placeholder for the gaussian
cmap = tf.placeholder(tf.float32,shape=[None,368,368,1],name='gaussian_peak')

# placeholder for sigma
# sigma = tf.placeholder(tf.float32,name='sigma')

# placeholder for the dropout probability
dropprob = tf.placeholder(tf.float32,name='dropout')

# placeholder for the predicted heatmap
# pred = tf.placeholder(tf.float32,shape=[T,None,46,46,outclass])

#load the model
net = model.Net(outclass=outclass,T=T,prob=dropprob)

# create the graph for the feed forwatd network
predict_heatmaps = net.forward(image,cmap)


                             #****************BUILDING THE GRAPH*********************



saver = tf.train.Saver()

sigma = 0.01
results = []

with tf.Session() as sess:
        
    #restore the model

    print('.............................................Restoring model.....................................')

    saver.restore(sess,model_dir)
    print('.............................................Model Restored.....................................')
    for i in range(5): #going over the sigmas

    #modify into the sessions process
        result = []  # save sigma and pck
        result.append(sigma)
        pck_all = []
        # for step in range(len(dl)//batch_size):

        # get the inputs for the placeholders
            # images, label, center = dl()

        images = np.full((1,368,368,T*3), 1.0)
        center = np.full((1,368,368,1), 1.0)
        label = np.full((1,T,46,46,outclass),1.0)

        prediction = sess.run(predict_heatmaps,feed_dict={image:images,cmap:center,dropprob:1.0})  # get a list size: temporal * 4D Tensor shape=[T,batchsize,46,46,outclass]

        #no gradient calculation so no need to run trainer
        
    
            #ignoring the initial heatmap(used as a prior)
        prediction =  prediction[1:]

            # calculate pck
        pck = lstm_pm_evaluation(label, prediction, sigma=sigma, temporal=T)
            
        pck_all.append(pck)


        print('sigma ==========> ' + str(sigma))
        print('===PCK evaluation in test dataset is ' + str(sum(pck_all) / len(pck_all)))
        result.append(str(sum(pck_all) / len(pck_all)))
        results.append(result)

        sigma += 0.01

    results = pd.DataFrame(results)
    results.to_csv(str(save_dir) +  'test_pck0.csv', header=['Sigma','Avg.Pck'], index=None, sep='\t')

