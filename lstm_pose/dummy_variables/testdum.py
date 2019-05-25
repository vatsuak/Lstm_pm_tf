#training block 


#need to modify to tensorflow
# from data.handpose_data2 import UCIHandPoseDataset

from util import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from time import gmtime, strftime
import tensorflow as tf
import model
import numpy as np
from tqdm import tqdm

import pandas as pd
from DataLoader import *


# hyper parameter
T = 5
outclass = 21
learning_rate = 8e-6
batch_size = 1
epochs = 1
begin_epoch = 0
save_dir = './ckptdummy0/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

dataset_train = Feeder(data_dir='./data/', label_dir='./labels/', train=True, temporal=T, joints=outclass)
dl_train = DataLoader(dataset_train,batch_size)

    
                            #***********************Placeholders*********************
    
#placeholder for the input image
image = tf.placeholder(tf.float32,shape=[None,368,368,T*3],name='temporal_info')

# the output prediction should come out as 46*46*21
label_map = tf.placeholder(tf.float32,shape=[None,T,46,46,outclass])

# placeholder for the gaussian
cmap = tf.placeholder(tf.float32,shape=[None,368,368,1],name='gaussian_peak')

# placeholder for the dropuout probability
dropprob = tf.placeholder(tf.float32,name='dropout')

# Build model
net = model.Net(outclass=outclass,T=T,prob=dropprob)


                            #****************BUILDING THE GRAPH*********************


# the output predicted
predict_heatmaps = net.forward(image, cmap)  # lis of size (temporal + 1 ) * 4D Tensor

#optimizer
optim = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999)

#loss calculation 
criterion = tf.losses.mean_squared_error  # loss function MSE  

total_loss = calc_loss(predict_heatmaps, label_map, criterion, temporal=T)

#gradient computation and back prop

trainer = optim.minimize(total_loss)

saver = tf.train.Saver()

def train():

    with tf.Session() as sess:
        
        #initilialize all the weights and biases
        
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        for epoch in range(begin_epoch, epochs + 1):

            images = np.full((1,368,368,T*3), 1.0)
            center = np.full((1,368,368,1), 1.0)
            label = np.full((1,T,46,46,outclass),1.0)


            print(strftime("%Y-%m-%d    %H:%M:%S", gmtime())+'   epoch....................................' + str(epoch+1))
            # for step, (images, label_map, center_map, imgs) in enumerate(train_dataset):



            # ******************** calculate and save loss of each joints ********************
            sess.run(trainer,feed_dict={image:images,label_map:label,cmap:center,dropprob:0.5})

                #for test
                # sess.run(trainer,feed_dict={image:images,label_map:lbl,cmap:cm})
                
                            #  ************************* save model per 10 epochs  *************************
            if epoch % 10 == 0:
                saver.save(sess, os.path.join(save_dir, 'lstm_pm_epoch{:d}.ckpt'.format(epoch)))
                #..............................Validation begins...................................

                validate(sess,predict_heatmaps)

    print('train done!')


def validate(sess, predict_heatmaps):

    print('Validation:')
    sigmas = [(i+1)/100 for i in range(5)]   # set the number of sigmas needed to calculate over
    results =  []
    pck_tot = 0
    for sigma in tqdm(sigmas): #going over the sigmas

    #modify into the sessions process
        result = []  # save sigma and pck
        result.append(sigma)
        
        # for step in range(len(dl)//batch_size):

        # # get the inputs for the placeholders
        # images, label, center = dlobj()# there is just one batch of all the images together

        images = np.full((1,368,368,T*3), 1.0)
        center = np.full((1,368,368,1), 1.0)
        label = np.full((1,T,46,46,outclass),1.0)

            # get the prediction from the saved model
        prediction = sess.run(predict_heatmaps,feed_dict={image:images,cmap:center,dropprob:1.0})

        #no gradient calculation so no need to run trainer
        
        

            #ignoring the initial heatmap(used as a prior)
        prediction =  prediction[1:]

            # calculate pck
        pck = lstm_pm_evaluation(label, prediction, sigma=sigma, temporal=T)
        pck_tot += pck
        # pck_all.append(pck)


        # print('sigma ==========> ' + str(sigma))
        # print('===PCK evaluation in test dataset is ' + str(sum(pck_all) / len(pck_all)))
    # print('===PCK evaluation in validation dataset is ' + str(pck))

        result.append(str(pck))
        results.append(result)

        # results.append(pck)
    print('===PCK evaluation in validation dataset is ' + str(pck_tot/len(sigmas)))
    results = pd.DataFrame(results)
    results.to_csv('test_pck.csv', header=['Sigma','Batch_PCK'], index=None,sep='\t')


if __name__ == '__main__':
    train()
