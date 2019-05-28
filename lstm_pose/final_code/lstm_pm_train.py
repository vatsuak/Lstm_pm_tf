#training block 

from util import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from time import gmtime, strftime
import tensorflow as tf
import model
import numpy as np
import pandas as pd
from DataLoader import *
from tqdm import tqdm



# hyper parameter
T = 5
outclass = 21
learning_rate = 8e-6
batch_size = 4                   # batch size* temporal must be atleast total number of images in the dataset otherwise the batches will be reported as the same images in a cyclic manner
epochs = 100
begin_epoch = 0
save_dir = './ckpt/'                        # to save model
data_dir = './train_data/'                        # the train data dir
label_dir = './labels/001L0.json'           # the label dir

save_dir_val= './validation_info/'          # dir to save the validation info i.e.the csv
data_dir_val = './validation_data/'                # dir to find the validation dataset
label_dir_val = './labels/001L0.json'       # dir to find the validation labels should point to a json file


#dataset for training
print('Training Dataset')
dataset_train = Feeder(data_dir=data_dir, label_dir=label_dir, train=True, temporal=T, joints=outclass)
dl_train = DataLoader(dataset_train,batch_size)

#dataset for validation 
print('Validation Dataset')
dataset_valid = Feeder(data_dir=data_dir_val, label_dir=label_dir_val, train=False, temporal=T, joints=outclass)
dl_valid = DataLoader(dataset_valid,len(os.listdir(data_dir_val)))# make sure that all the images are used as a batch process for validation

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if not os.path.exists(save_dir_val):
    os.mkdir(save_dir_val)

    
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

            # im = np.full((1,368,368,T*3), 1.0)
            # cm = np.full((1,368,368,1), 1.0)
            # lbl = np.full((1,T,46,46,outclass),1.0)


            print(strftime("%Y-%m-%d    %H:%M:%S", gmtime())+'   epoch....................................' + str(epoch+1))
            # for step, (images, label_map, center_map, imgs) in enumerate(train_dataset):

            #iterations over the batches
            '''
            step:the order/number of the batch
            images  : 4D Tensor  Batch_size  *   width(368)  *  height(368)* (temporal * 3)
            #currently: Batch_size  *  (temporal * 3)  *  width(368)  *  height(368)
            
            label_map : 5D Tensor: Batch_size * (joints+1) *   45 * 45  * Temporal    
            #currently:  Batch_size  *  Temporal        * (joints+1) *   45 * 45
            center_map : 4D Tensor  Batch_size * width(368) * height(368) *  1
            # currently: Batch_size  *  1  * width(368) * height(368)
            '''
            print('==Training')
            for step in range(len(dl_train)//batch_size):

                # get the inputs for the placeholders
                images, label, center = dl_train()

            # ******************** calculate and save loss of each joints ********************
                sess.run(trainer,feed_dict={image:images,label_map:label,cmap:center,dropprob:0.5})

                #for test
                # sess.run(trainer,feed_dict={image:images,label_map:lbl,cmap:cm})
                
                if step % 10 == 0:
                    print('--step .....' + str(step+1))
                    print('--loss ' + str(float(sess.run(total_loss,feed_dict={image:images,label_map:label,cmap:center,dropprob:0.5}))))
                # print('--loss ' + str(float(sess.run(total_loss,feed_dict={image:im,label_map:lbl,cmap:cm}))))

                #  ************************* validate and save model per 10 epochs  *************************
            if epoch % 10 == 0:
                saver.save(sess, os.path.join(save_dir, 'lstm_pm_epoch{:d}.ckpt'.format(epoch)))


                #..............................Validation begins...................................

            if epoch %  10 == 0:
                validate(dl_valid, sess, predict_heatmaps, epoch, save_dir_val)

    print('train done!')


def validate(dlobj, sess, predict_heatmaps, epoch, save_dir_val):

    print('==Validation')
    sigmas = [(i+1)/100 for i in range(5)]   # set the number of sigmas needed to calculate over    
    results =  []
    pck_tot = 0

    for sigma in tqdm(sigmas): #going over the sigmas

        result = []  # save sigma and pck for a particular sigma
        result.append(sigma)
        # pck_all = []
        # for step in range(len(dl)//batch_size):

        # # get the inputs for the placeholders
        images, label, center = dlobj()# there is just one batch of all the images together

        # images = np.full((1,368,368,T*3), 1.0)
        # center = np.full((1,368,368,1), 1.0)
        # label = np.full((1,T,46,46,outclass),1.0)

            # get the prediction from the saved model
        prediction = sess.run(predict_heatmaps,feed_dict={image:images,cmap:center,dropprob:1.0})

        #no gradient calculation so no need to run trainer
        
        

            #ignoring the initial heatmap(used as a prior)
        prediction =  prediction[1:]

            # calculate pck
        pck = lstm_pm_evaluation(label, prediction, sigma=sigma, temporal=T)

        # caculate the total PCK
        pck_tot += pck

        # pck_all.append(pck)

        # Add the pck according to the sigma
        result.append(str(pck))
        results.append(result)

        # print('sigma ==========> ' + str(sigma))
        # print('===PCK evaluation in test dataset is ' + str(sum(pck_all) / len(pck_all)))
        # print('===PCK evaluation in validation dataset is ' + str(pck))
        # result.append(str(sum(pck_all) / len(pck_all)))
        # results.append(result)

    print('--PCK evaluation in validation dataset is ' + str(pck_tot/len(sigmas)))
    # print(results)
    results = pd.DataFrame(results)
    results.to_csv(save_dir_val + str('test_pck_epoch_{}.csv'.format(epoch)), header=['Sigma','Avg. Pck'], index=None,sep='\t')


if __name__ == '__main__':
    train()
