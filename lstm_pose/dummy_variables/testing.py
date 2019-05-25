import model
import tensorflow as tf
import numpy as np
from util import *

T = 5
outclass = 21
image = tf.placeholder(tf.float32,shape=[None,368,368,T*3],name='temporal_info')
cmap = tf.placeholder(tf.float32,shape=[None,368,368,1],name='gaussian_peak')
dropprob = tf.placeholder(tf.float32,name='dropout')
# label_map = tf.placeholder(tf.float32,shape=[None,T,46,46,outclass])

net = model.Net(outclass=outclass,T=T,prob=dropprob)
predict_heatmaps = net.forward(image,cmap)
#optimizer
# optim = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)

    #loss calculation 
# criterion = tf.losses.mean_squared_error  # loss function MSE  
    
# total_loss = calc_loss(predict_heatmaps, label_map, criterion, temporal=T)
    
    #gradient computation and back prop
    
# trainer = optim.minimize(total_loss)
# print([n.name for n in tf.get_default_graph().as_graph_def().node])
saver = tf.train.Saver()
with tf.Session() as sess:
    im = np.full((3,368,368,T*3), 1.0)
    cm = np.full((3,368,368,1), 1.0)

    # out = net.forward(image,cmap)
    # sess.run(tf.global_variables_initializer())
    # maps = sess.run(out,feed_dict={image:im,cmap:cm})

    # for m in maps:
    #    print(m.shape)
    print('restoring model')

    saver.restore(sess,'./ckptdummy/lstm_pm_epoch0.ckpt')

    maps = sess.run(predict_heatmaps,feed_dict={image:im,cmap:cm,dropprob:0.5})

    for m in maps:
       print(m.shape)

