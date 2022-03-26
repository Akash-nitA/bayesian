'''
    in bayesian modelling we try to predict the distribution of the weights and biases and the prior distribution of the weights and biases.
    we minimize the negative log likelihood of the data given the weights and biases.

'''


# import libraries
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

# data preprocessing
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
x_train=tf.cast(tf.reshape(x_train/255,(-1,784)),tf.float32)
y_train=tf.one_hot(y_train,depth=10)
x_test=tf.cast(tf.reshape(x_test/255,(-1,784)),tf.float32)
y_test=tf.one_hot(y_test,depth=10)
opt=tf.keras.optimizers.Adam(learning_rate=0.05)


w=tf.Variable(tf.ones([784,10]))
b=tf.Variable(tf.ones([10]))

# joint distribution of weights,biases and predictions
def factor():
    w= yield tfp.distributions.Normal(loc=tf.zeros([784,10]),scale=tf.ones([784,10]))
    b= yield tfp.distributions.Normal(loc=tf.zeros([10]),scale=tf.ones([10]))
    y_pred= yield tfp.distributions.Multinomial(1,probs=tf.nn.softmax(tf.matmul(x_train,w)+b))

model=tfp.distributions.JointDistributionCoroutineAutoBatched(factor)

l,f,pred=model.sample()

# take the log probability of the joint distribution as loss 
loss= (lambda l,f: model.log_prob(l,f,y_train))

# minimize the loss
tfp.math.minimize(lambda: -loss(w,b),5000,opt)

# draw a random sample of weights and biases from the joint distribution
w_t,b_t,x_pred=model.sample(value=(w,b))

# predict the train and test data output using the drawn sample
y_pred=tf.nn.softmax(tf.matmul(x_train,w_t)+b_t)
y_pred_test=tf.nn.softmax(tf.matmul(x_test,w_t)+b_t)

# calculate the accuracy of the prediction
train_accuracy=np.sum(tf.argmax(y_pred,axis=1)==tf.argmax(y_train,axis=1))/len(y_train)
test_accuracy=np.sum(tf.argmax(y_pred_test,axis=1)==tf.argmax(y_test,axis=1))/len(y_test)

print("train_accuracy: ",train_accuracy)
print("test_accuracy: ",test_accuracy)

'''
train_accuracy:  0.9385333333333333
test_accuracy:  0.9257
'''
    















