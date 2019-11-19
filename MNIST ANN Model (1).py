
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


tf.__version__


# In[3]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST/", one_hot = True)


# In[4]:


sess = tf.InteractiveSession()


# In[5]:


x  = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# In[6]:


W = tf.Variable(tf.zeros([784, 10], tf.float32))
b = tf.Variable(tf.zeros([10], tf.float32))


# In[7]:


sess.run(tf.global_variables_initializer())


# In[8]:


tf.matmul(x, W) + b


# In[9]:


y = tf.nn.softmax(tf.matmul(x, W) + b)


# In[10]:


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))


# In[11]:


train_step = tf.train.GradientDescentOptimizer(.5).minimize(cross_entropy)


# In[12]:


for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict = {x: batch[0], y_: batch[1]})


# In[13]:


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = accuracy.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels}) * 100
print("Final accuracy of this simple ANN model is: {} %".format(acc))


# In[14]:


sess.close()

