
# 
# This graph describes the problem that we are trying to solve visually. We want to create and train a model that takes an image of a hand written digit as input and predicts the class of that digit, that is, it predicts the digit or it predicts the class of the input image.
# 
# ![Hand Written Digits Classification](images/1_1.png)

# ### Import TensorFlow

# In[18]:


import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
print('Using TensorFlow version', tf.__version__)


# # Task 2: The Dataset
# ### Import MNIST

# In[19]:


from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) =  mnist.load_data() 


# ### Shapes of Imported Arrays

# In[10]:


print('x_train shape:', x_train.shape) #amt of examples in the datase, 28 rows, 28 columns
print('y_train shape:', y_train.shape) 
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)


# ### Plot an Image Example

# In[43]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.imshow(x_train[59999], cmap='binary')
plt.show()


# ### Display Labels

# In[20]:


y_train[59999]


# In[19]:


print(set(y_train)) #prints the classes that we have


# # Task 3: One Hot Encoding
# After this encoding, every label will be converted to a list with 10 elements and the element at index to the corresponding class will be set to 1, rest will be set to 0:
# 
# | original label | one-hot encoded label |
# |------|------|
# | 5 | [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] |
# | 7 | [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] |
# | 1 | [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] |
# 
# ### Encoding Labels

# In[21]:


from tensorflow.keras.utils import to_categorical

y_train_encoded = to_categorical(y_train) #to_categorical function encodes y_train and y_test
y_test_encoded = to_categorical(y_test)


# ### Validated Shapes

# In[22]:


print('y_train_encoded shape:', y_train_encoded.shape)
print('y_test_encoded shape:' , y_test_encoded.shape)

#its still the same amount of examples for both training and test set
#each example is a 10 dimensional vector as per the y value


# ### Display Encoded Labels

# In[35]:


y_train_encoded[0]


# # Task 4: Neural Networks
# 
# ### Linear Equations
# 
# ![Single Neuron](images/1_2.png)
# 
# The above graph simply represents the equation:
# 
# \begin{equation}
# y = w1 * x1 + w2 * x2 + w3 * x3 + b
# \end{equation}
# 
# Where the `w1, w2, w3` are called the weights and `b` is an intercept term called bias. The equation can also be *vectorised* like this:
# 
# \begin{equation}
# y = W . X + b
# \end{equation}
# 
# Where `X = [x1, x2, x3]` and `W = [w1, w2, w3].T`. The .T means *transpose*. This is because we want the dot product to give us the result we want i.e. `w1 * x1 + w2 * x2 + w3 * x3`. This gives us the vectorised version of our linear equation.
# 
# A simple, linear approach to solving hand-written image classification problem - could it work?
# 
# ![Single Neuron with 784 features](images/1_3.png)
# 
# ### Neural Networks
# 
# ![Neural Network with 2 hidden layers](images/1_4.png)
# 
# This model is much more likely to solve the problem as it can learn more complex function mapping for the inputs and outputs in our dataset.

# # Task 5: Preprocessing the Examples
# 
# ### Unrolling N-dimensional Arrays to Vectors

# In[23]:


import numpy as np

#reshaping the vectors for our x train models using np.reshape, y value represents the desired shape
x_train_reshaped = np.reshape(x_train,(60000,784))
x_test_reshaped = np.reshape(x_test, (10000,784))

#seeing if they got reshaped
print('x_train_reshaped:' , x_train_reshaped.shape)
print('x_test_reshaped shape:' , x_test_reshaped.shape)


# ### Display Pixel Values

# In[39]:


print(set(x_train_reshaped[0]))
#this prints values available in the 0th/first example of data fed to TensorFlow


# ### Data Normalization

# In[24]:


import numpy as np
x_mean = np.mean(x_train_reshaped)
x_std = np.std(x_train_reshaped)

epsilon = 1e-10
x_train_norm =  (x_train_reshaped - x_mean) / (x_std + epsilon)
x_test_norm = (x_test_reshaped - x_mean) / (x_std + epsilon)


# ### Display Normalized Pixel Values

# In[ ]:


print(set(x_train_norm[0]))
(/much, smaller, values, because, we, struck, down, distribution, of, the, scale)


# # Task 6: Creating a Model
# ### Creating the Model

# In[5]:


#creating neural network with sequential class defined in keras and add layers
#two hidden layers with 128 nodes each and one output layer with 10 nodes for the 10 classes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#Input layer is just the input examples when using 'Sequential' class
#output of first line of dense layer is the input of the second line of dense layer.
#no need to specify input shape for second dense layer
#128,128,10 are all the number of nodes for each respetive dense node
#Each line of code within the sequential class is a "layer"
model = Sequential([
    Dense(128, activation = 'relu', input_shape =(784,)),
    Dense(128, activation ='relu'),
    Dense(10, activation ='softmax')
])


# ### Activation Functions
# 
# The first step in the node is the linear sum of the inputs:
# \begin{equation}
# Z = W . X + b
# \end{equation}
# 
# The second step in the node is the activation function output:
# 
# \begin{equation}
# A = f(Z)
# \end{equation}
# 
# Graphical representation of a node where the two operations are performed:
# 
# ![ReLU](images/1_5.png)
# 
# ### Compiling the Model

# In[8]:


#Predicted output and actual outputs given difference, should be minimized for more acurate model
#optimization algorithm minimizes this difference 

model.compile(
    optimizer= 'sgd',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model.summary() #displays architecture of the model
#model shows first dense layer with 128 nodes and dense_1 with 128 nodes and so on


# # Task 7: Training the Model
# 
# ### Training the Model

# In[26]:


#epoch is an iteration of all the examples going through the model
#its going to go thru the examples 8 times as epoch is set to 8
# the more epochs, the more accurate

model.fit(x_train_norm, y_train_encoded, epochs=8)


# ### Evaluating the Model

# In[41]:


#making the sure the model didnt memorize the examples 
#if this accuracy is high then thats good but if low then model just memorized examples
#does forward pass to understand the prediction of the model and compares with the actual labels
loss, accuracy = model.evaluate(x_test_norm, y_test_encoded)
if accuracy * 100 > 89:
    print("Training was successful! Accuracy was:", str(accuracy*100) + '%')
    #print("Training was successful! Accuracy was:", accuracy*100)
elif accuracy * 100 < 89:
    print("Training failed. Accuracy was:", accuracy*100)



# # Task 8: Predictions
# 
# ### Predictions on Test Set

# In[40]:


preds = model.predict(x_test_norm)
print('Shape of predictions:', preds.shape)


# ### Plotting the Results

# In[46]:


plt.figure(figsize=(12,12))

#gt represents groundtest
start_index = 0

for x in range(25):
    plt.subplot(5, 5, x+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    pred = np.argmax(preds[start_index+x])
    gt = y_test[start_index+x]
    
    col = 'g'
    if pred != gt:
        col = 'r'
        
    plt.xlabel('i={}, pred = {}, gt = {}'.format(start_index+x, pred, gt), color=col)
    plt.imshow(x_test[start_index+x], cmap='binary')
plt.show()
    


# In[48]:


plt.plot(preds[8])
plt.show()
#tells us the softmax probability when looking at index 8, the prediction is 5 and its indeed 5
#theres a super low probability of being 8 as observable in the graph


# In[ ]:





# In[ ]:




