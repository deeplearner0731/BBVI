from sklearn.model_selection import train_test_split as train_test
import autograd.numpy.random as npr
import numpy as np 

def load_data(seed=0,state=32):
  
    
    npr.seed(seed)
    
    data = np.load("./data/new_data1D.npz")
    #data = np.load("./data/moon_data.npz")
    x = data['x']
    y = data['y']
    ids = np.arange(x.shape[0])
    npr.shuffle(ids)

    x_train, x_test, y_train, y_test = train_test(x, y, test_size=0.3,random_state=state)
    mu = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train = (x_train-mu)/std
    x_test = (x_test-mu)/std
 
    return x_train, y_train, x_test, y_test