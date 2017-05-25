
## Intallation of Python, TensorFlow and Keras

0. If you need to remove old versions of Python

```
sudo rm -rf /Library/Frameworks/Python.framework
cd /usr/local/bin
ls -l . | grep '../Library/Frameworks/Python.framework' | awk '{print $9}' | xargs sudo rm
```
Also, you have to remove the program from Applications folder, for instance
```
sudo rm -rf "/Applications/Python 2.7"   
```

1. Python installation

	* https://www.python.org
	* Downloads -> Python 3.6.1 
	* Go to 'Downloads' folder and install the software

2. Open python in **terminal**. 
```
python2.7
python3.6
python                      # Open the default python installation
```
Change the default python installation by editing bash_profile in the **terminal**.
```
vim ~/.bash_profile
```
and add, for instance
```
alias python='python3'
```

3. Install pip.
```
sudo easy_install pip
```

4. Install TensorFlow library in Mac from **terminal**. More datails can be found in the [(official webpage)](https://www.tensorflow.org/install/)
```
pip install tensorflow	 # Python 2.7
pip3 install tensorflow	 # Python 3.x
```
You can add option ```–-user``` to install it in users folder. To verify the installation you can run a first program in **Python**
```	
import tensorflow as tf
hello = tf.constant('Hello world!')
sess = tf.Session()
sess.run(hello)
```
If installation failed, install the latest version of TensorFlow by issuing a command of the following format:
```
sudo pip  install --upgrade TF_BINARY_URL   # Python 2.7
sudo pip3 install --upgrade TF_BINARY_URL   # Python 3.x 
```
If installation failed again, try reinstalling it like this:
```
sudo pip uninstall tensorflow
pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py3-none-any.whl
```

5. Install Keras library from **terminal**.
```
sudo pip install keras		# Python 2.7
sudo pip3 install keras		# Python 3.x
```

You can start by running your first Neural Network program in **Python**
```
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Activation  
x = np.random.uniform(-1,1,[1000,100])
b = np.random.normal(0,1,100)
e = np.random.normal(0,1,1000)
u = np.dot(x,b)
y = u + e
x_train = x[range(200,1000),:]
x_test = x[range(0,200),:]
y_train = y[range(200,1000)]
y_test = y[range(0,200)]
model = Sequential()
model.add(Dense(50,input_dim=100,activation='relu')) 
model.add(Dense(units=10))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=800)
yHat = model.predict(x_test, batch_size=800)[:,0]
np.corrcoef(y_test,yHat)[0,1]
```
