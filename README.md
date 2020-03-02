# 1. TensorFlow in Practice Specilization

Hi all, I have been recently learning the specilization course of TensorFlow published by [deeplearning.ai](https://www.deeplearning.ai/) on [Coursera](https://www.coursera.org/). There are 4 courses in this specilization:
- [Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning](https://www.coursera.org/learn/introduction-tensorflow/home/welcome)
- [Convolutional Neural Networks in TensorFlow](https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/home/welcome)
- [Natural Language Processing in TensorFlow](https://www.coursera.org/learn/natural-language-processing-tensorflow/home/welcome)
- [Sequences, Time Series and Prediction](https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/home/welcome)

# 2. About this Repo
For this repo, I'll share my notes, codes, and answers of coding assignments. And you know, the most difficult problems in Python or any other programming languge are some "too specific problems" (e.g., spelling, capital, tab, or the inconsistence between your Python versiona and computer version). Also, I'll share some experiences that I overcome the specific problems.

The source code of each course of this specilization could be found in the 4 Jupyter Notebook (.ipynb) files and currently only one could be seen.

# 3. Progress Update
## 3.1 Intro to TensorFlow for AI, ML, and DL
Week 2 Exercise
```
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.99): # or just try aother parameter: logs.get('acc')
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, callbacks=[myCallback()])
```
Week 4 Exercise
```
def train_happy_sad_model():
    DESIRED_ACCURACY = 0.999
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('acc')>DESIRED_ACCURACY): # or just try aother parameter: logs.get('acc')
                print("\nReached 99.9% accuracy so cancelling training!")
                self.model.stop_training = True
    callbacks = myCallback()
    
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)), # 3 means rgb color
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(), # the pattern of Flatten + Hidder + Output Neuron
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])
    from tensorflow.keras.optimizers import RMSprop

    model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])
              
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1/255)
    
    train_generator = train_datagen.flow_from_directory(
          '/tmp/h-or-s',
        target_size=(150, 150), 
        batch_size=128,
        class_mode='binary')
        
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=8,  
        epochs=15,
        verbose=1,
        callbacks=[callbacks])
    return history.history['acc'][-1]
```
Finally, please feel free to contact me or leave me an issue in GitHub. It would be great to have some future discussions with all of you!  


----  
Xin Jin  
Senior Research Assistant  
Dept. Media and Communication, City Univ. HK  
[About](www.xjin.tech) | [LinkedIn](linkedin.com/in/xjin613/) | [Twitter](https://twitter.com/xjin_comm) | xin.jin@cityu.edu.hk
