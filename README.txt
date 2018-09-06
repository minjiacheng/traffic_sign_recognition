code adapted from https://github.com/preritj/Traffic-Sign-classifier

Need tensorflow 0.12 (yes the version from 2yrs ago) and Python 3.5 to run - incompatible with newer version of tf
Biggest learning outcome: when people say TF ver0.XX is incompatible with TF ver1.XX, they mean it! 
Code might still run but you will probably run into all sorts of weird errors and inconsistencies, major change of syntax and problem when save/loading model
Felt tensorflow is very much similar to keras, but working with TF helps with understanding Keras properly. TF definitely more complicated to use than Keras as you need 
to define every little detail but also offers more freedom. However you could define loss functions and model structures etc in a similar style in Keras too

A pure TF approach. 
run Train.py to train the model
run Test.py to make predictions on all of our own images in myData folder

used GTSRB (German traffic sign) which is a benchmark dataset. data all pickled up nicely
see report/classes.jpg for the 43 different classes model can predict
generated fake data by random augmentation so we have 3000 examples per class in training set
Good example for image augmentation - different linear transformations and dataset explorations etc that may be of interest
 
data are pickled and stored in traffic-signs-data folder

Place the images you want to test in myData folder
model saved to models folder
model structure is essentially VGGnet with minor tweaks
98% test accuracy (on 12k images)
works reasonably well in non german signs

see the original repo for detailed data exploration and explanation

strange bug: if only define the model architecture once then model will always predict everything as bicycle crossing
building for a 3rd time makes model return error "key variable not found"
work around for now is to define architecture twice

