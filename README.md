# OpenCVExample
This is simple example of usage OpenCV library.

Just download project and enjoy it :) (strongly for windows-users)

For running add to classpath lib from src\main\resources\..
(mb you will need to add -Djava.library.path=<path to *.dll> in case of errors with native libs)

Test file OpenCVExample-master\fire_test.txt contains some data about fires: temperature, humidity, wind power. 
And label column with count of fires. 
In our class we will test CvRTrees for prediction some combinations of input parameters.

We try to predict more that 0 fires, 5 fires and 10 fires.