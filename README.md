# dp-predict-stock
This is the final project of NTUST CS5144701  Practices of Deep Learning course.  
Used re-constructed news title data to predict stock by deep learning(LSTM).

And this project‘goal is to improve the performance of [this paper](https://www.ijcai.org/Proceedings/15/Papers/329.pdf)'s
result.That paper uses the CNN neural network.

## technology
+ [Reverb](http://reverb.cs.washington.edu/README.html)  
+ [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory)  
+ [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)  

## data source
maybe use [this news api](https://newsapi.org/).

## library
```
pip3 install jpype1   # mayebe not use
pip3 install newsapi-python
```

## How to Use reverb
Go to the website [Reverb](http://reverb.cs.washington.edu/README.html) and download jar file.

Prepare the newsdata.txt file. And run this comamnd:
```
java -Xmx512m -jar reverb.jar newsdata.txt > reverb_result.txt
```
