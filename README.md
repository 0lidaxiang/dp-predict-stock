# dp-predict-stock
This is the final project of NTUST CS5144701  Practices of Deep Learning course.  
Used re-constructed news title data to predict stock by deep learning(LSTM).

And this project‘goal is to improve the performance of [this paper](https://www.ijcai.org/Proceedings/15/Papers/329.pdf)'s
result.That paper uses the CNN neural network.

## Used Technology & Library
+ [Reverb](http://reverb.cs.washington.edu/README.html)  
+ [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory)  
+ [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)    
+ [Stanford-OpenIE-Python](https://github.com/philipperemy/Stanford-OpenIE-Python)  

#### How to Use reverb
Go to the website [Reverb](http://reverb.cs.washington.edu/README.html) and download jar file.

Prepare the newsdata.txt file. And run this comamnd:
```
java -Xmx512m -jar reverb.jar newsdata.txt > reverb_result.txt
```

## Dataset
This [financial-news-dataset](https://github.com/philipperemy/financial-news-dataset) is from Bloomberg and Reuters, including 450,341 news from Bloomberg and 109,110 news from Reuters.

Stock data is from  this api website - [alphavantage](https://www.alphavantage.co).

## Progress
1. [x] Decide using which dataset, including stock data and news data (2018-05-20).

2. [x] To know how to use reverb to split one news title sentence (2018-05-25).

3. [x] Process the news data: got the title and time (2018-05-27).

4. [x] Process the stock data: got the price (2018-05-28).

5. [ ] Using the Reverb technology to split a news title sentence of all news's data to the three part format (2018-05-29).

6. [ ] Generate the word_embedding model, using reverb-result  (2018-05-29).

7. [ ] Generate embedding_matrix, using Keras's tool (2018-05-30).
8. [ ] Construct the neural network model (2018-05-30).
9. [ ] Train and predict (2018-05-31).
10. [ ] If the predict's result dont bigger than 65%, to find and modify the errors (2018-06-01).
11. [ ] Generate the final project documents and slides  (2018-06-18).
