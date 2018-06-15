# dp-predict-stock
Final project of NTUST CS5144701  Practices of Deep Learning course.  

The goal of this project is to improve the performance of [this paper](https://www.ijcai.org/Proceedings/15/Papers/329.pdf) by using RNN-LSTM model instead of EB-CNN for prediction.


## Training Guide  
### Scripts in this project  
| Script                 | Description             |
| ---------------------- | ----------------------- |
| `process_data.ipynb`   | Fetch data from stock price API and news dataset and generate data frame pickle file `train.pkl`.
| `reverb.ipynb`         | Preprocess data for reverb OpenIE phrase extraction `reverb_pre.txt` and extract the result `reverb_result.txt` for training model `train_reduce.pkl`. 
| `word2Vec.ipynb`       | Produce word embedding model `word2vec.model`.
| `main.ipynb`           | Model training and evaluation.


### Reverb usage  
Download jar from the [Reverb Website](http://reverb.cs.washington.edu/README.html)
Prepare newsdata.txt and run:
```
java -Xmx512m -jar ./test-reverb/reverb.jar reverb_pre.txt reverb_result.txt
```


## About the Datasets
- News dataset: [financial-news-dataset](https://github.com/philipperemy/financial-news-dataset) includes 450,341 news from Bloomberg and 109,110 news from Reuters within the date range from 2000 to 2013.

- Stock price dataset: [alphavantage](https://www.alphavantage.co) API.


## References
+ [Reverb](http://reverb.cs.washington.edu/README.html)  
+ [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory)  
+ [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)    
+ [Stanford-OpenIE-Python](https://github.com/philipperemy/Stanford-OpenIE-Python)  


## Progress
1. [x] Decide using which dataset, including stock data and news data (2018-05-20).
2. [x] To know how to use reverb to split one news title sentence (2018-05-25).
3. [x] Process the news data: got the title and time (2018-05-27).
4. [x] Process the stock data: got the price (2018-05-28).

5. [x] Reconstruct the stock price data, only save the result that compared with last day price.If today's price is smaller than last days, it will be saved on 0 and other situation will be saved on 1 (2018-05-29).

6. [x] Using the Reverb technology to split a news title sentence of all news's data to the three part format (2018-05-29).

7. [x] Generate the word_embedding model, using reverb-result  (2018-05-30).
8. [x] Generate embedding_matrix, using Keras's tool (2018-05-30).
9. [x] Construct the neural network model(Event-LSTM) (2018-05-30).
10. [x] Train and predict (2018-05-30).
11. [ ] If the predict's result dont bigger than 65%, to find and modify the errors (2018-06-01).
12. [ ] Generate the final project documents and slides  (2018-06-18).
