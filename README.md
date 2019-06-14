## Weibo Post Emoji Prediction
[NJU's data mining course final contest in spring 2019](https://www.kaggle.com/c/mining-challenge-for-nju-introdm-2019/)

## Data description and preprocessing
The task is to predict a suitable emoji for users' posts. There are 72 different emoji (labels) in total, and the whole data consist of ~820k posts with their corresponding labels. The occurrence of labels are extremely imbalanced, with the most popular emoji *laugh to cry* appears hundreds of times more than rare emoji. In fact, almost all labels occurs over 10 times lesser than *laugh to cry*. Most of the post contents are written in Chinese with a minority in English, number, and other emoji which is not the label we would like to predict. This makes cleaning the data difficult. I just replace the *troublesome* English quotes (" and ') with ' ' and remove the end-of-line token. Then I leave all the work to [jieba](https://github.com/fxsjy/jieba) and [pkuseg](https://github.com/lancopku/pkuseg-python). The word vector I use comes from [this repository](https://github.com/Embedding/Chinese-Word-Vectors). Specifically, [skip gram w/ negative sampling](https://github.com/Embedding/Chinese-Word-Vectors#various-domains) on weibo, which is exactly the domain we would like to train our model on. I have tried on 300d vector with only word as context features, and 300d vector with word, character and ngram as features. The latter performs a little bit better.
## Model
### TextCNN
Adapt from gluonnlp model zoo: *IMDB review binary classification* [here](https://gluon-nlp.mxnet.io/model_zoo/sentiment_analysis/index.html?highlight=textcnn). I have read a great introduction to it but can not find it now. But [this blog](https://blog.csdn.net/chuchus/article/details/77847476) can serve as a good tutorial.

Each epoch takes around 2 minutes, and the VRAM consumption w.r.t. my hyperparameter setting is 2G. 
### Self attention sentence embedding
Adapt from gluonnlp tutorial [here](https://gluon-nlp.mxnet.io/examples/sentiment_analysis/sentiment_analysis.html). The webpage (which is actually a jupyter notebook) contains all codes and a detailed introduction.

Each epoch takes several minutes to ten minutes, and the VRAM consumption w.r.t. my hyperparameter setting is ~6.9G.
### bert
Adapt from gluonnlp model zoo [here](https://gluon-nlp.mxnet.io/model_zoo/bert/index.html). There are many blogs explaining what bert is and how it works. Due to insufficient memory, I can not truly apply it to this dataset. I run on CPU to get rid of superficial bugs, but I believe there are still some left out. Since I have run bert on AWS before, I think the program alone will consume more than 12G VRAM. Thus 8G VRAM on my laptop runs out in 3 sec. Switch to float16 might help. Besides, run on CPU will take around the same amount of memory.
## Performance and potential improvement
### TextCNN
- Achieve ~.17 f1 score w/o sufficient fine-tuning.

May be improved through the following methods, and more:
- From simple train, validation data separation (.9 and .1) to k-fold cross validation.
- Add more convolution layer, or change the kernel size, channel number.
- Change trainer, loss function. Especially, give appropriate weight to resolve data imbalance.

### Self attention sentence embedding
- Achieve ~.14 f1 score w/o sufficient fine-tuning.

May be improved through the following methods, and more:
- Adjust hyperparameters, especially the number of bi-lstm layers and the number of attention hops. Since most posts are fairly short. A lot of them tend to overfit badly.
- Find better weight vector for weighted softmax cross entropy loss. My current heuristic can be found in util.py.
  
### bert
With mixed precision (fp16 and fp32), bert still can not fit into 8G VRAM.
