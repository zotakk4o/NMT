# NMT
A university project for Neural Machine Translation from English to Bulgarian.

## Installation
1. Extract ```NMTModel.zip``` from model folder to the main directory of the project.
2. Extract ```wordsData_corpusData``` to the main directory of the project.
3. Run the project

## How to run the project
The following commands are supported:
 - python run.py train - trains the model on en_bg_data/train.en and en_bg_data/train.bg
 - python run.py extratrain - continues the training process on an already-existing model.
 - python run.py translate <source> <destination> - translates the <source> and saves it as <destination>
 - python run.py bleu <target> <test> - calculates BLEU score for the <test> corpus, compared to the <result> one.
