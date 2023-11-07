# construction-estimation-ml

### Setup

```
$ git clone https://github.com/sensviz/construction-estimation-ml
$ cd construction-estimation-ml
$ pip install -e requirements.txt
$ uvicorn app.main:app --reload to run the API
```
helpers/preprocess.py is to preprocess the data.

models/train.py is to train and test the mode.

main.py is to run the API.

routers/rooRouters.py contain endpoints.

The endpoint train requires the following Inputs


| KEY | DESCRIPTION                                        |
|-----|----------------------------------------------------|
|lr       | The learning Rate is given to the model              |
|epochs   | The number of epochs model should be trained on      |
|variable | Target variables such as cost or quantity            |
|split    | the percentage to which you want to split the dataset|
|Data     | Dataset in a .csv format                             |


This app helps you train the model to predict the cost or quantity from the dataset you provide to it. It will automatically do the preprocessing and training.
