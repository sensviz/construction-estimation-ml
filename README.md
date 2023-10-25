# construction-estimation-ml

### Setup

```
$ git clone https://github.com/sensviz/construction-estimation-ml
$ cd construction-estimation-ml
$ pip insttall -e requirements.txt
$ uvicorn app.main:app --reload to run the API
```
helpers/preprocess.py is to preprocess the data.

models/train.py is to train and test the mode.

main.py is to run the API.

routers/rooRouters contain endpoints.
