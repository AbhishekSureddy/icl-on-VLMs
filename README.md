# icl-on-VLMs

The repository has code to perform In-context learning on Vision tasks like image captioning, Visual Q&A

## Setting Environment

- On a create a new python virtual environment (python version = 3.9.20)
```
module load conda/latest
conda create --name open-flamingo python=3.9.20
```
- Once the environment is created, activate it using the below command
```
conda activate open-flamingo
```
- To install the requirements, run the below command
```
pip install -r open-flamingo-requirements.txt
```

## Gathering Incontext examples

refer to flamingo/rice.py or RICE.ipynb to understand how we can extract the similar in-context demonstrations. referred from: https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/eval/rices.py

## Image Captioning

- related files: flamingo/flamingo_e2e_captioning.py
- refer to flamingo/flamingo_e2e.ipynb 