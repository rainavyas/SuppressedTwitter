# SuppressedTwitter
Twitter Emotion Classifier with suppressed projection matrix singular values for robustness.

# Objective

NLP classification of twitter tweets into one of six emotions: love, joy, fear, anger, surprise, sadness.
The dataset is described in https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp?select=train.txt

The model training is adapted for robustness against adversarial attacks. This is achieved by suppressing the singular value sizes of the Transformer architecture projection matrices. 


# Requirements

python3.4 or above

## Necessary Packages (installation with PyPI)

pip install torch, torchvision

pip install transformers


# Training

Fork the repository (and clone).

Run the _train.py_ scripts with desired arguments in your terminal. For example, to train an ELECTRA-based classifier:

_python ./train.py electra_trained.th electra --B=8 --lr=0.00001 --epochs=5 --suppression=0.1_

# Experimental Results

| Epochs | Test Accuracy (%) | Singular Value Sum |
| :-----------------: | :-----------------: | :-----------------: |
20 | 92.3 | 527 |
40 | 91.1 | 253 |
60 | 87.1 | 120 |


Note: Training without suppression achieved an accuracy of 93.3% and had a singular value sum of 1361.

### Training Details

- Initialise encoder with _electra_
- Batch Size = 16
- Epochs = Dependent on level of suppression
- Learning Rate = 1e-5
- Suppression cost function ceofficient = 10.0
