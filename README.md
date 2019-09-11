# Google Landmark Recognition 2019 13th Solution

## Requirement
- pytroch
- cv2
- scikit-learn==0.20.3
- scipy==1.2.0
- numpy==1.16.1
- pandas==0.24.2
- joblib==0.13.2

## Model Architecture
SE-ResNeXt50 + ArcFace

## Module Structure
- common
  - logger.py
  - util.py
- model
  - loss.py
  - metrics.py
  - model_util.py : save and load model weight
  - model.py
- config.py
- dataset.py
- main.py
- preprocessing.py : random sampling of training
- tvp.py : train, validate and predict

## How to run
```
python -m src.main
```
