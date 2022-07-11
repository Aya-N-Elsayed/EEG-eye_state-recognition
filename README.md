# EEG-eye_state-recognition
Using Baden-Wuerttemberg Cooperative State University dataset

## Results
![image](https://user-images.githubusercontent.com/33070648/178268767-a5e6c17b-f547-45ef-9a0f-426d9bf1d035.png)

To run the code:
python main.py --classifier  KNN --scaling 1  --dataset  EEGEyeState.ARFF
python main.py --classifier LinearSVM --scaling 1  --dataset  EEGEyeState.ARFF
python main.py --classifier RBFSVM --scaling 1  --dataset  EEGEyeState.ARFF
python main.py --classifier DT --scaling 1  --dataset  EEGEyeState.ARFF
