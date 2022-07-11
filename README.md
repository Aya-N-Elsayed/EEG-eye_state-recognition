# EEG-eye_state-recognition
Using Baden-Wuerttemberg Cooperative State University dataset
## Dataset
The data set consists of 14 EEG values and a value indicating the eye state. The eye state was detected via a camera during the EEG measurement and added later manually to the file after analysing the video frames. '1' indicates the eye-closed and '0' the eye-open state. 
## Results
![image](https://user-images.githubusercontent.com/33070648/178268767-a5e6c17b-f547-45ef-9a0f-426d9bf1d035.png)

## Conclusion

## To run the code:
For the Arguments:__
--classifier-- for 'classifier name'__
--dataset-- for 'dataset name'__
--scaling--  for 'feature scaling enter 1'__
Examples:
>python main.py --classifier  KNN --scaling 1  --dataset  EEGEyeState.ARFF
>python main.py --classifier LinearSVM --scaling 1  --dataset  EEGEyeState.ARFF
>python main.py --classifier RBFSVM --scaling 1  --dataset  EEGEyeState.ARFF
>python main.py --classifier DT --scaling 1  --dataset  EEGEyeState.ARFF
