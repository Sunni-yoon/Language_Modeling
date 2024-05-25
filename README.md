# Language_Modeling (인공신경망과 딥러닝)


  
--------------------------------------------------------------------------------------
## 파일 설명

```main.py``` : 모델 학습 & 테스트 전체 <br/>
```dataset.py``` : MNIST 데이터셋 전처리 <br/>
```model.py``` : LeNet-5, Custom MLP, Regularized LeNet-5 세가지 모델 구현 <br/>
```plot_visualization.py``` : main 결과값 plot 시각화 <br/>

--------------------------------------------------------------------------------------

## Plot the average loss values for training and validation <br/>
**[Training Details]**  <br/>
Epochs : 10  <br/>
Batch Size : 128  <br/>
Embedding Dimension : 128  <br/>
Hidden Dimension : 128  <br/>
Number of Layers : 2  <br/>
Learning Rate : 0.0005  <br/>
Dropout Probability : 0.5  <br/>
Validation Split : 20% <br/>
### RNN 
* Epoch 1, Train Loss: 2.2965, Val Loss: 1.9917 <br/>
* Epoch 2, Train Loss: 2.1075, Val Loss: 1.9374 <br/>
* Epoch 3, Train Loss: 2.0647, Val Loss: 1.9115 <br/>
* Epoch 4, Train Loss: 2.0422, Val Loss: 1.8949 <br/>
* Epoch 5, Train Loss: 2.0264, Val Loss: 1.8839 <br/>
* Epoch 6, Train Loss: 2.0158, Val Loss: 1.8749 <br/>
* Epoch 7, Train Loss: 2.0067, Val Loss: 1.8704 <br/>
* Epoch 8, Train Loss: 1.9993, Val Loss: 1.8619 <br/>
* Epoch 9, Train Loss: 1.9934, Val Loss: 1.8589 <br/>
* Epoch 10, Train Loss: 1.9873, Val Loss: 1.8566 <br/>
![rnn_training_validation_loss](https://github.com/Sunni-yoon/Language_Modeling/assets/118954283/14783fc3-fdda-4b3e-a019-1781c504f095)

### LSTM
* Epoch 1/10, Train Loss: 2.1926, Val Loss: 1.8468 <br/>
* Epoch 2/10, Train Loss: 1.8664, Val Loss: 1.7499 <br/>
* Epoch 3/10, Train Loss: 1.7883, Val Loss: 1.7037 <br/>
* Epoch 4/10, Train Loss: 1.7464, Val Loss: 1.6784 <br/>
* Epoch 5/10, Train Loss: 1.7185, Val Loss: 1.6620 <br/>
* Epoch 6/10, Train Loss: 1.6984, Val Loss: 1.6514 <br/>
* Epoch 7/10, Train Loss: 1.6834, Val Loss: 1.6398 <br/>
* Epoch 8/10, Train Loss: 1.6706, Val Loss: 1.6339 <br/>
* Epoch 9/10, Train Loss: 1.6602, Val Loss: 1.6248 <br/>
* Epoch 10/10, Train Loss: 1.6510, Val Loss: 1.6201 <br/>
![lstm_training_validation_loss](https://github.com/Sunni-yoon/Language_Modeling/assets/118954283/a274aa18-4234-4b63-87b8-11d9375c0f9f)

--------------------------------------------------------------------------------------

## Compare the language generation performances of vanilla RNN and LSTM in terms of loss values for validation dataset
As observed, the CharLSTM consistently shows lower validation loss compared to CharRNN across all epochs. <br/>
* [RNN] 1.8566 > [LSTM] 1.6201 <br/>
<br/>
The validation loss for CharRNN decreases gradually but stabilizes at a higher value compared to CharLSTM. <br/>
This indicates that the CharLSTM model generalizes better to the validation dataset and can generate more plausible text sequences. <br/>
It suggests that LSTM's ability to capture long-term dependencies in sequences contributes to its better performance in language generation tasks compared to RNN. 

