# Language_Modeling (인공신경망과 딥러닝)

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
* As observed, the CharLSTM consistently shows lower validation loss compared to CharRNN across all epochs. <br/>
  * [RNN] 1.8566 > [LSTM] 1.6201 <br/>
* The validation loss for CharRNN decreases gradually but stabilizes at a higher value compared to CharLSTM. <br/>
* This indicates that the CharLSTM model generalizes better to the validation dataset and can generate more plausible text sequences. <br/>
* It suggests that LSTM's ability to capture long-term dependencies in sequences contributes to its better performance in language generation tasks compared to RNN. <br/>

--------------------------------------------------------------------------------------

## Generating at least 100 length of 5 different samples from different seed characters
**1. love** <br/>
Generated Text : <br/>
loved men and beow'd. Now you dake not,He as he is sher, if my farewal sat 'twas ome of Rome, let that 

**2. mind** <br/>
Generated Text : <br/>
mind of seif a thing! I stouting was say them you dranred maves within people:The marreur goistings for <br/>

**3. child** <br/>
Generated Text : <br/>
child is resby my from my good of blage than betber. BUCKINGHAM:Your partot wath? our ibhance, bloody h <br/>

**4. good** <br/>
Generated Text : <br/>
good confast.CLARENCE:I holy bonicter in offineit is my noble cares to hate undocchess the brow with <br/>  

**5. happy** <br/>
Generated Text : <br/>
happy but of your bear you breath of him,A voices King apony the what.QUEEN MARGARET:I hators You hav

--------------------------------------------------------------------------------------

## Generating at least 100 length of 5 different samples from different seed characters
 
**Temperatures : 0.1 / 0.5 / 1.0 / 2.0 / 5.0** <br/>
**word : happy** <br/>

* Temperature 0.1 : <br/>
  * Generated Text : happy the country and the country to the son the consul of the world of the consul and the country so the <br/>
  * At a temperature of 0.1, the generated text is very repetitive and lacks diversity. <br/>

* Temperature 0.5 :
  * Generated Text : happy is such horse of the service:For the holy for with the king and do so streak of the must let the n <br/>
  * At a temperature of 0.5, the text shows some diversity while maintaining consistency. The generated text has more variety and some meaningful structure but can still be somewhat repetitive. <br/>

* Temperature 1.0 : <br/>
  * happy to the wife. LADY ANNE:If, But other duse this thought him, perserful wintanted To lose me, my lo <br/>
  * At a temperature of 1.0, the model generates more diverse text while maintaining coherence. This setting allows the model to produce plausible and varied text sequences without being too conservative or too random. <br/>

* Temperature 2.0 : <br/>
  * happysry?To-muldingsser, draw Gringlim:-oden has, I, he, ruve dliled unmoy muving, and uf thrumedady',-a <br/>
  * At a temperature of 2.0, the text becomes more diverse but loses consistency. The model generates creative but less meaningful output. <br/>

* Temperature 5.0 : <br/>
  * happyhnleyiemr;!:wAdPhPrS?Ckno! wh byxRo eNtonduusook'r;; minf  mBqlei!Esvvoum ZanK AefocDaj.! Edos-de. <br/>
  * At a temperature of 5.0, the generated text is highly random. The model samples almost uniformly from the probability distribution, resulting in nonsensical output. <br/>
  
 <br/>
 
**Discussion** <br/>
* The temperature parameter in the softmax function controls the randomness of the generated text. <br/>
* A lower temperature (<1) makes the model more conservative, resulting in repetitive and less diverse text. <br/>
* A higher temperature (>1) increases diversity but can lead to less coherent and more random outputs. <br/> 
* As shown in the results above, the model generates different values depending on the temperature parameter. Therefore, adjusting the temperature parameter can help generate more plausible results. <br/>
* Generally, a temperature around 1.0 provides a good balance for generating plausible and varied text.
