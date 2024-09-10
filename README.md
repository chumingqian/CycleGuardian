# CycleGuardian
A  lightwight  Framework  for the  Respiratory Sound  Classification


Auscultation plays a pivotal role in early respiratory and pulmonary disease diagnosis.
Despite the emergence of deep learning-based methods for automatic respiratory
sound classification post-Covid-19, limited datasets impede performance
enhancement. Distinguishing between normal and abnormal respiratory sounds poses
challenges due to the coexistence of normal respiratory components and noise
components in both types. Moreover, different abnormal respiratory sounds exhibit
similar anomalous features, hindering their differentiation. Besides, existing state-of-
the-art models suffer from excessive parameter size, impeding deployment on
resource-constrained mobile platforms.



To address these issues, in this  project, we design a  lightweight network CycleGuardian and propose a framework based on an improved
deep clustering and contrastive learning. We first generate a hybrid spectrogram for
feature diversity and grouping spectrograms to facilitating intermittent abnormal sound
capture.Then, CycleGuardian integrates a deep clustering module with a similarity-
constrained clustering component to improve the ability to capture abnormal features
and a contrastive learning module with group mixing for enhanced abnormal feature
discernment. Multi-objective optimization enhances overall performance during
training. In experiments we use the ICBHI2017 dataset, following the official split
method and without any pre-trained weights, our method achieves Sp: 88.92 $\%$,
Se: 33.33$\%$, and Score: 61.12$\%$ with a network model size of 27M, comparing to
the current model, our method leads by nearly 5$\%$, achieving the current best
performances.Additionally, we deploy the network on Android devices, showcasing a
comprehensive intelligent respiratory sound auscultation system.


# trian 

change the num_worker to  suit your  machine;  also, you may  change the  batch size, but it will  affect the final  score.

```python
python  train_lab8_6_3CycleGuardian_specAug.py    --data_dir ./data/ICBHI_final_database --dataset_split_file ./data/patient_trainTest6_4.txt --model_path ./models_out --lr_h 0.001 --lr_l 0.001 --batch_size 128 --num_worker 15 --start_epochs 0 --epochs 600
```

