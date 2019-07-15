## Gluon Classifier

classification network implemented by Gluon. The training script was generally adopted from Gluoncv.

### Supported:
1. ShuffleNet
2. more is on the way

### Results:
model_name| papers| ours 
--|:--:|--:  
ShuffleNet0.3_g3| 67.4 | 63.1

#### Difference with papers
for shufflenet, we trained half batches as the papers, and we use train/val2017, but not 2012.



