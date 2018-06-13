from collections import defaultdict
labels = defaultdict(list)

with open('train_val_label.txt','r') as f:
 for line in f:
  labels[line[0:8]].append(line)

ID = []
for (k,v) in labels.items():
 ID.append(k)

ID_train = []
ID_val = []
for i in range(0, len(ID)):
 if i%8 == 0:
  ID_val.append(ID[i])
 else:
  ID_train.append(ID[i])
 
train_list = file('train_list.txt',"a+")
val_list = file('val_list.txt',"a+")

for i in range(0,len(ID_val)):
 tmp = labels[ID_val[i]]
 for j in range(0,len(tmp)):
  val_list.write(tmp[j])
val_list.close()

for i in range(0,len(ID_train)):
 tmp = labels[ID_train[i]]
 for j in range(0,len(tmp)):
  train_list.write(tmp[j])
train_list.close()
