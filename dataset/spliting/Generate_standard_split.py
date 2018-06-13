trainval_label = 'train_val_label.txt'
test_label = 'test_label.txt'
trainval_label_file = file(trainval_label, "a+")
test_label_file = file(test_label, "a+")

list_dict = {}
with open('Full_Label.txt','r') as t:
 for line in t:
  key = line.split()[0]
  list_dict[key] = line

num = 0
with open('train_val_list.txt','r') as s:
 for line in s:
  num = num + 1
  if num % 10000 == 0:
   print num
  tmp = line.split()[0]
  trainval_label_file.write(list_dict[tmp]) 

num = 0
with open('test_list.txt','r') as s:
 for line in s:
  num = num + 1
  if num % 10000 == 0:
   print num
  tmp = line.split()[0]
  test_label_file.write(list_dict[tmp]) 
