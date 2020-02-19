import numpy as np

label_no = np.load('../label_no.npy')
label_yes = np.load('../label_yes.npy')

label_test = label_yes[-470:]

counter1 = 0
counter2 = 0

for i in range(len(label_test)):
    if label_test[i] == 0:
        counter1 = counter1 + 1
    elif label_test[i] == 1:
        counter2 = counter2 + 1

print(counter1)
print(counter2)