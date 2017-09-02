import numpy as np
import pylab
import os

file1 = os.path.join('./logs_liberty', 'notredame.txt')
file2 = os.path.join('./logs_liberty_default', 'notredame.txt')
def loadData(fileName):
    with open(fileName, 'r') as f:
        file = open(fileName, 'r')
        data = file.read()
        nlines = data.split("\n")
    list_fpr95 = [line for line in nlines if len(line) > 0]
    x = range(len(list_fpr95))
    y = np.asarray(list_fpr95)
    y = y.astype(np.float32)

    return (x,y)

x1, y1 = loadData(file1)
x2, y2 = loadData(file2)

print('default: ', (100*np.min(y2)))
print('variance: ', 100*np.min(y1))
pylab.plot(x1, y1, color="blue", linewidth=1.0, linestyle="-", label="variance_constraint")
pylab.plot(x2, y2, color="green", linewidth=1.0, linestyle="-", label="default_loss")

pylab.legend(loc='lower right')

pylab.xlim(-0.5, 10.5)
pylab.xlabel('Epoches', size=15)
pylab.ylabel('False Positive Rate at 95% Recall', size=15)

pylab.show()

