import numpy as np
from sklearn.metrics import f1_score

class FireMetrics():
    def __init__(self):
        self.pres = []   # shape: data_len * num_classes
        self.labels = [] # shape: data_len

        

    def update(self, pres, labels):
        self.pres.extend(pres)
        self.labels.extend(labels)

    def reset(self):
        self.pres = []
        self.labels = []


    def _get_numpy_id(self):
        pres = np.array(self.pres)
        labels = np.array(self.labels)

        if len(pres.shape)==2:
            if pres.shape[1]>1:
                pres = np.argmax(pres, axis=1)
            else:
                pres = pres[:,0]
        
        if len(labels.shape)==2:
            if labels.shape[1]>1:
                labels = np.argmax(labels, axis=1)
            else:
                labels = labels[:,0]

        return pres,labels


    def get_acc(self):
        pres,labels = self._get_numpy_id()

        right = (pres == labels).sum()
        acc = right/len(pres)
        return acc

    def get_F1(self, mode='macro'):
        # mode:  macro  micro  weighted
        pres,labels = self._get_numpy_id()

        f1 = f1_score(labels, pres, average=mode)

        return f1
