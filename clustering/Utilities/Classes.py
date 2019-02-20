import numpy as np

class Point:
    '''
    :var pos is position in carthesian coordinates
    '''
    def __init__(self, pos):
        self.pos = pos

    def dist(self, other): #uses euclidian distance for now.
        return np.linalg.norm(np.subtract(self.pos, other.pos))

    def vectorAdd(self, other):
        if(len(self.pos) == len(other.pos)):
            self.pos = self.pos + other.pos
            return
        print("Dimensions don't agree.")

class Set:
    '''
    :var points is a set of point objects
    '''

    def __init__(self, dim):
        '''
        constructor that sets all vars to null.
        '''
        self.dim = dim
        self.pList = []
        self.empty = True

    def __add__(self, other):
        self.pList.append(other)
        if self.empty:
            self.posMat = other.pos
            self.empty = False
        else:
            self.posMat = np.vstack([self.posMat, other.pos])
    def getKernel(self):
        kernel = Point(np.zeros(self.dim))
        for e in self.pList:
            kernel.vectorAdd(e)
            kernel.pos = kernel.pos/len(self.pList)
        return kernel

