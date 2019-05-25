from random import random


class Matrix:
    '''
    Matrix object for storing two dimensional data
    '''

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = [[0 for j in range(cols)] for i in range(rows)]

    @staticmethod
    def from_array(arr):
        '''
        Create Matrix object from one-dimensional Array
        '''
        m = Matrix(len(arr), 1)
        for i in range(m.rows):
            m.data[i][0] = arr[i]
        return m

    def to_array(self):
        '''
        Create a one-dimensional array from a Matrix object
        '''
        arr = []
        for i in range(self.rows):
            for j in range(self.cols):
                arr.append(self.data[i][j])
        return arr

    def randomize(self):
        '''
        Set the data of the Matrix to random number between -1 and 1
        '''
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = random() * 2 - 1

    def add(self, n):
        '''
        Add a Matrix or a scalar to another Matrix element-wise.
        '''
        if isinstance(n, Matrix):
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] += n.data[i][j]
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] += n

    @staticmethod
    def subtract(a, b):
        '''Return a new matrix object a - b'''
        result = Matrix(a.rows, a.cols)
        for i in range(result.rows):
            for j in range(result.cols):
                result.data[i][j] = a.data[i][j] - b.data[i][j]
        return result

    @staticmethod
    def static_mult(a, b):
        '''
        Perform matrix product between two Matrix objects.
        Returns a Matrix
        '''
        # Matrix product
        if (a.cols != b.rows):
            print("Columns of A must match rows of B")
            return
        result = Matrix(a.rows, b.cols)
        for i in range(result.rows):
            for j in range(result.cols):
                suma = 0
                for k in range(a.cols):
                    suma += a.data[i][k] * b.data[k][j]
                result.data[i][j] = suma
        return result

    def mult(self, n):
        '''
        Multiplies a Matrix by a scalar or by another Matrix
        element-wise
        '''
        if isinstance(n, Matrix):
            # Hadamard product
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] *= n.data[i][j]
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] *= n

    @staticmethod
    def transpose(a):
        '''
        Returns the transpose of a Matrix
        '''
        result = Matrix(a.cols, a.rows)
        for i in range(a.rows):
            for j in range(a.cols):
                result.data[j][i] = a.data[i][j]
        return result

    def map(self, func):
        '''
        Applies a function to each element of a Matrix
        '''
        for i in range(self.rows):
            for j in range(self.cols):
                val = self.data[i][j]
                self.data[i][j] = func(val)

    @staticmethod
    def static_map(matrix, func):
        '''
        Applies a function to each element of a Matrix, but
        this is a static method
        '''
        result = Matrix(matrix.rows, matrix.cols)
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                val = matrix.data[i][j]
                result.data[i][j] = func(val)
        return result
