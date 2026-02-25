# data from https://www.kaggle.com/datasets/spscientist/students-performance-in-exams/data
import csv

class Vector:
    def __init__(self, arr):
        self.arr = arr

    def dimensions(self):
        return len(self.arr)
    
    def get_scalar_multiple(self, a):
        copy = []
        for x in self.arr:
            copy.append(x * a)
        return Vector(copy)
    
    def subtract_vector(self, a):
        if a.dimensions() != self.dimensions():
            raise 'Dimensions do not match'
        
        for i in range(a.dimensions()):
            self.arr[i] -= a.arr[i]

    def stringify(self):
        string = '<'
        dimensions = self.dimensions()
        for i in range(dimensions):
            string += str(self.arr[i])
            if i != dimensions - 1:
                string += ', '
        string += '>'
        return string

# dot product of two vectors
def dot_product(v1, v2):
    if v1.dimensions() != v2.dimensions():
        raise 'Dimensions do not match'
    
    sum = 0
    for i in range(v1.dimensions()):
        sum += v1.arr[i] * v2.arr[i]

    return sum

def add_vectors(v1, v2):
    if v1.dimensions() != v2.dimensions():
        raise 'Dimensions do not match'
    
    result = Vector([])
    for i in range(v1.dimensions()):
        result.arr.append(v1.arr[i] + v2.arr[i])

    return result

# fill array of n size with x value
def fill_array(x, n):
    arr = [x] * n
    return arr

features = 5
# assign numbers to each value of a feature
# string[][]
feature_indexes = []
for _ in range(features):
    feature_indexes.append([])

def determine_feature_value(i, str):
    try:
        index = feature_indexes[i].index(str)
        return index
    except ValueError:
        # not found
        feature_indexes[i].append(str)
        return len(feature_indexes[i]) - 1
    
def lookup_feature_index(feature_index, index):
    return feature_indexes[feature_index][index]

feature_vectors = []
math_scores = []
reading_scores = []
writing_scores = []

with open("StudentsPerformance.csv", 'r') as file:
    csvreader = csv.reader(file)
    headers = next(csvreader)
    for row in csvreader:
        feature_values = []
        for i in range(features):
            value = determine_feature_value(i, row[i])
            feature_values.append(value)
            
        feature_vectors.append(Vector(feature_values))
        math_scores.append(int(row[5]))
        reading_scores.append(int(row[6]))
        writing_scores.append(int(row[7]))

# number of training examples
N = len(feature_vectors)
print(f"Parsed {N} CSV examples")

# gradient descent fn to modify w and b for each epoch
# w is vector, b is whole number, y is array of results we're training on (either math, reading, or writing scores)
def update_w_and_b(w, b, y, alpha):
    dl_dw = Vector(fill_array(0, features))
    dl_db = 0

    for i in range(N):
        error = y[i] - (dot_product(w, feature_vectors[i]) + b)
        sub_dl_dw = feature_vectors[i].get_scalar_multiple(-2 * error)
        dl_dw = add_vectors(sub_dl_dw, dl_dw)

        sub_dl_db = -2 * error
        dl_db += sub_dl_db

    w.subtract_vector(dl_dw.get_scalar_multiple(alpha * (1 / float(N))))
    b -= dl_db * alpha * (1 / float(N))

    return w, b

# y is the array of scores we want to train on, math, reading, or writing
# this is the gradient descent algorithm that iterates for each epoch and updates w and b
def train(y, alpha, epochs):
    w = Vector(fill_array(0, features))
    b = 0
    for e in range(epochs):
        w, b = update_w_and_b(w, b, y, alpha)

        if e % 400 == 0:
            print(f"epoch: {e} loss: {calc_loss(w, b, y)}")
    
    return w, b

# squared loss
def calc_loss(w, b, y):
    sum = 0
    for i in range(N):
        prediction = dot_product(w, feature_vectors[i]) + b
        sum += (y[i] - prediction) ** 2
    return sum / float(N)

ALPHA = 0.001
EPOCHS = 15000

w_math, b_math = train(math_scores, ALPHA, EPOCHS)
w_reading, b_reading = train(reading_scores, ALPHA, EPOCHS)
w_writing, b_writing = train(writing_scores, ALPHA, EPOCHS)

print(f"Math | Weights: {w_math.stringify()} Bias: {b_math}")
print(f"Reading | Weights: {w_reading.stringify()} Bias: {b_reading}")
print(f"Writing | Weights: {w_writing.stringify()} Bias: {b_writing}")

def predict(w, b, x):
    if x.dimensions() != w.dimensions():
        raise 'Dimensions do not match'
    
    return dot_product(w, x) + b

print(predict(w_math, b_math, Vector([0, 0, 0, 0, 0])))
print(predict(w_reading, b_reading, Vector([0, 0, 3, 0, 0])))
print(predict(w_writing, b_writing, Vector([1, 4, 5, 2, 1])))
