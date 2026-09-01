# locally weighted linear regression
# data from https://www.kaggle.com/datasets/spscientist/students-performance-in-exams/data
# n = # features
# m = # examples
# j = feature index
# i = example index

import csv
import math

def parse_gender(str):
    if str == "female":
        return 0
    else:
        return 1

def parse_race(str):
    if str == "group A":
        return 0
    elif str == "group B":
        return 1
    elif str == "group C":
        return 2
    elif str == "group D":
        return 3
    elif str == "group E":
        return 4
    else:
        raise Exception('unexpected race ' + str)

def parse_education(str):
    if str == "bachelor's degree":
        return 0
    elif str == "some college":
        return 1
    elif str == "master's degree":
         return 2
    elif str == "associate's degree":
         return 3
    elif str == "high school":
        return 4
    elif str == "some high school":
        return 5
    else:
        raise 'unexpected eduation ' + str

def parse_lunch(str):
    if str == "standard":
        return 0
    elif str == "free/reduced":
        return 1
    else:
        raise 'unexpected lunch ' + str

def parse_test_prep(str):
    if str == "completed":
        return 0
    elif str == 'none':
        return 1
    else:
        raise 'unexpected test prep ' + str

n = 5 # number features
examples = []
Y = []

with open("StudentsPerformance.csv", 'r') as file:
    csvreader = csv.reader(file)
    headers = next(csvreader)
    for row in csvreader:
        # x_0 is always 1
        example = [1, parse_gender(row[0]), parse_race(row[1]), parse_education(row[2]), parse_lunch(row[3]), parse_test_prep(row[4])]
        y = int(row[6])

        examples.append(example)
        Y.append(y)

m = len(examples)

if len(examples) != len(Y):
    raise 'examples and Y not equal len'

def dot_product(a, b):
    if len(a) != len(b):
        raise 'dot_product invalid dimensions'
    sum = 0
    for i in range(len(a)):
        sum += a[i] * b[i]
    return sum

def is_int(a):
    return isinstance(a, int) and not isinstance(a, bool)

def h(x, weights):
    return dot_product(x, weights)

def calc_loss(theta, x):
    sum = 0
    for i in range(m):
        x_i = examples[i]
        diff = h(x_i, theta) - Y[i]
        squared = diff ** 2
        w_i = w(x_i, x)
        product = squared * w_i
        sum += product / m
    return sum


def subtract_vectors(a, b):
    if len(a) != len(b):
        raise 'subtract_vectors invalid dimensions'

    c = []
    for i in range(len(a)):
        c.append(a[i] - b[i])
    return c

TAU = 1

# x_i example to evaluate weight of
# x location of prediction
def w(x_i, x):
    diff = subtract_vectors(x_i, x)
    squared = dot_product(diff, diff)
    return math.exp(-1 * squared / (2 * TAU * TAU))

def scale_vector(c, a):
    for i in range(len(a)):
        a[i] *= c
    return a

ALPHA = 0.01

# x = location of prediction to inference from
def gradient_descent(theta, x):
    new_theta = list(theta)

    for i in range(m):
        x_i = examples[i]
        y_i = Y[i]

        diff = h(x_i, theta) - y_i
        w_i = 2 * w(x_i, x)
        for j in range(n + 1):
            mod = diff * w_i * x_i[j] * ALPHA / m
            new_theta[j] -= mod

    return new_theta

EPOCHS = 10000

def gradient_inference(x):
    theta = [0] * (n + 1)
    if len(x) == n:
        x.insert(0, 1)

    for e in range(EPOCHS):
        if e % 100 == 0:
            print("Epoch :", str(e))
            print("Loss :", str(calc_loss(theta, x)))
        theta = gradient_descent(theta, x)

    hypothesis = h(x, theta)
    print("Theta :", theta)
    print("Hypothesis :", str(hypothesis))
    loss = calc_loss(theta, x)
    print("Loss :", str(loss))

gradient_inference([0, 1, 0, 0, 1])
