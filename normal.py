# linear regression implemented via normal equation
# data from https://www.kaggle.com/datasets/spscientist/students-performance-in-exams/data
# n = # features
# m = # examples
# j = feature index
# i = example index

import csv

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
        example = [[parse_gender(row[0])], [parse_race(row[1])], [parse_education(row[2])], [parse_lunch(row[3])], [parse_test_prep(row[4])]]
        y = int(row[5])

        examples.append(example)
        Y.append([y])

m = len(examples)

if len(examples) != len(Y):
    raise 'examples and Y not equal len'

def transpose_matrix(A):
    rows = len(A)
    cols = len(A[0])
    B = []
    for _ in range(cols):
        B.append([0] * rows)
    for r in range(rows):
        for c in range(cols):
            B[c][r] = A[r][c]
    return B

def dot_product(a, b):
    if len(a) != len(b):
        raise 'dot_product invalid dimensions'
    sum = 0
    for i in range(len(a)):
        sum += a[i] * b[i]
    return sum

def matrix_multiplication(A, B):
    if len(A[0]) != len(B):
        raise 'matrix_multiplication invalid dimensions'

    C = []
    for r in range(len(A)):
        row = A[r]
        result = []
        for c in range(len(B[0])):
            col = []
            for r2 in range(len(B)):
                col.append(B[r2][c])

            result.append(dot_product(row, col))
        C.append(result)
    return C

def matrix_dimensions(A):
    rows = len(A)
    cols = len(A[0])
    return [rows, cols]

# only part used ai 
def invert_matrix(A):
    n = len(A)
    # augment A with identity matrix
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]

    for i in range(n):
        # partial pivoting
        pivot = max(range(i, n), key=lambda r: abs(aug[r][i]))
        if abs(aug[pivot][i]) < 1e-12:
            print(aug[pivot][i])
            raise ValueError("Matrix is singular")
        aug[i], aug[pivot] = aug[pivot], aug[i]

        pivot_val = aug[i][i]
        aug[i] = [x / pivot_val for x in aug[i]]

        for r in range(n):
            if r != i:
                factor = aug[r][i]
                aug[r] = [aug[r][c] - factor * aug[i][c] for c in range(2 * n)]

    return [row[n:] for row in aug]

def normal_equation():
    X = []
    for x in examples:
        transpose = transpose_matrix(x)[0]
        transpose.insert(0, 1) # x0 is intercept (1)
        X.append(transpose)

    X_transpose = transpose_matrix(X)

    inverted = invert_matrix(matrix_multiplication(X_transpose, X))

    weights = transpose_matrix(matrix_multiplication(matrix_multiplication(inverted, X_transpose), Y))[0]
    return weights

weights = normal_equation()

x = [1, 1, 2, 1, 0, 1]

def h(x, weights):
    return dot_product(x, weights)

print(h(x, weights))
