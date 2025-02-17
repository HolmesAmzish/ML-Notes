import random

points = []
for i in range(1000):
    x1 = random.randint(0, 1000)
    x2 = random.randint(0, 1000)
    if pow((x1 - 500), 2) + pow((x2 - 500), 2) <= 62500:
        tag = 1
    else:
        tag = 0
    points.append((x1, x2, tag))

with open('points.csv', 'w') as file:
    for point in points:
        file.write(str(point[0]) + ',' + str(point[1]) + ',' + str(point[2]) + '\n')

print("All points have been written to the file.")