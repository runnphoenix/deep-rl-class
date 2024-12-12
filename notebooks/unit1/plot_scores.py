import csv
import matplotlib.pyplot as plt

scores = []
with open('scores.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        scores.append(row)

scores = [float(score) for score in scores[0]]
print(type(scores), len(scores))

plt.plot(scores)
plt.show()
