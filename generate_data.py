from typing import List, Callable, Tuple
import random
from sklearn.preprocessing import MinMaxScaler
import numpy as np

Genome = List[int]
Population = List[Genome]

NUM_PLANTS = 9
NUM_FIELDS = 3
PRODUCTION_QTY = 100

plant_cost = [21, 21, 22, 45, 27, 22, 25, 21, 21]
print(f'plant cost: {plant_cost}, len: {len(plant_cost)}')
plant_qty = [0.6, 0.7, 0.2, 0.4, 0.3, 0.95, 0.9, 0.8, 0.4]
print(f'plant quality: {plant_qty}, len: {len(plant_qty)}')
# field_cost = [((plant_cost[int(i/NUM_FIELDS)])*random.random()) for i in range(NUM_PLANTS*NUM_FIELDS)]
field_cost = [round((plant_cost[int(i/NUM_FIELDS)])*random.random(), 1) for i in range(NUM_PLANTS*NUM_FIELDS)]
print(f'field cost: {field_cost}, len: {len(field_cost)}')
data = np.array(field_cost).reshape(-1, 1)
scaler = MinMaxScaler()
scaled_cost = scaler.fit_transform(data)
scaled_cost = scaled_cost.flatten()
field_qty = [round(0.1 + (scaled_cost[i]*random.uniform(0.5, 0.9)), 1) for i in range(NUM_PLANTS*NUM_FIELDS)]
print(f'field quality: {field_qty}, len: {len(field_qty)}')
Plant_capacity = [int(PRODUCTION_QTY/8) for _ in range(NUM_PLANTS*NUM_FIELDS)]
print(f'capacity uniform: {Plant_capacity}')

# Save to external file
with open(".\data\plant_cost.txt", "w") as file:
    file.write("\n".join(str(item) for item in plant_cost))

my_list = []
with open(".\data\plant_cost.txt", "r") as file:
    for line in file:
        item = line.strip()  # Remove leading/trailing whitespace

        # Convert the item to integer if possible, otherwise convert to float
        try:
            item = int(item)
        except ValueError:
            try:
                item = float(item)
            except ValueError:
                pass

        my_list.append(item)

if plant_cost == my_list:
    print("Match!")


with open(".\data\plant_qty.txt", "w") as file:
    file.write("\n".join(str(item) for item in plant_qty))

my_list = []
with open(".\data\plant_qty.txt", "r") as file:
    for line in file:
        item = line.strip()  # Remove leading/trailing whitespace

        # Convert the item to integer if possible, otherwise convert to float
        try:
            item = int(item)
        except ValueError:
            try:
                item = float(item)
            except ValueError:
                pass

        my_list.append(item)

if plant_qty == my_list:
    print("Match!")

with open(".\data\\field_cost.txt", "w") as file:
    file.write("\n".join(str(item) for item in field_cost))

my_list = []
with open(".\data\\field_cost.txt", "r") as file:
    for line in file:
        item = line.strip()  # Remove leading/trailing whitespace

        # Convert the item to integer if possible, otherwise convert to float
        try:
            item = int(item)
        except ValueError:
            try:
                item = float(item)
            except ValueError:
                pass

        my_list.append(item)

if field_cost == my_list:
    print("Match!")

with open(".\data\\field_qty.txt", "w") as file:
    file.write("\n".join(str(item) for item in field_qty))

my_list = []
with open(".\data\\field_qty.txt", "r") as file:
    for line in file:
        item = line.strip()  # Remove leading/trailing whitespace

        # Convert the item to integer if possible, otherwise convert to float
        try:
            item = int(item)
        except ValueError:
            try:
                item = float(item)
            except ValueError:
                pass

        my_list.append(item)

if field_qty == my_list:
    print("Match!")