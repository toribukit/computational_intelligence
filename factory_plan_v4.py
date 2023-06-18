import time
from functools import partial
from random import choice, choices, randint, randrange, random
from typing import List, Callable, Tuple
from collections import namedtuple
import random
import numpy as np
# Set the random seed
# random.seed(420)

Genome = List[int]
Population = List[Genome]
FitnessFunc = Callable[[Genome], int]
PopulateFunc = Callable[[], Population]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunct = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunct = Callable[[Genome], Genome]

POPULATION_SIZE = 30
NUM_PLANTS = 9
NUM_FIELDS = 3
PRODUCTION_QTY = 100
ADDITIONAL_COST_PER_FIELD = 50
EVAL_ROUND = 10

Plant_capacity = [int(PRODUCTION_QTY/8) for _ in range(NUM_PLANTS*NUM_FIELDS)]

MAX_PROD_QTY = PRODUCTION_QTY + int(PRODUCTION_QTY*0.25)
plant_cost = []
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

        plant_cost.append(item)

plant_qty = []
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

        plant_qty.append(item)

field_cost = []
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

        field_cost.append(item)

field_qty = []
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

        field_qty.append(item)

print(f'plant cost: {plant_cost}, len: {len(plant_cost)}')
print(f'plant quality: {plant_qty}, len: {len(plant_qty)}')
print(f'field cost: {field_cost}, len: {len(field_cost)}')
print(f'field quality: {field_qty}, len: {len(field_qty)}')

def encode(num_plants: int, num_fields:int, list_prod: [int], prod_qty: int) -> Genome:
    binary_representation = bin(prod_qty)[2:]
    len_binary = len(binary_representation)
    gen = []
    for i in range(num_plants*num_fields):
        binary_string = bin(list_prod[i])[2:]
        if len(binary_string) < len_binary:
            binary_string = '0' * (len_binary - len(binary_string)) + binary_string
        binary_digits = [int(bit) for bit in binary_string]
        # print(binary_digits)
        gen.extend(binary_digits)
    return gen


def generate_genome(num_plants: int, field_per_plant:int, prod_qty: int) -> Genome:
    remaining_qty = prod_qty
    list_prod = []
    for i in range(num_plants*field_per_plant):
        qty_plant = random.randint(0, remaining_qty)
        if i == ((num_plants*field_per_plant) - 1):
            qty_plant = remaining_qty

        if qty_plant > Plant_capacity[i]:
            qty_plant = Plant_capacity[i]

        remaining_qty -= qty_plant
        list_prod.append(qty_plant)
    # print(list_prod)
    gen = encode(num_plants, field_per_plant, list_prod, prod_qty)
    return gen

# print(f'genome: {generate_genome(NUM_PLANTS, PRODUCTION_QTY)}')

def decode(num_plants: int, gen: Genome, prod_qty: int) -> List[int]:
    binary_representation = bin(prod_qty)[2:]
    len_binary = len(binary_representation)

    prod = [0 for _ in range(num_plants)]
    for i in range(num_plants):
        offset1 = i*len_binary
        offset2 = (i+1)*7
        binary_digits = gen[offset1: offset2]
        # print(binary_digits)
        binary_string = ''.join(str(bit) for bit in binary_digits)
        decimal_value = int(binary_string, 2)
        prod[i] = decimal_value

    return prod

# genome= generate_genome(NUM_PLANTS, PRODUCTION_QTY)
# print(f'genome: {genome}')
# list_prod = decode(NUM_PLANTS, genome, PRODUCTION_QTY)
# print(list_prod)

def generate_population(size: int, num_plants: int, field_per_plant:int, prod_qty: int) -> Population:
    return [generate_genome(num_plants,field_per_plant, prod_qty) for _ in range(size)]

population_gen = generate_population(POPULATION_SIZE, NUM_PLANTS, NUM_FIELDS, PRODUCTION_QTY)

print(population_gen)
for i in range(POPULATION_SIZE):
    dec_res = decode(NUM_PLANTS*NUM_FIELDS, population_gen[i], PRODUCTION_QTY)
    print(f'decode: {dec_res} ; SUM {sum(dec_res)}')

def fitness(genome: Genome, cost_f: [float], qty_f: [float], field_per_plant: int, num_plant:int, total_qty: int, cost_p: [int], qty_p: [float] = plant_qty) -> float:
    total_field = field_per_plant*num_plant

    list_prod = decode(total_field, genome, total_qty)

    if sum(list_prod) < total_qty:
        raise ValueError(f"production decode must equal or greater than total quantity. list prod: {list_prod} and total qty: {total_qty}")

    value = 0
    deficit = 0
    cost_field = []
    cost_plant = [0 for _ in range(num_plant)]
    for i in range(total_field):
        # print(f'field: {i}')
        temp_cost = cost_f[i] * list_prod[i]
        temp_qty = qty_f[i]
        # print(f"quality: {temp_qty}")
        val_random = random.random()
        if val_random > temp_qty:
            props = val_random - temp_qty
            failed = 1 + (int(list_prod[i]*props))
            deficit += failed
            # print(f"deficit: {failed} | deficit: {deficit}")
            # temp_cost += (deficit*ADDITIONAL_COST_PER_FIELD)
        cost_field.append(temp_cost)

        cost_plant[int(i/field_per_plant)] += list_prod[i]*cost_p[int(i/field_per_plant)]
        cost_plant[int(i/field_per_plant)] += (list_prod[i] * (1.0 - qty_p[int(i/field_per_plant)])) * cost_p[int(i/field_per_plant)]

    # print(f'prod real: {(sum(list_prod) - deficit)}')
    if (sum(list_prod) - deficit) < total_qty:
        additional_cost = (total_qty - (sum(list_prod) - deficit)) * ADDITIONAL_COST_PER_FIELD
    else:
        additional_cost = 0
    # print(f'cost field: {sum(cost_field)}, cost plant: {sum(cost_plant)}, additional cost: {additional_cost}')
    value = sum(cost_field) + sum(cost_plant) + additional_cost
    return value
def fitness_fixed(genome: Genome, cost_f: [float], qty_f: [float], field_per_plant: int, num_plant:int, total_qty: int, cost_p: [int]) -> float:
    total_field = field_per_plant*num_plant

    list_prod = decode(total_field, genome, total_qty)
    if sum(list_prod) < total_qty:
        raise ValueError(f"production decode must equal or greater than total quantity. list prod: {list_prod} and total qty: {total_qty}")

    value = 0
    cost_field = []
    cost_plant = [0 for _ in range(num_plant)]
    for i in range(total_field):
        temp_cost = cost_f[i] * list_prod[i]
        cost_field.append(temp_cost)
        cost_plant[int(i/field_per_plant)] += list_prod[i]*cost_p[int(i/field_per_plant)]
    value = sum(cost_field) + sum(cost_plant)
    return value

# print(f"fitness: {fitness(population_gen[1], field_cost, field_qty, NUM_FIELDS, NUM_PLANTS, PRODUCTION_QTY, plant_cost)}")

def fitness_avg(genome: Genome, cost_f: [float], qty_f: [float], field_per_plant: int, num_plant:int, total_qty: int, cost_p: [int], eval_round: int = EVAL_ROUND) -> float:
    value_hist = []
    for _ in range(eval_round):
        val = fitness(genome, cost_f, qty_f, field_per_plant, num_plant, total_qty, cost_p)
        value_hist.append(val)

    return (sum(value_hist) / len(value_hist))
def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    fitness_val = [fitness_func(genome) for genome in population]
    min_fitness = min(fitness_val)
    # print(min_fitness)
    adjusted_weights = [min_fitness / fitness for fitness in fitness_val]
    # print(adjusted_weights)

    return choices(
        population=population,
        weights=adjusted_weights,
        k=2
    )

def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("Genomes a and b must be of the same length")

    length = len(a)
    if length < 2:
        return a, b

    p = randint(1, length-1)
    return a[0:p]+b[p:], b[0:p]+a[p:]

def uniform_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    # Ensure parents have the same length
    if len(a) != len(b):
        raise ValueError("Genomes a and b must be of the same length")
    # Create empty offspring chromosomes
    offspring1 = [0] * len(a)
    offspring2 = [0] * len(b)

    # Iterate over each gene (bit)
    for i in range(len(a)):
        # Randomly select a parent for the gene
        if random.random() < 0.5:
            offspring1[i] = a[i]
            offspring2[i] = b[i]
        else:
            offspring1[i] = b[i]
            offspring2[i] = a[i]
    offspring1 = perform_correction(offspring1, NUM_FIELDS, NUM_PLANTS, PRODUCTION_QTY, Plant_capacity)
    offspring2 = perform_correction(offspring2, NUM_FIELDS, NUM_PLANTS, PRODUCTION_QTY, Plant_capacity)

    return offspring1, offspring2

def perform_correction(offspring:Genome, field_per_plant: int, num_plant:int, total_qty: int, capacity: [int], max_qty: int = MAX_PROD_QTY) -> Genome:
    total_field = field_per_plant * num_plant
    list_prod = decode(total_field, offspring, total_qty)

    for i in range(len(list_prod)):
        if list_prod[i] > capacity[i]:
            list_prod[i] = capacity[i]

    if sum(list_prod) > max_qty or sum(list_prod) < total_qty:
        diff = total_qty - (sum(list_prod))

        while diff != 0:
            field_index = random.randint(0, len(list_prod) - 1)

            if diff > 0:
                increment = min(diff, (capacity[field_index] - list_prod[field_index]))
                list_prod[field_index] += increment
                diff -= increment
            else:
                decrement = min(-diff, list_prod[field_index])
                list_prod[field_index] -= decrement
                diff += decrement

    offspring = encode(num_plant,field_per_plant, list_prod,total_qty)

    return offspring

# fitness_func=partial(
#         fitness, cost_f=field_cost, qty_f=field_qty, field_per_plant=NUM_FIELDS, num_plant=NUM_PLANTS, total_qty=PRODUCTION_QTY, cost_p=plant_cost
#     )
# parents = selection_pair(population_gen, fitness_func)
# offspring_a, offspring_b = uniform_crossover(parents[0], parents[1])
# dec_res = decode(NUM_PLANTS*NUM_FIELDS, offspring_a, PRODUCTION_QTY)
# print(f'decode offspring 1: {dec_res} ; SUM {sum(dec_res)}')
# dec_res = decode(NUM_PLANTS*NUM_FIELDS, offspring_b, PRODUCTION_QTY)
# print(f'decode offspring 2: {dec_res} ; SUM {sum(dec_res)}')
# dec_res = decode(NUM_PLANTS*NUM_FIELDS, parents[0], PRODUCTION_QTY)
# print(f'decode parent 1: {dec_res} ; SUM {sum(dec_res)}')
# dec_res = decode(NUM_PLANTS*NUM_FIELDS, parents[1], PRODUCTION_QTY)
# print(f'decode parent 2: {dec_res} ; SUM {sum(dec_res)}')

# offspring_a_correct = perform_correction(offspring_a, NUM_FIELDS, NUM_PLANTS, PRODUCTION_QTY, Plant_capacity)
# dec_res = decode(NUM_PLANTS*NUM_FIELDS, offspring_a_correct, PRODUCTION_QTY)
# print(f"decode correction offspring 1: {dec_res} ; SUM {sum(dec_res)}")
#
# offspring_b_correct = perform_correction(offspring_b, NUM_FIELDS, NUM_PLANTS, PRODUCTION_QTY, Plant_capacity)
# dec_res = decode(NUM_PLANTS*NUM_FIELDS, offspring_b_correct, PRODUCTION_QTY)
# print(f"decode correction offspring 2: {dec_res} ; SUM {sum(dec_res)}")



def mutation(genome: Genome, num: int = 2, probability: float= 0.5) -> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        # print(genome)
        # print(genome[index])
        genome[index] = genome[index] if random.random() > probability else abs(genome[index]-1)
    genome = perform_correction(genome, NUM_FIELDS, NUM_PLANTS, PRODUCTION_QTY, Plant_capacity)
    return genome


# mutation_func = mutation
# dec_res = decode(NUM_PLANTS*NUM_FIELDS, offspring_a, PRODUCTION_QTY)
# print(f'decode offspring 1: {dec_res} ; SUM {sum(dec_res)}')
# offspring_a = mutation_func(offspring_a)
# dec_res = decode(NUM_PLANTS*NUM_FIELDS, offspring_a, PRODUCTION_QTY)
# print(f'decode offspring 1: {dec_res} ; SUM {sum(dec_res)}')

def run_evolution(
        populate_func: PopulateFunc,
        fitness_func: FitnessFunc,
        selection_func: SelectionFunc = selection_pair,
        crossover_funct: CrossoverFunct = uniform_crossover,
        mutation_func: MutationFunct = mutation,
        generation_limit: int = 100
) -> Tuple[Population, int]:
    population = populate_func()

    for i in range(generation_limit):
        population = sorted(population, key=lambda genome: fitness_func(genome), reverse=False)
        print(f"generations: {i}")
        dec_res = decode(NUM_PLANTS * NUM_FIELDS, population[0], PRODUCTION_QTY)
        print(f"fitness: {fitness_avg(population[0], field_cost, field_qty, NUM_FIELDS, NUM_PLANTS, PRODUCTION_QTY, plant_cost)} ; SUM {sum(dec_res)}")

        next_generation = population[0:8]

        for j in range(int(len(population) / 2) -4):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_funct(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]

        population = next_generation

    population = sorted(
        population,
        key=lambda genome: fitness_func(genome),
        reverse=False
    )
    print(f"generations: {i}")
    dec_res = decode(NUM_PLANTS * NUM_FIELDS, population[0], PRODUCTION_QTY)
    print(f"fitness: {fitness_avg(population[0], field_cost, field_qty, NUM_FIELDS, NUM_PLANTS, PRODUCTION_QTY, plant_cost)}; SUM {sum(dec_res)}")


    return population, i

start = time.time()
population, generation = run_evolution(
    populate_func=partial(
        generate_population, size=POPULATION_SIZE, num_plants=NUM_PLANTS, field_per_plant=NUM_FIELDS, prod_qty=PRODUCTION_QTY
    ),
    fitness_func=partial(
        fitness_avg, cost_f=field_cost, qty_f=field_qty, field_per_plant=NUM_FIELDS, num_plant=NUM_PLANTS, total_qty=PRODUCTION_QTY, cost_p=plant_cost
    ),
    generation_limit=200
)
end = time.time()

print(f"number of generations : {generation}")
print(f"time: {end - start}s")
dec_res = decode(NUM_PLANTS*NUM_FIELDS, population[0], PRODUCTION_QTY)
print(f'Best Solutions: {dec_res} ; SUM {sum(dec_res)}')
print(f"fitness avg: {fitness_avg(population[0], field_cost, field_qty, NUM_FIELDS, NUM_PLANTS, PRODUCTION_QTY, plant_cost)}")
print(f"fitness fixed: {fitness_fixed(population[0], field_cost, field_qty, NUM_FIELDS, NUM_PLANTS, PRODUCTION_QTY, plant_cost)}")