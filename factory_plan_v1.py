import time
from functools import partial
from random import choice, choices, randint, randrange, random
from typing import List, Callable, Tuple
from collections import namedtuple
import random

Genome = List[int]
Population = List[Genome]
FitnessFunc = Callable[[Genome], int]
PopulateFunc = Callable[[], Population]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunct = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunct = Callable[[Genome], Genome]

POPULATION_SIZE = 10
NUM_PLANTS = 3
PRODUCTION_QTY = 100

Plant_capacity = [PRODUCTION_QTY for _ in range(NUM_PLANTS)]

def generate_genome(num_plants: int, prod_qty: int) -> Genome:
    binary_representation = bin(prod_qty)[2:]
    len_binary = len(binary_representation)

    remaining_qty = prod_qty
    gen = []
    for i in range(num_plants):
        qty_plant = random.randint(0, remaining_qty)
        if i == (num_plants - 1):
            qty_plant = remaining_qty

        if qty_plant > Plant_capacity[i]:
            qty_plant = Plant_capacity[i]

        remaining_qty -= qty_plant
        binary_string = bin(qty_plant)[2:]
        if len(binary_string) < len_binary:
            binary_string = '0' * (len_binary - len(binary_string)) + binary_string
        binary_digits = [int(bit) for bit in binary_string]
        gen.extend(binary_digits)

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
        print(binary_digits)
        binary_string = ''.join(str(bit) for bit in binary_digits)
        decimal_value = int(binary_string, 2)
        prod[i] = decimal_value

    return prod

genome= generate_genome(NUM_PLANTS, PRODUCTION_QTY)
print(f'genome: {genome}')
list_prod = decode(NUM_PLANTS, genome, PRODUCTION_QTY)
print(list_prod)

def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]

def fitness(genome: Genome, things: [Thing], weight_limit: int) -> int:
    if len(genome) != len(things):
        raise ValueError("genome and things must be of the same length")

    weight = 0
    value = 0

    for i, thing in enumerate(things):
        if genome[i] == 1:
            weight += thing.weight
            value += thing.value

            if weight > weight_limit:
                return 0

    return value

def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    return choices(
        population=population,
        weights=[fitness_func(genome) for genome in population],
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

def mutation(genome: Genome, num: int = 1, probability: float= 0.5) -> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index]-1)
    return genome

def run_evolution(
        populate_func: PopulateFunc,
        fitness_func: FitnessFunc,
        fitness_limit: int,
        selection_func: SelectionFunc = selection_pair,
        crossover_funct: CrossoverFunct = single_point_crossover,
        mutation_func: MutationFunct = mutation,
        generation_limit: int = 100
) -> Tuple[Population, int]:
    population = populate_func()

    for i in range(generation_limit):
        population = sorted(population, key=lambda genome: fitness_func(genome), reverse=True)

        if fitness_func(population[0]) >= fitness_limit:
            break

        next_generation = population[0:2]

        for j in range(int(len(population) / 2) -1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_funct(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]

        population = next_generation

    population = sorted(
        population,
        key=lambda genome: fitness_func(genome),
        reverse=True
    )

    return population, i

start = time.time()
population, generation = run_evolution(
    populate_func=partial(
        generate_population, size=10, genome_length=len(more_things)
    ),
    fitness_func=partial(
        fitness, things=more_things, weight_limit=3000
    ),
    fitness_limit=1310,
    generation_limit=100
)
end = time.time()

def genome_to_things(genome: Genome, things: [Thing]) -> [Thing]:
    result = []
    for i, thing in enumerate(things):
        if genome[i] == 1:
            result += [thing.name]
    return result

print(f"number of generations : {generation}")
print(f"time: {end - start}s")
print(f"best solution: {genome_to_things(population[0], more_things)}")