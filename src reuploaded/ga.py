import copy
import heapq
import metrics
import multiprocessing.pool as mpool
import os
import random
import shutil
import time
import math

width = 200
height = 16

class Individual_Grid(object):
    __slots__ = ["genome", "_fitness"]

    def __init__(self, genome):
        self.genome = copy.deepcopy(genome)
        self._fitness = None

    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        coefficients = dict(
            meaningfulJumpVariance=1.0,  
            negativeSpace=0.6,           
            pathPercentage=0.8,          
            emptyPercentage=0.4,         
            linearity=-0.5,              
            solvability=4.0              
        )
        
        level = self.to_level()
        # Bonus for powerups and coins
        powerup_count = sum(row.count('M') for row in level)
        powerup_bonus = min(powerup_count * 0.5, 2.0)
        
        coin_count = sum(row.count('o') + row.count('?') for row in level)
        coin_bonus = min(coin_count * 0.1, 1.5)
        
        # Penalize any accidental gaps in the ground row
        gap_penalty = 0
        ground_row = level[height-1]
        gap_count = ground_row.count('-')
        if gap_count > 10:
            gap_penalty = (gap_count - 10) * 0.2
        
        self._fitness = (sum(coefficients[m] * measurements[m] for m in coefficients)
                         + powerup_bonus + coin_bonus - gap_penalty)
        return self

    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    # This instance method defers to the static mutation operator.
    def mutate(self, genome):
        return Individual_Grid.mutate_static(genome)

    @staticmethod
    def mutate_static(genome):
        left = 1
        right = width - 1

        # --- Enforce the Ground Row ---
        for x in range(left, right):
            genome[height-1][x] = "X"

        # --- Feature Probabilities ---
        p_pipe = 0.008  # Pipe probability
        p_block = 0.06  # Bblock probability
        p_coin = 0.08   # Coin probability

        # Track feature positions
        pipe_positions = []     # Store (x_pos, height) tuples for pipes
        item_block_positions = [] # Store x positions of item blocks
        enemy_positions = []    # Store x positions of enemies

        # --- First Pass: Place Pipes ---
        # Start after a safe distance from Mario
        x = left + 12  # Increased starting gap
        min_pipe_spacing = 8   # Increased minimum pipe spacing
        
        while x < right - 8:
            if random.random() < p_pipe:
                # Strict pipe spacing check
                if not pipe_positions or (x - pipe_positions[-1][0] >= min_pipe_spacing):
                    pipe_height = random.randint(2, 4)
                    pipe_positions.append((x, pipe_height))
                    # Place the pipe
                    top_row = (height - 1) - pipe_height
                    genome[top_row][x] = "T"
                    for py in range(top_row + 1, height - 1):
                        genome[py][x] = "|"
                    x += min_pipe_spacing  # Force minimum spacing
                    continue
            x += 1

        # --- Second Pass: Place Blocks and Coins ---
        min_block_spacing = 4  # Minimum spaces between item blocks
        last_block_x = left

        for x in range(left + 6, right - 5):
            # Check if this column has a pipe or is too close to one
            is_near_pipe = any(abs(x - pipe_pos[0]) <= 2 for pipe_pos in pipe_positions)
            
            if not is_near_pipe:  # Only place blocks if not near a pipe
                # Place item blocks with minimum spacing
                if random.random() < p_block and (x - last_block_x >= min_block_spacing):
                    y_pos = random.randint(height-7, height-3)
                    if genome[y_pos][x] == "-":
                        block_type = random.choice(["?", "B", "M"])
                        genome[y_pos][x] = block_type
                        item_block_positions.append(x)
                        last_block_x = x

                # Place coins with consideration for blocks
                if random.random() < p_coin and (x not in item_block_positions):
                    y_pos = random.randint(height-8, height-4)
                    if genome[y_pos][x] == "-":
                        genome[y_pos][x] = "o"

        # --- Third Pass: Place Enemies (Goombas) ---
        # Define segments between pipes for enemy placement
        segments = []
        last_x = left
        for pipe_pos in pipe_positions:
            if pipe_pos[0] - last_x > 1:
                segments.append((last_x, pipe_pos[0] - 1))
            last_x = pipe_pos[0] + 1
        
        if right - last_x > 1:
            segments.append((last_x, right - 1))

        # Place enemies in segments with strict control
        for seg_start, seg_end in segments:
            seg_width = seg_end - seg_start + 1
            
            # Special handling for starting segment
            if seg_start == left:
                # Maximum 2 enemies in starting area
                num_enemies = min(2, seg_width // 8)
            else:
                # For other segments, keep very sparse
                if seg_width >= 16:
                    num_enemies = 2
                elif seg_width >= 8:
                    num_enemies = 1
                else:
                    num_enemies = 0

            # Place enemies with strict spacing rules
            valid_positions = list(range(seg_start + 3, seg_end - 2))
            random.shuffle(valid_positions)
            
            min_enemy_spacing = 4  # Minimum blocks between enemies
            placed_enemies = 0
            
            for pos in valid_positions:
                if placed_enemies >= num_enemies:
                    break
                    
                # Check if position is valid (not too close to other enemies or item blocks)
                if (all(abs(pos - e_pos) >= min_enemy_spacing for e_pos in enemy_positions) and
                    all(abs(pos - b_pos) >= 2 for b_pos in item_block_positions)):
                    if genome[height-2][pos] == "-":
                        genome[height-2][pos] = "E"
                        enemy_positions.append(pos)
                        placed_enemies += 1

        return genome

    def generate_children(self, other):
        new_genome = copy.deepcopy(self.genome)
        left = 1
        right = width - 1
        # Column-based crossover using two crossover points.
        crossover_points = sorted([random.randint(left+5, right-5) for _ in range(2)])
        for y in range(height):
            for x in range(left, right):
                if x < crossover_points[0] or x >= crossover_points[1]:
                    new_genome[y][x] = self.genome[y][x]
                else:
                    new_genome[y][x] = other.genome[y][x]
                # Make sure that an enemy on row height-2 is only kept if the ground in that column is solid.
                if y == height - 2 and new_genome[y][x] == "E" and new_genome[height-1][x] != "X":
                    new_genome[y][x] = "-"
        # Apply mutation to the child.
        new_genome = self.mutate(new_genome)
        return (Individual_Grid(new_genome),)

    def to_level(self):
        return self.genome

    @classmethod
    def empty_individual(cls):
        # Build an empty level: air everywhere, then enforce the ground row.
        g = [["-" for col in range(width)] for row in range(height)]
        g[height-1] = ["X"] * width
        # Place fixed markers (Mario's start, flag, etc.)
        g[height-2][0] = "m"
        g[7][-1] = "v"
        for row in range(8, 14):
            g[row][-1] = "f"
        for row in range(14, height):
            g[row][-1] = "X"
        return cls(g)

    @classmethod
    def random_individual(cls):
        # Start from an empty level and then decorate it with features.
        ind = cls.empty_individual()
        new_genome = copy.deepcopy(ind.genome)
        new_genome = cls.mutate_static(new_genome)
        return cls(new_genome)


def offset_by_upto(val, variance, min=None, max=None):
    val += random.normalvariate(0, variance**0.5)
    if min is not None and val < min:
        val = min
    if max is not None and val > max:
        val = max
    return int(val)


def clip(lo, val, hi):
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val


class Individual_DE(object):
    __slots__ = ["genome", "_fitness", "_level"]

    def __init__(self, genome):
        self.genome = list(genome)
        heapq.heapify(self.genome)
        self._fitness = None
        self._level = None

    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=0.5,
            emptyPercentage=0.6,
            linearity=-0.5,
            solvability=2.0
        )
        penalties = 0
        if len(list(filter(lambda de: de[1] == "6_stairs", self.genome))) > 5:
            penalties -= 2
        self._fitness = sum(coefficients[m] * measurements[m] for m in coefficients) + penalties
        return self

    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    def mutate(self, genome):
        left = 1
        right = width - 1
        
        for y in range(height):
            for x in range(left, right):
                if random.random() < 0.1:
                    if y > height - 4:
                        genome[y][x] = random.choice(["X", "B", "?", "-"])
                    elif y > height - 8:
                        genome[y][x] = random.choice(["X", "B", "?", "o", "-", "-", "-"])
                    else:
                        genome[y][x] = random.choice(["o", "-", "-", "-", "-"])
                    if genome[y][x] in ["X", "B", "?"]:
                        if y < height - 1 and genome[y+1][x] == "-":
                            genome[y+1][x] = "X"
                    if y == height - 1 and genome[y][x] == "-":
                        gap_width = random.randint(2, 4)
                        for i in range(gap_width):
                            if x + i < right:
                                genome[y][x+i] = "-"
        return genome

    def generate_children(self, other):
        pa = random.randint(0, len(self.genome) - 1)
        pb = random.randint(0, len(other.genome) - 1)
        a_part = self.genome[:pa] if len(self.genome) > 0 else []
        b_part = other.genome[pb:] if len(other.genome) > 0 else []
        ga = a_part + b_part
        b_part = other.genome[:pb] if len(other.genome) > 0 else []
        a_part = self.genome[pa:] if len(self.genome) > 0 else []
        gb = b_part + a_part
        return Individual_DE(self.mutate(ga)), Individual_DE(self.mutate(gb))

    def to_level(self):
        if self._level is None:
            base = Individual_Grid.empty_individual().to_level()
            for de in sorted(self.genome, key=lambda de: (de[1], de[0], de)):
                x = de[0]
                de_type = de[1]
                if de_type == "4_block":
                    y = de[2]
                    breakable = de[3]
                    base[y][x] = "B" if breakable else "X"
                elif de_type == "5_qblock":
                    y = de[2]
                    has_powerup = de[3]
                    base[y][x] = "M" if has_powerup else "?"
                elif de_type == "3_coin":
                    y = de[2]
                    base[y][x] = "o"
                elif de_type == "7_pipe":
                    h = de[2]
                    base[height - h - 1][x] = "T"
                    for y in range(height - h, height):
                        base[y][x] = "|"
                elif de_type == "0_hole":
                    w = de[2]
                    for x2 in range(w):
                        base[height - 1][clip(1, x + x2, width - 2)] = "-"
                elif de_type == "6_stairs":
                    h = de[2]
                    dx = de[3]
                    for x2 in range(1, h + 1):
                        for y in range(x2 if dx == 1 else h - x2):
                            base[clip(0, height - y - 1, height - 1)][clip(1, x + x2, width - 2)] = "X"
                elif de_type == "1_platform":
                    w = de[2]
                    h = de[3]
                    madeof = de[4]
                    for x2 in range(w):
                        base[clip(0, height - h - 1, height - 1)][clip(1, x + x2, width - 2)] = madeof
                elif de_type == "2_enemy":
                    base[height - 2][x] = "E"
            self._level = base
        return self._level

    @classmethod
    def empty_individual(_cls):
        g = []
        return Individual_DE(g)

    @classmethod
    def random_individual(_cls):
        elt_count = random.randint(8, 128)
        g = [random.choice([
            (random.randint(1, width - 2), "0_hole", random.randint(1, 8)),
            (random.randint(1, width - 2), "1_platform", random.randint(1, 8), random.randint(0, height - 1), random.choice(["?", "X", "B"])),
            (random.randint(1, width - 2), "2_enemy"),
            (random.randint(1, width - 2), "3_coin", random.randint(0, height - 1)),
            (random.randint(1, width - 2), "4_block", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "5_qblock", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "6_stairs", random.randint(1, height - 4), random.choice([-1, 1])),
            (random.randint(1, width - 2), "7_pipe", random.randint(2, height - 4))
        ]) for i in range(elt_count)]
        return Individual_DE(g)

Individual = Individual_Grid

def generate_successors(population):
    results = []
    # Sort by fitness (highest first)
    population = sorted(population, key=lambda x: x.fitness(), reverse=True)
    elite_count = int(len(population) * 0.1)
    results.extend(population[:elite_count])
    while len(results) < len(population):
        parent1 = random.choice(population[:len(population)//2])
        parent2 = random.choice(population[:len(population)//2])
        children = parent1.generate_children(parent2)
        results.extend(children)
    return results

def ga():
    pop_limit = 480
    batches = os.cpu_count()
    if pop_limit % batches != 0:
        print("It's ideal if pop_limit divides evenly into " + str(batches) + " batches.")
    batch_size = int(math.ceil(pop_limit / batches))
    with mpool.Pool(processes=os.cpu_count()) as pool:
        init_time = time.time()
        population = [Individual.random_individual() if random.random() < 0.9
                      else Individual.empty_individual()
                      for _ in range(pop_limit)]
        population = pool.map(Individual.calculate_fitness, population, batch_size)
        init_done = time.time()
        print("Created and calculated initial population statistics in:", init_done - init_time, "seconds")
        generation = 0
        start = time.time()
        best_fitness_history = []  # Track best fitness per generation for our new break condition.
        print("Use ctrl-c to terminate this loop manually.")
        
        max_generations = 50
        max_stagnant_generations = 10
        
        try:
            while True:
                now = time.time()
                best = max(population, key=Individual.fitness)
                current_fitness = best.fitness()
                print("Generation:", generation)
                print("Max fitness:", current_fitness)
                print("Average generation time:", (now - start) / (generation + 1))
                print("Net time:", now - start)
                
                # Update our best fitness history.
                best_fitness_history.append(current_fitness)
                if len(best_fitness_history) > 10:
                    # Check if improvement over the past 10 generations is less than threshold.
                    if best_fitness_history[-1] - best_fitness_history[0] < 1.0:
                        print("Breaking: Insufficient improvement over the last 10 generations.")
                        break
                    # Keep the history window limited to 10 generations.
                    best_fitness_history.pop(0)
                
                generation += 1
                
                if generation >= max_generations:
                    print("Stopping: Max generations reached.")
                    break
                
                next_population = generate_successors(population)
                gentime = time.time()
                next_population = pool.map(Individual.calculate_fitness, next_population, batch_size)
                popdone = time.time()
                print("Generated successors in:", gentime - now, "seconds")
                print("Calculated fitnesses in:", popdone - gentime, "seconds")
                population = next_population
        except KeyboardInterrupt:
            pass
    return population

if __name__ == "__main__":
    final_gen = sorted(ga(), key=Individual.fitness, reverse=True)
    best = final_gen[0]
    print("Best fitness: " + str(best.fitness()))
    now = time.strftime("%m_%d_%H_%M_%S")
    for k in range(10):
        with open("levels/" + now + "_" + str(k) + ".txt", 'w') as f:
            for row in final_gen[k].to_level():
                f.write("".join(row) + "\n")
