import random
import numpy as np
from typing import List, Tuple, Optional
from metrics import metrics

class Individual_Grid:
    def __init__(self, width=200, height=16):
        self.width = width
        self.height = height
        self.genome = [['-' for x in range(width)] for y in range(height)]
        self.fitness = None
        
    def generate_children(self, other) -> List['Individual_Grid']:
        # Single point crossover
        child = Individual_Grid(self.width, self.height)
        crossover_point = random.randint(0, self.width-1)
        
        for y in range(self.height):
            for x in range(self.width):
                if x < crossover_point:
                    child.genome[y][x] = self.genome[y][x]
                else:
                    child.genome[y][x] = other.genome[y][x]
        
        return [child]

    def mutate(self) -> None:
        """Apply random mutations to the genome with improved constraints"""
        mutation_rate = 0.05
        MIN_JUMP_HEIGHT = 3
        MAX_JUMP_HEIGHT = 6
        
        for x in range(2, self.width-2):  # Don't modify edges
            if random.random() < mutation_rate:
                # Randomly select a vertical position
                y = random.randint(self.height-MAX_JUMP_HEIGHT, self.height-MIN_JUMP_HEIGHT)
                
                # Only modify if space is empty and surroundings are clear
                if (self.genome[y][x] == '-' and 
                    all(self.genome[y+dy][x] == '-' for dy in [-1, 1] if 0 <= y+dy < self.height)):
                    
                    # Different probabilities for different elements
                    element = random.choices(
                        ['-', '?', 'B', 'o', 'E'],
                        weights=[0.6, 0.15, 0.1, 0.1, 0.05]
                    )[0]
                    
                    # Special handling for enemies
                    if element == 'E':
                        # Only place on ground and with spacing
                        if (self.genome[self.height-1][x] == 'X' and 
                            all(self.genome[self.height-2][x+dx] != 'E' 
                                for dx in range(-3, 4) if 0 <= x+dx < self.width)):
                            self.genome[self.height-2][x] = 'E'
                    else:
                        self.genome[y][x] = element
                            
        # Maintain Mario and flag positions
        self.genome[self.height-2][1] = 'm'  # Mario
        
        # Maintain flagpole
        flag_x = self.width-2
        flag_height = 5
        for y in range(self.height-2, self.height-flag_height-1, -1):
            self.genome[y][flag_x] = 'f'
        self.genome[self.height-flag_height-1][flag_x] = 'v'

    def _apply_constrained_mutation(self, x: int, y: int) -> None:
        ground_level = self.height - 1

        if y == ground_level:
            self.genome[y][x] = random.choice(['X', '|'])
            if self.genome[y][x] == '|':
                self._validate_pipe_placement(x, y)
        elif y > ground_level - 4:  
            if self._is_between_pipes(x):
                if self._count_goombas_between_pipes(x) < 3:
                    self.genome[y][x] = 'E'
        else:  
            self.genome[y][x] = random.choice(['-', '?', 'B', 'o'])
            
    def _validate_pipe_placement(self, x: int, y: int) -> None:
        min_pipe_distance = 5
        for dx in range(-min_pipe_distance, min_pipe_distance + 1):
            check_x = x + dx
            if 0 <= check_x < self.width and check_x != x:
                if self.genome[y][check_x] == '|':
                    self.genome[y][x] = 'X'
                    return
                    
    def _is_between_pipes(self, x: int) -> bool:
        left_pipe = right_pipe = None
        for dx in range(x-1, -1, -1):
            if self.genome[self.height-1][dx] == '|':
                left_pipe = dx
                break
        for dx in range(x+1, self.width):
            if self.genome[self.height-1][dx] == '|':
                right_pipe = dx
                break
        return left_pipe is not None and right_pipe is not None

    def _count_goombas_between_pipes(self, x: int) -> int:
        count = 0
        left_x = right_x = x
        while left_x > 0 and self.genome[self.height-1][left_x] != '|':
            left_x -= 1
        while right_x < self.width and self.genome[self.height-1][right_x] != '|':
            right_x += 1
            
        for check_x in range(left_x, right_x):
            for y in range(self.height):
                if self.genome[y][check_x] == 'E':
                    count += 1
        return count

    def calculate_fitness(self) -> float:
        """Calculate fitness with improved metrics"""
        level_lines = self.to_level()
        level_metrics = metrics(level_lines)
        
        if not level_metrics['solvability']:
            return 0.0
            
        fitness = 0.0
        
        # Basic metrics with adjusted weights
        fitness += level_metrics['pathPercentage'] * 4.0
        fitness += level_metrics['decorationPercentage'] * 2.0
        fitness += level_metrics['negativeSpace'] * 1.5
        
        # Reward meaningful jumps
        fitness += level_metrics['meaningfulJumps'] * 2.0
        
        # Penalize extreme linearity but not too harshly
        if level_metrics['linearity'] > 0.8:
            fitness -= (level_metrics['linearity'] - 0.8) * 1.5
        
        # Add bonus for good decoration balance
        if 0.2 <= level_metrics['decorationPercentage'] <= 0.4:
            fitness += 2.0
        
        # Ensure non-negative fitness
        return max(0.0, fitness)

    def _check_pipe_constraints(self) -> int:
        violations = 0
        last_pipe_x = -float('inf')
        
        for x in range(self.width):
            if self.genome[self.height-1][x] == '|':
                if x - last_pipe_x < 5:  # Min pipe distance
                    violations += 1
                last_pipe_x = x
                
                # Check for floating pipes
                if any(self.genome[y][x] == '|' for y in range(self.height-1)):
                    violations += 1
                    
        return violations

    def _check_goomba_constraints(self) -> int:
        violations = 0
        current_goombas = 0
        in_pipe_section = False
        
        for x in range(self.width):
            if self.genome[self.height-1][x] == '|':
                if current_goombas > 3:
                    violations += current_goombas - 3
                current_goombas = 0
                in_pipe_section = True
            elif in_pipe_section:
                if any(self.genome[y][x] == 'E' for y in range(self.height)):
                    current_goombas += 1
                    
        return violations

    def _check_block_constraints(self) -> int:
        violations = 0
        
        for y in range(self.height):
            for x in range(self.width):
                if self.genome[y][x] in ['?', 'B']:
                    # Check for ground contact
                    if y == self.height - 1:
                        violations += 1
                    # Check for adjacency to pipes
                    if any(self.genome[y+dy][x+dx] == '|' 
                          for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]
                          if 0 <= y+dy < self.height and 0 <= x+dx < self.width):
                        violations += 1
                        
        return violations

    def _has_unreachable_blocks(self) -> bool:
        """Check for unreachable blocks"""
        MAX_JUMP_HEIGHT = 4
        for x in range(self.width):
            lowest_block = -1
            for y in range(self.height-1):
                if self.genome[y][x] in ['?', 'B', 'o']:
                    if lowest_block == -1:
                        lowest_block = y
                    if y < self.height-MAX_JUMP_HEIGHT-1:  # Too high
                        return True
        return False

    def _has_invalid_pipe_placement(self) -> bool:
        """Check for invalid pipe placement"""
        for x in range(self.width):
            for y in range(self.height-1):
                if self.genome[y][x] == '|':
                    # Check if pipe has ground support
                    if y < self.height-1 and self.genome[y+1][x] not in ['|', 'X']:
                        return True
                    # Check if pipe top is properly placed
                    if self.genome[y][x] == 'T' and y > 0 and self.genome[y-1][x] != '-':
                        return True
        return False

    def to_level(self) -> List[str]:
        """Convert genome to level string format as list of lines"""
        lines = []
        for y in range(self.height):
            # Ensure each line is exactly width characters long
            line = ''.join(self.genome[y][:self.width])  # Truncate if too long
            line = line.ljust(self.width, '-')  # Pad with empty space if too short
            lines.append(line)
        return lines

    def to_level_string(self) -> str:
        """Convert genome to single string format for file saving"""
        lines = self.to_level()
        return '\n'.join(lines) + '\n'  # Add final newline for Unity compatibility

    def from_level(self, level_str: str) -> None:
        """Load genome from level string format"""
        self.genome = [list(row) for row in level_str.split('\n')]
        self.height = len(self.genome)
        self.width = len(self.genome[0])

class Individual_DE:
    def __init__(self, width=200, height=16):
        self.width = width
        self.height = height
        self.genome = []  # List of design elements
        self.fitness = None
        
    def generate_children(self, other) -> List['Individual_DE']:
        child = Individual_DE(self.width, self.height)
        
        # Variable point crossover
        crossover_points = sorted(random.sample(
            range(min(len(self.genome), len(other.genome))), 
            random.randint(1, 3)
        ))
        
        current_parent = self
        current_point = 0
        
        for point in crossover_points:
            child.genome.extend(current_parent.genome[current_point:point])
            current_point = point
            current_parent = other if current_parent == self else self
            
        child.genome.extend(current_parent.genome[current_point:])
        
        return [child]

    def mutate(self) -> None:
        # Element addition/removal
        if random.random() < 0.3:
            if random.random() < 0.5 and self.genome:
                # Remove random element
                del self.genome[random.randrange(len(self.genome))]
            else:
                # Add new random element
                self.genome.append(self._generate_random_element())
                
        # Element modification
        for element in self.genome:
            if random.random() < 0.1:
                self._mutate_element(element)

    def _generate_random_element(self) -> dict:
        element_type = random.choice(['hole', 'platform', 'enemy', 'coin', 'block', 'pipe', 'stairs'])
        
        if element_type == 'hole':
            return {
                'type': 'hole',
                'width': random.randint(2, 4),
                'x': random.randint(0, self.width-4)
            }
        elif element_type == 'platform':
            return {
                'type': 'platform',
                'width': random.randint(2, 6),
                'height': random.randint(3, self.height-4),
                'x': random.randint(0, self.width-6),
                'block_type': random.choice(['?', 'X', 'B'])
            }
        elif element_type == 'enemy':
            return {
                'type': 'enemy',
                'x': random.randint(0, self.width-1)
            }
        elif element_type == 'coin':
            return {
                'type': 'coin',
                'x': random.randint(0, self.width-1),
                'height': random.randint(2, self.height-4)
            }
        elif element_type == 'block':
            return {
                'type': 'block',
                'x': random.randint(0, self.width-1),
                'height': random.randint(2, self.height-4),
                'breakable': random.choice([True, False])
            }
        elif element_type == 'pipe':
            return {
                'type': 'pipe',
                'x': random.randint(0, self.width-1),
                'height': random.randint(2, 4)
            }
        else:  # stairs
            return {
                'type': 'stairs',
                'x': random.randint(0, self.width-4),
                'height': random.randint(2, 4),
                'direction': random.choice([-1, 1])
            }

    def _mutate_element(self, element: dict) -> None:
        """Modify element properties within constraints"""
        if element['type'] == 'hole':
            element['width'] = max(2, min(4, element['width'] + random.randint(-1, 1)))
        elif element['type'] == 'platform':
            element['width'] = max(2, min(6, element['width'] + random.randint(-1, 1)))
            element['height'] = max(3, min(self.height-4, element['height'] + random.randint(-1, 1)))
        elif element['type'] in ['coin', 'block']:
            element['height'] = max(2, min(self.height-4, element['height'] + random.randint(-1, 1)))
        elif element['type'] == 'pipe':
            element['height'] = max(2, min(4, element['height'] + random.randint(-1, 1)))
        elif element['type'] == 'stairs':
            element['height'] = max(2, min(4, element['height'] + random.randint(-1, 1)))
            if random.random() < 0.1:
                element['direction'] *= -1

    def calculate_fitness(self) -> float:
        """Calculate fitness based on level metrics"""
        # Get the level as a list of strings
        level_lines = self.to_level()  # This now returns a list of strings
        
        # Calculate metrics
        level_metrics = metrics(level_lines)  # metrics expects a list of strings
        
        if not level_metrics['solvability']:
            return 0.0
            
        fitness = 0.0
        
        # Basic level design rewards
        fitness += level_metrics['pathPercentage'] * 3.0
        fitness += level_metrics['decorationPercentage'] * 2.0
        fitness += level_metrics['negativeSpace'] * 1.5
        
        # Gameplay element rewards
        fitness += level_metrics['meaningfulJumps'] * 0.5
        fitness += min(level_metrics['jumps'], 10) * 0.3  # Cap jump rewards
        
        # Level flow penalties
        if level_metrics['linearity'] > 0.8:  # Penalize too much linearity
            fitness -= (level_metrics['linearity'] - 0.8) * 2.0
            
        return max(0.0, fitness)  # Ensure non-negative fitness

    def _count_elements(self) -> dict:
        counts = {'hole': 0, 'platform': 0, 'enemy': 0, 'coin': 0, 
                 'block': 0, 'pipe': 0, 'stairs': 0}
        for element in self.genome:
            counts[element['type']] += 1
        return counts

    def to_level(self) -> str:
        """Convert design elements to level string"""
        # Initialize empty level
        level = [['-' for x in range(self.width)] for y in range(self.height)]
        
        # Add ground
        for x in range(self.width):
            level[self.height-1][x] = 'X'
            
        # Apply design elements in order
        for element in self.genome:
            self._apply_element(level, element)
            
        return '\n'.join(''.join(row) for row in level)

    def _apply_element(self, level: List[List[str]], element: dict) -> None:
        """Apply a single design element to the level grid"""
        x = element['x']
        
        if element['type'] == 'hole':
            # Create holes in ground level, keeping track of minimum safety width
            for hole_x in range(x, min(x + element['width'], self.width)):
                if hole_x > 1 and hole_x < self.width - 2:  # Leave edges solid
                    level[self.height-1][hole_x] = '-'
                    
        elif element['type'] == 'platform':
            # Create platforms at specified height
            y = element['height']
            block_type = element['block_type']
            for platform_x in range(x, min(x + element['width'], self.width)):
                if 0 <= y < self.height - 1:  # Don't place at ground level
                    level[y][platform_x] = block_type
                    
        elif element['type'] == 'enemy':
            # Place enemy (Goomba) on ground if valid position
            if level[self.height-1][x] == 'X':  # Only place on solid ground
                # Check for nearby pipes to enforce goomba constraints
                left_pipe = right_pipe = False
                for dx in range(-3, 4):  # Check 3 blocks left and right
                    check_x = x + dx
                    if 0 <= check_x < self.width:
                        if level[self.height-1][check_x] == '|':
                            if dx < 0:
                                left_pipe = True
                            else:
                                right_pipe = True
                
                # Only place if between pipes and not too many goombas
                if left_pipe and right_pipe:
                    goomba_count = 0
                    for check_x in range(x-3, x+4):
                        if 0 <= check_x < self.width:
                            if level[self.height-2][check_x] == 'E':
                                goomba_count += 1
                    
                    if goomba_count < 3:  # Maximum 3 goombas between pipes
                        level[self.height-2][x] = 'E'
                    
        elif element['type'] == 'coin':
            # Place floating coins, making sure they're not intersecting with other elements
            y = element['height']
            if 0 <= y < self.height - 1:  # Don't place at ground level
                if level[y][x] == '-':  # Only place in empty space
                    level[y][x] = 'o'
                    
        elif element['type'] == 'block':
            # Place question or breakable blocks
            y = element['height']
            if 0 <= y < self.height - 1:  # Don't place at ground level
                if level[y][x] == '-':  # Only place in empty space
                    block_type = 'B' if element['breakable'] else '?'
                    # Check for vertical spacing
                    can_place = True
                    for dy in [-1, 0, 1]:
                        check_y = y + dy
                        if 0 <= check_y < self.height:
                            if level[check_y][x] not in ['-', 'o']:  # Allow overlap with coins
                                can_place = False
                                break
                    if can_place:
                        level[y][x] = block_type
                    
        elif element['type'] == 'pipe':
            # Place pipes, ensuring proper spacing and ground contact
            height = element['height']
            if x < self.width - 1 and level[self.height-1][x] == 'X':  # Need space and ground
                # Check for minimum pipe spacing
                can_place = True
                for dx in range(-4, 5):  # Check 4 blocks left and right
                    check_x = x + dx
                    if 0 <= check_x < self.width:
                        if level[self.height-1][check_x] == '|':
                            can_place = False
                            break
                            
                if can_place:
                    # Place pipe base
                    level[self.height-1][x] = '|'
                    level[self.height-1][x+1] = '|'
                    # Place pipe body
                    for y in range(self.height-2, self.height-height-1, -1):
                        if y >= 0:
                            level[y][x] = '|'
                            level[y][x+1] = '|'
                    # Place pipe top
                    if self.height-height-1 >= 0:
                        level[self.height-height-1][x] = 'T'
                        level[self.height-height-1][x+1] = 'T'
                        
        elif element['type'] == 'stairs':
            # Create ascending/descending stairs
            height = element['height']
            direction = element['direction']
            for h in range(height):
                stair_x = x + (h * direction)
                if 0 <= stair_x < self.width:
                    # Build stairs from ground up
                    for y in range(self.height-1, self.height-h-2, -1):
                        if y >= 0:
                            level[y][stair_x] = 'X'

def generate_successors(population: List[Individual_Grid], num_children: int) -> List[Individual_Grid]:
    """Implement both tournament and roulette wheel selection"""
    successors = []
    
    # Tournament selection
    tournament_size = 5
    for _ in range(num_children // 2):
        parent1 = _tournament_select(population, tournament_size)
        parent2 = _tournament_select(population, tournament_size)
        
        children = parent1.generate_children(parent2)
        for child in children:
            child.mutate()
            successors.append(child)
            
    # Roulette wheel selection
    total_fitness = sum(ind.fitness for ind in population)
    for _ in range(num_children - len(successors)):
        parent1 = _roulette_select(population, total_fitness)
        parent2 = _roulette_select(population, total_fitness)
        
        children = parent1.generate_children(parent2)
        for child in children:
            child.mutate()
            successors.append(child)
            
    return successors

def _tournament_select(population: List[Individual_Grid], 
                    tournament_size: int) -> Individual_Grid:
    """Select individual through tournament selection"""
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda ind: ind.fitness)

def _roulette_select(population: List[Individual_Grid], 
                    total_fitness: float) -> Individual_Grid:
    """Select individual through roulette wheel selection"""
    r = random.uniform(0, total_fitness)
    current_sum = 0
    for ind in population:
        current_sum += ind.fitness
        if current_sum > r:
            return ind
    return population[-1]  # Fallback

def to_level(self) -> List[str]:
    """Convert genome to level string format as list of lines"""
    lines = []
    for y in range(self.height):
        # Ensure each line is exactly width characters long
        line = ''.join(self.genome[y][:self.width])  # Truncate if too long
        line = line.ljust(self.width, '-')  # Pad with empty space if too short
        lines.append(line)
    return lines

def to_level_string(self) -> str:
    """Convert genome to single string format for file saving"""
    lines = self.to_level()
    return '\n'.join(lines) + '\n'  # Add final newline for Unity compatibility

def ga():
    """Main genetic algorithm loop"""
    population_size = 100
    num_generations = 50
    population = []
    
    # Variables for early stopping
    best_fitness = float('-inf')
    generations_without_improvement = 0
    max_generations_without_improvement = 10
    
    # Initialize population
    for _ in range(population_size):
        if random.random() < 0.7:
            ind = random_individual()
        else:
            ind = empty_individual()
        population.append(ind)
    
    # Evolution loop
    try:
        for generation in range(num_generations):
            # Calculate fitness
            for ind in population:
                if ind.fitness is None:
                    ind.fitness = ind.calculate_fitness()
            
            # Sort population
            population.sort(key=lambda ind: ind.fitness or 0.0, reverse=True)
            
            # Check for improvement
            current_best_fitness = population[0].fitness
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                generations_without_improvement = 0
                print(f"Generation {generation}: New Best Fitness = {best_fitness}")
            else:
                generations_without_improvement += 1
                print(f"Generation {generation}: Best Fitness = {current_best_fitness} (No improvement for {generations_without_improvement} generations)")
            
            # Early stopping check
            if generations_without_improvement >= max_generations_without_improvement:
                print(f"Stopping early: No improvement for {max_generations_without_improvement} generations")
                break
            
            # Generate next population
            next_population = []
            
            # Elitism
            elitism_count = population_size // 10
            next_population.extend(population[:elitism_count])
            
            # Generate remaining through selection/crossover/mutation
            children = generate_successors(population, population_size - elitism_count)
            next_population.extend(children)
            
            population = next_population
            
            # Save best level
            with open("levels/last.txt", "w") as f:
                f.write(population[0].to_level_string())
            
            # Save samples
            if generation % 10 == 0:
                for i, ind in enumerate(population[:5]):
                    with open(f"levels/gen{generation}_sample{i}.txt", "w") as f:
                        f.write(ind.to_level_string())
                        
    except Exception as e:
        print(f"Error during evolution: {str(e)}")
        raise
        
    return population[0]

def random_individual() -> Individual_Grid:
    ind = Individual_Grid()
    # Initialize with empty spaces
    ind.genome = [['-' for x in range(ind.width)] for y in range(ind.height)]
    
    # Add solid ground
    for x in range(ind.width):
        ind.genome[ind.height-1][x] = 'X'
    
    # Constants for better level design
    MIN_JUMP_HEIGHT = 3
    MAX_JUMP_HEIGHT = 6
    BLOCK_SPAWN_CHANCE = 0.04
    PIPE_SPAWN_CHANCE = 0.08
    ENEMY_SPAWN_CHANCE = 0.02
    MIN_PIPE_SPACING = 8
    MIN_ENEMY_SPACING = 5
    FLAGPOLE_SAFETY_ZONE = 10
    
    # First, place all pipes to establish boundaries for other elements
    pipe_positions = []
    x = 5  # Start after initial area
    while x < ind.width - FLAGPOLE_SAFETY_ZONE:
        if random.random() < PIPE_SPAWN_CHANCE and (not pipe_positions or x - pipe_positions[-1] >= MIN_PIPE_SPACING):
            pipe_height = random.randint(2, 3)
            # Place single pipe, starting one tile ABOVE ground
            for y in range(ind.height-2, ind.height-pipe_height-2, -1):
                ind.genome[y][x] = '|'
            # Add pipe top
            ind.genome[ind.height-pipe_height-2][x] = 'T'
            pipe_positions.append(x)
            x += MIN_PIPE_SPACING
        else:
            x += 1
    
    # Place blocks between pipes
    last_block_type = None
    last_block_x = 0
    
    for i in range(len(pipe_positions) + 1):
        start_x = 2 if i == 0 else pipe_positions[i-1] + 2
        if i == len(pipe_positions):
            end_x = ind.width - FLAGPOLE_SAFETY_ZONE
        else:
            end_x = pipe_positions[i] - 2
        
        for x in range(start_x, end_x):
            if random.random() < BLOCK_SPAWN_CHANCE:
                if x >= ind.width - FLAGPOLE_SAFETY_ZONE:
                    continue
                
                block_y = ind.height - random.randint(MIN_JUMP_HEIGHT, MAX_JUMP_HEIGHT)
                
                if ind.genome[block_y][x] == '-':
                    if (block_y > 0 and block_y < ind.height-1 and 
                        all(ind.genome[y][x] == '-' for y in [block_y-1, block_y+1])):
                        
                        available_types = ['?', 'B', 'o']
                        if last_block_type == 'B' and x - last_block_x < 3:
                            available_types.remove('B')
                        elif last_block_type == '?' and x - last_block_x < 3:
                            available_types.remove('?')
                        
                        block_type = random.choices(
                            available_types,
                            weights=[0.5, 0.3, 0.2][:len(available_types)]
                        )[0]
                        
                        ind.genome[block_y][x] = block_type
                        last_block_type = block_type
                        last_block_x = x
    
    # Place enemies with safe zone check
    last_enemy_x = 0
    enemy_limit = ind.width - FLAGPOLE_SAFETY_ZONE  # Define safe zone boundary
    if pipe_positions:  # If we have pipes, use the last pipe as boundary
        enemy_limit = min(enemy_limit, pipe_positions[-1])
    
    for x in range(2, enemy_limit):
        if (x - last_enemy_x > MIN_ENEMY_SPACING and 
            random.random() < ENEMY_SPAWN_CHANCE and 
            ind.genome[ind.height-1][x] == 'X' and
            x not in pipe_positions):
            
            clear_area = True
            for dx in [-1, 0, 1]:
                check_x = x + dx
                if 0 <= check_x < ind.width:
                    if ind.genome[ind.height-2][check_x] != '-':
                        clear_area = False
                        break
            
            if clear_area:
                ind.genome[ind.height-2][x] = 'E'
                last_enemy_x = x
    
    # Add Mario at start
    ind.genome[ind.height-2][1] = 'm'
    
    # Add flagpole
    flag_x = ind.width-2
    flag_height = 5
    
    # Add pole
    for y in range(ind.height-2, ind.height-flag_height-1, -1):
        ind.genome[y][flag_x] = 'f'
    
    # Add top
    ind.genome[ind.height-flag_height-1][flag_x] = 'v'
    
    return ind
    
def empty_individual() -> Individual_Grid:
    """Create an empty individual with just ground"""
    ind = Individual_Grid()
    
    # Initialize with empty spaces
    ind.genome = [['-' for x in range(ind.width)] for y in range(ind.height)]
    
    # Add solid ground
    for x in range(ind.width):
        ind.genome[ind.height-1][x] = 'X'
    
    # Add Mario's start position
    ind.genome[ind.height-2][1] = 'm'
    
    # Add goal
    ind.genome[ind.height-1][ind.width-2] = 'X'  # Ground under flag
    ind.genome[ind.height-2][ind.width-2] = 'v'  # Flagpole
    for y in range(ind.height-3, -1, -1):  # Flag
        ind.genome[y][ind.width-2] = 'f'
    
    return ind

if __name__ == '__main__':
    print("Starting genetic algorithm...")
    best = ga()
    print(f"Final Best Fitness: {best.fitness}")