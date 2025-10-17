import gc
import copy
import time
import random
import itertools
import itertools
import numpy as np
from collections import deque

str_label = '6555565333335532222355322222355322222235632221222365322222235532222235532222355333335655556'
pos_index = {
    (-5, 0):0, (-5, 1):1, (-5, 2):2, (-5, 3):3, (-5, 4):4, (-5, 5):5, 
    (-4, -1):6, (-4, 0):7, (-4, 1):8, (-4, 2):9, (-4, 3):10, (-4, 4):11, (-4, 5):12, 
    (-3, -2):13, (-3, -1):14, (-3, 0):15, (-3, 1):16, (-3, 2):17, (-3, 3):18, (-3, 4):19, (-3, 5):20, 
    (-2, -3):21, (-2, -2):22, (-2, -1):23, (-2, 0):24, (-2, 1):25, (-2, 2):26, (-2, 3):27, (-2, 4):28, (-2, 5):29, 
    (-1, -4):30, (-1, -3):31, (-1, -2):32, (-1, -1):33, (-1, 0):34, (-1, 1):35, (-1, 2):36, (-1, 3):37, (-1, 4):38, (-1, 5):39, 
    (0, -5):40, (0, -4):41, (0, -3):42, (0, -2):43, (0, -1):44, (0, 0):45, (0, 1):46, (0, 2):47, (0, 3):48, (0, 4):49, (0, 5):50, 
    (1, -5):51, (1, -4):52, (1, -3):53, (1, -2):54, (1, -1):55, (1, 0):56, (1, 1):57, (1, 2):58, (1, 3):59, (1, 4):60, 
    (2, -5):61, (2, -4):62, (2, -3):63, (2, -2):64, (2, -1):65, (2, 0):66, (2, 1):67, (2, 2):68, (2, 3):69, 
    (3, -5):70, (3, -4):71, (3, -3):72, (3, -2):73, (3, -1):74, (3, 0):75, (3, 1):76, (3, 2):77, 
    (4, -5):78, (4, -4):79, (4, -3):80, (4, -2):81, (4, -1):82, (4, 0):83, (4, 1):84, 
    (5, -5):85, (5, -4):86, (5, -3):87, (5, -2):88, (5, -1):89, (5, 0):90
}
index_pos = {value: key for key, value in pos_index.items()}
owner_map = {None:'0', 'black':'1', 'white':'2'}

# hexagon_label to hexagon_board
def board_to_str(hexagon_board):
    hexagon_str = '0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'
    byte_str = bytearray(hexagon_str, 'utf-8')
    for pos, hex_info in hexagon_board.items():
        byte_str[pos_index[pos]] = ord(owner_map[hex_info['owner']])
    return byte_str.decode('utf-8')

def str_to_board(hexagon_str):
    hexagon_board = {}
    for index in range(91):
        owner = 'black' if hexagon_str[index]=='1' else 'white'
        hexagon_board[index_pos[index]] = {'selected': True, 'owner':owner}
    return hexagon_board

class Node:
    def __init__(self, hexagon_str, turn, round):
        self.hexagon_str = hexagon_str
        self.turn = turn
        self.round = round
        self.new_children = deque() #存無限大
        self.children = [] #n!=0
        self.w = 0  # wins
        self.n = 0  # simulations 

    def add_children(self, child):
        if child.n == 0:
            self.new_children.append(child)
        else:
            self.children.append(child)

    def max_UCB_child(self, c_param, inference):
        if self.new_children and not inference:
            new_child = self.new_children.popleft()
            self.children.append(new_child)
            return new_child
        else:
            max_child = None
            max_UCB = float('-inf')
            for child in self.children:

                if self.turn == 'black':       # black turn
                    exploitation = child.w / child.n

                elif self.turn == 'white':     # white turn
                    exploitation = (child.n - child.w) / child.n

                exploration = np.sqrt(2 * np.log(self.n) / child.n)
                UCB = exploitation + c_param * exploration

                if UCB > max_UCB:
                    max_child = child
                    max_UCB =  UCB

            return max_child

class MCTree:
    def __init__(self):
        # tree structure
        self.root = None
        self.tree_dict = {}
           
    def select(self):
        path = [self.root]
        current_node = self.root
        
        while current_node.children or current_node.new_children:
            # turn 1~30 and c_param 2.0~0.5
            c_param = -0.0517*(current_node.round)+2.0517 
            current_node = current_node.max_UCB_child(c_param, False)
            path.append(current_node)

        return path, path[-1]

    #----------------------- Roll Out Start -----------------------#
    # randomly play to end
    def rollout(self, node):
        def select_hexes_by_random(hexagon_label):
            """Selects hexes randomly based on the current round and label availability."""
            selected_hexes = []
            available_labels = [key for key, value in hexagon_label.items() if value]
            if available_labels:
                selected_label = random.choice(available_labels)
                available_hexes = hexagon_label[selected_label]
                n = int(selected_label)  # Determine the number of hexagons to select based on their labels.
            
                # Randomly select n hexes, select all remaining hexes if fewer than n are available
                if len(available_hexes) > n:
                    selected_hexes.extend(random.sample(available_hexes, n))
                else:
                    selected_hexes.extend(available_hexes)

                for hexes in selected_hexes:
                    hexagon_label[selected_label].remove(hexes)

            return selected_hexes
        
        def calculate_connected_areas(hexagon_board, owner):
            """Calculates the largest connected area of hexes of the specified color."""
            def dfs(row, col, visited):
                if (row, col) in visited or not (row, col) in hexagon_board or hexagon_board[(row, col)]['owner'] != owner:
                    return 0
                visited.add((row, col))
                count = 1
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
                    next_row, next_col = row + dr, col + dc
                    count += dfs(next_row, next_col, visited)
                return count

            visited = set()
            max_area = 0
            for (row, col), info in hexagon_board.items():
                if (row, col) not in visited and info['owner'] == owner:
                    area = dfs(row, col, visited)
                    if area > max_area:
                        max_area = area
            return max_area
        
        byte_str = bytearray(node.hexagon_str, 'utf-8')
        hexagon_label = {'1':[], '2':[], '3':[], '5':[], '6':[]}
        current_turn = node.turn
        for index in range(91):
            if node.hexagon_str[index] == '0':
                hexagon_label[str_label[index]].append(index)

        while 1:
            selected_hexes = select_hexes_by_random(hexagon_label)
            if not selected_hexes: 
                current_hexagon_board = str_to_board(byte_str.decode('utf-8'))
                black_area = calculate_connected_areas(current_hexagon_board, 'black')  
                white_area = calculate_connected_areas(current_hexagon_board, 'white') 
                black_win = True if black_area>white_area else False
                return black_win
            
            for index in selected_hexes:
                byte_str[index] = ord('1') if current_turn=='black' else ord('2')
            current_turn = 'white' if current_turn=='black' else 'black' 

    #----------------------- Roll Out End -----------------------#

    #----------------------- Expand Start -----------------------#
    def expand(self, node):

        def create_node(combination):
            byte_str = bytearray(node.hexagon_str, 'utf-8')
            for index in combination:
                byte_str[index] = ord(owner_map[node.turn])
            next_hexagon_str = byte_str.decode('utf-8')

            if next_hexagon_str in self.tree_dict:
                node.add_children(self.tree_dict[next_hexagon_str])
            else:
                next_round = node.round+1
                next_node = Node(next_hexagon_str, next_turn, next_round)
                node.add_children(next_node)
                self.tree_dict[next_hexagon_str] = next_node

        hexagon_label = {'1':[], '2':[], '3':[], '5':[], '6':[]}
        next_turn = 'white' if node.turn == 'black' else 'black'
        for index in range(91):
            if node.hexagon_str[index] == '0':
                hexagon_label[str_label[index]].append(index)

        if node.round == 1:
            # select one hexagon in label 2
            available_hexes = random.sample(hexagon_label['2'], 18) #36個選18個
            combinations = list(itertools.combinations(available_hexes, 1))
            for combination in combinations:
                create_node(combination)
                
        elif node.round < 10:
            available_labels = [key for key, value in hexagon_label.items() if value and key!='5' and key!='6']
            for label in available_labels:
                if label=='3' and len(hexagon_label[label])>12:  
                    available_hexes = random.sample(hexagon_label[label], 12)
                elif label=='2' and len(hexagon_label[label])>14:
                    available_hexes = random.sample(hexagon_label[label], 16)
                else:
                    available_hexes = hexagon_label[label]

                if len(available_hexes) > int(label):
                    combinations = list(itertools.combinations(available_hexes, int(label)))
                    for combination in combinations:
                        create_node(combination)
                else:
                    create_node(available_hexes)

        else:
            available_labels = [key for key, value in hexagon_label.items() if value]

            for label in available_labels:
                if label=='5' and len(hexagon_label[label])>10:
                    available_hexes = random.sample(hexagon_label[label], 10)
                else:
                    available_hexes = hexagon_label[label]

                if len(available_hexes) > int(label):
                    combinations = list(itertools.combinations(available_hexes, int(label)))
                    for combination in combinations:
                        create_node(combination)
                else:
                    create_node(available_hexes)
    # ------------------------ Expand End ---------------------------#

    def backpropagate(self, path, black_win):
        for node in path: 
            node.n += 1
            if black_win:
                node.w += 1

    def next_action(self, hexagon_board, current_round):
        
        hexagon_str = board_to_str(hexagon_board)
        print('\ncurrent_round', current_round)
        print('hexagon_str', hexagon_str)
        if current_round==35: return []
        start_time = time.time()

        if hexagon_str in self.tree_dict:
                self.root = self.tree_dict[hexagon_str]
                print('before: ', 'w: ', self.root.w, 'n: ', self.root.n, 'child_num: ', len(self.root.children)+len(self.root.new_children))
        else:
            turn = 'white' if current_round%2==0 else 'black'
            self.root = Node(hexagon_str, turn, current_round)
            self.tree_dict[hexagon_str] = self.root
            
        while 1:
            current_time = time.time()
            
            if current_time-start_time > 28: break

            path, next_node = self.select()
            if next_node.n == 0: 
                black_win = self.rollout(next_node)
                self.backpropagate(path, black_win)
            else:
                self.expand(next_node)
                current_round += 1
        
        selected_hexes = []
        c_param = -0.0345*(self.root.round)+1.5345
        print(c_param)
        next_node = self.root.max_UCB_child(c_param, True)
        for index in range(91):
            if self.root.hexagon_str[index] == next_node.hexagon_str[index]: continue
            selected_hexes.append((index_pos[index], hexagon_board[index_pos[index]]))
        print('after: ', 'w: ', self.root.w, 'n: ', self.root.n, 'child_num: ', len(self.root.children)+len(self.root.new_children))
        print('next: ', 'w: ', next_node.w, 'n: ', next_node.n, 'child_num: ', len(next_node.children)+len(next_node.new_children), 'c_param: ', c_param)
        return selected_hexes
    
MCTS = MCTree()
