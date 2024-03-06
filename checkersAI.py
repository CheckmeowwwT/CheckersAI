from keras import models, layers
from keras.models import load_model
import random
import numpy as np
import math
from copy import deepcopy

import pygame
import sys

class PygameCheckers:
    def __init__(self, board):
        pygame.init()
        self.board = board
        self.rows, self.cols = 8, 8
        self.square_size = 80  
        self.screen = pygame.display.set_mode((self.cols * self.square_size, self.rows * self.square_size))
        self.colors = [pygame.Color("white"), pygame.Color("gray")]

    def draw_board(self):
        for row in range(self.rows):
            for col in range(self.cols):
                color = self.colors[((row + col) % 2)]
                pygame.draw.rect(self.screen, color, (col * self.square_size, row * self.square_size, self.square_size, self.square_size))

    def draw_pieces(self):
        for y, row in enumerate(self.board.board):
            for x, cell in enumerate(row):
                center = ((x * self.square_size + self.square_size//2), (y * self.square_size + self.square_size//2))
                if cell in [1, 3]: 
                    pygame.draw.circle(self.screen, pygame.Color("red"), center, self.square_size//2 - 10)
                elif cell in [2, 4]:  
                    pygame.draw.circle(self.screen, pygame.Color("black"), center, self.square_size//2 - 10)

    def update_display(self):
        self.draw_board()
        self.draw_pieces()
        pygame.display.flip()

class CheckerBoard:
    def __init__(self):
        self.board = self.create_board()
        self.current_player = 1  
        self.next_player = 2
        self.must_jump_from = None  

    def create_board(self):
        board = [[None for _ in range(8)] for _ in range(8)]
        for row in range(3):
            for col in range(row % 2, 8, 2):
                board[row][col] = 2  
            for col in range((5 + row) % 2, 8, 2):
                board[row + 5][col] = 1  
        return board

    def print_board(self):
        piece_symbols = {None: '.', 1: 'x', 2: 'o', 3: 'X', 4: 'O'}  # Simple symbols for each piece type
        print('  a b c d e f g h')
        print(' +-----------------+')
        for y, row in enumerate(self.board):
            print(f"{8 - y}|{' '.join(piece_symbols[piece] for piece in row)}|")
        print(' +-----------------+')

    def available_moves(self, player):
        if self.must_jump_from:
            return self.get_jumps_from_position(*self.must_jump_from, player)

        all_moves = []
        all_jumps = []
        for y, row in enumerate(self.board):
            for x, cell in enumerate(row):
                if cell in [player, player + 2]:  
                    jumps = self.get_jumps_from_position(x, y, player)
                    if jumps:
                        all_jumps.extend(jumps)
                    elif not self.must_jump_from: 
                        moves = self.get_moves_from_position(x, y, player)
                        all_moves.extend(moves)

        return all_jumps if all_jumps else all_moves

    def get_moves_from_position(self, x, y, player):
        moves = []
        directions = []

        if self.board[y][x] == 1: 
            directions.extend([(1, -1), (1, 1)])  

        elif self.board[y][x] == 2:  
            directions.extend([(-1, -1), (-1, 1)])  

       

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8 and self.board[ny][nx] is None:
                moves.append(((x, y), (nx, ny)))

        return moves

    def get_jumps_from_position(self, x, y, player):
        jumps = []
        directions = []
        if self.board[y][x] == 1:  
            directions = [(1, -1), (1, 1)]
        elif self.board[y][x] == 2:
            directions = [(-1, -1), (-1, 1)]
        elif self.board[y][x] in [3, 4]:  # Kings
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  

        for dx, dy in directions:
            nx, ny = x + 2 * dx, y + 2 * dy
            mx, my = x + dx, y + dy  
            if (0 <= nx < 8 and 0 <= ny < 8 and self.board[ny][nx] is None and
                    self.board[my][mx] in [3 - player, 4 if 3 - player == 1 else 3]):
                jumps.append(((x, y), (nx, ny)))
        return jumps

    def switch_player(self):
        
        self.current_player, self.next_player = self.next_player, self.current_player

    def make_move(self, move):
        (x1, y1), (x2, y2) = move
        player = self.board[y1][x1]
        self.board[y2][x2] = player
        self.board[y1][x1] = None

        
        if player == 1 and y2 == 7:  
            self.board[y2][x2] = 3 
        elif player == 2 and y2 == 0:  
            self.board[y2][x2] = 4  

     
        if abs(x2 - x1) > 1:  
            self.board[(y1 + y2) // 2][(x1 + x2) // 2] = None
            further_jumps = self.get_jumps_from_position(x2, y2, player)
            if further_jumps:
                self.must_jump_from = (x2, y2)  
                return  

        self.must_jump_from = None  
        self.switch_player() 

    def check_winner(self):
        player1, player2 = False, False
        for row in self.board:
            for cell in row:
                if cell == 1 or cell == 3:
                    player1 = True
                elif cell == 2 or cell == 4:
                    player2 = True
        if not player1:
            return 2  
        if not player2:
            return 1  
        return None  

    def random_move(self, player):
        available = self.available_moves(player)
        return random.choice(available) if available else None

    def is_terminal(self):
   
        if self.check_winner() is not None:
            return True
        if not self.available_moves(self.current_player):
            return True  
        return False 

    def is_draw(self):
        
        if self.check_winner() is None and not self.available_moves(self.current_player):
            return True
        return False

    def result(self, player):
     
        winner = self.check_winner()
        if winner is None:
            return 0  
        return 1 if winner == player else -1

class CheckersAI:
    def __init__(self):
        self.model = self.build_checkers_model()
        self.gamma = 0.95 
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

    def build_checkers_model(self):
        model = models.Sequential([
            layers.Flatten(input_shape=(8, 8, 1)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='tanh')  #
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def predict_move(self, board):
        if np.random.rand() <= self.epsilon:
            
            available_moves = board.available_moves(board.current_player)
            return random.choice(available_moves) if available_moves else None
        else:
           
            root = MCTSNode(state=deepcopy(board))  
            best_move = MCTS(root, iterations=1000)  
            return best_move

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, state, action, reward, next_state, done):

        state_input = np.array([self.encode_board(state)]).reshape(-1, 8, 8, 1)
        next_state_input = np.array([self.encode_board(next_state)]).reshape(-1, 8, 8, 1)


        future_reward = 0
        if not done:
            future_reward = self.gamma * np.amax(self.model.predict(next_state_input)[0])

     
        target_reward = reward + future_reward

       
        target = self.model.predict(state_input)

        
        target[0][0] = target_reward  

        
        self.model.fit(state_input, target, epochs=1, verbose=0)

       
        self.update_epsilon()

    def encode_board(self, board_instance):
        
        encoded_board = np.zeros((8, 8, 1), dtype=np.float32)
        for y, row in enumerate(board_instance.board):
            for x, cell in enumerate(row):
                encoded_board[y, x, 0] = cell if cell is not None else 0
        return encoded_board

    def play_self_game(self):
        board = CheckerBoard()  
        move_history = []

        while not board.is_terminal():
            current_state = board  
            move = self.predict_move(board)  

            if move:
                board.make_move(move)
                next_state = board  
                reward = 0
                done = board.is_terminal()
               
                move_history.append((current_state, move, reward, next_state, done))
            else:
                break 

      
        for current_state, move, reward, next_state, done in move_history:
           
            self.train(current_state, move, reward, next_state, done)

    def train_from_self_play(self, number_of_games=100):
        for _ in range(number_of_games):
            self.play_self_game()
            



class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state  
        self.parent = parent
        self.move = move  
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = state.available_moves(state.current_player)  

    def select_child(self):
       
       
        exploration_constant = math.sqrt(2)

       
        best_score = -float('inf')
        best_child = None

        for child in self.children:
            
            ucb1_score = child.wins / child.visits + exploration_constant * math.sqrt(
                math.log(self.visits) / child.visits)

            if ucb1_score > best_score:
                best_score = ucb1_score
                best_child = child

        return best_child

    def add_child(self, move, state):
      
        child = MCTSNode(state=state, parent=self, move=move)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result):
       
        self.visits += 1
        self.wins += result


def MCTS(root, iterations):
    for _ in range(iterations):
        node = root
        state = deepcopy(node.state)

  
        while node.untried_moves == [] and node.children != []:  
            node = node.select_child()
            state.make_move(node.move)


        if node.untried_moves and not state.is_terminal():
            move = random.choice(node.untried_moves)
            state.make_move(move)
            new_node = node.add_child(move, deepcopy(state))

            
            while not state.is_terminal():
                available_moves = state.available_moves(state.current_player)
                if available_moves:
                    move = random.choice(available_moves)
                    state.make_move(move)
                else:
                    break

          
            result = state.result(new_node.state.current_player if new_node.state else None)
            while new_node is not None:
                new_node.update(result)
                new_node = new_node.parent

        if not root.children:
           return none

   
    return max(root.children, key=lambda c: c.wins / c.visits).move


def simulate_ai_game_with_pygame(ai):
    board = CheckerBoard()
    pg_checker = PygameCheckers(board)
    game_over = False

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True 

        if not board.check_winner() and not board.is_terminal():
            pg_checker.update_display()
            move = ai.predict_move(board)  
            if move:
                print(f"AI moves from {move[0]} to {move[1]}")
                board.make_move(move)
            else:
                print("No moves available.")

            pygame.time.wait(500)  
        else:
           
            winner = board.check_winner()
            if winner:
                print(f"Game over. Winner: Player {winner}")
            else:
                print("Game ended in a draw.")

            pg_checker.update_display()  
            pygame.time.wait(1500)  

          
            waiting_for_close = True
            while waiting_for_close:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting_for_close = False
                        game_over = True
                    elif event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                        
                        waiting_for_close = False
                        game_over = True

    pygame.quit()


if __name__ == '__main__':
    AI = CheckersAI()
    #AI.train_from_self_play(number_of_games=100)
    # Save the model after training
    #AI.model.save('my_checkers_model.h5')

    simulate_ai_game_with_pygame(AI)
   



