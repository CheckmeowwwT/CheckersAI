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
        self.square_size = 80  # Adjust as per your screen size
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
                if cell in [1, 3]:  # Player 1 and kings
                    pygame.draw.circle(self.screen, pygame.Color("red"), center, self.square_size//2 - 10)
                elif cell in [2, 4]:  # Player 2 and kings
                    pygame.draw.circle(self.screen, pygame.Color("black"), center, self.square_size//2 - 10)

    def update_display(self):
        self.draw_board()
        self.draw_pieces()
        pygame.display.flip()

class CheckerBoard:
    def __init__(self):
        self.board = self.create_board()
        self.current_player = 1  # Player 1 starts
        self.next_player = 2
        self.must_jump_from = None  # Track if a multi-jump is in progress

    def create_board(self):
        board = [[None for _ in range(8)] for _ in range(8)]
        for row in range(3):
            for col in range(row % 2, 8, 2):
                board[row][col] = 2  # Player 2 pieces
            for col in range((5 + row) % 2, 8, 2):
                board[row + 5][col] = 1  # Player 1 pieces
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
                if cell in [player, player + 2]:  # Check for player's pieces and kings
                    jumps = self.get_jumps_from_position(x, y, player)
                    if jumps:
                        all_jumps.extend(jumps)
                    elif not self.must_jump_from:  # If not in a multi-jump, add regular moves
                        moves = self.get_moves_from_position(x, y, player)
                        all_moves.extend(moves)

        return all_jumps if all_jumps else all_moves

    def get_moves_from_position(self, x, y, player):
        moves = []
        directions = []

        if self.board[y][x] == 1:  # Normal piece for Player 1
            directions.extend([(1, -1), (1, 1)])  # Only down left and down right

        elif self.board[y][x] == 2:  # Normal piece for Player 2
            directions.extend([(-1, -1), (-1, 1)])  # Only up left and up right

        # Check for Kings and allow all directions if necessary, depending on your implementation

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8 and self.board[ny][nx] is None:
                moves.append(((x, y), (nx, ny)))

        return moves

    def get_jumps_from_position(self, x, y, player):
        jumps = []
        directions = []
        if self.board[y][x] == 1:  # Player 1's normal pieces
            directions = [(1, -1), (1, 1)]
        elif self.board[y][x] == 2:  # Player 2's normal pieces
            directions = [(-1, -1), (-1, 1)]
        elif self.board[y][x] in [3, 4]:  # Kings
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # Can move & jump in all directions

        for dx, dy in directions:
            nx, ny = x + 2 * dx, y + 2 * dy
            mx, my = x + dx, y + dy  # Middle square for potential capture
            if (0 <= nx < 8 and 0 <= ny < 8 and self.board[ny][nx] is None and
                    self.board[my][mx] in [3 - player, 4 if 3 - player == 1 else 3]):
                jumps.append(((x, y), (nx, ny)))
        return jumps

    def switch_player(self):
        # Toggle between the two players
        self.current_player, self.next_player = self.next_player, self.current_player

    def make_move(self, move):
        (x1, y1), (x2, y2) = move
        player = self.board[y1][x1]
        self.board[y2][x2] = player
        self.board[y1][x1] = None

        # Promote to King if reaching the opposite side
        if player == 1 and y2 == 7:  # Player 1 reaches the bottom
            self.board[y2][x2] = 3  # Promote to King
        elif player == 2 and y2 == 0:  # Player 2 reaches the top
            self.board[y2][x2] = 4  # Promote to King

        # Handle jumps
        if abs(x2 - x1) > 1:  # A jump was made
            self.board[(y1 + y2) // 2][(x1 + x2) // 2] = None
            further_jumps = self.get_jumps_from_position(x2, y2, player)
            if further_jumps:
                self.must_jump_from = (x2, y2)  # Must continue jumping
                return  # Further jumps available, don't switch player yet

        self.must_jump_from = None  # No further jumps, reset
        self.switch_player()  # Switch turns

    def check_winner(self):
        player1, player2 = False, False
        for row in self.board:
            for cell in row:
                if cell == 1 or cell == 3:
                    player1 = True
                elif cell == 2 or cell == 4:
                    player2 = True
        if not player1:
            return 2  # Player 2 wins
        if not player2:
            return 1  # Player 1 wins
        return None  # No winner yet

    def random_move(self, player):
        available = self.available_moves(player)
        return random.choice(available) if available else None

    def is_terminal(self):
        # Check if the game has ended either by a player winning or a draw.
        if self.check_winner() is not None:
            return True  # A player has won the game.
        if not self.available_moves(self.current_player):
            return True  # No available moves for the current player, indicating a draw.
        return False  # The game is still ongoing.

    def is_draw(self):
        # The game is considered a draw if there are no legal moves for the current player,
        # and no player has won the game yet.
        if self.check_winner() is None and not self.available_moves(self.current_player):
            return True
        return False

    def result(self, player):
        # Evaluate the game result from the perspective of 'player'
        winner = self.check_winner()
        if winner is None:
            return 0  # Game is ongoing
        return 1 if winner == player else -1

class CheckersAI:
    def __init__(self):
        self.model = self.build_checkers_model()
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

    def build_checkers_model(self):
        model = models.Sequential([
            layers.Flatten(input_shape=(8, 8, 1)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='tanh')  # Output a score for the board state
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def predict_move(self, board):
        if np.random.rand() <= self.epsilon:
            # Exploration: Random move
            available_moves = board.available_moves(board.current_player)
            return random.choice(available_moves) if available_moves else None
        else:
            # Exploitation: Use MCTS
            root = MCTSNode(state=deepcopy(board))  # Create a root node for MCTS
            best_move = MCTS(root, iterations=1000)  # Perform MCTS with specified iterations
            return best_move

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, state, action, reward, next_state, done):
        # Prepare the inputs for the network
        # Note: The action is not directly used to adjust the network since we're predicting state values, not actions.
        state_input = np.array([self.encode_board(state)]).reshape(-1, 8, 8, 1)
        next_state_input = np.array([self.encode_board(next_state)]).reshape(-1, 8, 8, 1)

        # Predict the future discounted reward
        future_reward = 0
        if not done:
            future_reward = self.gamma * np.amax(self.model.predict(next_state_input)[0])

        # The target is the reward received plus the future discounted reward
        target_reward = reward + future_reward

        # Predict the current reward for all actions
        target = self.model.predict(state_input)

        # Update the rewards for the action taken
        target[0][0] = target_reward  # Assuming a single output node predicting the state value

        # Train the model
        self.model.fit(state_input, target, epochs=1, verbose=0)

        # Update epsilon for the epsilon-greedy strategy
        self.update_epsilon()

    def encode_board(self, board_instance):
        # Correct and simplify encoding function
        encoded_board = np.zeros((8, 8, 1), dtype=np.float32)
        for y, row in enumerate(board_instance.board):
            for x, cell in enumerate(row):
                encoded_board[y, x, 0] = cell if cell is not None else 0
        return encoded_board

    def play_self_game(self):
        board = CheckerBoard()  # This is a CheckerBoard instance
        move_history = []

        while not board.is_terminal():
            current_state = board  # Directly use the board instance instead of encoding it here
            move = self.predict_move(board)  # Continue using the board instance for predictions

            if move:
                board.make_move(move)
                next_state = board  # Similarly, directly use the updated board instance
                reward = 0
                done = board.is_terminal()
                # Instead of encoding states here, store the board instances to be encoded within train
                move_history.append((current_state, move, reward, next_state, done))
            else:
                break  # No valid moves available

        # Process the move history and train
        for current_state, move, reward, next_state, done in move_history:
            # Directly pass CheckerBoard instances to train; train will handle encoding
            self.train(current_state, move, reward, next_state, done)

    def train_from_self_play(self, number_of_games=100):
        for _ in range(number_of_games):
            self.play_self_game()
            # Optionally evaluate AI performance here to track improvements



class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state  # Represents the CheckerBoard state
        self.parent = parent
        self.move = move  # The move that led to this state
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = state.available_moves(state.current_player)  # Dynamically generate available moves

    def select_child(self):
        # This constant is often used in the UCB1 formula to balance exploration and exploitation.
        # You might need to adjust it based on the specifics of your game and how exploration is valued.
        exploration_constant = math.sqrt(2)

        # UCB1 formula for each child: wins / visits + sqrt(2 * log(parent_visits) / visits)
        # Choose the child node that maximizes this value.
        best_score = -float('inf')
        best_child = None

        for child in self.children:
            # Compute the UCB1 score for this child
            ucb1_score = child.wins / child.visits + exploration_constant * math.sqrt(
                math.log(self.visits) / child.visits)

            if ucb1_score > best_score:
                best_score = ucb1_score
                best_child = child

        return best_child

    def add_child(self, move, state):
        # Adds a new child node for the move
        child = MCTSNode(state=state, parent=self, move=move)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result):
        # Update this node - increment the visit count and update the win count based on the result
        self.visits += 1
        self.wins += result


def MCTS(root, iterations):
    for _ in range(iterations):
        node = root
        state = deepcopy(node.state)  # Deep copy to simulate without affecting actual game state

        # Selection
        while node.untried_moves == [] and node.children != []:  # Node is fully expanded and non-terminal
            node = node.select_child()
            state.make_move(node.move)

        # Expansion
        if node.untried_moves and not state.is_terminal():
            move = random.choice(node.untried_moves)
            state.make_move(move)
            new_node = node.add_child(move, deepcopy(state))

            # Simulation from new_node
            while not state.is_terminal():
                available_moves = state.available_moves(state.current_player)
                if available_moves:
                    move = random.choice(available_moves)
                    state.make_move(move)
                else:
                    break

            # Backpropagation from new_node
            result = state.result(new_node.state.current_player if new_node.state else None)
            while new_node is not None:
                new_node.update(result)
                new_node = new_node.parent

        if not root.children:
            # Handle the case where no children are present
            # This might involve logging an error, returning a default move, or raising a more informative error
            return None  # or some default move if applicable

    # After iterations, return the move with the highest win ratio
    return max(root.children, key=lambda c: c.wins / c.visits).move


def simulate_ai_game_with_pygame(ai):
    board = CheckerBoard()
    pg_checker = PygameCheckers(board)
    game_over = False

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True  # Exit the game loop if the window is closed

        if not board.check_winner() and not board.is_terminal():
            pg_checker.update_display()
            move = ai.predict_move(board)  # Adjust this method accordingly
            if move:
                print(f"AI moves from {move[0]} to {move[1]}")
                board.make_move(move)
            else:
                print("No moves available.")

            pygame.time.wait(500)  # Slow down the game for visualization
        else:
            # Game has ended, print the result and wait for the user to close the window
            winner = board.check_winner()
            if winner:
                print(f"Game over. Winner: Player {winner}")
            else:
                print("Game ended in a draw.")

            pg_checker.update_display()  # Make sure the final board position is shown
            pygame.time.wait(1500)  # Wait a moment before entering the wait loop

            # Wait for the user to close the window
            waiting_for_close = True
            while waiting_for_close:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting_for_close = False
                        game_over = True
                    elif event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                        # Exit if any key is pressed or the mouse is clicked
                        waiting_for_close = False
                        game_over = True

    pygame.quit()


if __name__ == '__main__':
    AI = CheckersAI()
    #AI.train_from_self_play(number_of_games=100)
    # Save the model after training
    #AI.model.save('my_checkers_model.h5')

    simulate_ai_game_with_pygame(AI)
    #print("Model saved successfully.")



