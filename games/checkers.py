import datetime
import math
import os

import gym
import numpy
import torch

from .abstract_game import AbstractGame

class MuZeroConfig:
    def __init__(self):
         # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game



        ### Game
        self.observation_shape = (4, 8, 8)  # Dimensions of the game observation, must be 3 (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = [i for i in range(8 * 8)]  # Fixed list of all possible actions. You should only edit the length
        self.players = [i for i in range(2)]  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "random"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_actors = 2  # Number of simultaneous threads self-playing to feed the replay buffer
        self.max_moves = 121  # Maximum number of moves if game is not finished before
        self.num_simulations = 400  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size
        
        # Residual Network
        self.downsample = False  # Downsample observations before representation network (See paper appendix Network Architecture)
        self.blocks = 6  # Number of blocks in the ResNet
        self.channels = 128  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 4  # Number of channels in policy head
        self.resnet_fc_reward_layers = [64]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [64]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [64]  # Define the hidden layers in the policy head of the prediction network
        
        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network



        ### Training
        self.results_path = os.path.join(os.path.dirname(__file__), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.training_steps = 10000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 512  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 50  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 1  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.training_device = "cuda" if torch.cuda.is_available() else "cpu"  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.002  # Initial learning rate
        self.lr_decay_rate = 0.9  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000



        ### Replay Buffer
        self.window_size = 10000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 121  # Number of game moves to keep for every batch element
        self.td_steps = 121  # Number of steps in the future to take into account for calculating the target value
        self.use_last_model_value = False  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)

        # Prioritized Replay (See paper appendix Training)
        self.PER = True  # Select in priority the elements in the replay buffer which are unexpected for the network
        self.use_max_priority = False  # If False, use the n-step TD error as initial priority. Better for large replay buffer
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1
        self.PER_beta = 1.0



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = 8  # Desired self played games per training step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
    def visit_softmax_temperature_fn(self, trained_steps):
            """
            Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
            The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.
            Returns:
                Positive float.
            """
            if trained_steps < 0.5 * self.training_steps:
                return 1.0
            elif trained_steps < 0.75 * self.training_steps:
                return 0.5
            else:
                return 0.25


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Checkers()

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        
        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def close(self):
        """
        Properly close the game.
        """
        pass

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        valid = False
        while not valid:
            valid, action = self.env.human_input_to_action()
        return action
    
    def action_to_string(self, action):
        """
        Convert an action number to a string representing the action.
        Args:
            action_number: an integer from the action space.
        Returns:
            String representing the action.
        """
        return self.env.action_to_human_input(action)


class Checkers:

    DIRECTIONS = {'nw':(1,1),'ne':(-1,1),'sw':(1,-1),'se':(-1,-1)}
    def __init__(self):
        self.board_size = 8
        self.board = numpy.zeros((self.board_size, self.board_size)).astype(int)
        self.player = 1
        self.is_jump =False
        self.board_markers = [
            chr(x) for x in range(ord("A"), ord("A") + self.board_size)
        ]

    def to_play(self):
        if self.is_jump:
            return self.player
        else:
            return 0 if self.player ==1 else 1
       

    def reset(self):
        self.board = numpy.zeros((self.board_size, self.board_size)).astype(int)
        self.is_jump = False
        self.player = 1
        return self.get_observation()

    def step(self, action):
        x0 = int(action[0])
        y0 = int(action[1])
        x1 = int(action[2])
        y1 = int(action[3])
        self.is_jump = bool(action[4])
        self.board[x1][y1] = self.board[x0][y0]
        self.board[x0][y0]= 0
        if (y1 == 7 and self.board[x1][y1] == 1) or (y1 == 0 and self.board[x1][y1] == -1):
            self.board[x1][y1] *= 2
        if self.is_jump:
            x_space = (x1-x0)/2
            y_space = (y1-y0)/2
            self.board[x0+x_space][y0+y_space] = 0
            self.is_jump = can_jump(x1,y1)
        
        done = self.is_finished()

        reward = 1 if done else 0
        if not self.is_jump:
            self.player *= -1
        

        return self.get_observation(), reward, done

    def can_jump(self,x,y):
        if self.player == np.sign(curr_piece) and (np.abs(curr_piece)>1 or self.player == dy): #checks if selected piece is right color
            for d in DIRECTIONS:
                dx = DIRECTIONS[d][0] 
                dy = DIRECTIONS[d][1] 
                if  last_jumped_x+2*dx<8 and 
                    last_jumped_x+2*dx>0 and 
                    last_jumped_y+2*dy<8 and 
                    last_jumped_y+2*dy>0 and 
                    (np.sign(player) == -1*np.sign(self.board[x+dx][y+dy] and 
                    self.board[last_jumped_x+2*dx][last_jumped_y+2*dy])==0):
                    return True
        return False

    def get_observation(self):
        board_player1 = numpy.where(self.board == 1, 1.0, 0.0)
        board_player2 = numpy.where(self.board == -1, 1.0, 0.0)
        board_to_play = numpy.full((8, 8), self.player).astype(float)
        board_jump = numpy.full((8, 8), self.is_jump).astype(float)
        return numpy.array([board_player1, board_player2, board_to_play,board_jump])

    def legal_actions(self, last_jumped_x, last_jumped_y):
        legal = []
        curr_piece = self.board[last_jumped_y][last_jumped_x]
        
        if self.is_jump:
            #this condition is only true if the last move was a jump
                #==> should be the same player's turn
            curr_loc = str(last_jumped_x*10+last_jumped_y)
            for d in DIRECTIONS:
                dx = DIRECTIONS[d][0]
                dy = DIRECTIONS[d][1]
                if player == np.sign(curr_piece) and (np.abs(curr_piece)>1 or player == dy): #checks if selected piece is right color
                    
                    if  last_jumped_x+2*dx<8 and 
                        last_jumped_x+2*dx>0 and 
                        last_jumped_y+2*dy<8 and 
                        last_jumped_y+2*dy>0 and 
                        (np.sign(player) == -1*np.sign(self.board[last_jumped_x+dx][last_jumped_y+dy] and 
                        self.board[last_jumped_x+2*dx][last_jumped_y+2*dy])==0):
                        legal.append(curr_loc + str(last_jumped_x+2*dx) +str(last_jumped_y+2*dy)+'1') 
                        
            return legal
        else:
            return legal_directions(self.player)
            
    def legal_directions(self,player)
        legal = []
       
        for x in range(self.board_size): #x
            for y in range(self.board_size): # y
                curr_piece = self.board[x][y]
                curr_loc = str(10*x+y)
                for d in DIRECTIONS:
                    dx = DIRECTIONS[d][0]
                    dy = DIRECTIONS[d][1]
                    if player == np.sign(curr_piece) and (np.abs(curr_piece)>1 or player == dy): #checks if selected piece is right color
                        
                        if  x+2*dx<8 and 
                            x+2*dx>0 and 
                            y+2*dy<8 and 
                            y+2*dy>0 and 
                            (np.sign(player) == -1*np.sign(self.board[x+dx][y+dy] and 
                            self.board[x+2*dx][y+2*dy])==0):
                            if not self.is_jump:    
                                legal = [] #empty legal of previous moves since jumps must be made if possible
                            legal.append(curr_loc + str(x+2*dx) +str(y+2*dy)+'1') 
                            
                            self.is_jump = True
                    
                    if not self.is_jump and
                            x+dx<8 and 
                            x+dx>0 and 
                            y+dy<8 and 
                            y+dy>0 and  
                            self.board[x+dx][y+dy] == 0 :

                            legal.append(curr_loc + str(x+dx) +str(y+dy)+'0')
                    
                   
        return legal
    def is_finished(self):
        has_legal_actions = False
        
        has_red =False
        has_black = False
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.board[x][y]<0:
                    has_black=True
                    has_legal_actions = len(legal_actions(-1))>0
                    self.is_jump = self.player == -1 and self.is_jump
                elif self.board[x][y]>0:
                    has_red = True
                    has_legal_actions = len(legal_actions(1))>0
                    self.is_jump = self.player == 1 and self.is_jump
        
        return has_black and has_red and not has_legal_actions

    def render(self):
        marker = "  "
        for i in range(self.board_size):
            marker = marker + self.board_markers[i] + " "
        print(marker)
        for row in range(self.board_size):
            print(chr(ord("A") + row), end=" ")
            for col in range(self.board_size):
                ch = self.board[row][col]
                if ch == 0:
                    print(".", end=" ")
                elif ch == 1:
                    print("X", end=" ")
                elif ch == -1:
                    print("O", end=" ")
                elif ch == 2:
                    print("♔", end=" ")
                elif ch == -2:
                    print("♚", end=" ")
            print()

    def human_input_to_action(self):
        human_input = input("Enter an action: ")
        if (
            len(human_input) == 4
            and human_input[0] in self.board_markers
            and human_input[1] in self.board_markers
            and human_input[2] in self.board_markers
            and human_input[3] in self.board_markers
        ):
            x0 = ord(human_input[0]) - 65
            y0 = ord(human_input[1]) - 65
            x1 = ord(human_input[2]) - 65
            y1 = ord(human_input[3]) - 65
            jump = (x1+1-x0)%2 # if it is jump should be 1 else 0
            
            if  np.sign(self.board[x0][y0]) == self.player
                and x1 <8 and x1>=0 and y1 <8 and y1>=0 
                and self.board[x1][y1] == 0:
                if self.is_jump and (y1-y0)%2==0 and (x1-x0)%2==0:
                    return True, "{}{}{}{}{}".format(x0,y0,x1,y1,jump)
                elif not self.is_jump and (y1-y0)%2==1 and (x1-x0)%2==1:
                    return True, "{}{}{}{}{}".format(x0,y0,x1,y1,jump)
        return False, -1

    def action_to_human_input(self, action):
        x0 = chr(65+int(action[0]))
        y0 = chr(65+int(action[1]))
        x1 = chr(65+int(action[2]))
        y1 = chr(65+int(action[3]))
        
        return "{}{}{}{}".format(x0,y0,x1,y1)

