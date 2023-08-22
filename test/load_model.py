import torch

import simplifiedMuZero.net2.models2 as models
from games.tictactoe import Game, MuZeroConfig

from game_tournament import load_model

config = MuZeroConfig()

muzero_2net_checkpoint_path = r"C:\Users\chunchang\workspace\muzero-general\results\tictactoe\2023-08-21--22-01-34\muzero_2net\model.checkpoint"
muzero_2net_model = load_model(models.MuZeroNetwork_2net, muzero_2net_checkpoint_path, config)

