

class RHEAIndividual:
    def __init__(self, L:int, discount_factor:double, forword_model, state, play_id:int,
                 seed, heuristic):
        self.state = state
        self.L = L
        self.discount_factor = discount_factor
        self.forword_model = forword_model
        self.play_id = play_id
        self.seed = seed
        self.heuristic = heuristic