from games.simple_grid import MuZeroConfig

if __name__ == "__main__":
    config = MuZeroConfig()
    config.results_path /= "config_test"
    print(config.results_path)