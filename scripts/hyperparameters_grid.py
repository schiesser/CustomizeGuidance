HYPERPARAMETER_GRID = {
    "constant": {},

    "linear": {},

    "exponential": {},

    "APG": {
        "momentum_value": [0.0],
        "eta": [-1.0],
        "norm_threshold": [10.0, 15.0, 20.0],
    },

    "zero_star": {
        "zero_steps": [0, 1, 2, 4],
        "use_zero_init": [False, True],
    },

    "rectified_pp": {
        "lambda_max": [1.1, 1.3, 1.5],
        "gamma": [1.5, 2.0, 3.0],
    },

    "SMC": {
        "lambda_param": [1.0, 1.5, 2.0],
        "k": [1.0, 2.0, 3.0],
    },
}