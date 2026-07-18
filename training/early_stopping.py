class EarlyStopping:

    def __init__(
        self,
        patience=4,
        min_delta=0.0
    ):

        self.patience = patience

        self.best_score = -1

        self.counter = 0
        self.min_delta = min_delta

    def __call__(
        self,
        score
    ):

        # Consider an improvement only if it exceeds min_delta
        if score > (self.best_score + self.min_delta):

            self.best_score = score

            self.counter = 0

            return False

        self.counter += 1

        return (
            self.counter >= self.patience
        )