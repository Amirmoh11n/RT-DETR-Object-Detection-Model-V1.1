class EarlyStopping:

    def __init__(
        self,
        patience=4
    ):

        self.patience = patience

        self.best_score = -1

        self.counter = 0

    def __call__(
        self,
        score
    ):

        if score > self.best_score:

            self.best_score = score

            self.counter = 0

            return False

        self.counter += 1

        return (
            self.counter >= self.patience
        )