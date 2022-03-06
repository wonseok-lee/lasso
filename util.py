class Record:
    def __init__(self):
        self.obj = []
        self.loss = []
        self.time = []
        self.beta = []

    def add_obj(self, val):
        self.obj.append(val)

    def add_loss(self, val):
        self.loss.append(val)

    def add_time(self, val):
        self.time.append(val)

    def add_beta(self, val):
        self.beta.append(val)

    def get_time(self):
        return self.time[0]

