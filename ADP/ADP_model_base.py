from ADP import ADP_Config, ADP_Data


class BaseADP_model:
    def __init__(self, config: ADP_Config | None = None):
        if config is None:
            config = ADP_Config()
        self.config = config
        self.data = ADP_Data(config)
        self.X, self.noise, self.beta, self.Y = self.data.Initializе_param()
        self.n = config.n
        self.d = config.d
        self.N_loc = config.N_loc
        self.N_lin = config.N_lin
        self.N_J = config.N_J
        self.N_phi = config.N_phi
        self.lam = config.lam
        self.kernel = config.kernel
        self.a = config.a
        self.h_min = config.h_min

    def Model_initialize(self):
        pass

    def Model_step_0(self):
        pass

    def Model_step_k(self):
        pass

    def Model_fit(self):
        pass
