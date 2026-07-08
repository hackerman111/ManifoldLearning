from adp.characteristics import ADP_Characteristics
from adp.data import Data


class EDR:
    def __init__(self) -> None:
        self.Data = Data()
        self.Characteristics = ADP_Characteristics()

    pass

    def Mean_Calculate(self):
        pass

    def Local_Mean_Calculate(self, weights):
        pass

    def Kernel_Calculate(self, distance):
        pass

    def Generate_Direction(self):
        pass

    def Generate_Anisotropic_Direction(self, beta_previous, h_k, rho_k):
        pass

    def Weight_Calculate(self, h, rho=None, beta=None):
        pass

    def H0_Calculate(self):
        pass

    def H_Update(self, h_previous):
        pass

    def Step_k_Condition(self, h_k):
        pass

    def Rho_Calculate(self, beta_previous, h_k):
        pass

    def Generate_Data(self):
        pass

    def Beta_Normalize(self, beta):
        pass

    def I_Calculate(self, weights, local_mean, directions):
        pass

    def U_Calculate(self, weights, local_mean, directions):
        pass

    def Average_Derivative_Statistics_Calculate(self, weights, local_mean, directions):
        pass

    def L_Calculate(self, I, U, beta):
        pass

    def Beta_Calculate(self, I, U, l, beta_previous=None):
        pass

    def Objective_Calculate(self, I, U, l, beta):
        pass

    def Alternating_Minimization(self, I, U, beta_initial=None):
        pass
