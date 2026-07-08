from adp.edr import EDR


class EDR_ADP_single_index(EDR):
    def run_average_derivative(self, beta=None):
        self.Generate_Data()
        beta_k, h_0 = self.run_step_0(beta)
        h_k = h_0

        while self.Step_k_Condition(h_k):
            beta_k, h_k, rho_k = self.run_step_k(beta_k, h_k, beta, h_0)

        return beta_k

    def run_step_0(self, beta=None):
        h_0 = self.H0_Calculate()
        directions = self.Generate_Direction()

        weights = self.Weight_Calculate(h_0)
        local_mean = self.Local_Mean_Calculate(weights)

        I = self.I_Calculate(weights, local_mean, directions)
        U = self.U_Calculate(weights, local_mean, directions)

        beta_0 = self.Alternating_Minimization(I, U)
        beta_0 = self.Beta_Normalize(beta_0)
        self.Characteristics.Step_0_Characteristics_Save(beta, beta_0, h_0)
        return beta_0, h_0

    def run_step_k(self, beta_previous, h_previous, beta=None, h_0=None):
        h_k = self.H_Update(h_previous)
        rho_k = self.Rho_Calculate(beta_previous, h_k)
        directions = self.Generate_Anisotropic_Direction(beta_previous, h_k, rho_k)

        weights = self.Weight_Calculate(h_k, rho_k, beta_previous)
        local_mean = self.Local_Mean_Calculate(weights)

        I = self.I_Calculate(weights, local_mean, directions)
        U = self.U_Calculate(weights, local_mean, directions)

        beta_k = self.Alternating_Minimization(I, U, beta_previous)
        beta_k = self.Beta_Normalize(beta_k)
        self.Characteristics.Step_k_Characteristics_Save(beta, beta_k, rho_k, h_0, h_k)
        return beta_k, h_k, rho_k

    def run(self, beta=None):
        return self.run_average_derivative(beta)
