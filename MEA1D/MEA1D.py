import matplotlib.pyplot as plt
import numpy as  np
from scipy.integrate import solve_bvp
from MEA1D_conf import *

class MEA:

    def __init__(self):

        self.U = 1.15

        self.ya_lambda = 0
        self.yb_lambda = 0

    def dydx(self, x, y):
        phi_e, j_e, phi_p, j_p, T, j_T, lambda_, j_lambda, x_H2O, j_H2O, x_H2, j_H2, x_O2, j_O2, s, j_s = [y[i_var, :]for i_var in range(16)]
        dphi_e, dj_e, dphi_p, dj_p, dT, dj_T, dlambda, dj_lambda, dx_H2O, dj_H2O, dx_H2, dj_H2, dx_O2, dj_O2, ds, dj_s = [np.zeros(y.shape[1]) for _ in range(16)]

        # Boundary conditions
        j_p[11*2] = 0
        j_p[11*4:] = 0
        x_H2O[22:33] = 0
        x_H2[22:] = 0
        j_H2[22:] = 0
        self.ya_lambda = lambda_[11]
        self.yb_lambda = lambda_[43]

        # AGDL
        C = P_A / (R * T[:11])  # total interstitial gas concentration
        dphi_e[:11] = -j_e[:11] / sigma_e_GDL  # electron flux: j_e = -sigma_e*grad(phi_e)
        dT[:11] = -j_T[:11] / k_GDL  # heat flux: j_T = -k*grad(T)
        dx_H2O[:11] = -j_H2O[:11] / (C * D_H2O_A(eps_p_GDL, tau_GDL, s[:11], T[:11]))  # water vapor flux: j_H2O = -C*D_H2O*grad(x_H2O)
        dx_H2[:11] = -j_H2[:11] / (C * D_H2(eps_p_GDL, tau_GDL, s[:11], T[:11]))  # hydrogen flux: j_H2 = -C*D_H2*grad(x_H2)
        dj_T[:11] = -j_e[:11] * dphi_e[:11]  # conservation of heat: div(j_T) = S_T

        # ACL
        C = P_A / (R * T[11:22])  # total interstitial gas concentration
        x_sat = P_sat(T[11:22]) / P_A  # saturation water vapor mole fraction
        lambda_eq = sorption(x_H2O[11:22] / x_sat)  # equilibrium water content of ionomer
        S_ad = k_ad(lambda_[11:22], lambda_eq, T[11:22]) / (L[2] * V_m) * (lambda_eq - lambda_[11:22])  # absorption/desorption reaction rate
        eta = phi_e[11:22] - phi_p[11:22] + T[11:22] * DeltaS_HOR / (2 * F) + R * T[11:22] / (2 * F) * np.log(x_H2[11:22] * P_A / P_ref)  # overpotential
        i = BV(i_0_HOR(T[11:22]), a_ACL, T[11:22], beta_HOR, eta)  # electrochemical reaction rate
        S_F = i / (2 * F)  # Faraday's law
        dphi_e[11:22] = -j_e[11:22] / sigma_e_CL  # electron flux: j_e = -sigma_e*grad(phi_e)
        dphi_p[11:22] = -j_p[11:22] / sigma_p(eps_i_CL, lambda_[11:22],T[11:22])  # proton flux: j_p = -sigma_p*grad(phi_p)
        dT[11:22] = -j_T[11:22] / k_CL  # heat flux: j_T = -k*grad(T)
        dlambda[11:22] = (-j_lambda[11:22] + xi(lambda_[11:22]) / F * j_p[11:22]) * V_m / D_lambda(eps_i_CL,lambda_[11:22],T[11:22])  # dissolved water flux: j_lambda = -D_lambda/V_m*grad(lambda)+xi/F*j_p
        dx_H2O[11:22] = -j_H2O[11:22] / (C * D_H2O_A(eps_p_CL, tau_CL, s[11:22], T[11:22]))  # water vapor flux: j_H2O = -C*D_H2O*grad(x_H2O)
        dx_H2[11:22] = -j_H2[11:22] / (C * D_H2(eps_p_CL, tau_CL, s[11:22], T[11:22]))  # hydrogen flux: j_H2 = -C*D_H2*grad(x_H2)
        dj_e[11:22] = -i  # conservation of electrons: div(j_e) = S_e
        dj_p[11:22] = i  # conservation of protons: div(j_p) = S_p
        dj_T[11:22] = -j_e[11:22] * dphi_e[11:22] - j_p[11:22] * dphi_p[11:22] + i * eta - S_F * T[11:22] * DeltaS_HOR + H_ad * S_ad  # conservation of heat: div(j_T) = S_T
        dj_lambda[11:22] = S_ad  # conservation of dissolved water: div(j_lambda) = S_lambda
        dj_H2O[11:22] = -S_ad  # conservation of water vapor: div(j_H2O) = S_H2O
        dj_H2[11:22] = -S_F  # conservation of hydrogen: div(j_H2) = S_H2

        # MEM
        dphi_p[22:33] = -j_p[22:33] / sigma_p(1, lambda_[22:33], T[22:33])  # proton flux: j_p = -sigma_p*grad(phi_p)
        dT[22:33] = -j_T[22:33] / k_PEM  # heat flux: j_T = -k*grad(T)
        dlambda[22:33] = (-j_lambda[22:33] + xi(lambda_[22:33]) / F * j_p[22:33]) * V_m / D_lambda(1, lambda_[22:33], T[22:33])  # dissolved water flux: j_lambda = -D_lambda/V_m*grad(lambda)+xi/F*j_p
        dj_T[22:33] = -j_p[22:33] * dphi_p[22:33]  # conservation of heat: div(j_T) = S_T

        # CCL
        C = P_C / (R * T[33:44])  # total interstitial gas concentration
        x_sat = P_sat(T[33:44]) / P_C  # saturation water vapor mole fraction
        S_ec = gamma_ec(x_H2O[33:44], x_sat, s[33:44], T[33:44]) * C * (x_H2O[33:44] - x_sat)  # evaporation/condensation reaction rate
        lambda_eq = sorption(x_H2O[33:44] / x_sat)  # equilibrium water content of ionomer
        S_ad = k_ad(lambda_[33:44], lambda_eq, T[33:44]) / (L[4] * V_m) * (lambda_eq - lambda_[33:44])  # absorption/desorption reaction rate
        eta = -(DeltaH - T[33:44] * DeltaS_ORR) / (2 * F) + R * T[33:44] / (4 * F) * np.log(x_O2[33:44] * P_C / P_ref) - ( phi_e[33:44] - phi_p[33:44])  # overpotential
        i = BV(i_0_ORR(T[33:44], x_O2[33:44]), a_CCL, T[33:44], beta_ORR, eta)  # electrochemical reaction rate
        S_F = i / (2 * F)  # Faraday's law
        dphi_e[33:44] = -j_e[33:44] / sigma_e_CL  # electron flux: j_e = -sigma_e*grad(phi_e)
        dphi_p[33:44] = -j_p[33:44] / sigma_p(eps_i_CL, lambda_[33:44],T[33:44])  # proton flux: j_p = -sigma_p*grad(phi_p)
        dT[33:44] = -j_T[33:44] / k_CL  # heat flux: j_T = -k*grad(T)
        dlambda[33:44] = (-j_lambda[33:44] + xi(lambda_[33:44]) / F * j_p[33:44]) * V_m / D_lambda(eps_i_CL,lambda_[33:44],T[33:44])  # dissolved water flux: j_lambda = -D_lambda/V_m*grad(lambda)+xi/F*j_p
        dx_H2O[33:44] = -j_H2O[33:44] / (C * D_H2O_C(eps_p_CL, tau_CL, s[33:44], T[33:44]))  # water vapor flux: j_H2O = -C*D_H2O*grad(x_H2O)
        dx_O2[33:44] = -j_O2[33:44] / (C * D_O2(eps_p_CL, tau_CL, s[33:44], T[33:44]))  # oxygen flux: j_O2 = -C*D_O2*grad(x_O2)
        ds[33:44] = -j_s[33:44] * V_w / D_s(kappa_CL, s[33:44], T[33:44])  # liquid water flux: j_s = -D_s/V_w*grad(s)
        dj_e[33:44] = i  # conservation of electrons: div(j_e) = S_e
        dj_p[33:44] = -i  # conservation of protons: div(j_p) = S_p
        dj_T[33:44] = -j_e[33:44] * dphi_e[33:44] - j_p[33:44] * dphi_p[33:44] + i * eta - S_F * T[33:44] * DeltaS_ORR + H_ad * S_ad + H_ec * S_ec  # conservation of heat: div(j_T) = S_T
        dj_lambda[33:44] = S_F + S_ad  # conservation of dissolved water: div(j_lambda) = S_lambda
        dj_H2O[33:44] = -S_ec - S_ad  # conservation of water vapor: div(j_H2O) = S_H2O
        dj_O2[33:44] = -S_F / 2  # conservation of oxygen: div(j_O2) = S_O2
        dj_s[33:44] = S_ec  # conservation of liquid water: div(j_s) = S_s

        # CGDL
        C = P_C / (R * T[44:55])  # total interstitial gas concentration
        x_sat = P_sat(T[44:55]) / P_C  # saturation water vapor mole fraction
        S_ec = gamma_ec(x_H2O[44:55], x_sat, s[44:55], T[44:55]) * C * (x_H2O[44:55] - x_sat)  # evaporation/condensation reaction rate
        dphi_e[44:55] = -j_e[44:55] / sigma_e_GDL  # electron flux: j_e = -sigma_e*grad(phi_e)
        dT[44:55] = -j_T[44:55] / k_GDL  # heat flux: j_T = -k*grad(T)
        dx_H2O[44:55] = -j_H2O[44:55] / (C * D_H2O_C(eps_p_GDL, tau_GDL, s[44:55], T[44:55]))  # water vapor flux: j_H2O = -C*D_H2O*grad(x_H2O)
        dx_O2[44:55] = -j_O2[44:55] / (C * D_O2(eps_p_GDL, tau_GDL, s[44:55], T[44:55]))  # oxygen flux: j_O2 = -C*D_O2*grad(x_O2)
        ds[44:55] = -j_s[44:55] * V_w / D_s(kappa_GDL, s[44:55], T[44:55])  # liquid water flux: j_s = -D_s/V_w*grad(s)
        dj_T[44:55] = -j_e[44:55] * dphi_e[44:55] + H_ec * S_ec  # conservation of heat: div(j_T) = S_T
        dj_H2O[44:55] = -S_ec  # conservation of water vapor: div(j_H2O) = S_H2O
        dj_s[44:55] = S_ec  # conservation of liquid water: div(j_s) = S_s

        return [dphi_e, dj_e,
                dphi_p, dj_p,
                dT, dj_T,
                dlambda, dj_lambda,
                dx_H2O, dj_H2O,
                dx_H2, dj_H2,
                dx_O2, dj_O2,
                ds, dj_s]

    def y_init(self, x):
        y = np.zeros((16, len(x)))
        for n in range(5):
            for i in range(11):
                if n == 0:
                    y[:, n * 11 + i] = self.y_init_agdl(x)
                    if i > 0:
                        dy = 1.6e-5 * np.array(self.dydx(x, y))[:, n * 11 + i - 1]
                        y[:, n * 11 + i] = y[:, n * 11 + i] + dy
                elif n == 1:
                    y[:, n * 11 + i] = self.y_init_acl(x)
                    if i > 0:
                        dy = 1.0e-6 * np.array(self.dydx(x, y))[:, n * 11 + i - 1]
                        y[:, n * 11 + i] = y[:, n * 11 + i] + dy
                elif n == 2:
                    y[:, n * 11 + i] = self.y_init_mem(x)
                    if i > 0:
                        dy = 2.5e-6 * np.array(self.dydx(x, y))[:, n * 11 + i - 1]
                        y[:, n * 11 + i] = y[:, n * 11 + i] + dy
                elif n == 3:
                    y[:, n * 11 + i] = self.y_init_ccl(x)
                    if i > 0:
                        dy = 1.0e-6 * np.array(self.dydx(x, y))[:, n * 11 + i - 1]
                        y[:, n * 11 + i] = y[:, n * 11 + i] + dy
                else:
                    y[:, n * 11 + i] = self.y_init_cgdl(x)
                    if i > 0:
                        dy = 1.6e-5 * np.array(self.dydx(x, y))[:, n * 11 + i - 1]
                        y[:, n * 11 + i] = y[:, n * 11 + i] + dy
        return y

    def bcfun(self,ya, yb):
        res = np.zeros(len(ya))
        Neq = int(ya.shape[0] / 2)  # number of second order derivative

        # ELECTRONS
        res[0] = ya[0]
        res[1] = yb[0] - self.U

        # PROTONS
        res[2] = ya[3]
        res[3] = yb[3]

        # TEMPERATURE
        res[4] = ya[4] - T_A
        res[5] = yb[4] - T_C

        # DISSOLVED WATER
        res[6] = ya[6] - self.ya_lambda
        res[7] = yb[6] - self.yb_lambda

        # WATER VAPOR
        res[8] = ya[8] - x_H2O_A
        res[9] = yb[8] - x_H2O_C

        # HYDROGEN
        res[10] = ya[10] - x_H2_A
        res[11] = yb[10]

        # OXYGEN
        res[12] = ya[12]
        res[13] = yb[12] - x_O2_C

        # LIQUID WATER
        res[14] = ya[15]
        res[15] = yb[15]

        return np.array(res)

    def y_init_agdl(self, x):
        T = (T_C + T_A) / 2
        x_H2O = x_H2O_A
        x_H2 = x_H2_A
        return np.array([0, 0, 0, 0, T, 0, 0, 0, x_H2O, 0, x_H2, 0, 0, 0, 0, 0])

    def y_init_acl(self,x):
        lambda_ = sorption(0.9)
        T = (T_C + T_A) / 2
        x_H2O = x_H2O_A
        x_H2 = x_H2_A
        return np.array([0, 0, 0, 0, T, 0, lambda_, 0, x_H2O, 0, x_H2, 0, 0, 0, 0, 0])

    def y_init_mem(self,x):
        phi_e = 0
        phi_p = 0
        T = (T_C + T_A) / 2
        lambda_ = sorption(0.7)
        x_H2O = 0
        x_H2 = 0
        x_O2 = 0
        s = 0

        return [phi_e, 0, phi_p, 0, T, 0, lambda_, 0, x_H2O, 0, x_H2, 0, x_O2, 0, s, 0]

    def y_init_ccl(self,x):
        phi_e = self.U
        phi_p = -self.U
        T = (T_C + T_A) / 2
        lambda_ = sorption(0.9)
        x_H2O = x_H2O_C
        x_H2 = 0
        x_O2 = x_O2_C
        s = s_C
        return [phi_e, 0, phi_p, 0, T, 0, lambda_, 0, x_H2O, 0, x_H2, 0, x_O2, 0, s, 0]

    def y_init_cgdl(self,x):
        phi_e = self.U
        phi_p = 0
        T = (T_C + T_A) / 2
        lambda_ = 0
        x_H2O = x_H2O_C
        x_H2 = 0
        x_O2 = x_O2_C
        s = s_C

        return [phi_e, 0, phi_p, 0, T, 0, lambda_, 0, x_H2O, 0, x_H2, 0, x_O2, 0, s, 0]


if __name__ == '__main__':

    solutions = []

    for Ustack in U:
        model = MEA()
        model.U = Ustack
        x_init = np.hstack([np.linspace(0,1.60e-4,12)[:11],
                            np.linspace(1.60e-4,1.70e-4,12)[:11],
                            np.linspace(1.70e-4,1.95e-4,12)[:11],
                            np.linspace(1.95e-4,2.05e-4,12)[:11],
                            np.linspace(2.05e-4,3.65e-4,12)[:11]])
        y0 = model.y_init(x_init)
        sol = solve_bvp(fun=model.dydx,bc=model.bcfun,x=x_init,y=y0,verbose=2)
        solutions.append(sol)

    fig, ax = plt.subplots(nrows=4,ncols=2,figsize=(14,12))
    titles = [r'$\phi_e$',r'$\phi_p$',r'$T$',r'$\lambda$',r'$x_{H_2O}$',r'$x_{H_2}$',r'$x_{O_2}$',r'$s$']
    i_sol = 0
    for sol in solutions:
        for i_var in range(8):
            ax[i_var//2,i_var%2].plot(x_init,sol.y[i_var*2,:55],label = f"U = {U[i_sol]:.2f}")
            ax[i_var//2,i_var%2].set_title(titles[i_var])
            ax[i_var//2,i_var%2].axvline(x = x_init[11], linestyle='--',color='#DE2341')
            ax[i_var//2,i_var%2].axvline(x = x_init[44], linestyle='--',color='#DE2341')
            ax[i_var//2,i_var%2].axvline(x = x_init[33], linestyle='--',color='#A063D3')
            ax[i_var//2,i_var%2].axvline(x = x_init[22], linestyle='--',color='#A063D3')
        i_sol += 1

    handles, labels = ax[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # 为 legend 留出下方空间

    # fig, ax = plt.subplots(nrows=4,ncols=2,figsize=(14,12))
    # for sol in solutions:
    #     for i_var in range(8):
    #         ax[i_var//2,i_var%2].plot(x_init,sol.y[i_var*2-1,:55])
    #         ax[i_var//2,i_var%2].set_title(titles[i_var])
    #         ax[i_var//2,i_var%2].axvline(x = 1.60e-4, linestyle='--',color='#DE2341')
    #         ax[i_var//2,i_var%2].axvline(x = 2.05e-4, linestyle='--',color='#DE2341')
    #         ax[i_var//2,i_var%2].axvline(x = 1.70e-4, linestyle='--',color='#A063D3')
    #         ax[i_var//2,i_var%2].axvline(x = 1.95e-4, linestyle='--',color='#A063D3')
    #
    # handles, labels = ax[0][0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.01))
    # plt.tight_layout(rect=[0, 0.1, 1, 1])  # 为 legend 留出下方空间

    plt.show()