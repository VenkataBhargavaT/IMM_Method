import numpy as np
import math as mt

class ctra_model:
    def __init__(self, SamplingData):
        self.td = SamplingData.td
        self.dpsi_swthr = 0.001
        self.var_ddpsi = 0.0001
        self.var_jerk = 0.001
        self.lk = 0.0
        # the state convention is [x, y, psi, dpsi, vog, aog]
        self.state = np.matrix([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
        self.P = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.01, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])*1
        self.R = np.matrix([[1, 0], [0, 1]])
        self.Rv = np.matrix([[1*0.001, 0], [0, 0.01]])
        self.scaleq = 1.0
        '''
        self.Q = np.matrix([[ts ** 5 / 5.0, 0.0, ts ** 4 / 4.0, 0.0, ts ** 3 / 3.0, 0.0],
                            [0.0, ts ** 5 / 5.0, 0.0, ts ** 4 / 4.0, 0.0, ts ** 3 / 3.0],
                            [ts ** 4 / 4.0, 0.0, ts ** 3 / 3.0, 0.0, ts ** 2 / 2.0, 0.0],
                            [0.0, ts ** 4 / 4.0, 0.0, ts ** 3 / 3.0, 0.0, ts ** 2 / 2.0],
                            [ts ** 3 / 3.0, 0.0, ts ** 2 / 2.0, 0.0, 1 * ts, 0.0],
                            [0.0, ts ** 3 / 3.0, 0.0, ts ** 2 / 2.0, 0.0, 1 * ts]]) * q
        '''


    def get_q(self):
        ts = self.td
        vog = self.state[4, 0]
        psi = self.state[2, 0]
        sin_psi = np.sin(psi)
        cos_psi = np.cos(psi)
        var_ddpsi = self.var_ddpsi
        var_jerk = self.var_jerk
        qk = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

        qk[0, 0] = (var_ddpsi*vog**2*sin_psi**2+var_jerk*cos_psi**2)*ts**5/20
        qk[0, 1] = (var_jerk - var_ddpsi*vog**2)*(sin_psi*cos_psi)*(ts**5/20)
        qk[0, 2] = -var_ddpsi*vog*sin_psi*(ts**4/8.0)
        qk[0, 3] = -var_ddpsi*vog*sin_psi*(ts**3/6.0)
        qk[0, 4] = (var_jerk * cos_psi * (ts ** 4 / 8.0))
        qk[0, 5] = var_jerk * cos_psi * (ts ** 3 / 6.0)
        qk[1, 0] = qk[0, 1]
        qk[1, 1] = (var_ddpsi*vog**2*cos_psi**2+var_jerk*sin_psi**2)*(ts**5/20)
        qk[1, 2] = var_ddpsi*vog*cos_psi*(ts**4/8.0)
        qk[1, 3] = var_ddpsi*vog*cos_psi*(ts**3/6.0)
        qk[1, 4] = var_jerk*sin_psi*(ts**4/8.0)
        qk[1, 5] = var_jerk*sin_psi*(ts**3/6.0)
        qk[2, 0] = qk[0, 2]
        qk[2, 1] = qk[1, 2]
        qk[2, 2] = var_ddpsi*(ts**3/3)
        qk[2, 3] = var_ddpsi*(ts**2/2)
        qk[2, 4] = 0
        qk[2, 5] = 0
        qk[3, 0] = qk[0, 3]
        qk[3, 1] = qk[1, 3]
        qk[3, 2] = qk[2, 3]
        qk[3, 3] = var_ddpsi*ts
        qk[3, 4] = 0
        qk[3, 5] = 0
        qk[4, 0] = qk[0, 4]
        qk[4, 1] = qk[1, 4]
        qk[4, 2] = 0
        qk[4, 3] = 0
        qk[4, 4] = var_jerk*(ts**3/3.0)
        qk[4, 5] = var_jerk*(ts**2/2.0)
        qk[5, 0] = qk[0, 5]
        qk[5, 1] = qk[1, 5]
        qk[5, 2] = qk[2, 5]
        qk[5, 3] = qk[3, 5]
        qk[5, 4] = qk[4, 5]
        qk[5, 5] = var_jerk*ts

        qk *= self.scaleq
        '''
        if np.abs(self.state[3, 0]) >self.dpsi_swthr:
            qk = qk * 10 / ts ** 4
        else:
            qk = qk*10/ts**4
        '''

        return qk

    def pred_state(self, u):
        ts = self.td
        psi = self.state[2, 0]
        dpsi = self.state[3, 0]
        vog = self.state[4, 0]
        aog = self.state[5, 0]
        # dont shuffle the below terms
        self.state[4, 0] += aog*ts
        self.state[2, 0] += dpsi*ts
        vog_u = self.state[4,0]
        psi_u = self.state[2,0]

        if np.abs(dpsi)>self.dpsi_swthr:
            self.state[0, 0] += (vog_u * dpsi * np.sin(psi_u) + aog * np.cos(psi_u) - vog * dpsi * np.sin(
                psi) - aog * np.cos(psi)) * (1.0 / dpsi ** 2)
            self.state[1, 0] += (-vog_u * dpsi * np.cos(psi_u) + aog * np.sin(psi_u) + vog * dpsi * np.cos(
                psi) - aog * np.sin(psi)) * (1.0 / dpsi ** 2)
        else:
            self.state[0, 0] += vog*np.cos(psi)*ts+aog*np.cos(psi)*ts**2/2.0
            self.state[1, 0] += vog*np.sin(psi)*ts+aog*np.sin(psi)*ts**2/2.0

        gam = np.matrix([[0, 0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])

        self.state += np.matmul(gam, u)
        xn = self.state
        return xn

    def ctra_f_jacob(self):
        j_f = np.matrix(np.eye(6, 6))
        dt = self.td
        dpsi = self.state[3, 0]
        psi = self.state[2, 0]
        vog = self.state[4, 0]
        aog = self.state[5, 0]
        if abs(dpsi)>self.dpsi_swthr:
            '''
            j_f[0, 2] = (vog+aog*dt)*(np.cos(psi+dpsi*dt)/dpsi) - vog*(np.cos(psi)/dpsi) - aog*(np.sin(psi+dpsi*dt)/(dpsi**2)) + aog*np.sin(psi)/(dpsi**2)
            j_f[0, 3] = (vog+aog*dt)*((dpsi*dt*np.cos(psi+dpsi*dt)-np.sin(psi+dpsi*dt))/dpsi**2) + vog*(np.sin(psi)/dpsi**2)
            + aog*((-dpsi**2*dt*np.sin(psi+dpsi*dt)-2*dpsi*np.cos(psi+dpsi*dt))/dpsi**3) + aog*np.cos(psi)*2/dpsi**3
            j_f[0, 4] = np.sin(psi+dpsi*dt)/dpsi - np.sin(psi)/dpsi
            j_f[0, 5] = dt*np.sin(psi+dpsi*dt)/dpsi + (np.cos(psi+dpsi*dt) - np.cos(dpsi))/dpsi**2
            j_f[1, 2] = (vog+aog*dt)*(np.sin(psi+dpsi*dt)/dpsi) - vog*np.sin(psi)/dpsi + aog*np.cos(psi+dpsi*dt)/(dpsi**2) - aog*np.cos(psi)/dpsi**2
            j_f[1, 3] = (vog+aog*dt)*((dpsi*dt*np.sin(psi+dpsi*dt)+np.cos(psi+dpsi*dt))/dpsi**2) - vog*np.cos(psi)/dpsi**2
            +  aog*((dpsi**2*dt*np.cos(psi+dpsi*dt)- 2*dpsi*np.sin(psi+dpsi*dt))/dpsi**3) + aog*np.sin(psi)*2/dpsi**3
            j_f[1, 4] = -(np.cos(psi+dpsi*dt)/dpsi) + np.cos(psi)/dpsi
            j_f[1, 5] = -dt*np.cos(psi+dpsi*dt)/dpsi + (np.sin(psi+dpsi*dt)-np.sin(psi))/dpsi**2
            j_f[2, 3] = dt
            j_f[4, 5] = dt
            '''
            j_f[0, 2] = -vog*np.sin(psi)*dt - vog*dpsi*np.cos(psi)*dt**2/2 - aog*np.sin(psi)*dt**2/2.0 - aog*np.cos(psi)*dpsi*dt**3/3.0
            j_f[0, 3] = -vog*np.sin(psi)*dt**2/2.0 - aog*np.sin(psi)*dt**3/3.0
            j_f[0, 4] = np.cos(psi)*dt - dpsi*np.sin(psi)*dt**2/2.0
            j_f[0, 5] = np.cos(psi)*dt**2/2.0 - dpsi*np.sin(psi)*dt**3/3.0
            j_f[1, 2] = vog*np.cos(psi)*dt - vog*dpsi*np.sin(psi)*dt**2/2.0 + aog*np.cos(psi)*dt**2/2.0 - aog*np.sin(psi)*dpsi*dt**3/3.0
            j_f[1, 3] = vog*np.cos(psi)*dt**2/2.0 + aog*np.cos(psi)*dt**3/3.0
            j_f[1, 4] = np.sin(psi)*dt + dpsi*np.cos(psi)*dt**2/2.0
            j_f[1, 5] = np.sin(psi)*dt**2/2.0 + dpsi*np.cos(psi)*dt**3/3.0
            j_f[2, 3] = dt
            j_f[4, 5] = dt

        else:
            j_f = np.matrix([[1.0, 0.0, -vog*dt*np.sin(psi)-aog*np.sin(psi)*dt**2/2.0, -vog*np.sin(psi)*dt**2/2.0-aog*np.sin(psi)*dt**3/3.0, dt*np.cos(psi), np.cos(psi)*dt**2/2.0],
                             [0.0, 1.0, vog*dt*np.cos(psi)+aog*np.cos(psi)*dt**2/2.0,  vog*np.cos(psi)*dt**2/2.0+aog*np.cos(psi)*dt**3/3.0, dt*np.sin(psi), np.sin(psi)*dt**2/2.0],
                             [0.0, 0.0, 1.0, dt, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 1.0, dt], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

        return j_f

    def predict_statecov(self):
        pr_cov_q = self.get_q()
        f_jacob = self.ctra_f_jacob()
        p_u = np.matmul(f_jacob, np.matmul(self.P, np.transpose(f_jacob)))+pr_cov_q
        self.P = p_u
        return p_u

    def kal_update(self, z, H, r):
        x = self.state
        p = self.P
        hph = np.matmul(H, np.matmul(p, np.matrix.transpose(H)))
        s_i = np.linalg.inv(hph + r)
        k = np.matmul(np.matmul(p, np.matrix.transpose(H)), s_i)
        innov = z - np.matmul(H, x)
        x = x + np.matmul(k, innov)
        pu_inter = (np.matrix(np.eye(6, 6)) - np.matmul(k, H))
        pu = np.matmul(pu_inter, p)
        # pu = p - np.matmul(k, np.matmul(hph+r, np.transpose(k)))
        innov_stat = np.matmul(np.transpose(innov), np.matmul(s_i, innov))
        meas_dim = len(z)
        s_det = np.linalg.det(hph + r)
        likelihood = np.exp(-0.5 * innov_stat[0, 0]) / (np.sqrt((2 * mt.pi)**meas_dim * s_det))
        self.lk = likelihood
        self.P = pu
        self.state = x
        return











