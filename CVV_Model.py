import numpy as np
import math as mt

class cvcv_model:

    def __init__(self, sampdata):
        self.sig_r = 0.1
        self.sig_tita = 3.0 * (mt.pi / 180.0)
        self.sig_x = 0.1
        self.sig_y = 0.01
        self.td = sampdata.td
        self.lk = 0.0
        ts = sampdata.td
        q = 0.1 #1/ts
        # the state convention is [x, y, vx, vy]
        self.state = np.matrix([[1.0], [0.0], [10.0], [0.0]])
        self.P = np.matrix([[10.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 10.0, 0.0], [0.0, 0.0, 0.0, 10.0]])
        self.R = np.matrix([[0.1, 0], [0, 0.01]])
        self.Rv = np.matrix([[1, 0], [0, 1]]) * 0.001
        self.Q = np.matrix([[ts ** 3 / 3.0, 0.0, ts ** 2 / 2.0, 0.0], [0.0, ts ** 3 / 3.0, 0.0, ts ** 2 / 2.0],
                            [ts ** 2 / 2.0, 0.0, 1 * ts, 0.0],
                            [0.0, ts ** 2 / 2.0, 0.0, 1 * ts]]) * q

    # descrete state constant velocity model
    def pred_state(self, u):
        a1 = u[0, 0]
        a2 = u[1, 0]
        dt = self.td
        # B = np.matrix([[dt, 0], [0.0, dt], [0.0, 0.0], [0.0, 0.0]])
        B = np.matrix([[dt ** 2 / 2.0, 0], [0.0, dt ** 2 / 2.0], [dt, 0.0], [0.0, dt]])
        xn = np.matmul(self.f_jacob_cvcv(), self.state) + np.matmul(B, u)
        self.state = xn
        return xn

    def f_jacob_cvcv(self):
        dt = self.td
        return np.matrix([[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    def predict_statecov(self):
        dt = self.td
        p_u = self.f_jacob_cvcv() * self.P * np.transpose(self.f_jacob_cvcv()) + self.Q
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
        pu_inter = (np.matrix(np.eye(4, 4)) - np.matmul(k, H))
        pu = np.matmul(pu_inter, p)
        # pu = p - np.matmul(k, np.matmul(hph+r, np.transpose(k)))
        innov_stat = np.matmul(np.transpose(innov), np.matmul(s_i, innov))
        likelihood = np.exp(-0.5*innov_stat[0, 0])/np.sqrt(2*mt.pi*np.linalg.det(hph+r))
        self.lk = likelihood
        self.P = pu
        self.state = x
        return

    def kal_nonlin_update(self, innov, H, r):
        x = self.state
        p = self.P
        hph = np.matmul(H, np.matmul(p, np.matrix.transpose(H)))
        s_i = np.linalg.inv(hph + r)
        k = np.matmul(np.matmul(p, np.matrix.transpose(H)), s_i)
        x = x + np.matmul(k, innov)
        pu_inter = (np.matrix(np.eye(4, 4)) - np.matmul(k, H))
        pu = np.matmul(pu_inter, p)
        # pu = p - np.matmul(k, np.matmul(hph+r, np.transpose(k)))
        self.P = pu
        self.state = x
        return x
