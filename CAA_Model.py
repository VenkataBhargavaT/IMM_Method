import numpy as np
import math as mt


class caa_model:

    def __init__(self, sampdata):
        self.td = sampdata.td
        ts = self.td
        self.lk = 0.0
        sx = 1
        sy = 1
        # the state convention is [x, y, vx, vy, ax, ay]
        self.state = np.matrix([[1.0], [0.0], [10.0], [0.0], [0.0], [0.0]])
        self.P = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])*1
        self.R = np.matrix([[1, 0], [0, 1]])
        self.Rv = np.matrix([[1, 0], [0, 1]]) * 0.001
        self.Q = np.matrix([[sx * ts ** 5 / 5.0, 0.0, sx * ts ** 4 / 4.0, 0.0, sx * ts ** 3 / 3.0, 0.0],
                            [0.0, sy * ts ** 5 / 5.0, 0.0, sy * ts ** 4 / 4.0, 0.0, sy * ts ** 3 / 3.0],
                            [sx * ts ** 4 / 4.0, 0.0, sx * ts ** 3 / 3.0, 0.0, sx * ts ** 2 / 2.0, 0.0],
                            [0.0, sy * ts ** 4 / 4.0, 0.0, sy * ts ** 3 / 3.0, 0.0, sy * ts ** 2 / 2.0],
                            [sx * ts ** 3 / 3.0, 0.0, sx * ts ** 2 / 2.0, 0.0, sx * 1 * ts, 0.0],
                            [0.0, sy * ts ** 3 / 3.0, 0.0, sy * ts ** 2 / 2.0, 0.0, sy * 1 * ts]])
    def scaleq(self, sx, sy):
        ts = self.td
        self.Q = np.matrix([[sx * ts ** 5 / 5.0, 0.0, sx * ts ** 4 / 4.0, 0.0, sx * ts ** 3 / 3.0, 0.0],
                            [0.0, sy * ts ** 5 / 5.0, 0.0, sy * ts ** 4 / 4.0, 0.0, sy * ts ** 3 / 3.0],
                            [sx * ts ** 4 / 4.0, 0.0, sx * ts ** 3 / 3.0, 0.0, sx * ts ** 2 / 2.0, 0.0],
                            [0.0, sy * ts ** 4 / 4.0, 0.0, sy * ts ** 3 / 3.0, 0.0, sy * ts ** 2 / 2.0],
                            [sx * ts ** 3 / 3.0, 0.0, sx * ts ** 2 / 2.0, 0.0, sx * 1 * ts, 0.0],
                            [0.0, sy * ts ** 3 / 3.0, 0.0, sy * ts ** 2 / 2.0, 0.0, sy * 1 * ts]])
        return


    def pred_state(self, u):
        # a1 = u[0, 0]
        # a2 = u[1, 0]
        dt = self.td
        # B = np.matrix([[dt, 0], [0.0, dt], [0.0, 0.0], [0.0, 0.0]])
        B = np.matrix([[dt ** 2 / 2.0, 0], [0.0, dt ** 2 / 2.0], [dt, 0.0], [0.0, dt], [1.0, 0.0], [0.0, 1.0]])
        xn = np.matmul(self.caa_f_jacob_caa(), self.state) + np.matmul(B, u)
        self.state = xn
        return xn

    def caa_f_jacob_caa(self):
        dt = self.td
        return np.matrix([[1.0, 0.0, dt, 0.0, dt ** 2 / 2.0, 0.0], [0.0, 1.0, 0.0, dt, 0.0, dt ** 2 / 2],
                          [0.0, 0.0, 1.0, 0.0, dt, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                          [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    def predict_statecov(self):
        dt = self.td
        f_jacob = self.caa_f_jacob_caa()
        p_u =  np.matmul(f_jacob, np.matmul(self.P, np.transpose(f_jacob))) + self.Q
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
        if np.linalg.det(hph+r) < 1e-9:
            print("det of hph_r is small",np.linalg.det(hph+r))
            print("det of hph_r is: ", np.linalg.det(hph ))
            print("det of r is small", np.linalg.det( r))
            likelihood = np.exp(-0.5 * innov_stat[0, 0]) / np.sqrt(2 * mt.pi * 0.0001)
        else:
            meas_dim = len(z)
            s_det = np.linalg.det(hph + r)
            likelihood = np.exp(-0.5 * innov_stat[0, 0]) / (np.sqrt((2 * mt.pi)**meas_dim * s_det))
        self.lk = likelihood
        self.P = pu
        self.state = x
        return

