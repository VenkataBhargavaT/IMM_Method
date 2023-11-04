import numpy as np
import math as mt
from CAA_Model import caa_model
from CTRA_Model import ctra_model
from CVV_Model import cvcv_model

# from ekf_auto import SamplingData
class samplingdata:
    '''
    Define the start time, delta time cycle and end time.
    '''
    ts = 0.
    td = 0.1
    tf = 20.

def generate_truth_data(data, xki):
    sampletime = np.arange(data.ts, data.tf, data.td)
    count = 0
    xk = xki
    st = np.zeros([4, len(sampletime)])
    inp_u = np.zeros([2, len(sampletime)])
    tita = 0.0
    for t in sampletime:
        if t <= 0.0:  # 4.0
            v = 10.0
            deltita = 0.0
            tita = tita + deltita
        elif (t > 2.0) and (t <= 4.0):  # 6.0
            v = 10.0
            deltita = ((90.0 - 0) / 20.0) * mt.pi / 180.0 * 0
            # v = 10.0
            # deltita = 0.0
            tita = np.min([tita + deltita, 90 * mt.pi / 180.0])
        elif (t > 7.0) and (t <= 9.0):
            v = 10.0
            deltita = ((0.0 - 90.0) / 20.0) * mt.pi / 180.0 * 0
            tita = np.max([tita + deltita, 0 * mt.pi / 180.0])
        else:
            v = 10.0
            deltita = 0.0

            tita = np.max([tita + deltita, 0 * mt.pi / 180.0])

        vx = v * mt.cos(tita)
        vy = v * mt.sin(tita)
        # u = np.matrix([[0], [0]])
        # xkp = Desc_State(xk, u, data.td)

        xkp = np.matrix([[xk[0, 0] + vx * data.td], [xk[1, 0] + vy * data.td], [vx], [vy]])
        xk = xkp

        st[:, count] = np.array([xkp[0, 0], xkp[1, 0], xkp[2, 0], xkp[3, 0]])
        inp_u[:, count] = np.array([vx * 0, vy * 0])
        count += 1
    print("count value:", count)

    return inp_u, st


# descrete state constant velocity model
def Desc_State(X, u, T):
    # B = np.matrix([[T, 0], [0, T], [0, 0], [0, 0]])
    B = np.matrix([[T ** 2 / 2.0, 0], [0, T ** 2 / 2.0], [T, 0], [0, T]])
    Xn1 = np.matmul(F(T), X) + np.matmul(B, u)
    return Xn1


def F(dt):
    return np.matrix([[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def ground_truth_data(data, xki):
    sampletime = np.arange(data.ts, data.tf, data.td)
    count = 0
    st = np.zeros([4, len(sampletime)])
    caa = caa_model(data)
    cvv = cvcv_model(data)
    ctra = ctra_model(data)

    caa.state = np.matrix([[xki[0, 0]], [xki[1, 0]], [xki[2, 0]], [xki[3, 0]], [0.0], [0.0]])
    cvv.state = np.matrix([[xki[0, 0]], [xki[1, 0]], [xki[2, 0]], [xki[3, 0]]])
    u = np.matrix([[0], [0]])
    for t in sampletime:
        if t <= 0.0:  # 4.0
            a = 0
            caa.state[4, 0] = a
            caa.state[5, 0] = 0
            xkp = caa.pred_state(u)
        elif (t > 2.0) and (t < 5.0):  # 6.0
            a = 4.0
            caa.state[4, 0] = a
            caa.state[5, 0] = 0
            xkp = caa.pred_state(u)
            '''
            elif t==4.0:
                cvv.state = np.matrix([[caa.state[0, 0]], [caa.state[1, 0]], [caa.state[2, 0]], [caa.state[3, 0] ]])
                xkp = cvv.pred_state(u)
            elif (t > 4.0) and (t < 6.0):
                xkp = cvv.pred_state(u)
            elif t==6.0:
                cvv.state[3, 0] = 0.0
                caa.state = np.matrix([[cvv.state[0, 0]], [cvv.state[1, 0]], [cvv.state[2, 0]], [cvv.state[3, 0]], [0.0],
                                       [0.0]])
                xkp = caa.pred_state(u)
            elif (t > 7.0) and (t <= 9.0):
                a = -4.0
                caa.state[4, 0] = a
                xkp = caa.pred_state(u)
            '''
        elif t == 5.0:
            dpsi = 10.0*np.pi/180
            vog = np.sqrt(caa.state[2, 0]**2+caa.state[3, 0]**2)
            ctra.state = np.matrix([[caa.state[0, 0]], [caa.state[1, 0]], [0], [dpsi], [vog], [0]])
            ctra.pred_state(u)
            xkp = np.matrix([[ctra.state[0, 0]], [ctra.state[1, 0]], [ctra.state[4, 0]*np.cos(ctra.state[2, 0])],
                             [ctra.state[4, 0]*np.sin(ctra.state[2, 0])]])
        elif (t > 5.0) and (t < 6.0):
            ctra.pred_state(u)
            xkp = np.matrix([[ctra.state[0, 0]], [ctra.state[1, 0]], [ctra.state[4, 0] * np.cos(ctra.state[2, 0])],
                             [ctra.state[4, 0] * np.sin(ctra.state[2, 0])]])
        elif t == 6.0:
            dpsi = -10.0 * np.pi / 180
            ctra.state[3, 0] = dpsi
            ctra.pred_state(u)
            xkp = np.matrix([[ctra.state[0, 0]], [ctra.state[1, 0]], [ctra.state[4, 0] * np.cos(ctra.state[2, 0])],
                             [ctra.state[4, 0] * np.sin(ctra.state[2, 0])]])
        elif (t > 6.0) and (t < 7.0):
            ctra.pred_state(u)
            xkp = np.matrix([[ctra.state[0, 0]], [ctra.state[1, 0]], [ctra.state[4, 0] * np.cos(ctra.state[2, 0])],
                             [ctra.state[4, 0] * np.sin(ctra.state[2, 0])]])
        elif t == 7.0:
            vog = ctra.state[4, 0]
            psi = ctra.state[2, 0]
            caa.state = np.matrix([[ctra.state[0, 0]], [ctra.state[1, 0]], [vog*np.cos(psi)], [vog*np.sin(psi)], [0.0], [0.0]])
            xkp = caa.pred_state(u)

        else:
            a = 0
            caa.state[4, 0] = a
            caa.state[5, 0] = 0
            xkp = caa.pred_state(u)

        # desc_state_ctrv(xk, u, data.td)
        #xk = xkp
        st[:, count] = np.array([xkp[0, 0], xkp[1, 0], xkp[2, 0], xkp[3, 0]])

        count += 1
    print("count value:", count)

    return st


# descrete state ctrv model
def desc_state_ctrv(X, u, T):
    w = X[4, 0]
    gam = np.matrix([[T ** 2 / 2.0, 0, 0], [0, T ** 2 / 2.0, 0], [T, 0, 0], [0, T, 0], [0, 0, T]])
    xn1 = np.matmul(fctrv(w, T), X) + np.matmul(gam, u)
    return xn1


def fctrv(w, dt):
    if np.abs(w) > 0.0001:
        st = np.matrix([[1.0, 0.0, np.sin(w * dt) / w, (np.cos(w * dt) - 1) / w, 0.0],
                          [0.0, 1.0, (1 - np.cos(w * dt)) / w, np.sin(w * dt) / w, 0.0],
                          [0.0, 0, np.cos(w * dt), -np.sin(w * dt), 0], [0.0, 0.0, np.sin(w * dt), np.cos(w * dt), 0],
                          [0, 0, 0, 0, 1.0]])
    else:
        st = np.matrix([[1.0, 0.0, dt, 0.0, 0.0], [0.0, 1.0, 0.0, dt, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])
    return st







