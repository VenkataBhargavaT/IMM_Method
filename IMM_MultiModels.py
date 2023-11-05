import numpy as np
import math as mt
from multimodels import cvcv_model
from multimodels import ctra_model
from multimodels import caa_model
#from multimodels import ground_truth_data
import matplotlib.pyplot as plt
from multimodels import samplingdata


'''
This is the main file to run the IMM Algorithm. Two math models CTRA, CAA models are used by the IMM tracker to choose the proper model based on the scenario. 
'''
class IMM:
    def __init__(self):
        self.mrk_tr_prob = np.matrix([[0.8, 0.2], [0.2, 0.8]])
        self.cond_mod_prob = np.matrix([[0.98, 0.02], [0.02, 0.98]])
        self.model_prob = np.matrix([[0.6], [0.4]])

    def predict_model_prob(self):
        '''
        tot_mod_prob = np.matmul(self.mrk_tr_prob, self.model_prob)
        con_mod_prob = self.mrk_tr_prob # used just as initialization purpose and is overwritten later
        for r in range(self.mrk_tr_prob.shape[0]):
            for c in range(self.mrk_tr_prob.shape[1]):
                con_mod_prob[r, c] = self.mrk_tr_prob[r, c]*self.model_prob[c, 0]/tot_mod_prob[r, 0]
        self.cond_mod_prob = con_mod_prob
        '''

        tot_mod_prob = self.mrk_tr_prob[0, 0] * self.model_prob[0, 0] + self.mrk_tr_prob[1, 0] * self.model_prob[1, 0]
        self.cond_mod_prob[0, 0] = self.mrk_tr_prob[0, 0] * self.model_prob[0, 0] / tot_mod_prob
        self.cond_mod_prob[0, 1] = self.mrk_tr_prob[1, 0] * self.model_prob[1, 0] / tot_mod_prob

        tot_mod_prob = self.mrk_tr_prob[0, 1] * self.model_prob[0, 0] + self.mrk_tr_prob[1, 1] * self.model_prob[1, 0]
        self.cond_mod_prob[1, 0] = self.mrk_tr_prob[0, 1] * self.model_prob[0, 0] / tot_mod_prob
        self.cond_mod_prob[1, 1] = self.mrk_tr_prob[1, 1] * self.model_prob[1, 0] / tot_mod_prob
        return

    def update_model_prob(self, likelihood):
        tot_mod_prob = 0
        '''
        for r in range(self.model_prob.shape[0]):
            tot_mod_prob += self.model_prob[r, 0]*likelihood[r, 0]

        for r in range(self.model_prob.shape[0]):
            self.model_prob[r, 0] = self.model_prob[r, 0] * likelihood[r, 0]/tot_mod_prob
        '''
        mod1_prob = likelihood[0, 0] * (
                    self.mrk_tr_prob[0, 0] * self.model_prob[0, 0] + self.mrk_tr_prob[1, 0] * self.model_prob[1, 0])
        mod2_prob = likelihood[1, 0] * (
                    self.mrk_tr_prob[0, 1] * self.model_prob[0, 0] + self.mrk_tr_prob[1, 1] * self.model_prob[1, 0])
        tot_mod_prob = mod2_prob + mod1_prob
        self.model_prob[0, 0] = mod1_prob / tot_mod_prob
        self.model_prob[1, 0] = mod2_prob / tot_mod_prob
        return

def ground_truth_data(data, xki):
    sampletime = np.arange(data.ts, data.tf, data.td)
    count = 0
    st = np.zeros([4, len(sampletime)])
    caa = caa_model(data)
    cvv = cvcv_model(data)
    ctra = ctra_model(data)

    caa.state = np.matrix([[xki[0, 0]], [xki[1, 0]], [xki[2, 0]], [xki[3, 0]], [0.0], [0.0]])
    cvv.state = np.matrix([[xki[0, 0]], [xki[1, 0]], [xki[2, 0]], [xki[3, 0]]])
    vogi = np.sqrt(xki[2, 0]**2 + xki[3, 0]**2)
    ctra.state = np.matrix([[xki[0, 0]], [xki[1, 0]], [0.0], [0.0], [vogi], [0.0]])
    u = np.matrix([[0], [0]])
    for t in sampletime:
        if t <= 0.0:  # 4.0
            a = 0
            caa.state[4, 0] = a
            caa.state[5, 0] = 0
            xkp = caa.pred_state(u)
        elif (t > 6.0) and (t < 5.0):  # 6.0
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
        elif t == 7.0:
            dpsi = 10.0*np.pi/180
            vog = np.sqrt(caa.state[2, 0]**2+caa.state[3, 0]**2)
            ctra.state = np.matrix([[caa.state[0, 0]], [caa.state[1, 0]], [0], [dpsi], [vog], [0]])
            ctra.pred_state(u)
            xkp = np.matrix([[ctra.state[0, 0]], [ctra.state[1, 0]], [ctra.state[4, 0]*np.cos(ctra.state[2, 0])],
                             [ctra.state[4, 0]*np.sin(ctra.state[2, 0])]])
        elif (t > 7.0) and (t < 9.0):
            ctra.pred_state(u)
            xkp = np.matrix([[ctra.state[0, 0]], [ctra.state[1, 0]], [ctra.state[4, 0] * np.cos(ctra.state[2, 0])],
                             [ctra.state[4, 0] * np.sin(ctra.state[2, 0])]])
        elif t == 9.0:
            dpsi = -10.0 * np.pi / 180
            ctra.state[3, 0] = dpsi
            ctra.pred_state(u)
            xkp = np.matrix([[ctra.state[0, 0]], [ctra.state[1, 0]], [ctra.state[4, 0] * np.cos(ctra.state[2, 0])],
                             [ctra.state[4, 0] * np.sin(ctra.state[2, 0])]])
        elif (t > 9.0) and (t < 11.0):
            ctra.pred_state(u)
            xkp = np.matrix([[ctra.state[0, 0]], [ctra.state[1, 0]], [ctra.state[4, 0] * np.cos(ctra.state[2, 0])],
                             [ctra.state[4, 0] * np.sin(ctra.state[2, 0])]])
        elif t == 11.0:
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


if __name__ == '__main__':
    inp = samplingdata

    ini_state = np.matrix([[10.0], [0.0], [10.0], [0.0]])
    Xd_true = ground_truth_data(inp, ini_state)
    # model1: 'caa'
    # the state convention is [x, y, vx, vy, ax, ay]
    caa = caa_model(inp)
    caa.state = np.matrix([[10.0], [0.0], [10.0], [0.0], [0.0], [0.0]])
    caa.scaleq(0.1, 0.01)
    # model2: 'ctra':
    # the state convention is [x, y, psi, dpsi, vog, aog]

    ctra = ctra_model(inp)
    ctra.state = np.matrix([[10.0], [0.0], [0.0], [0.0], [10.0], [0.0]])
    ctra.scaleq = 1000.0*0.5

    # xkp = st.state
    sig_r = 0.1
    sig_tita = 3 * (mt.pi / 180.0)

    H = np.matrix([[1.0, 0., 0., 0.], [0., 1., 0., 0.]])
    Hv = np.matrix([[0.0, 0., 1.0, 0.], [0., 0., 0., 1.]])
    Hd = np.matrix([[1.0, 0.], [0., 1.]])
    H_st6 = np.matrix([[1.0, 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0.]])
    # H_st6 = np.matrix([[1.0, 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0.], [0., 0., 1., 0., 0., 0.],
    #                   [0., 0., 0., 1., 0., 0.]])
    timesamples = np.arange(inp.ts, inp.tf, inp.td)
    ctra_est = np.zeros([4, len(timesamples)])
    caa_est = np.zeros([4, len(timesamples)])
    aog_est = np.zeros([1, len(timesamples)])
    psi_est = np.zeros([1, len(timesamples)])
    dpsi_est = np.zeros([1, len(timesamples)])
    x_pred = np.zeros([4, len(timesamples)])
    store_z_true = np.zeros([2, len(timesamples)])
    store_z_meas = np.zeros([2, len(timesamples)])
    storecaa_dist_y_std = np.zeros([1, len(timesamples)])
    storecaa_dist_x_std = np.zeros([1, len(timesamples)])
    storectra_dist_y_std = np.zeros([1, len(timesamples)])
    storectra_dist_x_std = np.zeros([1, len(timesamples)])
    store_likelihood = np.zeros([2, len(timesamples)])
    store_vy_std = np.zeros([1, len(timesamples)])
    store_vx_std = np.zeros([1, len(timesamples)])
    store_imm_mod_prob = np.zeros([2, len(timesamples)])

    # model_trans_prob = np.matrix([[0.98, 0.02], [0.2, 0.98]])
    imm = IMM()

    for sampleidx in range(0, len(timesamples)):

        u = np.matrix([[0, 0]])
        store_imm_mod_prob[:, sampleidx] = np.array([imm.model_prob[0, 0], imm.model_prob[1, 0]])
        # predict imm model prob
        imm.predict_model_prob()

        if sampleidx==0:
            cond_mod_prob = imm.cond_mod_prob

            '''               
            caa_pos = caa.state
            caa_pos_cov = caa.P
    
            ctra_pos = ctra.state
            ctra_pos_cov = ctra.P
            '''

            caa_pos = np.matrix([[caa.state[0, 0]], [caa.state[1, 0]]])
            caa_pos_cov = np.matrix([[caa.P[0, 0], caa.P[0, 1]], [caa.P[1, 0], caa.P[1, 1]]])

            ctra_pos = np.matrix([[ctra.state[0, 0]], [ctra.state[1, 0]]])
            ctra_pos_cov = np.matrix([[ctra.P[0, 0], ctra.P[0, 1]], [ctra.P[1, 0], ctra.P[1, 1]]])

            imm_int_caa_pos = caa_pos * cond_mod_prob[0, 0] + ctra_pos * cond_mod_prob[0, 1]
            caa_cov_sprd = np.matmul(caa_pos - imm_int_caa_pos, np.matrix.transpose(caa_pos - imm_int_caa_pos))
            ctra_cov_sprd = np.matmul(ctra_pos - imm_int_caa_pos, np.matrix.transpose(ctra_pos - imm_int_caa_pos))
            imm_int_caa_pos_cov = cond_mod_prob[0, 0] * (caa_pos_cov + caa_cov_sprd) + cond_mod_prob[0, 1] * (
                        ctra_pos_cov + ctra_cov_sprd)

            imm_int_ctra_pos = caa_pos * cond_mod_prob[1, 0] + ctra_pos * cond_mod_prob[1, 1]
            caa_cov_sprd = np.matmul(caa_pos - imm_int_ctra_pos, np.matrix.transpose(caa_pos - imm_int_ctra_pos))
            ctra_cov_sprd = np.matmul(ctra_pos - imm_int_ctra_pos, np.matrix.transpose(ctra_pos - imm_int_ctra_pos))
            imm_int_ctra_pos_cov = cond_mod_prob[1, 0] * (caa_pos_cov + caa_cov_sprd) + cond_mod_prob[1, 1] * (
                        ctra_pos_cov + ctra_cov_sprd)


            caa.state[0, 0] = imm_int_caa_pos[0, 0]
            caa.state[1, 0] = imm_int_caa_pos[1, 0]
            caa.P[0, 0] = imm_int_caa_pos_cov[0, 0]
            caa.P[0, 1] = imm_int_caa_pos_cov[0, 1]
            caa.P[1, 0] = imm_int_caa_pos_cov[1, 0]
            caa.P[1, 1] = imm_int_caa_pos_cov[1, 1]

            ctra.state[0, 0] = imm_int_ctra_pos[0, 0]
            ctra.state[1, 0] = imm_int_ctra_pos[1, 0]
            ctra.P[0, 0] = imm_int_ctra_pos_cov[0, 0]
            ctra.P[0, 1] = imm_int_ctra_pos_cov[0, 1]
            ctra.P[1, 0] = imm_int_ctra_pos_cov[1, 0]
            ctra.P[1, 1] = imm_int_ctra_pos_cov[1, 1]
            '''
            caa.state = imm_int_caa_pos
            caa.P = imm_int_caa_pos_cov
    
            ctra.state = imm_int_ctra_pos
            ctra.P = imm_int_ctra_pos_cov
            '''

        caa.pred_state(u.T)
        caa.predict_statecov()

        ctra.pred_state(u.T)
        ctra.predict_statecov()
        x_pred[:, sampleidx] = np.array([caa.state[0, 0], caa.state[1, 0], caa.state[2, 0], caa.state[3, 0]])

        # tita = mt.atan2(st.state[1, 0], st.state[0, 0])
        # rng = np.max([mt.sqrt(st.state[0, 0] ** 2 + st.state[1, 0] ** 2), 1.0])
        tita = mt.atan2(Xd_true[1, sampleidx], Xd_true[0, sampleidx])
        rng = np.max([mt.sqrt(Xd_true[0, sampleidx] ** 2 + Xd_true[1, sampleidx] ** 2), 1.0])
        p2c_diff = np.matrix([[mt.cos(tita), -rng * mt.sin(tita)], [rng * mt.sin(tita), mt.cos(tita)]])
        polar_var = np.matrix([[sig_r ** 2, 0.0], [0.0, (sig_tita) ** 2]])
        meas_cov = np.matmul(p2c_diff, np.matmul(polar_var, p2c_diff.T))
        caa.R = meas_cov
        ctra.R = meas_cov
        # polar_var = np.matrix([[sig_r ** 2 * mt.cos(tita) ** 2 + (rng * mt.sin(tita)) ** 2 * sig_tita ** 2, 0],
        #                        [0, sig_r ** 2 * mt.sin(tita) ** 2 + (rng * mt.cos(tita)) ** 2 * sig_tita ** 2]])

        meas_noise = np.matrix(np.transpose(np.random.multivariate_normal((0, 0), polar_var, (1, 1))))
        # meas_noise = np.matmul(p2c_diff, meas_noise)

        z_true = np.transpose(np.matrix(Xd_true[0:2, sampleidx]))
        z = z_true + meas_noise.T

        '''
        z_true = np.transpose(np.matrix(Xd_true[0:4, sampleidx]))
        k = 0.1
        #z = z_true + np.matrix([meas_noise[0, 0]*k, meas_noise[1, 0]*k, np.random.randn(1).__getitem__(0)*0.1,
        #                        np.random.randn(1).__getitem__(0)*0.1]).T
        z = z_true + np.matrix([np.random.randn(1).__getitem__(0)*k, np.random.randn(1).__getitem__(0)*k, np.random.randn(1).__getitem__(0) * k,
                                np.random.randn(1).__getitem__(0) * k]).T
        meas_noise_cov = np.matrix(np.eye(4, 4))*k**2

        meas_noise_cov[0, 0] = meas_cov[0, 0] * k**2
        meas_noise_cov[0, 1] = meas_cov[0, 1] * k**2
        meas_noise_cov[1, 0] = meas_cov[1, 0] * k**2
        meas_noise_cov[1, 1] = meas_cov[1, 1] * k**2
        '''
        # caa model update
        caa.kal_update(z, H_st6, meas_cov * 10)
        # caa.kal_update(z, H_st6, meas_noise_cov)

        # ctra model update
        ctra.kal_update(z, H_st6, meas_cov * 10)
        # ctra.kal_update(z, H_st6, meas_noise_cov)


        ctra_est[:, sampleidx] = np.array([ctra.state[0, 0], ctra.state[1, 0], ctra.state[4, 0]*np.cos(ctra.state[2, 0]),
                                        ctra.state[4, 0]*np.sin(ctra.state[2, 0])])
        dpsi_est[:, sampleidx] = ctra.state[3, 0]
        caa_est[:, sampleidx] = np.array([caa.state[0, 0], caa.state[1, 0], caa.state[2, 0], caa.state[3, 0]])
        #ctra_est[:, sampleidx] = np.array([ctra.state[0, 0], ctra.state[1, 0], ctra.state[2, 0], ctra.state[3, 0]])

        likeli_hood = np.matrix([[caa.lk], [ctra.lk]])
        imm.update_model_prob(likeli_hood)
        store_likelihood[:, sampleidx] = np.array([caa.lk, ctra.lk])

        aog_est[:, sampleidx] = ctra.state[5, 0]
        psi_est[:, sampleidx] = ctra.state[2, 0]
        ru, rs, rv = np.linalg.svd(caa.P)
        # if np.linalg.det(ru) > 0:
        storecaa_dist_x_std[:, sampleidx] = mt.sqrt(rs[0])
        storecaa_dist_y_std[:, sampleidx] = mt.sqrt(rs[1])

        ru, rs, rv = np.linalg.svd(ctra.P)
        storectra_dist_x_std[:, sampleidx] = mt.sqrt(rs[0])
        storectra_dist_y_std[:, sampleidx] = mt.sqrt(rs[1])

        store_z_true[:, sampleidx] = np.array([z_true[0, 0], z_true[1, 0]])
        store_z_meas[:, sampleidx] = np.array([z[0, 0], z[1, 0]])

    # print("model 1 Q:", caa.Q)
    # print("model 2 Q:", ctra.Q)

    plt.figure(111)
    plt.plot(timesamples, store_imm_mod_prob[0, :], 'r--')
    plt.plot(timesamples, store_imm_mod_prob[1, :], 'b--')
    plt.legend(["model caa ", "model ctra"])
    plt.xlabel('time t(sec)')
    plt.ylabel('imm model prob')
    plt.title("IMM model prob vs time")

    plt.figure(211)
    plt.plot(timesamples, np.transpose(caa_est[0, :] - Xd_true[0, :]), 'r--')
    plt.plot(timesamples, np.transpose(ctra_est[0, :] - Xd_true[0, :]), 'b--')
    plt.plot(timesamples, np.transpose(store_z_meas[0, :] - Xd_true[0, :]), 'g--')
    # plt.plot(timesamples, np.transpose(Xd_true[0, :]), 'k--')
    plt.xlabel('time (sec)')
    plt.ylabel('object pos x (m) ')

    plt.figure(311)
    plt.plot(timesamples, np.transpose(caa_est[1, :]), 'r--')
    plt.plot(timesamples, np.transpose(ctra_est[1, :]), 'b--')
    plt.plot(timesamples, np.transpose(Xd_true[1, :]), 'k--')
    plt.plot(timesamples, np.transpose(store_z_meas[1, :]), 'g--')
    plt.xlabel('time (sec)')
    plt.ylabel('object pos y (m) ')

    plt.figure(411)
    plt.plot(timesamples, np.transpose(caa_est[2, :]), 'r--')
    plt.plot(timesamples, np.transpose(ctra_est[2, :]), 'b--')
    plt.plot(timesamples, np.transpose(Xd_true[2, :]), 'k--')
    plt.xlabel('time (sec)')
    plt.ylabel('object vrelx (m/s) ')

    plt.figure(511)
    plt.plot(timesamples, np.transpose(caa_est[3, :]), 'r--')
    plt.plot(timesamples, np.transpose(ctra_est[3, :]), 'b--')
    plt.plot(timesamples, np.transpose(Xd_true[3, :]), 'k--')
    plt.xlabel('time (sec)')
    plt.ylabel('object vrely (m/s) ')

    plt.figure(611)
    plt.plot(np.transpose(caa_est[1, :]), np.transpose(caa_est[0, :]), 'r--')
    plt.plot(np.transpose(ctra_est[1, :]), np.transpose(ctra_est[0, :]), 'b--')
    plt.plot(np.transpose(Xd_true[1, :]), np.transpose(Xd_true[0, :]), 'k--')
    plt.xlabel('object posy')
    plt.ylabel('object posx (m) ')

    plt.figure(711)
    plt.plot(timesamples, np.transpose(storecaa_dist_x_std))
    plt.plot(timesamples, np.transpose(storecaa_dist_y_std))
    plt.plot(timesamples, np.transpose(storectra_dist_x_std))
    plt.plot(timesamples, np.transpose(storectra_dist_y_std))
    plt.legend(['caa x std', 'caa y std', 'ctra x std', 'ctra y std'])
    plt.xlabel('time')
    plt.ylabel('object posx std (m) ')

    plt.figure(811)
    plt.plot(timesamples, np.transpose(store_likelihood[0, :]), 'r--')
    plt.plot(timesamples, np.transpose(store_likelihood[1, :]), 'b--')
    plt.xlabel('time')
    plt.ylabel('object likelihood')

    plt.figure(911)
    plt.plot(timesamples, np.transpose(dpsi_est[0, :]), 'r--')
    plt.plot(timesamples, np.transpose(psi_est[0, :]), 'b--')
    plt.xlabel('time')
    plt.ylabel('object dpsi')


    plt.show()

