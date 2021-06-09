from skmultiflow.drift_detection import DDM
import matplotlib.pyplot as plt
import warnings
import time
import numpy as np
from skmultiflow.trees import HoeffdingTreeClassifier
from guppy import hpy
import arff
import pandas
from skmultiflow.data import DataStream

warnings.filterwarnings('ignore')
plt.style.use("seaborn-whitegrid")

# Global variable
TRAINING_SIZE = 1
grace = 1000
ignore = 0

elec_data = arff.load("elecNormNew.arff")
elec_df = pandas.DataFrame(elec_data)
elec_df.columns = ['date', 'day', 'period', 'nswprice', 'nswdemand', 'vicprice', 'vicdemand', 'transfer', 'class']
mapping = {"day":{"1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7}, "class": {"UP": 0, "DOWN": 1}}
elec_df = elec_df.replace(mapping)

elec_full_df = pandas.concat([elec_df] * 200)

STREAM_SIZE = elec_full_df.shape[0]

elec_stream = DataStream(elec_full_df, name="elec")
elec_stream.prepare_for_use()

X_train, y_train = elec_stream.next_sample(TRAINING_SIZE)

ht = HoeffdingTreeClassifier()

ht.partial_fit(X_train, y_train)

n_global = ignore + TRAINING_SIZE  # Cumulative Number of observations
d_ddm = 0
w_ddm = 0
TP_ddm = []
FP_ddm = []
RT_ddm = []
DIST_ddm = []
mem_ddm = []
retrain = False
grace_end = n_global
detect_end = n_global
mine_pr = []
mine_std = []
mine_alpha = []
pr_min = []
std_min = []
pi = []
mine_x_mean = []
mine_sum = []
mine_threshold = []
pred_grace_ht = []
pred_grace_ht_p = []
ht_p = None
ML_accuracy = 0

ddm = DDM()
h = hpy()
while elec_stream.has_more_samples():
    n_global += 1

    X_test, y_test = elec_stream.next_sample()
    y_predict = ht.predict(X_test)

    ddm_start_time = time.time()
    ddm.add_element(y_test != y_predict)
    ML_accuracy += 1 if y_test == y_predict else 0
    ddm_running_time = time.time() - ddm_start_time
    RT_ddm.append(ddm_running_time)
    if (n_global > grace_end):
        if (n_global > detect_end):
            if ht_p is not None:
                drift_point = detect_end - 2 * grace
                print("Accuracy of ht: " + str(np.mean(pred_grace_ht)))
                print("Accuracy of ht_p: " + str(np.mean(pred_grace_ht_p)))
                if (np.mean(pred_grace_ht_p) > np.mean(pred_grace_ht)):
                    print("TP detected at: " + str(drift_point))
                    TP_ddm.append(drift_point)
                    ht = ht_p
                else:
                    print("FP detected at: " + str(drift_point))
                    FP_ddm.append(drift_point)
                ht_p = None
                pred_grace_ht = []
                pred_grace_ht_p = []
            if ddm.detected_warning_zone():
                w_ddm += 1
            if ddm.detected_change():
                d_ddm += 1
                ht_p = HoeffdingTreeClassifier()
                grace_end = n_global + grace
                detect_end = n_global + 2 * grace
        else:
            pred_grace_ht.append(y_test == y_predict)
            pred_grace_ht_p.append(y_test == ht_p.predict(X_test))

    if ht_p is not None:
        ht_p.partial_fit(X_test, y_test)
    ht.partial_fit(X_test, y_test)
    print("N_global: " + str(n_global))
x = h.heap()
mem_ddm.append(x.size)

print("Number of drifts detected by ddm: " + str(d_ddm))
print("TP by ddm:" + str(len(TP_ddm)))
print("FP by ddm:" + str(len(FP_ddm)))
print("Mean RT  %s seconds" % np.mean((ddm_running_time)))
print("Mean Memory by ddm:" + str(mem_ddm))
print("Accuracy by DDM:" + str(ML_accuracy / STREAM_SIZE))
