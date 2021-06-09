from AutoDDM import AutoDDM
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
tolerance = 500
ignore = 0

weather_data = arff.load('weatherAUS.arff')
weather_df = pandas.DataFrame(weather_data)


weather_full_df = pandas.concat([weather_df] * 150)

STREAM_SIZE = weather_full_df.shape[0]

weather_stream = DataStream(weather_full_df, name="weather")
weather_stream.prepare_for_use()

X_train, y_train = weather_stream.next_sample(TRAINING_SIZE)

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

ddm = AutoDDM(tolerance=tolerance)
h = hpy()
while weather_stream.has_more_samples():
    n_global += 1

    X_test, y_test = weather_stream.next_sample()
    y_predict = ht.predict(X_test)

    ddm_start_time = time.time()
    ddm.add_element(y_test != y_predict, n_global)
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
                    ddm.detect_TP(drift_point)
                    ht = ht_p
                else:
                    print("FP detected at: " + str(drift_point))
                    FP_ddm.append(drift_point)
                    ddm.detect_FP(n_global)
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
x = h.heap()
mem_ddm.append(x.size)

print("Number of drifts detected by ddm: " + str(d_ddm))
print("TP by ddm:" + str(len(TP_ddm)))
print("FP by ddm:" + str(len(FP_ddm)))
print("Mean RT  %s seconds" % np.mean((ddm_running_time)))
print("Mean Memory by ddm:" + str(mem_ddm))
print("Accuracy by DDM:" + str(ML_accuracy / STREAM_SIZE))
