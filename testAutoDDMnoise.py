from AutoDDM import AutoDDM
from skika.data.reccurring_concept_stream import RCStreamType, RecurringConceptStream, conceptOccurence
import matplotlib.pyplot as plt
import warnings
import time
import numpy as np
import random
import collections
from skmultiflow.trees import HoeffdingTreeClassifier
from guppy import hpy
import pandas


warnings.filterwarnings('ignore')
plt.style.use("seaborn-whitegrid")

# Global variable
TRAINING_SIZE = 1
STREAM_SIZE = 10000000
grace = 1000
tolence = 500
DRIFT_INTERVALS = [50000]
concepts = [0, 1, 2]
RANDOMNESS = 100
ignore = 0
rounds = 3

data_types = [RCStreamType.SINE, RCStreamType.AGRAWAL, RCStreamType.RBF]
noises = [0, 0.1, 0.2, 0.3]
Settings = []
Precisions = []
Precisions_std = []
Recalls = []
Recalls_std = []
F1s = []
F1s_std = []
F2s = []
F2s_std = []
Accuracies = []
Accuracies_std = []
Times = []
Times_std = []
Delays = []
Delays_std = []
Memories = []
Memories_std = []

for a in data_types:
    for b in noises:
        total_D_mine = []
        total_TP_mine = []
        total_FP_mine = []
        total_RT_mine = []
        total_DIST_mine = []
        total_mean_mem = []
        precisions = []
        recalls = []
        f1_scores = []
        f2_scores = []
        accuracies = []
        seeds = []
        for k in range(rounds):
            #seed = seeds[k]
            seed = random.randint(0, 10000)
            print('Seed: ' + str(seed))
            seeds.append(seed)
            keys = []
            actuals = [0]
            concept_chain = {0: 0}
            current_concept = 0
            for i in range(1, STREAM_SIZE + 1):
                for j in DRIFT_INTERVALS:
                    if i % j == 0:
                        if i not in keys:
                            keys.append(i)
                            randomness = random.randint(0, RANDOMNESS)
                            d = i + ((randomness * 1) if (random.randint(0, 1) > 0) else (randomness * -1))
                            concept_index = random.randint(0, len(concepts) - 1)
                            while concepts[concept_index] == current_concept:
                                concept_index = random.randint(0, len(concepts) - 1)
                            concept = concepts[concept_index]
                            concept_chain[d] = concept
                            actuals.append(d)
                            current_concept = concept

                            i2 = i + 17000
                            keys.append(i2)
                            randomness = random.randint(0, RANDOMNESS)
                            d = i2 + ((randomness * 1) if (random.randint(0, 1) > 0) else (randomness * -1))
                            concept_index = random.randint(0, len(concepts) - 1)
                            while concepts[concept_index] == current_concept:
                                concept_index = random.randint(0, len(concepts) - 1)
                            concept = concepts[concept_index]
                            concept_chain[d] = concept
                            actuals.append(d)
                            current_concept = concept

            x = collections.Counter(concept_chain.values())
            print(x)

            concept_0 = conceptOccurence(id=0, difficulty=6, noise=0,
                                         appearences=x[0], examples_per_appearence=max(DRIFT_INTERVALS))
            concept_1 = conceptOccurence(id=1, difficulty=6, noise=0,
                                         appearences=x[1], examples_per_appearence=max(DRIFT_INTERVALS))
            concept_2 = conceptOccurence(id=2, difficulty=6, noise=0,
                                         appearences=x[2], examples_per_appearence=max(DRIFT_INTERVALS))
            # concept_3 = conceptOccurence(id=3, difficulty=6, noise=0,
            #                              appearences=x[2], examples_per_appearence=max(DRIFT_INTERVALS))
            desc = {0: concept_0, 1: concept_1, 2: concept_2}

            datastream = RecurringConceptStream(
                rctype=a,
                num_samples=STREAM_SIZE,
                noise=b,
                concept_chain=concept_chain,
                seed=seed,
                desc=desc,
                boost_first_occurance=False)

            # X_train, y_train = datastream.next_sample(TRAINING_SIZE)
            X_train = []
            y_train = []
            for i in range(0, ignore + TRAINING_SIZE):
                if i < ignore:
                    continue
                X, y = datastream.next_sample()
                X_train.append(X[0])
                y_train.append(y[0])

            X_train = np.array(X_train)
            y_train = np.array(y_train)

            ht = HoeffdingTreeClassifier()

            ht.partial_fit(X_train, y_train)

            n_global = ignore + TRAINING_SIZE  # Cumulative Number of observations
            d_mine = 0
            w_mine = 0
            TP_mine = []
            FP_mine = []
            RT_mine = []
            mem_mine = []
            grace_end = n_global
            detect_end = n_global
            DIST_mine = []
            ML_accuracy = 0

            mineDDM = AutoDDM(tolence=tolence)
            h = hpy()
            while datastream.has_more_samples():
                n_global += 1

                X_test, y_test = datastream.next_sample()
                y_predict = ht.predict(X_test)
                mine_start_time = time.time()
                mineDDM.add_element(y_test != y_predict, n_global)
                ML_accuracy += 1 if y_test == y_predict else 0

                mine_running_time = time.time() - mine_start_time
                RT_mine.append(mine_running_time)
                if (n_global > grace_end):
                    if mineDDM.detected_warning_zone():
                        w_mine += 1
                    if mineDDM.detected_change():
                        d_mine += 1
                        drift_point = key = min(actuals, key=lambda x: abs(x - n_global))
                        if (drift_point != 0 and drift_point not in TP_mine and abs(drift_point - n_global) <= tolence):
                            print("A true positive detected at " + str(n_global))
                            DIST_mine.append(n_global - drift_point)
                            TP_mine.append(drift_point)
                            ht = HoeffdingTreeClassifier()
                            mineDDM.detect_TP(n_global)
                            grace_end = n_global + grace
                        else:
                            print("A false positive detected at " + str(n_global))
                            mineDDM.detect_FP(n_global)
                            FP_mine.append(drift_point)
                ht.partial_fit(X_test, y_test)
                if n_global % 10000 == 0:
                    # For Travis CI
                    print("N_global: " + str(n_global))
            x = h.heap()
            mem_mine.append(x.size)

            print("Round " + str(k + 1) + " out of 30 rounds")
            print("Actual drifts:" + str(len(actuals)))

            print("Number of drifts detected by mine: " + str(d_mine))
            total_D_mine.append(d_mine)
            print("TP by mine:" + str(len(TP_mine)))
            total_TP_mine.append(len(TP_mine))
            print("FP by mine:" + str(len(FP_mine)))
            total_FP_mine.append(len(FP_mine))
            precision = len(TP_mine) / (len(TP_mine) + len(FP_mine))
            recall = len(TP_mine) / (len(actuals) - 1)
            if (precision + recall) == 0:
                f1 = 0
                f2 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)
                f2 = 5 * precision * recall / (4 * precision + recall)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            f2_scores.append(f2)
            print("Precision: ", precision)
            print("Recall: ", recall)
            print("F1: ", f1)
            print("F2: ", f2)
            print("Accuracy by mine:" + str(ML_accuracy/STREAM_SIZE))
            accuracies.append(ML_accuracy/STREAM_SIZE)
            print("Mean RT  %s seconds" % (np.nanmean(mine_running_time)))
            total_RT_mine.append(np.nanmean(mine_running_time))
            print("Mean DIST by mine:" + str(np.nanmean(DIST_mine)))
            total_DIST_mine.append(np.nanmean(DIST_mine))
            print("Mean Memory by mine:" + str(mem_mine))
            total_mean_mem.append(mem_mine)

        print("Overall result:")
        print("Stream size: " + str(STREAM_SIZE))
        print("Drift intervals: " + str(DRIFT_INTERVALS))
        print("Actual drifts:" + str(len(actuals)))
        print("Seeds: " + str(seeds))

        print("Overall result for mine:")

        print("D: ", str(total_D_mine))
        print("Average Drift Detected: ", str(np.nanmean(total_D_mine)))
        print("Drift Detected Standard Deviation: ", str(np.std(total_D_mine)))

        print("TP: ", str(total_TP_mine))
        print("Average TP: ", str(np.nanmean(total_TP_mine)))
        print("TP Standard Deviation: ", str(np.std(total_TP_mine)))

        print("FP: ", str(total_FP_mine))
        print("Average FP: ", str(np.nanmean(total_FP_mine)))
        print("FP Standard Deviation: ", str(np.std(total_FP_mine)))

        print("Precisions: " + str(precisions))
        Precisions.append(np.nanmean(precisions))
        print("Average: " + str(np.nanmean(precisions)))
        Precisions_std.append(np.std(precisions))
        print("Deviation: " + str(np.std(precisions)))

        print("Recalls: " + str(recalls))
        Recalls.append(np.nanmean(recalls))
        print("Average: " + str(np.nanmean(recalls)))
        Recalls_std.append(np.std(recalls))
        print("Deviation: " + str(np.std(recalls)))

        print("F1 scores: " + str(f1_scores))
        F1s.append(np.nanmean(f1_scores))
        print("Average: " + str(np.nanmean(f1_scores)))
        F1s_std.append(np.std(f1_scores))
        print("Deviation: " + str(np.std(f1_scores)))

        print("F2 scores: " + str(f2_scores))
        F2s.append(np.nanmean(f2_scores))
        print("Average: " + str(np.nanmean(f2_scores)))
        F2s_std.append(np.std(f2_scores))
        print("Deviation: " + str(np.std(f2_scores)))

        print("Accuracies: " + str(accuracies))
        Accuracies.append(np.nanmean(accuracies))
        print("Average: " + str(np.nanmean(accuracies)))
        Accuracies_std.append(np.std(accuracies))
        print("Deviation: " + str(np.std(accuracies)))

        print("RT: ", str(total_RT_mine))
        Times.append(np.nanmean(total_RT_mine))
        print("Average RT: ", str(np.nanmean(total_RT_mine)))
        Times_std.append(np.std(total_RT_mine))
        print("RT Standard Deviation: ", str(np.std(total_RT_mine)))

        print("DIST: ", str(total_DIST_mine))
        Delays.append(np.nanmean(total_DIST_mine))
        print("Average DIST: ", str(np.nanmean(total_DIST_mine)))
        Delays_std.append(np.std(total_DIST_mine))
        print("DIST Standard Deviation: ", str(np.std(total_DIST_mine)))

        print("MEM: ", str(total_mean_mem))
        Memories.append(np.nanmean(total_mean_mem))
        print("Average Memory: ", str(np.nanmean(total_mean_mem)))
        Memories_std.append(np.std(total_mean_mem))
        print("Memory Standard Deviation: ", str(np.std(total_mean_mem)))

        Settings.append(str(a) + ' + ' + str(b))

dict = {'Settings': Settings, 'Precisions': Precisions, 'Precisions_std': Precisions_std, 'Recalls': Recalls, 'Recalls_std': Recalls_std,
        'F1s': F1s, 'F1s_std': F1s_std, 'F2s': F2s, 'F2s_std': F2s_std, 'Accuracies': Accuracies, 'Accuracies_std': Accuracies_std,
        'Times': Times, 'Times_std': Times_std, 'Delays': Delays, 'Delays_std': Delays_std, 'Memories': Memories, 'Memories_std': Memories_std}

df = pandas.DataFrame(dict)
df.to_csv('autoDDM_Noise.csv')


