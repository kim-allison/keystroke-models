import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math # new
from operator import add
random_seed = 1998
#np.random.seed(0)
import time

# 1) Importing online data
data = pd.read_csv("/data/online_data.csv")
data = data[data.columns[:-3]] # remove return key from data set (if data set is in correct format)
individuals = data["subject"].unique() # NumPy: returns the sorted unique elements
# individuals = array of user ids ('unique' ids), s002 to s057
# array(['s002', 's003', 's004', 's005', 's007', 's008', 's010', 's011',
#        's012', 's013', 's015', 's016', 's017', 's018', 's019', 's020',
#        's021', 's022', 's024', 's025', 's026', 's027', 's028', 's029',
#        's030', 's031', 's032', 's033', 's034', 's035', 's036', 's037',
#        's038', 's039', 's040', 's041', 's042', 's043', 's044', 's046',
#        's047', 's048', 's049', 's050', 's051', 's052', 's053', 's054',
#        's055', 's056', 's057'], dtype=object)

columns_to_drop = [i for i in data.columns if "DD." in i] # remove down-down columns
columns_to_keep = data.columns.drop(columns_to_drop)
data = data[columns_to_keep]
individuals = data["subject"].unique()

'''
These are the extracted data we use to figure out the mean and stdev.
*depends on seed and num_samples/ repetitions.
'''
def get_random_samples(df, user, session, num_samples = 10):
  usable_rows = (df["subject"] == user) & (df["sessionIndex"] == session)
  df = df[usable_rows]
  global random_seed 
  df = df.sample(frac=1, random_state = random_seed)
  df.reset_index(inplace=True, drop=True)
  df = df.loc[:num_samples-1]
  df = df[df.columns[3:]]
  return df

# 2) Stage 1 (online data): training and testing
'''
  Calculate z-score for each individual value
  and add them all up.
'''
def calculate_sample_z_score(sample, avg, stdv):
  total_z_score = 0

  for i in range(len(sample)):
    total_z_score += abs((sample[i] - avg[i]) / stdv[i])
  return total_z_score

def update_confusion_matrix(confusion_matrix, avg, stdv, bound, samples, true_user):
  for i in samples.index:
    z_score = calculate_sample_z_score(samples.loc[i], avg, stdv) / math.sqrt(len(samples.loc[i]))
    print(z_score)
    if z_score > bound:
      confusion_matrix[true_user+"R"] = confusion_matrix[true_user+"R"] + 1
    else: # z-score sum is within bounds! yay, accept
      confusion_matrix[true_user+"A"] = confusion_matrix[true_user+"A"] + 1
  return confusion_matrix

'''
Extract useful data, and compute the mean and variance for 'repetitions' amount of 
trials.
'''
def get_probability_trained_bound(df, user, session, repetitions = 10):
    # right now, we are assuming it's a normal distribution
    usable_rows = (df["subject"] == user) & (df["sessionIndex"] == session) # filter 1 user, 1 session
    df = df[usable_rows] # usable = chosen user and session
    global random_seed 
    # return df.size (1100)
    df = df.sample(frac=1, random_state = random_seed) # pandas: return a random sample, where frac = fraction of axis items to return
    # return df.size (1100)??? <Q> why sample? -> randomizes rows
    df.reset_index(inplace=True, drop=True) # resets index 
    df = df.loc[:repetitions-1] # get only 'repetitions' amount of rows
    df = df[df.columns[3:]] # remove subject, sessionIndex, rep

    average_row = {}
    stdv = {}
    for i in df.columns:
        average_row[i] = df[i].mean()
        stdv[i] = df[i].std()
    df = df.append(average_row, ignore_index=True)
    df = df.append(stdv, ignore_index=True)

    probability_trained_bound = df.loc[repetitions:] # mean, stdev
    probability_trained_bound = probability_trained_bound.rename(index={repetitions:"avg", repetitions+1:"stdv"})
    return probability_trained_bound

def test_probability_trained_bound(df, user, bounds_to_test, repetitions_train = 10, repetitions_test = 50):
  confusion_matrix_progression = {"TA":[], "TR":[], "FA":[], "FR":[]}
  
  for bound in bounds_to_test:
    print(bound)
    confusion_matrix = {"TA":0, "TR":0, "FA":0, "FR":0}
    np.random.seed(0) 
    training_session = np.random.randint(1, 9)
    start = time.time()
    avg_stdv = get_probability_trained_bound(df, user, training_session, repetitions_train)
    end = time.time()
    print(end - start)
    trained_average = avg_stdv.loc["avg"]
    trained_stdv = avg_stdv.loc["stdv"]

    for session in range(1, 9):
      if session == training_session:
        continue
      random_sample = get_random_samples(df, user, session, repetitions_test)
      confusion_matrix = update_confusion_matrix(confusion_matrix, trained_average, trained_stdv, bound, random_sample, "T")
      
    for test_user in df["subject"].unique():
      if test_user == user:
        continue
      for session in range(1, 9):
        random_sample = get_random_samples(df, test_user, session, repetitions_test)
        confusion_matrix = update_confusion_matrix(confusion_matrix, trained_average, trained_stdv, bound, random_sample, "F")
    
    for key in confusion_matrix_progression.keys():
      confusion_matrix_progression[key].append(confusion_matrix[key])

  return confusion_matrix_progression

# main (online data)
bounds = np.linspace(0, 10, 11) # return evenly spaced numbers over a specified interval, so 10 evenly spaced samples from 0 to 10
  # array([ 0.        ,  1.11111111,  2.22222222,  3.33333333,  4.44444444,
  #         5.55555556,  6.66666667,  7.77777778,  8.88888889, 10.        ])

total_cm = {}

for user_i in range(0,len(individuals)):
  confusion_matrix_progression_probability = test_probability_trained_bound(data, individuals[user_i], bounds, 10, 50) 
  # plt.plot(np.array(confusion_matrix_progression_probability['FA']) / (confusion_matrix_progression_probability['FA'][0] + confusion_matrix_progression_probability['FR'][0]))
  # plt.plot(np.array(confusion_matrix_progression_probability['TA']) / (confusion_matrix_progression_probability['TA'][0] + confusion_matrix_progression_probability['TR'][0]))
  # plt.show()

  print('user cm: ', confusion_matrix_progression_probability)
  # print('total cm: ', total_cm)
  if len(total_cm) == 0:
    total_cm = confusion_matrix_progression_probability
  else:
    for rate in confusion_matrix_progression_probability:
      added_prob = list(map(add, total_cm[rate], confusion_matrix_progression_probability[rate]))
      total_cm[rate] = added_prob

    # 'steeper' slop -> FA < TA, and we want to optimize this
    plt.plot(np.array(confusion_matrix_progression_probability['FA']) / (confusion_matrix_progression_probability['FA'][0] + confusion_matrix_progression_probability['FR'][0]), np.array(confusion_matrix_progression_probability['TA']) / (confusion_matrix_progression_probability['TA'][0] + confusion_matrix_progression_probability['TR'][0]))
    plt.plot(np.linspace(0, 1, 2), np.linspace(1, 0, 2), color = "black")
    plt.xlabel("FAR")
    plt.ylabel("TAR")
    plt.title(individuals[user_i])
    plt.show()

avg_cm = {}
# this step is intuitively reasonable, but unnecessary in this case
# since total_cm already sums up rates and is therefore ~ average.
for rate in total_cm:
  avg_prob = list(map(lambda x: x/len(individuals), total_cm[rate]))
  avg_cm[rate] = avg_prob
print('avg cm: ', avg_cm)

plt.plot(np.array(avg_cm['FA']) / (avg_cm['FA'][0] + avg_cm['FR'][0]),
        np.array(avg_cm['TA']) / (avg_cm['TA'][0] + avg_cm['TR'][0]))
plt.plot(np.linspace(0, 1, 2), np.linspace(1, 0, 2), color = "black")
plt.title("avg")
plt.xlabel("FAR")
plt.ylabel("TAR")
plt.show()

TAR_progression = np.array(avg_cm['TA']) / (avg_cm['TA'][0] + avg_cm['TR'][0])
FAR_progression = np.array(avg_cm['FA']) / (avg_cm['FA'][0] + avg_cm['FR'][0])

best_index = np.argmin(abs(1 - (TAR_progression + FAR_progression))) # which bound gives us a rate sum close to 1?
best_bound = bounds[best_index]
print(best_index)
print(best_bound)

# the best bound from the online data set = 8

# 3) Stage 2 (survey data): training and testing
survey_data = pd.read_csv("/data/survey_data_csv.csv")

'''
  Same as the one in the previous section --
  just needed a new one to index sample.
'''
def calculate_sample_z_score_survey(sample, avg, stdv):
  total_z_score = 0
  sample = sample.iloc[4:] # the survey data sample has other columns (subject and stuff)
  # print("Sample:\n")
  # print(type(sample))
  # <class 'pandas.core.series.Series'>
  # print(sample.iloc[4:])
  # print(sample)

  for i in range(len(sample)):
    total_z_score += abs((sample[i] - avg[i]) / stdv[i])
  return total_z_score

'''
  Same as the one in the previous section --
  changed the z-score function so that function call
  had to be different (lazy move).
'''
def update_confusion_matrix_survey(confusion_matrix, avg, stdv, bound, samples, true_user):
  for i in samples.index:
    z_score = calculate_sample_z_score_survey(samples.loc[i], avg, stdv) / math.sqrt(len(samples.loc[i]))
    if z_score > bound:
      confusion_matrix[true_user+"R"] = confusion_matrix[true_user+"R"] + 1
    else: # z-score sum is within bounds! yay, accept
      confusion_matrix[true_user+"A"] = confusion_matrix[true_user+"A"] + 1
  return confusion_matrix

def get_probability_trained_bound_survey(df, user, session):
  usable_rows = (df["subject"] == user) & (df["sessionIndex"] == session) # filter 1 user, 1 session
  df = df[usable_rows] # usable = chosen user and session
  
  global random_seed 
 
  #pandas: return a random sample, where frac = fraction of axis items to return
  #df = df.sample(frac=1, random_state = random_seed) 

  df.reset_index(inplace=True, drop=True) # resets index 
  df = df[df.columns[3:]] # remove subject, sessionIndex, rep

  average_row = {}
  stdv = {}
  for i in df.columns:
    average_row[i] = df[i].mean()
    stdv[i] = df[i].std()
  df = df.append(average_row, ignore_index=True)
  df = df.append(stdv, ignore_index=True)

  probability_trained_bound = df.loc[10:] # mean, stdev
  probability_trained_bound = probability_trained_bound.rename(index={10:"avg", 11:"stdv"})
  return probability_trained_bound

def test_probability_trained_bound_survey(df, user, bound, session_train):
  
  confusion_matrix = {"TA":0, "TR":0, "FA":0, "FR":0}
  start = time.time()
  avg_stdv = get_probability_trained_bound(df, user, session_train)
  end = time.time()
  print(end-start)
  trained_avg = avg_stdv.loc["avg"]
  trained_stdv = avg_stdv.loc["stdv"]

  # 1) TRUE: test 9 sessions data against mean/ stdev calculations
  # we are trying to identify a 'user' (testing for one user's sessions)
  usable_rows = (df["subject"] == user) & (df["sessionIndex"] == 3) # filter 1 user, 1 session
  test_sample = df[usable_rows] # usable = chosen user and session
  test_sample.reset_index(inplace = True)
  confusion_matrix = update_confusion_matrix_survey(confusion_matrix, trained_avg, trained_stdv, bound, test_sample, "T")

  # 2) FALSE: not user
  non_user_rows =  (df["subject"] != user)
  test_sample = df[non_user_rows]
  test_sample.reset_index(inplace = True)
  confusion_matrix = update_confusion_matrix_survey(confusion_matrix, trained_avg, trained_stdv, bound, test_sample, "F")

  #print(confusion_matrix)
  return confusion_matrix

# main (survey data)
TAR_list = []
FAR_list = []
# total_cm = {}
for users in survey_data["subject"].unique():
  #check if user has 30 inputs
  if(survey_data["subject"]==users).sum() != 30:
    continue
  confusion_matrix = test_probability_trained_bound_survey(survey_data, user = users, bound=8, session_train = 2)
  TAR = confusion_matrix['TA']/(confusion_matrix['TA'] + confusion_matrix['TR'])
  FAR = confusion_matrix['FA']/(confusion_matrix['FA'] + confusion_matrix['FR'])
  TAR_list.append(TAR)
  FAR_list.append(FAR)

print(np.array(TAR_list).min())
print(np.array(FAR_list).max())
print("")
print(np.array(TAR_list).mean()) # that's pretty good:)
print(np.array(FAR_list).mean())