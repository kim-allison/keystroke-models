import json
import csv
from collections import defaultdict

with open('/data/survey_data_json.json') as f:
  data = json.load(f)

# -----for loop time-----

KEYS = ['Digit5','KeyA','KeyE','KeyI','KeyL','KeyN','KeyO','KeyR','KeyT','Period','Shift']

real_data = data['data-survey-1'] # this will be different (changed)
ids = list(real_data.keys())

big_list = []

for id in ids:
    id_round = list(real_data[id].keys()) # ['round1', 'round10', 'round11', 'round12', 'round13', 'round14', 'round15', 'round16', 'round17', 'round18', 'round19'
    for round in id_round:

      info = {}
      info['subject'] = id[3:]
      info['rep'] = int(round[5:])
      info['sessionIndex'] = (int(round[5:])-1)//10+1
      # print(info)

      key_stuff = {}
      id_key = list(real_data[id][round].keys())
      for key in id_key:

        id_time = list(real_data[id][round][key].keys())
        #down
        down_time = real_data[id][round][key][id_time[0]]["time"]
        up_time = real_data[id][round][key][id_time[1]]["time"]

        if ("Shift" in key):
          key = "Shift"

        key_stuff['down'+key] = down_time/1000
        key_stuff['up'+key] = up_time/1000

      # hold time
      for some_key in KEYS:
        if 'Digit' in some_key: # 5
          info['H.five'] = key_stuff['up'+some_key]-key_stuff['down'+some_key]
        elif 'Period' in some_key: # period
          info['H.period'] = key_stuff['up'+some_key]-key_stuff['down'+some_key]
        elif 'Shift' in some_key: # shift
          pass
        elif 'R' in some_key: # r plus shift
          info['H.Shift.r'] = key_stuff['up'+some_key]-key_stuff['down'+some_key]
        else:
          info['H.'+some_key[-1].lower()] = key_stuff['up'+some_key]-key_stuff['down'+some_key]
      # print(info)

      # -----time to get key pairs
      # password: .tie5Roanl

      # NOTES:
      # I know I could have just used a for loop here
      # but I didn't want to use my brain...so
      # PLUS, this is going to be a problem if 'shift' is included
      # If I have time, I'll fix it:) -> SHIFT PROBLEM HAS BEEN FIXED!

      # KEYS = ['Digit5','KeyA','KeyE','KeyI','KeyL','KeyN','KeyO','KeyR','KeyT','Period','Shift']

      # UD.period.t
      info['UD.period.t'] = key_stuff['downKeyT']-key_stuff['upPeriod']

      # UD.t.i
      info['UD.t.i'] = key_stuff['downKeyI']-key_stuff['upKeyT']

      # i.e
      info['UD.i.e'] = key_stuff['downKeyE']-key_stuff['upKeyI']

      # e.five
      info['UD.e.five'] = key_stuff['downDigit5']-key_stuff['upKeyE']

      # TODO: five.Shift.r
      info['UD.five.Shift.r'] = key_stuff['downKeyR']-key_stuff['upDigit5']
      # *5R
      # info['UD.five.r'] = key_stuff['downr']-key_stuff['upfive']

      # Shift.r.o
      info['UD.Shift.r.o'] =  key_stuff['downKeyO']-key_stuff['upKeyR']
      # *Ro
      # info['UD.r.o'] = key_stuff['downo']-key_stuff['upr']

      # oa
      info['UD.o.a'] = key_stuff['downKeyA']-key_stuff['upKeyO']

      # an
      info['UD.a.n'] = key_stuff['downKeyN']-key_stuff['upKeyA']

      # nl
      info['UD.n.l'] = key_stuff['downKeyL']-key_stuff['upKeyN']

      big_list.append(info)

csv_file = '/data/survey_data_csv.csv'
# password: .tie5Roanl
csv_columns = ['subject','sessionIndex','rep']
h_columns = ['H.period','UD.period.t','H.t', 'UD.t.i','H.i', 'UD.i.e','H.e','UD.e.five','H.five','UD.five.Shift.r','H.Shift.r','UD.Shift.r.o','H.o','UD.o.a','H.a','UD.a.n','H.n','UD.n.l','H.l']
# ud_columns = ['UD.period.t', 'UD.t.i', 'UD.i.e', 'UD.e.five', 'UD.five.r', 'UD.r.o', 'UD.o.a', 'UD.a.n', 'UD.n.l']

csv_columns += h_columns

with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    for data in big_list:
        writer.writerow(data)