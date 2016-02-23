import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
from sklearn.cross_validation import StratifiedShuffleSplit
from copy import deepcopy


DATA_PATH = '/Users/vupham/data/sf_crimes/'

trainDF = pd.read_csv(DATA_PATH + 'train.csv')

xy_scaler = preprocessing.StandardScaler()
xy_scaler.fit(trainDF[["X", "Y"]])
trainDF[["X", "Y"]]=xy_scaler.transform(trainDF[["X", "Y"]])

trainDF = trainDF[abs(trainDF["Y"]) < 100]
trainDF["rot45_X"] = .707 * trainDF["Y"] + .707 * trainDF["X"]
trainDF["rot45_Y"] = .707 * trainDF["Y"] - .707 * trainDF["X"]
trainDF["rot30_X"] = (1.732/2) * trainDF["X"] + (1./2) * trainDF["Y"]
trainDF["rot30_Y"] = (1.732/2) * trainDF["Y"] - (1./2) * trainDF["X"]
trainDF["rot60_X"] = (1./2) * trainDF["X"] + (1.732/2) * trainDF["Y"]
trainDF["rot60_Y"] = (1./2) * trainDF["Y"] - (1.732/2) * trainDF["X"]
trainDF["radial_r"] = np.sqrt(np.power(trainDF["Y"], 2) + np.power(trainDF["X"], 2))

trainDF.index = range(len(trainDF))


def parse_time(x):
    DD = datetime.strptime(x,"%Y-%m-%d %H:%M:%S")
    time = DD.hour  # *60+DD.minute
    day = DD.day
    month = DD.month
    year = DD.year
    return time, day, month, year


def get_season(x):
    summer = 0
    fall = 0
    winter = 0
    spring = 0
    if x in [5, 6, 7]:
        summer = 1
    if x in [8, 9, 10]:
        fall = 1
    if x in [11, 0, 1]:
        winter = 1
    if x in [2, 3, 4]:
        spring = 1
    return summer, fall, winter, spring


def parse_data(df, logodds, logoddsPA):
    feature_list = df.columns.tolist()
    if "Descript" in feature_list:
        feature_list.remove("Descript")
    if "Resolution" in feature_list:
        feature_list.remove("Resolution")
    if "Category" in feature_list:
        feature_list.remove("Category")
    if "Id" in feature_list:
        feature_list.remove("Id")
    cleanData=df[feature_list]
    cleanData.index=range(len(df))
    print "Creating address features"
    address_features=cleanData["Address"].apply(lambda x: logodds[x])
    address_features.columns=["logodds"+str(x) for x in range(len(address_features.columns))]
    print "Parsing dates"
    cleanData["Time"], cleanData["Day"], cleanData["Month"], cleanData["Year"]=zip(*cleanData["Dates"].apply(parse_time))
#     dummy_ranks_DAY = pd.get_dummies(cleanData['DayOfWeek'], prefix='DAY')
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
#     cleanData["DayOfWeek"]=cleanData["DayOfWeek"].apply(lambda x: days.index(x)/float(len(days)))
    print "Creating one-hot variables"
    dummy_ranks_PD = pd.get_dummies(cleanData['PdDistrict'], prefix='PD')
    dummy_ranks_DAY = pd.get_dummies(cleanData["DayOfWeek"], prefix='DAY')
    cleanData["IsInterection"]=cleanData["Address"].apply(lambda x: 1 if "/" in x else 0)
    cleanData["logoddsPA"]=cleanData["Address"].apply(lambda x: logoddsPA[x])
    print "droping processed columns"
    cleanData=cleanData.drop("PdDistrict",axis=1)
    cleanData=cleanData.drop("DayOfWeek",axis=1)
    cleanData=cleanData.drop("Address",axis=1)
    cleanData=cleanData.drop("Dates",axis=1)
    feature_list=cleanData.columns.tolist()
    print "joining one-hot features"
    features = cleanData[feature_list].join(dummy_ranks_PD.ix[:,:]).join(dummy_ranks_DAY.ix[:,:]).join(address_features.ix[:,:])
    print "creating new features"
    features["IsDup"]=pd.Series(features.duplicated()|features.duplicated(take_last=True)).apply(int)
    features["Awake"]=features["Time"].apply(lambda x: 1 if (x==0 or (x>=8 and x<=23)) else 0)
    features["Summer"], features["Fall"], features["Winter"], features["Spring"]=zip(*features["Month"].apply(get_season))
    if "Category" in df.columns:
        labels = df["Category"].astype('category')
#         label_names=labels.unique()
#         labels=labels.cat.rename_categories(range(len(label_names)))
    else:
        labels = None
    return features, labels


addresses=sorted(trainDF["Address"].unique())
categories=sorted(trainDF["Category"].unique())
C_counts=trainDF.groupby(["Category"]).size()
A_C_counts=trainDF.groupby(["Address","Category"]).size()
A_counts=trainDF.groupby(["Address"]).size()
logodds={}
logoddsPA={}
MIN_CAT_COUNTS=2
default_logodds=np.log(C_counts/len(trainDF))-np.log(1.0-C_counts/float(len(trainDF)))
for addr in addresses:
    PA=A_counts[addr]/float(len(trainDF))
    logoddsPA[addr]=np.log(PA)-np.log(1.-PA)
    logodds[addr]=deepcopy(default_logodds)
    for cat in A_C_counts[addr].keys():
        if (A_C_counts[addr][cat]>MIN_CAT_COUNTS) and A_C_counts[addr][cat]<A_counts[addr]:
            PA=A_C_counts[addr][cat]/float(A_counts[addr])
            logodds[addr][categories.index(cat)]=np.log(PA)-np.log(1.0-PA)
    logodds[addr]=pd.Series(logodds[addr])
    logodds[addr].index=range(len(categories))


features, labels=parse_data(trainDF,logodds,logoddsPA)
print features.columns.tolist()
print len(features.columns)

collist=features.columns.tolist()
scaler = preprocessing.StandardScaler()
scaler.fit(features)
features[collist] = scaler.transform(features)

sss = StratifiedShuffleSplit(labels, train_size=0.5)
for train_index, test_index in sss:
    features_train,features_test=features.iloc[train_index],features.iloc[test_index]
    labels_train,labels_test=labels[train_index],labels[test_index]
features_test.index=range(len(features_test))
features_train.index=range(len(features_train))
labels_train.index=range(len(labels_train))
labels_test.index=range(len(labels_test))
features.index=range(len(features))
labels.index=range(len(labels))

print 'Training features: ', len(features.columns)

# print "all", log_loss(labels, model.predict_proba(features.as_matrix(),verbose=0))
# print "train", log_loss(labels_train, model.predict_proba(features_train.as_matrix(),verbose=0))
# print "test", log_loss(labels_test, model.predict_proba(features_test.as_matrix(),verbose=0))


testDF=pd.read_csv(DATA_PATH + "test.csv")
testDF[["X","Y"]]=xy_scaler.transform(testDF[["X", "Y"]])

#set outliers to 0
testDF["X"]=testDF["X"].apply(lambda x: 0 if abs(x)>5 else x)
testDF["Y"]=testDF["Y"].apply(lambda y: 0 if abs(y)>5 else y)

testDF["rot45_X"] = .707 * testDF["Y"] + .707 * testDF["X"]
testDF["rot45_Y"] = .707 * testDF["Y"] - .707 * testDF["X"]
testDF["rot30_X"] = (1.732/2) * testDF["X"] + (1./2) * testDF["Y"]
testDF["rot30_Y"] = (1.732/2) * testDF["Y"] - (1./2) * testDF["X"]
testDF["rot60_X"] = (1./2) * testDF["X"] + (1.732/2) * testDF["Y"]
testDF["rot60_Y"] = (1./2) * testDF["Y"] - (1.732/2) * testDF["X"]
testDF["radial_r"] = np.sqrt(np.power(testDF["Y"], 2) + np.power(testDF["X"], 2))

new_addresses=sorted(testDF["Address"].unique())
new_A_counts=testDF.groupby("Address").size()
only_new=set(new_addresses+addresses)-set(addresses)
only_old=set(new_addresses+addresses)-set(new_addresses)
in_both=set(new_addresses).intersection(addresses)
for addr in only_new:
    PA=new_A_counts[addr]/float(len(testDF)+len(trainDF))
    logoddsPA[addr]=np.log(PA)-np.log(1.-PA)
    logodds[addr]=deepcopy(default_logodds)
    logodds[addr].index=range(len(categories))
for addr in in_both:
    PA=(A_counts[addr]+new_A_counts[addr])/float(len(testDF)+len(trainDF))
    logoddsPA[addr]=np.log(PA)-np.log(1.-PA)

features_sub, _ = parse_data(testDF, logodds, logoddsPA)

collist = features_sub.columns.tolist()
print collist
print 'Test features: ', len(collist)

features_sub[collist] = scaler.transform(features_sub[collist])

np.savez('./crimes_data.npz', features_train=features_train.as_matrix(), labels_train=labels_train.as_matrix().T,
         features_val=features_test.as_matrix(), labels_val=labels_test.as_matrix().T,
         features_test=features_sub.as_matrix(),
         features=features.as_matrix(), labels=labels.as_matrix().T)

# predDF=pd.DataFrame(model.predict_proba(features_sub.as_matrix(),verbose=0),columns=sorted(labels.unique()))
# predDF.to_csv("crimeSF_NN_logodds.csv",index_label="Id",na_rep="0")
