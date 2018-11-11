import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

from sklearn.utils import shuffle  #to remove the sequence complete randomness

test=shuffle(test)
train=shuffle(train)

#sample=train.head() #to observe the various columns to understand the dataset

#f=train.Activity.value_counts()  >>> to observe if the data is symmetrical and not largely biased
#g=test.Activity.value_counts()   ditto^

#print(train.shape,test.shape) # to observe the dimensions of the data

# plotting the various data counts of categorical results to visually observe abopve symmetry >>plt.bar(f,g)[improve]'


#train_data=train.drop('Activity',axis=1).values
#test_data=test.drop('Activity',axis=1).values

train_label=train.Activity.values
test_label=test.Activity.values

#m=pd.DataFrame(train_label) #>>> to view the activity column alone after seperation

##Encoding the Activity column and transforming from categorical to numerical

from sklearn import preprocessing
encoder=preprocessing.LabelEncoder()
train_label_encoded=encoder.fit_transform(train_label)
test_label_encoded=encoder.fit_transform(test_label)

#classification models >>
# target variable is categorical and independent variables are numerical

# thus-
#Decission trees, SVM , NN(deep and other forms) ,Random Forest, Gradient boosting, Auto encoder

# STARTING WITH NEURAL NETWORK! 

# applying supervised neural network using multi-layered  perceptron

import sklearn.neural_network as nn
mlpSGD=nn.MLPClassifier(hidden_layer_sizes=(90,),max_iter=1000,alpha=1e-4,
                        solver='sgd',verbose=10,tol=1e-19,random_state=1,learning_rate_init=0.001) #activation function default relu
mlpADAM=nn.MLPClassifier(hidden_layer_sizes=(90,),max_iter=1000,alpha=1e-4,
                        solver='adam',verbose=10,tol=1e-19,random_state=1,learning_rate_init=0.001)
mlpLBFGS=nn.MLPClassifier(hidden_layer_sizes=(90,),max_iter=1000,alpha=1e-4,
                        solver='lbfgs',verbose=10,tol=1e-19,random_state=1,learning_rate_init=0.001)

# Fitting the model

nnmodel_SGD=mlpSGD.fit(train_data,train_label_encoded)
nnmodel_LBFGS=mlpLBFGS.fit(train_data,train_label_encoded)
nnmodel_ADAM=mlpADAM.fit(train_data,train_label_encoded)

#print(nnmodel_SGD.score(test_data,test_label_encoded)) >>>> 88.08%
#print(nnmodel_ADAM.score(test_data,test_label_encoded)) >>>> 89.08%
#print(nnmodel_LBFGS.score(test_data,test_label_encoded)) >>> 84.18%

#print('train_data',train.shape,'\n',train.columns) >>visualising basically
#print('test_data',test.shape,'\n',test.columns) >>visualizing basically

#print('Train Labels',train['Activity'].unique(),'\n Test Labels',test['Activity'].unique())
#print(len(train['Activity'].unique()))

#pd.crosstab(train.Activity,train.subject) >> over cross tabulation to observe any particaular anomaly of a subject

#sub=train.loc[train['subject']==1]
#print(sub.head())
# train.subject.value_counts()  >>>  to look at individual subject's activity distribution in total no. of subjects

###### PLOTTING A GRAPH

#fig=plt.figure(figsize=(80,80))
#ax1=fig.add_subplot(2,2,1)
#ax1=sb.stripplot(x='Activity',y=sub.iloc[:,0],data=sub,jitter=True)
#ax2=fig.add_subplot(2,2,3)
#ax2=sb.stripplot(x='Activity',y=sub.iloc[:,1],data=sub,jitter=True)
#plt.show()

#### LOGISTIC REGRESSION

train_df=pd.read_csv("train.csv")  #taking a copy to not overwrite initial train and test data
test_df=pd.read_csv("test.csv")

# Labelencoding by creating a function ourself

#unique_activities=train_df.Activity.unique()
#print("number of unique activities: {}".format(len(unique_activities)))
#
#replacer={}
#
#for i,activity in enumerate(unique_activities):
#    replacer[activity]=i
#
#train_df.Activity=train_df.Activity.replace(replacer)
#test_df.Activity=test_df.Activity.replace(replacer)
#train_df.head(10) just lookign

train_df=train_df.drop("subject",axis=1)
test_df=test_df.drop("subject",axis=1)

def get_all_data():
    train_values=train_df.values
    test_values=test_df.values
    np.random.shuffle(train_values)
    np.random.shuffle(test_values)
    X_train=train_values[:,:-1]
    X_test=test_values[:,:-1]
    y_train=train_values[:,-1]
    y_test=test_values[:,-1]
    
    return X_train,X_test,y_train,y_test

from sklearn.linear_model import LogisticRegression
X_train,X_test,y_train,y_test=get_all_data()

model=LogisticRegression(C=1)   # value of c can be changed and obseved for each variation
model.fit(X_train,y_train)  
model.score(X_test,y_test)  # gives the result accuracy >>> for C=10(86.98%) >>> C=1(87.28)  C basically defines overfitiing control
                                                                                            # change solver, penalty and some other hyperparameters and observe
##  But logically with som nay predictors, there is a high chnace of multicollinearity that would significantly affect the model
#>>>> SO WE TRY TRANSFORMATIONS ON DATA TO reduce this, by Principal Componenet Analysis

from sklearn.decomposition import PCA
X_train,X_test,y_train,y_test=get_all_data()
pca=PCA(n_components=200) #initializing the PCA reducing features to 200 from 561
pca.fit(X_train) # aplying the PCA

X_train=pca.transform(X_train)
X_test=pca.transform(X_test)

model.fit(X_train,y_train)
model.score(X_test,y_test) # 86.78%



# NOW USING Standard Scler to again transform the datset to reduce multicollinearity0

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train,X_test,y_train,y_test=get_all_data()

X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

model.fit(X_train,y_train)
model.score(X_test,y_test)  #>>>>>> 86.58%




# USING KERAS LIBRARY WITH Tensorflow Backend

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.np_utils import to_categorical


X_train,X_test,y_train,y_test=get_all_data()
X_train=X_train#.astype(float) # Internet
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)


####  SEARCH for tsne python and read it (another transformation of data method) : Understand how and which to use when

#-----

# Look for various activation functions of MLP/Keras nn Classifier ( softmax,relu and all that)

n_input=X_train.shape[1]  #number of eatures of model   shape[1] gives columns and shape[0]  gives rows as X_train.shape is a 2 element tuple
n_output=6  # Possible labels of movements
n_samples=X_train.shape[0]
n_hidden_units=40
Y_train=to_categorical(train_label_encoded)    #>>> must change variable name as type conversion is not done in this fucntion thus, new vareiables need to be assigned
Y_test=to_categorical(test_label_encoded)      # Alos must ensure the value is label encoded as required for two categorical function
print(Y_train.shape)
print(Y_test.shape)


# Creating the model

def create_model():
    model=Sequential()
    model.add(Dense(n_hidden_units,input_dim=n_input,activation='relu'))
    model.add(Dense(n_hidden_units,input_dim=n_input,activation='relu'))
    model.add(Dense(n_output,activation='softmax'))
    

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])  # COMPILING MODEL
    return model


 

estimator=KerasClassifier(build_fn=create_model,epochs=20,batch_size=10,verbose=False)
estimator.fit(X_train,Y_train)
print(" Score {}".format(estimator.score(X_test,Y_test)))


## >>>> RANDOM FOREST CLASSIFIER

from sklearn.ensemble import RandomForestClassifier
X_train,X_test,y_train,y_test=get_all_data()
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

model=RandomForestClassifier(n_estimators=500)
model.fit(X_train,y_train)
model.score(X_test,y_test)     ### >>>> 91.09%


#### SGD model over keras training 

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
 
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import SGD,Adam


# Creating the Features MATRIX

train_data=train.iloc[:,:561].values    # upto and excluding column subject check by f=list(train) then f[561]
test_data=test.iloc[:,:561].values

train_labels=train.iloc[:,562:].values
test_labels=test.iloc[:,562:].values

from sklearn import preprocessing
encoder=preprocessing.LabelEncoder()
train_label_encoded=encoder.fit_transform(train_labels)
test_label_encoded=encoder.fit_transform(test_labels)

model=Sequential()
model.add(Dense(64,activation='relu',input_dim=561))
model.add(Dropout(0.5))  # it reduces the number of weights that has to be passed to the next layer to reduce overfitting (dropout layer)
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6,activation='softmax'))

### to create 6 column encoding from 1 column encoded data VERY IMPT
train_labelss=np.zeros((len(train_labels),6))
test_labelss=np.zeros((len(test_labels),6))
for k in range(0,len(train_labels)):
    if train_labels[k]=='STANDING':
        train_labelss[k][0]=1
    elif train_labels[k]=='WALKING':
        train_labelss[k][1]=1
    elif train_labels[k]=='WALKING_UPSTAIRS':
        train_labelss[k][2]=1
    elif train_labels[k]=='WALKING_DOWNSTAIRS':
        train_labelss[k][3]=1
    elif train_labels[k]=='SITTING':
        train_labelss[k][4]=1
    else:
        train_labelss[k][5]=1

test_labelss=np.zeros((len(test_labels),6))
test_labelss=np.zeros((len(test_labels),6))
for k in range(0,len(test_labels)):
    if test_labels[k]=='STANDING':
        test_labelss[k][0]=1
    elif test_labels[k]=='WALKING':
        test_labelss[k][1]=1
    elif test_labels[k]=='WALKING_UPSTAIRS':
        test_labelss[k][2]=1
    elif test_labels[k]=='WALKING_DOWNSTAIRS':
        test_labelss[k][3]=1
    elif test_labels[k]=='SITTING':
        test_labelss[k][4]=1
    else:
        test_labelss[k][5]=1


sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
model.fit(train_data,train_labelss,epochs=30,batch_size=128)  #>> Problem of expecting (,6) getting (,1) solved by getting Y_train and Y_test from to_categorical fucntion
score=model.evaluate(test_data,test_labelss,batch_size=128)    # replace Y_train,Y_test with train_labelss, test_labelss after creating them special way
print(score)  # >>>>> 84.58% with Y_train and test but ajeeb with train_labelss

#also try with 'adam' as the optimizer (SHOULD IMPROVE ACCURACY)




##### >>>> Knearest Neighbor Classifier >>>>NOt robust thus bad accuracy

#clf=KNeighborsClassifier(n_neighbors=24)
#knnmodel=clf.fit(train_data,train_label_encoded)
#y_to_predict=clf.predict(test_data)
#
#acc=accuracy_score(test_label_encoded,y_to_predict)
#print(acc)
# 87%
#
#clf=KNeighborsClassifier(n_neighbors=24)
#knnmodel=clf.fit(train_data,train_labelss)
#y_to_predict=clf.predict(test_data)
#
#acc=accuracy_score(test_labelss,y_to_predict)
#print(acc)
#83.68%


#clf=KNeighborsClassifier(n_neighbors=24)
#knnmodel=clf.fit(train_data,Y_train)
#y_to_predict=clf.predict(test_data)
#
#acc=accuracy_score(Y_test,y_to_predict)
#print(acc)
##83.68%

#  TRAINING A SUPPORT VECTOR CLASSIFICATION

import numpy as np
import pandas as pd
import time

print("Number of features in Train: ",train.shape[1]) # 1 is for columns
print("Number of features in Train: ",train.shape[0])
print("Number of features in Test: ",test.shape[1])
print("Number of features in Test: ",test.shape[0])

trainData=train.drop(['subject','Activity'],axis=1).values
trainLabel=train.Activity.values

testData=test.drop(['subject','Activity'],axis=1).values
testLabel=test.Activity.values

print("Train Data shape:",trainData.shape)
print("Train Label Shape: ",trainLabel.shape)
print("Test Data shape:",testData.shape)
print("Test Label Shape: ",testLabel.shape)

print("Label Examples: ")
print(np.unique(trainLabel))


from sklearn import preprocessing
from sklearn import utils

ltrain=preprocessing.LabelEncoder()
ltest=preprocessing.LabelEncoder()

trainLabel=ltrain.fit_transform(trainLabel)
testLabel=ltest.fit_transform(testLabel)

print(np.unique(trainLabel))
print(np.unique(testLabel))
print("Train Label Shape: ",trainLabel.shape)
print("Test Label Shape: ",testLabel.shape)

print(utils.multiclass.type_of_target(testLabel))



import matplotlib.pyplot as plt
from sklearn.svm import SVC
# if inside a backend frontend situation like jupyter use %matplotlib inline
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.utils import shuffle

t0=time.clock()
# Create a RFE object and compute a cross vlidated SCORE

svc=SVC(kernel='linear')

# Accuracy scoring is basically proportional to the number of correct classifications

rfecv=RFECV(estimator=svc,step=1,cv=StratifiedKFold(6),scoring='accuracy')

# Before training the data it is good to shuffle the data

np.random.seed(1) #Basically to make the random numbers predicatble every time and unchanged
# BAsically it makes shuffle return the repeated items everytime.

print("Labels before shuffle:",testLabel[0:5])
testData,testLabel=shuffle(testData,testLabel)
trainData,trainLabel=shuffle(trainData,trainLabel)
print("Labels after Shuffle: ",trainLabel[0:5])

# train and Fit the data in the model

rfecv.fit(trainData,trainLabel)

print(" Optimal number of features: %d "% rfecv.n_features_)
print("Processing time",time.clock()-t0)

# Plotting number of features vs Cross validation Score

plt.figure(figsize=(32,24))
plt.xlabel("Number of features selected")
plt.ylabel("Cross Validation Score (number of correct classifications)")
plt.plot(range(1,len(rfecv.grid_scores_)+1),rfecv.grid_scores_)
plt.show()
 
print("Accuracy of the svm model on test data is :",rfecv.score(testData,testLabel))  #### 87.78%
print("Ranking of the features starting from Best first \n",rfecv.ranking_)

##### Tryting to get the  best features to look for any cross correlation
# masking the features to get only the best one
best_features=[]
for ix,value in enumerate(rfecv.support_):                         # print(rfecv.support_)  To see what it returns
    if value==True:
        best_features.append(testData[:,ix])
        
        
from pandas.tools.plotting import scatter_matrix
visualize=pd.DataFrame(np.asarray(best_features).T)
print(visualize.shape)
scatter_matrix(visualize.iloc[:,0:5],alpha=0.2,figsize=(16,16),diagonal='kde') # Only for 5 features presently

##### TENSOR FLOW MODEL #############

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer  # again basically label encoding in matrix format
from sklearn.utils import shuffle

train=shuffle(train)
test=shuffle(test)

train_features=train.iloc[:,:562].as_matrix()   #####Check if the sunject column needs to removed too as it is not worth
test_features=test.iloc[:,:562].values        ## as matrix and alues is same

binarizer=LabelBinarizer().fit(train["Activity"])
train_labels=binarizer.transform(train.Activity)
test_labels=binarizer.transform(test.Activity)

print(train_labels.shape)

### Defining the custom fucntions

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

def add_layer(inputs,input_size,output_size,activation=None):
    W=weight_variable([input_size,output_size])
    b=bias_variable([output_size])
    wxb=tf.matmul(inputs,W)+b
    
    if activation:
        return activation(wxb)
    
    else:
        return wxb
    
X=tf.placeholder(tf.float32,[None,562])
layer1=add_layer(X,562,1000,tf.nn.relu)
layer2=add_layer(layer1,1000,300,tf.nn.relu)
layer3=add_layer(layer2,300,50,tf.nn.relu)
output=add_layer(layer3,50,6)

y_=tf.placeholder(tf.float32,[None,6])


loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=output,logits=y_))
optimizer=tf.train.GradientDescentOptimizer(0.001)
train_step=optimizer.minimize(loss)

correct=tf.equal(tf.argmax(output,1),tf.argmax(y_,1))
score=tf.reduce_mean(tf.cast(correct,"float"))

sess=tf.Session()
init=tf.initialize_all_variables()
sess.run(init) 
   
for i in range(10000):
    batch=np.random.choice(train_features.shape[0],100)
    _,cost=sess.run([train_step,loss],feed_dict={X:train_features[batch],y_:train_labels[batch]})

print(sess.run(score,feed_dict={X: test_features,y_:test_labels}))

######>>>>>> loss value =0.1831 >>>>> accuracy approx 82%


#### KNN classifier 

trainLabels=train.Activity.values
trainData=train.drop("Activity",axis=1).values

testLabels=test.Activity.values
testData=test.drop("Activity",axis=1).values

print("Class labels stripped off")

# Transforming the non numerical labels to numerical labels using label encoding
from sklearn import preprocessing
le=preprocessing.LabelEncoder()

trainlabelsE=le.fit_transform(trainLabels)
testlabelsE=le.fit_transform(testLabels)

print("Labels encoded and transformed")

### >> APPLYING K NEAREST NEIGHBOURS

from sklearn.neighbors import KNeighborsClassifier as knn
import numpy as np

knnScoreDistance=np.zeros(51)
knnScoreUniform=np.zeros(51)

for num in range(5,51):
    knnclf=knn(n_neighbors=num,weights='distance')
    knnModel=knnclf.fit(trainData,trainlabelsE)
    knnScoreDistance[num]=knnModel.score(testData,testlabelsE)
    print("Testing set score for KNN_Distance(k=%d): %f" %(num,knnScoreDistance[num]))
    
for num in range(5,51):
    knnclf=knn(n_neighbors=num,weights='uniform')
    knnModel=knnclf.fit(trainData,trainlabelsE)
    knnScoreUniform[num]=knnModel.score(testData,testlabelsE)
    print("Testing set score for KNN_Uniform(k=%d): %f" %(num,knnScoreUniform[num]))
    
##### PLOTTINMG THE STUFF

import matplotlib.pyplot as plt
x=np.array(range(5,51))

plt.plot(x,knnScoreDistance[5:])
plt.plot(x,knnScoreUniform[5:])
plt.xlabel("No of Neighbors considered")
plt.ylabel("Test Data Accuracy")
plt.legend(['KNN Distance','KNN Unifrom'])
plt.show() 




######>>>>>CONFUSION MATRIX PLOTTING UNDERSTAND LATER
import itertools
import numpy as np
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

### >> USING DECISION TREE CLASSIFIER

from sklearn.tree import DecisionTreeClassifier              ########77.28%
from sklearn.metrics import accuracy_score,confusion_matrix

dtc=DecisionTreeClassifier(criterion='entropy')
tree=dtc.fit(trainData,trainlabelsE)
testPred=tree.predict(testData)

acc=accuracy_score(testlabelsE,testPred)
cfs=confusion_matrix(testlabelsE,testPred)

print("Accuracy: %f" %acc)

plt.figure()

class_names=le.classes_
plot_confusion_matrix(cfs,classes=class_names,title='DECISON TREE CONFUSION MATRIX, without Normalization')

##### USING DTC again but this time without any criteria


dtc=DecisionTreeClassifier()   ##### 77.78%
tree=dtc.fit(trainData,trainlabelsE)
testPred=tree.predict(testData)

acc=accuracy_score(testlabelsE,testPred)
cfs=confusion_matrix(testlabelsE,testPred)

print("Accuracy: %f" %acc)

plt.figure()

class_names=le.classes_
plot_confusion_matrix(cfs,classes=class_names,title='DECISON TREE CONFUSION MATRIX, without Normalization')


######  MULTIPLE CLASSIFIERS!!!

X,y=train_df.iloc[:,0:len(train_df.columns)-1],train_df.iloc[:,-1]
X_test,y_test=test_df.iloc[:,0:len(test_df.columns)-1],test_df.iloc[:,-1]

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score  ####For evaluation

classifiers=[DecisionTreeClassifier(),
             KNeighborsClassifier(7),
             SVC(),GaussianNB(),
             QuadraticDiscriminantAnalysis()]

names=[]
scores=[]

for clf in classifiers:
    clf=clf.fit(X,y)
    y_pred=clf.predict(X_test)
    
    names.append(clf.__class__.__name__)
    scores.append(accuracy_score(y_pred,y_test))
    
score_df=pd.DataFrame({'Model':names,'Scores': scores}).set_index('Model')
print(score_df)

### PLOTTING THE CLASSIFIER RESULTS

import matplotlib.pyplot as plt
ax=score_df.plot.bar()
ax.set_xticklabels(score_df.index,rotation=75,fontsize=10)

##### GRID SEARCH FOR SVC AS IT TURNED OUT TO BE THE BEST MODEL

from sklearn.model_selection import GridSearchCV
parameters={'kernel':['linear','rbf'],
            'C':[100,20,1,0.1]}
selector=GridSearchCV( SVC(),parameters,scoring='accuracy')
selector.fit(X,y)

print("Best parameter set found:")
print(selector.best_params_)
print("Detailed Grid Scores:")

means=selector.cv_results_['mean_test_score']
stds=selector.cv_results_['std_test_score']

for mean,std,params in zip(means,stds,selector.cv_results_['params']):
    print('%0.3f (+/- %0.3f) for %r' %(mean,std*2,params))
    print()


clf=SVC(kernel='linear',C=100).fit(X,y)
y_pred=clf.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test,y_pred))

##### >>>>> 87.38%




























                                                                                            
    
    













































