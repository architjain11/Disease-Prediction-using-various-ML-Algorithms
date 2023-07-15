<a name="br1"></a> 

Date of current version December 14, 2022.

**Disease Prediction by Juxtaposition of Multiple**

**Machine Learning Models**

<b>Archit Jain<sup>1</sup>, Faizan Ahmed Mohammed<sup>1</sup>, and Kapil Singh Baghel<sup>1</sup></b>

<sup>1</sup>Computer Engineering undergraduate at Netaji Subhas University of Technology, Dwarka

**ABSTRACT** The field of healthcare has a vast application of machine learning in this era. Traditional ways of diagnosing a

disease can only consider a limited number of factors, but with the upcoming technologies, we can consider thousands of

factors in the matter of seconds. Machine learning helps in making data-driven decisions relating to key trend and driving

research efficiency. We find the highest accuracy of prediction in case of multilayer perceptron with ReLU activation.

**1. INTRODUCTION**

a) We read the data and visualize it using bar plot. We

also check for missing values and handle, if any.

b) Original dataset is then split into two sets- training

and testing.

c) Our model is trained using training set. The

accuracy of each model is scored on the testing set

by manipulating the hyperparameters of each.

d) We plot confusion matrix using heatmap and use

ROC curve (receiver operator characteristic curve)

to find AUC (area under curve).

With advancements in the medical field, world class

treatment is reaching the patients and is saving lives.

Machine learning is playing a central role in the same with

its own significant developments easing its application.

Machine learning algorithms can make predictions after

considering many variables unlike traditional methods of

diagnosing which has a limit on the factors that can be

considered [11].

The dataset used in this report has symptoms represented by

binary 0 or 1 signifying whether that symptom is observed

or not and based on those values we form a prognosis.

e) We compare accuracy and AUC values of different

models to find the best one [6].

f) We display the predictions made by the best model

on the test set.

We follow various steps to reach our conclusion which are

depicted by the flowchart shown in Fig 1 below:

The major challenge in this field is extracting appropriate

information from our dataset and using them in an efficient

way so that we can handle cases of underfitting and

overfitting. This is also a sensitive field; medical conditions

should not be completely reliant on machine learning at its

current level but it serves as an efficient first or early

indicator of a potential disease. This facility would also

significantly lower disease detection costs and make it more

widely available to the public. In this report, we compare the

performance of various models on our dataset and check

which model suits best for our purposes.

**2. RELATED WORK**

This section will discuss some of the existing works on

different machine learning techniques related to disease

prediction.

There are many factors which are related to cause and

treatment of diabetes disease such as age, health history,

obesity, immune system strength, weakness, and many more

[1]. The objective of this study is to implement different

model to also find the best fit with significant accuracy to

diagnose diseases such as diabetes in patients [7]. Various

machine learning algorithms are also utilized for early-stage

*Fig 1- Flow of steps to compare ML models*

**pg. 1**



<a name="br2"></a> 

diabetes detection which include Random Forest Classifier **3. THEORY**

[1], Support Vector Machine [4], Decision Trees [7],

Logistic Regression,[5] K-Nearest Neighbors [3], Naïve 3.1 Machine Learning Models

Bayes Classifier [10], Multi-Layer Perceptron (MLP) [8].

**1. Random Forest Classifier**

Currently the approach to predict cardiovascular risk to not Random Forest [3] is a frequently used ML algorithm

so appropriate, Machine-learning gives us the chance to belonging to the class of supervised learning techniques.

improve accuracy by taking advantage of the complicated This model can be used for both types of problems-

relationship between risk factors [2]. This paper [2] studies Classification and Regression.

24,970 incident cardiovascular events (6.6%) occurred and As show in Fig 2, we observe multiple decision trees on

compared the different model with their accuracy like multiple subparts of a given dataset. Each decision tree

random forest +1.7%, logistic regression +3.2%, and neural below predicts an output (accuracy is predicted in our

networks +3.6%. The best performing algorithm was neural report). Then we take their average to improve the accuracy

networks which predicted a total of 4,998/7,404 cases of the overall prediction from the dataset.

(sensitivity 67.5%) and 53,458/75,585 non-cases (specificity

70\.7%), correctly predicting 355 (+7.6%) [2].

Focus is on implementation and study of performance by

models such as Naive Bayes, K-Nearest Neighbor (KNN)

and Random Forest classifier [1] based on the accuracy and

preciseness for chronic kidney disease or CKD prediction

[3][6]. The result of conducting the research is that the

performance of Random Forest classifier is relatively better

than both Naive Bayes and KNN [3].

The purpose of implementation of support vector machine

(SVM) is to develop decision support system to diagnosis

kidney disease patient [4]. Also, this paper focuses on

*Fig 2- Working of the Random Forest ML model*

methodology which consist of classification modelling and **2. Multinomial Naïve Bayes Classifier**

system development. Steps involved in a classification The Multinomial Naïve Bayes classifier [3] is used in the

model consists of data collection, its preparation and case of multinomial distributed data. This classifier assumes

grouping, and then finally classification [6]. The study no dependency between attributes ie. all attributes are

resulted in a trained model which can detect a chronic considered independent. This algorithm uses the concept of

condition of kidney disease based on several factors on SVM conditional independence with the assumption that the

with an outstanding accuracy of 98.34% [4].

attribute value of a given class is independent from the

values in other attributes. It is primarily used for document

Individual patient survival often depends on a complicated classification. The prediction is based on the conditional

relationship between multiple variables like symptoms of probability of an object as discussed below [10].

kidney failure, causes of kidney disease [4], medications,

and their interventions in case of Kidney Disease Prediction

[10]. Three data mining techniques (Artificial Neural

Networks (ANN), Decision tree and Logical Regression [5])

are used to evaluate the interaction between these variables

and the rate of patient’s survival. The performance

comparison of three of them are studied for extracting

knowledge in the form of classification rules from the data.

[5][6].

(푃(D|h . 푃 ℎ )

)

(

)

푃(ℎ|퐷) =

(*Eqn 1)*

푃(퐷)

P(h|D) is known as the posterior probability:

Conditional probability of the hypothesis h on

observed data D.

P(D|h) is known as the likelihood probability: It is the

probability of data D given that probability of a

hypothesis h being true.

Clinical decision support systems have also been installed

that combine various data mining techniques for prediction

of diseases such as diabetes and study its progression and

performance to various techniques on dataset [6].

P(h) is Prior Probability: Probability of hypothesis h

before the data is observed.

P(D) is Marginal Probability: Probability of given

evidence or data.

The above discussed studies give us a good insight on the

implementation of data mining into healthcare and the aim

of this study is to give a valuable cumulative insight as well.

**pg. 2**



<a name="br3"></a> 

**3. Logistic Regression**

available data. The new data blend to that cluster to which it

Logistic regression is a widely used algorithm whose main gets most resemblance. Fig 5 below shows the KNN

purpose is to predict the categorical dependent variable algorithm in a graph.

using given independent variables [5]. Calculated

probabilistic values lie in the range 0 and 1. In the figure

below the y value 0.8 indicates that the probability of that

event occurring is 80%. It uses a sigmoid function to plot on

graph as shown in Fig 3.

*Fig 5- Graphical representation of KNN Algorithm*

When a new data point is entered, the algorithm selects K

neighbors and calculates their Euclidean distance between

them, also, counting the number of datapoints in each of the

given categories. Then it assigns a the category to the new

datapoint for whom the number of neighbors is in majority.

**6. Support Vector Classifier**

Support Vector Machine algorithm [4] is one of the most

popularly used algorithms in the category of supervised

machine learning. It could be used for both applications-

*Fig 3- Graphical representation of Logistic Regression*

Regression and Classification.

푦

log [ ] = 푏 + 푏 푥 + 푏 푥 + ⋯ + 푏 푥 *(Eqn 2)*

0

1

1

2

2

푛

푛

1−푦

**4. Decision Tree Classifier**

It is an easy to implement, simple and widely used classifier.

High dimensional data can be efficiently handles and it

doesn’t require any previous domain knowledge or

parameter setting [7]. The results produced by our classifier

are easily interpretable and readable because of its

flowchart-like nature.

*Fig 6- Understanding of Support Vector Algorithm*

The main goal of Support Vector Algorithm is creating

a best fit decision boundary line to segregate n-

dimensional space into separate classes which will

help us easily classify the new unseen data point in its

correct category. This best decision boundary is called

hyperplane which helps to classify data points whereas

support vectors are datapoints closest to hyperplane.

*Fig 4- Model of a Decision Tree*

It basically lists out a set of all possible outcomes which

could later lead to more outcomes. In the above figure, the

node which cannot be further broken into more noes is

known as the leaf node whereas a decision node could be

broken into many other nodes.

**7. Multi-Layer Perceptron (Neural Network)**

**5. K-Nearest Neighbors**

Multi-Layer Perceptron (MLP) is one of the most

complicated approaches in supervised learning. It deals with

many layers of input nodes to give an output data. MLP has

KNN [3] is one of the simplest algorithms that comes under

the category of supervised learning. The working principle

of this algorithm is how well new data is resembled to

**pg. 3**



<a name="br4"></a> 

many layers which are interconnected to each other and with

this multilayered structure, we get our desired output. [8].

because of its better performance facilitated

by its easy to train perk.

The equation is given by,

푓(푥) = max(0, )

*(Eqn– 4)*

*Fig 7- Multi-Layer perceptron (Neural Network)*

As we can see, many nodes are connected to each

other forming a very complex structure of data. Such a

network of nodes can only be achieved from

supervised learning approach.

*Fig 9- ReLU Activation Function graph*

3\.2 Evaluation Metrics

A MLP has one input layer for each input. Between

output layer and input layer, there can as many as

hidden layers and nodes as needed to get the desired

output.

When creating a model or a group of models it is also

important to make sure that they are working sufficiently

well. For this we use evaluation metrics. The efficiency of

various models is evaluated using various metrics. The

evaluation metrics that we would be using are confusion

matrix and ROC curve and its AUC [6].

a) **Logistic Activation Function**

In this algorithm, we measure the dependent

and independent features of a dataset.

**1. Confusion Matrix**

퐿

A confusion matrix is a metric to determine the performance

of our classifier but it can only be determined for situations

where the actual values for test data are known as well. It is

also known as an error matrix.

푓(푥) =

*(Eqn- 3)*

1+푒−푘푥

ACTUAL VALUES

**Positive**

**Negative**

**Positive**

True Positive

(TP)

False Positive

(FP)

**Negative**

False Negative

(FN)

True Negative

(TN)

*Table 1- Confusion Matrix Representation*

The calculations that can performed using a confusion

matrix are:

*Fig 8- Logistic Activation Function graph*

Input is any real value and the output ranges

between 0 and 1. Output is dependent on the

input i.e., the greater the input the closer is

the output to 1.

**Accuracy** – Defines how correctly our model is working in

predicting the right class.

푇푃+푇푁

퐴푐푐푢푟푎푐푦 =

*(Eqn- 5)*

퐹푃+퐹푁+푇푃+푇푁

b) **Rectified Linear Unit**

**Error Rate** – Defines how frequently our model gives

The rectified linear unit or ReLU for short is

an activation function which is a linear

function that will give the input itself as the

output if positive and zero otherwise. This

model is used as the default activation

function for many types of neural networks

inaccurate predictions.

퐹푃+퐹푁

퐸푟푟표푟 푅푎푡푒 =

*(Eqn- 6)*

퐹푃+퐹푁+푇푃+푇푁

**pg. 4**



<a name="br5"></a> 

**Precision** - Defined as ratio of the correctly identified in the

positive class to the total positive identified in the positive

there are multiple missing values, we can even

consider deleting the entire row.

class.

2\. **Imputing missing values**

푇푃

푃푟푒푐ꢀ푠ꢀ표푛 =

*(Eqn-7)*

We can make a calculated guess about the missing

value and replace the missing value in our column

with the same.

푇푃+퐹푃

**Recall** – Tells us how many were predicted correctly by our

model out of the total positive classes.

.

a. Replacing with mean- This is most used

for numeric columns.

푇푃

푅푒푐푎푙푙 =

*(Eqn- 8)*

푇푃+퐹푁

b. Replacing with mode- Most occurring

value is imputed.

**2. ROC Curve**

A receiver operating characteristic curve, or ROC curve, is a

graph that gives us the performance of our model at all

possible threshold values. The curve is plotted as TPR vs

FPR at different thresholds of classification.

c. Replacing with median- The middlemost

value is used for imputing.

Thus, the curve consists of two parameters to plot:

d. Most frequent value imputed for

categorical data

a. True Positive Rate (TPR)

b. False Positive Rate (FPR)

e. Impute the value “missing”, which will be

considered separately.

4\.3 Model Training

**True Positive Rate (TPR)** is a same as recall and can be

We do implementation of various models to evaluate our

dataset [2]. While doing so we use various performance

metrics [6] for evaluation such as confusion matrix, AUC of

ROC curve along with the accuracy of model performance is

noted for comparison.

expressed as shown below:

푇푃

푇푃푅 =

*(Eqn- 9)*

푇푃+퐹푁

**False Positive Rate** (**FPR**) can be expressed as shown

below:

The basic steps involved in implementing any model are:

1\. Choosing the correct machine learning model.

퐹푃

퐹푃푅 =

*(Eqn- 10)*

퐹푃+푇푁

2\. Training the model on our training dataset.

**AUC** is short for "Area under the ROC Curve.” AUC

is a measure of the entire area under the ROC curve

from (0,0) to (1,1).

3\. Evaluating the trained model using various

evaluation metrics such as accuracy, or confusion

matrix, or area under ROC curve.

**4. METHODOLOGY**

Often the parameters of the model need to be tuned to handle

the cases of overfitting and underfitting to give us the most

ideal set of tuned parameters which help us fit our data well

onto our model.

4\.1 Dataset preparation

The dataset used here is available on Kaggle. It includes

4920 rows and 133 columns. All the columns are symptoms

with a binary value of 0 or 1 representing whether that

symptom is observed in that data value or not. Then we have

a prognosis of the disease along with it. As in our case,

classification modelling will consist of data collection,

preparation, grouping, classification, and extraction rules

[4].

The unseen data or new data (symptoms of a new patient, in

our case) are fed to the trained model to make a prediction.

For the model showing the highest accuracy (Multilayer

Perceptron, ReLU activation), we can see the evaluation

metrics as illustrated below:

4\.2 Data pre-processing and cleaning

**ACCURACY**

First, we need to pre-process our dataset, that is, turn it into

usable format [12]. We check for missing values, and find

none. Had we found missing values, we can handle them in

two ways:

mlp2=MLP(hidden\_layer\_sizes=(50,),activation='relu',max\_iter=12)

mlp2.fit(x\_train, y\_train)

accuracy = mlp2.score(x\_test, y\_test)

global\_accuracy['Multilayer Perceptron ReLU Activation']=accuracy

print('Accuracy is: ', end='')

print(accuracy)

1\. **Deleting missing values**

Accuracy is: 1.0

If missing value is of type MNAR (missing not at

random), then do not delete it. If of type MCAR

(missing completely at random), then delete it. If

*(Code sample #1)*

**pg. 5**



<a name="br6"></a> 

**CONFUSION MATRIX**

y\_pred=mlp2.predict(x\_test)

cm = confusion\_matrix(y\_test, y\_pred)

cm\_df = pd.DataFrame(cm)

plt.figure(figsize=(10,10))

sns.heatmap(cm\_df, annot=True, cmap='Blues')

plt.title('Confusion Matrix')

plt.ylabel('Actal Values')

plt.xlabel('Predicted Values')

plt.show()

4\.4 Results and Evaluation

MODEL

ACCURACY

0\.95443

0\.96613

0\.95135

0\.96182

0\.98830

0\.96490

0\.96490

AUC

Logistic Regression

Support Vector Machine

Multinomial Naïve Bayes

K-Nearest Neighbors

Random Forest Classifier

Decision Tree

1\.00000

1\.00000

0\.99976

1\.00000

0\.99983

0\.99949

0\.99865

Multilayer Perceptron

(Logistic Activation)

Multilayer Perceptron

(ReLU Activation)

1\.00000

1\.00000

*(Code sample #2)*

*Table 2- Accuracy and AUC values of various models*

The predictions as observed by ReLU activation is seen

below:

y\_pred = mlp2.predict(x\_test)

pd.DataFrame({'Actual': y\_test, 'Predicted': y\_pred})

**ROC CURVE**

plt.figure(figsize=(8,8))

y\_bin=label\_binarize(y\_test,classes=np.unique(y\_test))

false\_pos\_r={}

true\_pos\_r={}

th={}

auc\_val={}

pred\_prob=mlp2.predict\_proba(x\_test)

unique\_cl=np.unique(y\_test)

for i in range(len(unique\_cl)):

false\_pos\_r[i],true\_pos\_r[i],th[i]=roc\_curve(y\_bin[:,i],pred\_prob[:,i])

auc\_val[i]=auc(false\_pos\_r[i],true\_pos\_r[i])

plt.plot(false\_pos\_r[i],true\_pos\_r[i])

plt.xlabel("False Positive rate")

plt.ylabel("True Positive rate")

**Actual**

Acne

**Predicted**

Acne

**373**

**4916**

**1550**

**3081**

Acne

Acne

Hyperthyroidism

AIDS

Hyperthyroidism

AIDS

print('Average area under curve is: ', end='')

avg\_auc=statistics.mean(list(auc\_val.values()))

global\_auc\_val['Multilayer Perceptron ReLU Activation']=avg\_auc

print(avg\_auc)

**3857** Chronic cholestasis Chronic cholestasis

**...**

...

GERD

...

GERD

plt.legend([i for i in unique\_cl], bbox\_to\_anchor=(1, 1))

plt.show()

**1257**

**3346**

**3384**

**3290**

**1178**

Tuberculosis

Hepatitis D

Hypertension

Arthritis

Tuberculosis

Hepatitis D

Hypertension

Arthritis

1624 rows × 2 columns

*(Code sample #4)*

*(Code sample #3)*

**pg. 6**



<a name="br7"></a> 

**5. CONCLUSION**

**6. REFERENCES**

In our report, we compared different algorithms such as [1] Palimkar, Prajyot, Rabindra Nath Shaw, and Ankush

Ghosh. "Machine learning technique to prognosis diabetes

disease: random forest classifier approach." In Advanced

Computing and Intelligent Technologies, pp. 219-244.

Springer, Singapore, 2022.

Logistic Regression, Random Forest, Naïve Bayes, KNN,

Support Vector Machine (SVM), Decision Tree, and

Multilayer Perceptron based on AUC of its ROC curve and

accuracy evaluation metrics.

[2] Weng, Stephen F., Jenna Reps, Joe Kai, Jonathan M.

Garibaldi, and Nadeem Qureshi. "Can machine-learning

improve cardiovascular risk prediction using routine clinical

We observe that Multilayer Perceptron Model using ReLU

activation shows highest accuracy of 1.0 ie. 100% correct

predictions on our test dataset along with high area under data?." PloS one 12, no. 4 (2017): e0174944.

curve. One reason of that could be high number of columns

[3] Devika, R., Sai Vaishnavi Avilala, and V.

(which have symptoms) that need to be looked at and neural

networks perform well for those situations unlike decision

trees which need to make a decision at every column which

increases the depth of our tree (if depth is less, accuracy

suffers and if depth is more, data overfits in our tree).

Subramaniyaswamy. "Comparative study of classifier for

chronic kidney disease prediction using naive bayes, KNN

and random forest." In 2019 3rd International conference on

computing methodologies and communication (ICCMC), pp.

679-684. IEEE, 2019.

It was also observed that our dataset was uniform with same

[4] Ahmad, Mubarik, Vitri Tundjungsari, Dini Widianti,

number of tuples for each disease as was evident from the Peny Amalia, and Ummi Azizah Rachmawati. "Diagnostic

decision support system of chronic kidney disease using

support vector machine." In 2017 second international

conference on informatics and computing (ICIC), pp. 1-4.

IEEE, 2017.

bar plot. Manipulating the hyperparameters in different

models was showing that the data was fitting very well, so,

we can say that the differences between the observed values

and the model's predicted values are small and unbiased.

[5] Lakshmi, K. R., Y. Nagesh, and M. Veera Krishna.

"Performance comparison of three data mining techniques

for predicting kidney dialysis survivability." International

In some cases, we also observe that the AUC value is large,

whereas accuracy is relatively lower. One reason why this

might happen is if our classifier is achieving its good Journal of Advances in Engineering & Technology 7, no. 1

performance on the positive class (or high AUC) at the (2014): 242.

expense of high false negative rate (or high FNR). That is,

[6] Perveen, Sajida, Muhammad Shahbaz, Aziz Guergachi,

the ROC analysis tells us something about how well the

positive class sample is separated from other classes,

whereas the prediction accuracy gives us a hint on the actual

performance of the classifier.

and Karim Keshavjee. "Performance analysis of data mining

classification techniques to predict diabetes." Procedia

Computer Science 82 (2016): 115-121.

[7] Orabi, Karim M., Yasser M. Kamal, and Thanaa M.

Rabah. "Early predictive system for diabetes mellitus

disease." In Industrial Conference on Data Mining, pp. 420-

427\. Springer, Cham, 2016.

The Colab Notebook for entire implementation along with

dataset is available at the link below:

[*https://drive.google.com/drive/folders/1vlzjAbCLErwC3C-*](https://drive.google.com/drive/folders/1vlzjAbCLErwC3C-rDfQW6clUXVxyDxQC?usp=sharing%20)

[*rDfQW6clUXVxyDxQC?usp=sharing*](https://drive.google.com/drive/folders/1vlzjAbCLErwC3C-rDfQW6clUXVxyDxQC?usp=sharing%20)

[8] Zhang, Hanyu, Che-Lun Hung, William Cheng-Chung

Chu, Ping-Fang Chiu, and Chuan Yi Tang. "Chronic kidney

disease survival prediction with artificial neural networks."

In 2018 IEEE International Conference on Bioinformatics

and Biomedicine (BIBM), pp. 1351-1356. IEEE, 2018.

5\.1 Future Scope

There are different techniques which can be further applied

to improve the performance even more [9] to facilitate

effective and early detection of diseases.

[9] Singh, Smriti Mukesh, and Dinesh B. Hanchate.

"Improving disease prediction by machine learning." Int. J.

Res. Eng. Technol 5 (2018): 1542-1548.

1\. A larger, more diverse dataset can improve the

factors being taken into consideration unlike

traditional approach which is limited in those terms.

2\. Combining models can improve accuracy and

efficiency covering for each other’s limitations and

giving a more reliable outcome.

[10] Dulhare, Uma N., and Mohammad Ayesha. "Extraction

of action rules for chronic kidney disease using Naïve bayes

classifier." In 2016 IEEE International Conference on

**pg. 7**



<a name="br8"></a> 

Computational Intelligence and Computing Research and Aerospace Technology (ICECA), 2018, pp. 910-914,

(ICCIC), pp. 1-5. IEEE, 2016. doi: 10.1109/ICECA.2018.8474918.

[11] K. Shailaja, B. Seetharamulu and M. A. Jabbar, [12] S. Mohan, C. Thirumalai and G. Srivastava, "Effective

"Machine Learning in Healthcare: A Review," 2018 Second Heart Disease Prediction Using Hybrid Machine Learning

International Conference on Electronics, Communication Techniques," in *IEEE Access*, vol. 7, pp. 81542-81554,

2019,

doi:

10\.1109/ACCESS.2019.2923707

**pg. 8**

