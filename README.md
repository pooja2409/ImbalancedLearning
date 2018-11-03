# ImbalancedLearning<br>
Qn: Briefly describe the conceptual approach you chose! what are the trade-offs?<br>
Ans: The dataset is quite tricky. At first I performed a data analysis on the dataset which shows that the data is highly imbalanced and needs to be pre processed before use.<br>
if you dont balance the dataset and fit a model into this ver same dataset it will give an accuracy of 96%. Yes Thats because it will predict all the dataset as class 0, as the majority of dataset is 0.<br>
<br>
So we first need to balance the dataset, I did that by using the imblearn package of python. Which allows to oversample or undersample the data so that we can have a balanced dataset.<br>
Oversampling and undersampling in data analysis are techniques used to adjust the class distribution of a data set.
Now after oversampling the ratio of the dataset is 1.Now we can fit a machine learning model into this data and predict the values.<br>
Tree boosting has empirically proven to be efficient for predictive mining for both classification and regression.<br>
Here I have use XGBoost algorithm for classification of the dataset.<br>
Clearly accuracy cannot be a measure to check if the model is working fine or not in this case. so I used Recall and F1 score for measuring the model performance.
<br>
Qn: What's the model performance? What is the complexity? Where are the bottlenecks?<br>
Ans: The accuracy of the model is 60% and the recall is close to 0.50.<br>
The original sparse greedy algorithm doesn't use block storage. Thus to find the optimal split at each node, you needed to re-sort the data on each column. This ends up incurring a time complexity at each layer that is very crudely approximated by O(‖x‖0logn): basically, say you have ‖x‖0i nonzero entries for each feature 1≤i≤m; then at each layer you're sorting lists, each of length at most n, whose lengths sum to ∑mi=1‖x‖0i=‖x‖0 which can't take more than O(‖x‖0logn) time. Multiplying by K trees and d layers per tree gives you the original O(Kd‖x‖0logn) time complexity. The most time consuming part of the tree learning algorithm is getting the data in sorted order. This makes the time complexity of learning each tree O(n log n).<br>

Qn: If you had more time, what improvements would you make, and in what order of priority?<br>
Ans: If given more time I would try to study more about handling imbalanced dataset. using Cost sensitive learning.
A common scheme for this is to have the cost equal to the inverse of the proportion of the data-set that the class makes up. This increases the penalization as the class size decreases.
