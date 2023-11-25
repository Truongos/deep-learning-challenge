# deep-learning-challenge

Report:
For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.

The report should contain the following:

Overview of the analysis: Explain the purpose of this analysis.

Results: Using bulleted lists and images to support your answers, address the following questions:

Data Preprocessing

What variable(s) are the target(s) for your model?
What variable(s) are the features for your model?
What variable(s) should be removed from the input data because they are neither targets nor features?
Compiling, Training, and Evaluating the Model

How many neurons, layers, and activation functions did you select for your neural network model, and why?
Were you able to achieve the target model performance?
What steps did you take in your attempts to increase model performance?
Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.




Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

Requirements
-Preprocess the Data (30 points)
-Create a dataframe containing the charity_data.csv data , and identify the target and feature variables in the dataset (2 points)
-Drop the EIN and NAME columns (2 points)
-Determine the number of unique values in each column (2 points)
-For columns with more than 10 unique values, determine the number of data points for each unique value (4 points)
-Create a new value called Other that contains rare categorical variables (5 points)
-Create a feature array, X, and a target array, y by using the preprocessed data (5 points)
-Split the preprocessed data into training and testing datasets (5 points)
-Scale the data by using a StandardScaler that has been fitted to the training data (5 points)
-Compile, Train and Evaluate the Model (20 points)
-Create a neural network model with a defined number of input features and nodes for each layer (4 points)
-Create hidden layers and an output layer with appropriate activation functions (4 points)
-Check the structure of the model (2 points)
-Compile and train the model (4 points)
-Evaluate the model using the test data to determine the loss and accuracy (4 points)
-Export your results to an HDF5 file named AlphabetSoupCharity.h5 (2 points)
-Optimize the Model (20 points)
-Repeat the preprocessing steps in a new Jupyter notebook (4 points)
-Create a new neural network model, implementing at least 3 model optimization methods (15 points)
-Save and export your results to an HDF5 file named AlphabetSoupCharity_Optimization.h5 (1 point)
-Write a Report on the Neural Network Model (30 points)
-Write an analysis that includes a title and multiple sections, labeled with headers and subheaders (4 points)
-Format images in the report so that they display correction (2)
-Explain the purpose of the analysis (4)
-Answer all 6 questions in the results section (10)
-Summarize the overall results of your model (4)
-Describe how you could use a different model to solve the same problem, and explain why you would use that model (6)
