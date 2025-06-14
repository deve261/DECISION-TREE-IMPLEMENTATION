# DECISION-TREE-IMPLEMENTATION

COMPANY: CODTECH IT SOLUTIONS

NAME: YELLANKI DEVENDRA

INTERN ID: CT04DG224

DOMAIN: MACHINE LEARNING

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH

PROJECT DESCRIPTION: Decision Tree Classifier with Interactive User Interface
This project is a user-friendly desktop application designed to streamline the process of building, training, and visualizing decision tree classifiers using a graphical user interface (GUI). Developed in Python with the help of libraries such as Tkinter, Pandas, Scikit-learn, and Matplotlib, the application provides an intuitive way for users—especially those without a programming background—to interact with machine learning models.

At its core, the application enables users to import a CSV dataset, select a target column for classification, set a test size for model evaluation, and train a decision tree classifier. Once the model is trained, the application displays performance metrics including the classification report and model accuracy, and visualizes the decision tree structure graphically. Additionally, it allows for the export of the decision tree as an image file and offers a form-based interface for making predictions on new data.

Key Features and Functionalities
1. File Upload and Column Selection
The application supports CSV file import through a file dialog. Upon loading a dataset, it dynamically updates a dropdown list with the column names from the file, allowing users to select the target column (i.e., the label or class to be predicted). The last column is automatically preselected as the default target.

2. Adjustable Test Size
Users can fine-tune the test/train split ratio via an interactive slider ranging from 0.1 to 0.5. This allows flexible control over the proportion of data used for testing, aiding in better model evaluation and training dynamics.

3. Training and Evaluation
Once a dataset and target column are selected, and a test size is configured, users can train the model with a click. The backend uses Scikit-learn’s DecisionTreeClassifier to learn from the data. The application also handles preprocessing by encoding categorical features using one-hot encoding (via pd.get_dummies). Evaluation results are printed in a text box, showing the classification report (including precision, recall, and F1-score) and overall accuracy of the model.

4. Visualization
Using Matplotlib’s plot_tree, the trained decision tree is graphically displayed. This feature enables users to comprehend how the tree splits data based on feature values and makes decisions. This visualization is crucial for model interpretability and educational purposes.

5. Export Decision Tree Image
The GUI allows users to export the visualized decision tree as a PNG image. This is especially helpful for documentation, sharing results, or including the tree in reports or presentations.

6. Prediction of New Records
A highly interactive and valuable component of the app is the new record prediction form. When the model is trained, users can open a form-based interface that dynamically generates input fields based on the model’s features. After entering the appropriate values, the model predicts the output and displays the result in a popup window.

Technical Stack
Python: Core programming language.

Tkinter: For building the graphical user interface.

Pandas & NumPy: For data manipulation and numerical processing.

Scikit-learn: For machine learning functionality including model training, evaluation, and data splitting.

Matplotlib: For visualizing the decision tree.

Educational and Practical Value
This project is ideal for educational environments, particularly for teaching students how decision trees work, how to preprocess data, and how to evaluate a classifier. It also benefits business analysts and non-technical stakeholders who want to explore data-driven decision-making without delving into code.

By combining automation with interactivity, this tool empowers users to build robust machine learning models while maintaining full control over data inputs and parameters through a user-friendly interface. It bridges the gap between data science and usability, making machine learning more accessible and comprehensible to a wider audience.

#OUTPUT :



