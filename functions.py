from tabulate import tabulate
import pandas as pd
import numpy as np
import io
from scipy.stats import chi2_contingency
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns 
import matplotlib.pyplot as plt
import base64
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC,SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score


def display_pre_understanding(data, chat_history):
    table_message = table_show(data.head())
    chat_history.append("Displaying the first few rows of the data:" + table_message)
    data_info = dataframe_info(data)
    chat_history.append("Displaying DataFrame info:<br>"+ data_info)
    data_stats = dataframe_describe(data)
    chat_history.append("Displaying Basic Statistics: "+ data_stats)
    data_missing = missing_values_table(data)
    chat_history.append("Displaying Missing values in each column: "+ data_missing)
    data_duplicate = count_duplicate_rows(data)
    chat_history.append("Number of duplicate rows in the data: "+ str(data_duplicate))
    return
        
def table_show(data_head):
    table_message = tabulate(data_head, headers=data_head.columns, tablefmt="html", numalign="center", stralign="center")
    table_message = table_message.replace('<table>', '<table class="chat-table" style="border-collapse: collapse;">')
    return table_message

def dataframe_info(dataframe):
    buf = io.StringIO()
    dataframe.info(buf=buf)
    s = buf.getvalue()
    metadata_str = s.split("\n")[0:4] + s.split("\n")[-3:]
    metadata_df = pd.DataFrame([x.split(":") for x in metadata_str if ":" in x],columns=["Info", "Value"])
    df_info = pd.DataFrame({"Info": dataframe.columns,"Value": [str(dataframe[col].dtype) + " (non-null: " + str(dataframe[col].count()) + ")" for col in dataframe.columns]})
    df_combined = pd.concat([metadata_df, df_info], ignore_index=True)
    df_combined = df_combined.applymap(lambda x: "{:<20}".format(str(x)))
    table_message = tabulate(df_combined,headers=df_combined.columns,tablefmt="html",showindex=False)
    table_message = table_message.replace('<table>', '<table class="chat-table">')
    return table_message

def dataframe_describe(dataframe):
    df_describe = dataframe.describe()
    df_describe = df_describe.applymap(lambda x: "{:<20}".format(str(x)))
    table_message = tabulate(df_describe,headers=df_describe.columns,tablefmt="html",showindex=True,numalign="center",stralign="center")
    table_message = table_message.replace('<table>', '<table class="chat-table">')
    return table_message

def missing_values_table(dataframe):
    missing_values = dataframe.apply(lambda x: np.sum(x.isnull()))
    df_missing = pd.DataFrame({"Column Name": missing_values.index,"Missing Values": missing_values.values,})
    df_missing = df_missing.applymap(lambda x: "{:<20}".format(str(x)))
    table_message = tabulate(df_missing,headers=df_missing.columns,tablefmt="html",showindex=False,numalign="center",stralign="center")
    table_message = table_message.replace('<table>', '<table class="chat-table">')
    return table_message

def count_duplicate_rows(dataframe):
    return dataframe.duplicated().sum()

def important_preprocessing(dataframe,preprocessing_steps):
    if count_duplicate_rows(dataframe)>0:
        preprocessing_steps['Remove_Duplicates'] = 1
    if count_columns_with_outliers(dataframe)>0:
        preprocessing_steps['Remove_Outliers'] = 1
    if count_columns_with_missing_values(dataframe) >0:
        preprocessing_steps['Handling_Missing_Values'] = 1
    return preprocessing_steps

def remove_outliers_iqr(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    cleaned_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return cleaned_data

def count_columns_with_outliers(data):
    columns_with_outliers = 0
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            cleaned_data = remove_outliers_iqr(data, column)
            if len(cleaned_data) != len(data):
                columns_with_outliers += 1
    return columns_with_outliers

def count_columns_with_missing_values(data):
    missing_columns_count = 0
    for column in data.columns:
        if data[column].isnull().any():
            missing_columns_count += 1
    return missing_columns_count

def do_preprocessing(dataframe,chat_history,preprocessing_steps):
    if preprocessing_steps['Remove_Duplicates'] == 1:
        dataframe=label_encoder(dataframe)
        dataframe,count=remove_duplicates(dataframe)
        print(f"After removing duplicates: {dataframe.columns.tolist()}")
        chat_history.append(f"Removed {count} Duplicates")
            
    if preprocessing_steps['Remove_Outliers'] == 1:
        dataframe=label_encoder(dataframe)
        dataframe,count=remove_outliers(dataframe)
        print(f"After removing outliers: {dataframe.columns.tolist()}")
        chat_history.append(f"Removed {count} Outliers")

    if preprocessing_steps['Handling_Missing_Values'] == 1:
        dataframe=label_encoder(dataframe)
        dataframe=handle_missing_values_separately(dataframe)
        print(f"After handling missing values: {dataframe.columns.tolist()}")
        chat_history.append(f"Handled Missing Data")
        
    elif preprocessing_steps['Drop_Missing_Values'] == 1:
        dataframe=label_encoder(dataframe)
        dataframe,count=handle_missing_values(dataframe)
        print(f"After drop missing values: {dataframe.columns.tolist()}")
        chat_history.append(f"Removed {count} Missing Data")
        
    if preprocessing_steps['Scale_Columns'] == 1:
        dataframe=label_encoder(dataframe)
        dataframe=scale_columns(dataframe)
        print(f"After scaling columns: {dataframe.columns.tolist()}")
        chat_history.append(f"Scaled Data")
        
    if preprocessing_steps['Normalize_Columns'] == 1:
        dataframe=label_encoder(dataframe)
        dataframe=normalize_columns(dataframe)
        print(f"After normalize columns: {dataframe.columns.tolist()}")
        chat_history.append(f"Normalized Data")
        
    if preprocessing_steps['Handel_Skewness_of_the_data'] == 1:
        dataframe=label_encoder(dataframe)
        dataframe=handel_skewness_of_the_data(dataframe)
        print(f"After handle skewness: {dataframe.columns.tolist()}")
        chat_history.append(f"Handeled Skewness in the Data set")
        
    return dataframe

def label_encoder(dataset):
    print("Applying label encoding...")
    label_encoders = {}
    encoded_columns = []
    for col in dataset.select_dtypes(include=['object']).columns:
        if len(dataset[col].unique()) < 10:
            label_encoders[col] = LabelEncoder()
            encoded_col = f"{col}_encoded"
            dataset[encoded_col] = label_encoders[col].fit_transform(dataset[col])
            print(f"Current data type of column '{encoded_col}': {dataset[encoded_col].dtype}")
            if 'int' in str(dataset[encoded_col].dtype):
                dataset[encoded_col] = dataset[encoded_col].astype(int)
                print(f"Data type of column '{encoded_col}' converted to int.")
            elif 'float' in str(dataset[encoded_col].dtype):
                dataset[encoded_col] = dataset[encoded_col].astype(float)
                print(f"Data type of column '{encoded_col}' converted to float.")
            encoded_columns.append((col, encoded_col))
            print(f"Label encoding performed on column '{col}' and stored in '{encoded_col}'.")
        else:
            dataset.drop(col, axis=1, inplace=True)
            print(f"Column '{col}' dropped due to high cardinality.")
    for col, _ in encoded_columns:
        dataset.drop(col, axis=1, inplace=True)
        print(f"Original column '{col}' dropped after label encoding.")

    remaining_string_cols = dataset.select_dtypes(include=['object']).columns
    if not remaining_string_cols.empty:
        print("Dropping remaining string columns...")
        dataset.drop(remaining_string_cols, axis=1, inplace=True)
        print("Remaining string columns dropped.")

    dataset.columns = dataset.columns.str.replace('_encoded', '')

    print("Label encoding completed.")
    return dataset


def remove_duplicates(dataframe):
    duplicates_removed=dataframe.duplicated().sum()
    cleaned_data = dataframe.drop_duplicates()
    return cleaned_data,duplicates_removed

def remove_outliers(dataframe):
    for column in dataframe.columns:
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            q1 = dataframe[column].quantile(0.25)
            q3 = dataframe[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            clean_data = dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]
            outliers_removed = len(dataframe) - len(clean_data)
    return clean_data,outliers_removed

def handle_missing_values(dataframe):
    dataframe.dropna(axis=1, how='any', inplace=True)
    initial_rows = len(dataframe)
    missing_percentage = dataframe.isnull().mean() * 100
    columns_to_drop = missing_percentage[missing_percentage > 75].index
    dataframe.drop(columns=columns_to_drop, inplace=True)
    for column in dataframe.columns:
        if dataframe[column].isnull().mean() * 100 < 10:
            dataframe.dropna(subset=[column], inplace=True)
    dropped_rows = initial_rows - len(dataframe)
    return dataframe, dropped_rows

def scale_columns(dataframe):
    numerical_columns = dataframe.select_dtypes(include=['int64', 'float64', 'int','float','int32','float32']).columns
    scaler = MinMaxScaler()
    scaled_data = dataframe.copy()
    scaled_data[numerical_columns] = scaler.fit_transform(dataframe[numerical_columns])
    return scaled_data

def normalize_columns(dataframe):
    numerical_columns = dataframe.select_dtypes(include=['int64', 'float64', 'int','float','int32','float32']).columns
    normalized_data = dataframe.copy()
    for column in numerical_columns:
        normalized_data[column] = (dataframe[column] - dataframe[column].min()) / (dataframe[column].max() - dataframe[column].min())
    return normalized_data

def handel_skewness_of_the_data(dataframe):
    transformed_data = dataframe.copy()  
    for column in dataframe.columns:
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            if not (dataframe[column] <= 0).any():
                transformed_data[column] = np.log(dataframe[column])
    return transformed_data

def handle_missing_values_separately(data):
    data_types=identify_data_types(data)
    categorical_columns = data_types['categorical']['columns']
    numerical_columns = data_types['numerical']['columns']
    binary_columns = data_types['binary']['columns']
    print(f"categorical_columns: {categorical_columns}\nNumerical columns: {numerical_columns}\nBinary columns: {binary_columns}")

    for column in categorical_columns:
        if column in data.columns and data[column].isnull().any():
            print(f"In categorical columns: {data[column]}")
            if data[column].isnull().all():
                print(f"Skipping handling missing values for column '{column}' as it has no data.")
                continue
            data = select_best_imputation_method(data, column)

    for column in numerical_columns:
        if column in data.columns and pd.api.types.is_numeric_dtype(data[column]) and data[column].isnull().any():
            print(f"In numerical columns: {data[column]}")
            if data[column].isnull().all():
                print(f"Skipping handling missing values for column '{column}' as it has no data.")
                continue
            data = handle_missing_values_numerical(data, column)

    for column in binary_columns:
        if column in data.columns and pd.api.types.is_numeric_dtype(data[column]) and data[column].isnull().any():
            print(f"In binary columns: {data[column]}")
            if data[column].isnull().all():
                print(f"Skipping handling missing values for column '{column}' as it has no data.")
                continue
            data = handle_binary_mode(data, column)  
    return data

def handle_binary_mode(data, column):
    mode_value = data[column].mode()[0]
    data[column].fillna(mode_value, inplace=True)
    return data

def identify_data_types(df):
    categorical_cols = []
    numerical_cols = []
    text_cols = []
    binary_cols = []
    unique_threshold = len(df) * 0.10  
    for col in df.columns:
        unique_values = df[col].nunique()
        if pd.api.types.is_numeric_dtype(df[col]):
            if unique_values <= unique_threshold:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        elif pd.api.types.is_string_dtype(df[col]):
            text_cols.append(col)
        elif unique_values == 2: 
            binary_cols.append(col)
        else:
            categorical_cols.append(col)

    return {
        "categorical": {"columns": categorical_cols, "count": len(categorical_cols)},
        "numerical": {"columns": numerical_cols, "count": len(numerical_cols)},
        "text": {"columns": text_cols, "count": len(text_cols)},
        "binary": {"columns": binary_cols, "count": len(binary_cols)}
    }

def select_best_imputation_method(data, column):
    print(f"Handling missing values for column: {column}")
    mode_series = data[column].mode()
    if mode_series.empty:
        mode_value = 0
    else:
        mode_value = mode_series[0]
    mode_count = data[column].value_counts().get(mode_value, 0)
    total_values = len(data[column])
    mode_ratio = mode_count / total_values
    categorical_columns = [col for col in data.columns if pd.api.types.is_string_dtype(data[col]) and col != column]
    chi_square_association = False
    for categorical_col in categorical_columns:
        contingency_table = pd.crosstab(data[column], data[categorical_col])
        _, p_value, _, _ = chi2_contingency(contingency_table)
        if p_value < 0.05:  
            chi_square_association = True
            break
    if mode_ratio >= 0.75 or chi_square_association:
        data[column].fillna(mode_value, inplace=True)
    else:
        imputer = SimpleImputer(strategy='most_frequent')
        data[column] = imputer.fit_transform(data[[column]])
    return data

def handle_missing_values_numerical(data, column):
    skewness = data[column].skew()
    if skewness > 1:
        imputed_value = data[column].mode()[0]
        imputation_method = 'mode'
    elif skewness < -1:
        imputed_value = data[column].median()
        imputation_method = 'median'
    else:
        imputed_value = data[column].mean()
        imputation_method = 'mean'
    data[column].fillna(imputed_value, inplace=True)
    print(f"Missing values in numerical column '{column}' have been filled with {imputation_method}: {imputed_value}.")
    return data

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()

def do_vizualization(data,x_scale,y_scale,type,plot_data_list,chat_history4):
    plot_figure=[]
    if type == 'Bar':
        plot_figure,chat_history4 = generate_bar_plot(data, x_scale, y_scale,plot_data_list,chat_history4)
    elif type == 'Pie':
        plot_figure,chat_history4 = generate_pie_chart(data, x_scale,plot_data_list,chat_history4)
    elif type == 'Line':
        plot_figure,chat_history4 = generate_line_chart(data, x_scale, y_scale,plot_data_list,chat_history4)
    elif type == 'Box':
        plot_figure,chat_history4 = generate_box_plot(data, x_scale,plot_data_list,chat_history4)
    elif type == 'Histogram':
        plot_figure,chat_history4 = generate_histogram(data, x_scale,plot_data_list,chat_history4)
    elif type == 'Scatter':
        plot_figure,chat_history4 = generate_scatter_plot(data, x_scale, y_scale,plot_data_list,chat_history4)
    elif type == 'Countplot':
        plot_figure,chat_history4 = generate_count_plot(data, x_scale, y_scale,plot_data_list,chat_history4)
    return plot_figure,chat_history4

def generate_bar_plot(data, x_column, y_column, plot_data_list,chat_history4):
    try:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=data[x_column], y=data[y_column], color='purple')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(f'Bar Plot for {x_column} vs {y_column}')
        plt.legend([y_column], title='Legend')
        plot = plt.gcf()
        plot_data_list.append(fig_to_base64(plot))  # Assuming fig_to_base64 is defined elsewhere
        chat_history4.append(fig_to_base64(plot))
        plt.close()  # Close the plot to avoid displaying it
        return plot_data_list,chat_history4
    except KeyError as e:
        print(f"Error: One of the columns ({x_column}, {y_column}) is not found in the DataFrame.")
        return plot_data_list,chat_history4

def generate_pie_chart(data, x_column, plot_data_list,chat_history4):
    plt.figure(figsize=(8, 6))
    counts = data[x_column].value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'Pie Chart for {x_column}')
    plot = plt.gcf()
    plot_data_list.append(fig_to_base64(plot))
    chat_history4.append(fig_to_base64(plot))
    return plot_data_list,chat_history4

def generate_line_chart(data, x_column, y_column, plot_data_list,chat_history4):
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=data[x_column], y=data[y_column], color="red")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'Line Chart for {x_column} vs {y_column}')
    plot = plt.gcf()
    plot_data_list.append(fig_to_base64(plot))
    chat_history4.append(fig_to_base64(plot))
    return plot_data_list,chat_history4

def generate_box_plot(data, x_column, plot_data_list,chat_history4):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=x_column, data=data, color='blue',hue='Pclass')
    plt.xlabel(x_column)
    plt.ylabel('Values')
    plt.title(f'Box Plot for {x_column}')
    plot = plt.gcf()
    plot_data_list.append(fig_to_base64(plot))
    chat_history4.append(fig_to_base64(plot))
    return plot_data_list,chat_history4

def generate_histogram(data, x_column, plot_data_list,chat_history4):
    plt.figure(figsize=(8, 6))
    sns.histplot(data[x_column], color='darkred')
    plt.xlabel(x_column)
    plt.ylabel('Frequency')
    plt.title(f'Histogram for {x_column}')
    plot = plt.gcf()
    plot_data_list.append(fig_to_base64(plot))
    chat_history4.append(fig_to_base64(plot))
    return plot_data_list,chat_history4

def generate_scatter_plot(data, x_column, y_column, plot_data_list,chat_history4):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[x_column], y=data[y_column], color='darkred')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'Scatter Plot for {x_column} vs {y_column}')
    plot = plt.gcf()
    plot_data_list.append(fig_to_base64(plot))
    chat_history4.append(fig_to_base64(plot))
    return plot_data_list,chat_history4

def generate_count_plot(data, x_column, y_column, plot_data_list,chat_history4):
    plt.figure(figsize=(8, 6))
    if y_column == 'None':
        sns.countplot(x=x_column, data=data)
    else:
        sns.set_style('whitegrid')
        sns.countplot(x=x_column, hue=y_column, data=data)
    plot = plt.gcf()
    plot_data_list.append(fig_to_base64(plot))
    chat_history4.append(fig_to_base64(plot))
    return plot_data_list,chat_history4

def generate_heatmap(data, plot_data_list,chat_history4):
    # Filter numeric columns
    numeric_data = data.select_dtypes(include='number')
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Heatmap of Correlation Matrix')
    plt.xlabel('Features')
    plt.ylabel('Features')
    plot = plt.gcf()
    plot_data_list.append(fig_to_base64(plot))
    chat_history4.append(fig_to_base64(plot))
    return plot_data_list,chat_history4

def apply_pca(data, n_components=0.95):
    original_columns = data.columns.tolist()
    data_numeric = data.select_dtypes(include=['int64', 'float64', 'int','float','int32','float32'])
    categorical_cols = data.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        label_encoders = {}
        for col in categorical_cols:
            label_encoders[col] = LabelEncoder()
            data[col] = label_encoders[col].fit_transform(data[col])
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_numeric)
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data_scaled)
    num_components = min(len(original_columns), pca_data.shape[1])
    pca_df = pd.DataFrame(data=pca_data[:, :num_components], columns=[f"PC{i + 1}" for i in range(num_components)])
    pca_df.columns = original_columns[:num_components]
    return pca_df

def checking_and_handling_missing_values(dataset):
    print("Handling missing values...")
    missing_cols = dataset.columns[dataset.isnull().any()]
    if not missing_cols.empty:
        print("Missing values detected. Handling missing values...")
        for col in missing_cols:
            if dataset[col].dtype == 'object':
                dataset[col].fillna(dataset[col].mode()[0], inplace=True) 
            else:
                dataset[col].fillna(dataset[col].median(), inplace=True)  
        print("Missing values handled.")
    else:
        print("No missing values detected.")
    print("Missing value handling completed.")
    return dataset

def determine_model_type(y):
    if y.nunique() > 10:
        return 'regression'
    else:
        return 'classification'
        
def evaluate_top_models(dataset, output_column, important_cols,model_type='classification'):
    X = dataset[important_cols]
    y = dataset[output_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    regression_models = {
        'Linear': LinearRegression(),
        'Random Forest': RandomForestRegressor(),
        'Lasso': Lasso(),
        'Ridge': Ridge(),
        'SVM': SVR(),
        'Decision Tree': DecisionTreeRegressor()
    }
    classification_models = {
        'Logistic': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(),
        'K Nearest Neighbors': KNeighborsClassifier(),
    }
    models = classification_models if model_type.lower() == 'classification' else regression_models
    results = train_test_models(X_train, y_train, X_test, models)
    top_models = []
    for name, y_pred in results.items():
        if model_type.lower() == 'classification':
            report = classification_report(y_test, y_pred, output_dict=True)
            accuracy = report['accuracy']
            f1_score = report['weighted avg']['f1-score']
            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']
            top_models.append([name, accuracy, f1_score, precision, recall])
        else:
            r2 = r2_score(y_test, y_pred)
            top_models.append([name, r2])
    if model_type.lower() == 'classification':
        headers = ['Models', 'Accuracy', 'F1 Score', 'Precision', 'Recall']
    else:
        headers = ['Models', 'R2 Score']

    top_models_sorted = sorted(top_models, key=lambda x: x[1], reverse=True)
    top_3_models = [model[0] for model in top_models_sorted[:3]]

    # Create model_val dictionary for easier access
    model_val = {model[0]: {headers[i+1]: value for i, value in enumerate(model[1:])} for model in top_models_sorted}

    table_message = tabulate(top_models_sorted, headers=headers, tablefmt="html", numalign="center", stralign="center")
    table_message = table_message.replace('<table>', '<table class="chat-table" style="border-collapse: collapse;">')
    return top_3_models, table_message, model_val

def train_test_models(X_train, y_train, X_test, models):
    print("Training and testing models...")
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = y_pred
    print("Training and testing completed.")
    return results

def do_testing(selected_model, data, test_data, output_column, important_cols):
    test_data=label_encoder(test_data)
    model = None
    accuracy_table=[]
    model_type = determine_model_type(data[output_column])
    if selected_model == 'Linear':
        model = LinearRegression()
    elif selected_model == 'Lasso':
        model = Lasso()
    elif selected_model == 'Ridge':
        model = Ridge()
    elif selected_model == 'Logistic':
        model = LogisticRegression()
    elif selected_model == 'Naive Bayes':
        model = GaussianNB()
    elif selected_model == 'K Nearest Neighbors':
        model = KNeighborsClassifier()
    elif selected_model == 'Random Forest':
        if model_type== 'classification':
            model = RandomForestClassifier()
        else:
            model = RandomForestRegressor()
    elif selected_model == 'SVM':
        if model_type== 'classification':
            model = SVC()
        else:
            model = SVR()
    elif selected_model == 'Decision Tree':
        if model_type== 'classification':
            model = DecisionTreeClassifier()
        else:
            model = DecisionTreeRegressor()
    if model is None:
        print("Invalid model name")
        return accuracy_table

    X_train = data[important_cols]
    y_train = data[output_column]

    model.fit(X_train, y_train)

    test_data = test_data.dropna(subset=important_cols)

    X_test = test_data[important_cols]
    y_pred = model.predict(X_test)

    test_data['Predicted_' + output_column] = y_pred

    if model_type == 'classification':
        class_result = test_data[test_data['Predicted_' + output_column] == 1].shape[0]
        false_result = test_data[test_data['Predicted_' + output_column] == 0].shape[0]
        selected_model_metrics = {
            'Model': selected_model,
            'Model Type': model_type,
            f'{output_column} True': class_result,
            f'{output_column} False': false_result,
        }
        filtered_row_df = pd.DataFrame([selected_model_metrics])
    else:
        filtered_row_df = pd.DataFrame([test_data.head()])
    accuracy_table.append(f"On evaluating the test data the results are as follows")
    table_message = tabulate(filtered_row_df, headers=filtered_row_df.columns, tablefmt="html", numalign="center", stralign="center",showindex=False)
    accuracy_table.append(table_message.replace('<table>', '<table class="chat-table" style="border-collapse: collapse;">'))
    return accuracy_table