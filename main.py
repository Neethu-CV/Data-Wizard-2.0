from flask import Flask, jsonify, render_template, request, send_from_directory
import pandas as pd
from functions import *

app = Flask(__name__, template_folder='templates')

data = pd.DataFrame()
test_data = pd.DataFrame()
model_val = pd.DataFrame()
steps_status = {'step_1': 'active',  'step_2': 'inactive',  'step_3': 'inactive',  'step_4': 'inactive',  'step_5': 'inactive',  'step_6': 'inactive'  }
preprocessing_steps = {'Remove_Duplicates' : 0, 'Remove_Outliers' : 0, 'Handling_Missing_Values' : 0, 'Drop_Missing_Values' : 0, 'Scale_Columns' : 0, 'Normalize_Columns' : 0, 'Handel_Skewness_of_the_data':0 }
graph_options=['Line','Bar','Histogram','Pie','Box','Scatter','Countplot']
filename = None
test_filename = None
headers_code = None 
output_column= None
important_cols=None
model_type=None
selected_model=None
chat_history2 = []
chat_history3 = []
chat_history4 = []
chat_history5 = []
chat_history6 = []
graph_list=[]
best_model=[]
plot_data_list=[]
accuracy_table=[]
String_col = []
train_details = []
no_of_graph=0

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/')
def index():
    global data,filename,test_data,test_filename,steps_status, chat_history2, chat_history3, chat_history4, chat_history5, chat_history6,headers_code,preprocessing_steps,graph_options,graph_list,no_of_graph,output_column,best_model,plot_data_list,important_cols,model_type,selected_model,accuracy_table,String_col,model_val
    data, test_data, model_val = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    steps_status = {'step_1': 'active', 'step_2': 'inactive', 'step_3': 'inactive', 'step_4': 'inactive', 'step_5': 'inactive', 'step_6': 'inactive'}
    preprocessing_steps = {'Remove_Duplicates': 0, 'Remove_Outliers': 0, 'Handling_Missing_Values': 0, 'Drop_Missing_Values': 0, 'Scale_Columns': 0, 'Normalize_Columns': 0, 'Handel_Skewness_of_the_data': 0}
    graph_options = ['Line', 'Bar', 'Histogram', 'Pie', 'Box', 'Scatter', 'Countplot']
    filename, test_filename, important_cols, model_type, selected_model, headers_code, output_column = None, None, None, None, None, None, None
    chat_history2, chat_history3, chat_history4, chat_history5, chat_history6, graph_list, plot_data_list, accuracy_table, best_model, String_col = [], [], [], [], [], [], [], [], [], []
    no_of_graph = 0
    return render_template('index.html')

@app.route('/Home', methods=['POST'])
def home():
    global data,filename,test_data,test_filename,steps_status, chat_history2, chat_history3, chat_history4, chat_history5, chat_history6,headers_code,preprocessing_steps,graph_options,graph_list,no_of_graph,output_column,best_model,plot_data_list,important_cols,model_type,selected_model,accuracy_table,String_col,model_val
    data, test_data, model_val = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    steps_status = {'step_1': 'active', 'step_2': 'inactive', 'step_3': 'inactive', 'step_4': 'inactive', 'step_5': 'inactive', 'step_6': 'inactive'}
    preprocessing_steps = {'Remove_Duplicates': 0, 'Remove_Outliers': 0, 'Handling_Missing_Values': 0, 'Drop_Missing_Values': 0, 'Scale_Columns': 0, 'Normalize_Columns': 0, 'Handel_Skewness_of_the_data': 0}
    graph_options = ['Line', 'Bar', 'Histogram', 'Pie', 'Box', 'Scatter', 'Countplot']
    filename, test_filename, important_cols, model_type, selected_model, headers_code, output_column = None, None, None, None, None, None, None
    chat_history2, chat_history3, chat_history4, chat_history5, chat_history6, graph_list, plot_data_list, accuracy_table, best_model, String_col = [], [], [], [], [], [], [], [], [], []
    no_of_graph = 0
    return render_template('index.html')

@app.route('/Pre_understanding', methods=['POST'])
def step_2():
    global files,chat_history2,file,filename,data,test_filename,test_file,steps_status,test_data
    files = request.files
    chat_history2= []
    button_clicked = request.form['button_clicked']
    file = files['csv_file_1']
    filename = file.filename
    data = pd.read_csv(file)
    chat_history2.append(f"Uploaded file: {filename}")
    if button_clicked == '2_file':
        test_file = files['csv_file_2']
        test_filename = test_file.filename
        test_data = pd.read_csv(test_file)
        chat_history2.append(f"Uploaded file: {test_filename}")
    steps_status = {'step_1': 'done',  'step_2': 'active',  'step_3': 'inactive',  'step_4': 'inactive',  'step_5': 'inactive',  'step_6': 'inactive'  }
    display_pre_understanding(data, chat_history2)
    python_code=give_pythoncode(2)
    return render_template('step2.html', chat_history=chat_history2, steps_status=steps_status, python_code=python_code)

@app.route('/Pre_Understanding', methods=['POST'])
def step_2_done():
    global chat_history2,steps_status
    steps_status = {'step_1': 'done',  'step_2': 'active',  'step_3': 'inactive',  'step_4': 'inactive',  'step_5': 'inactive',  'step_6': 'inactive'  }
    python_code=give_pythoncode(2)
    return render_template('step2.html', chat_history=chat_history2, steps_status=steps_status, python_code=python_code)

@app.route('/Pre_processing', methods=['POST'])
def step_3():
    global data,preprocessing_steps,steps_status
    steps_status = {'step_1': 'done',  'step_2': 'done',  'step_3': 'active',  'step_4': 'inactive',  'step_5': 'inactive',  'step_6': 'inactive'  }
    preprocessing_steps=important_preprocessing(data,preprocessing_steps)
    python_code=give_pythoncode(3)
    return render_template('step3.html', steps_status=steps_status, preprocessing_steps=preprocessing_steps,python_code=python_code) 

@app.route('/Pre_Processing', methods=['POST'])
def step_3_done():
    global preprocessing_steps,steps_status
    steps_status = {'step_1': 'done',  'step_2': 'done',  'step_3': 'active',  'step_4': 'inactive',  'step_5': 'inactive',  'step_6': 'inactive'  }
    python_code=give_pythoncode(3)
    return render_template('step3.html', steps_status=steps_status, preprocessing_steps=preprocessing_steps,python_code=python_code) 

@app.route('/understanding', methods=['POST'])
def step_4():
    global data,steps_status,graph_options,plot_data_list,graph_list,no_of_graph,chat_history3 
    chat_history3=[]
    checked_values = request.form
    for key in checked_values:
        preprocessing_steps[key] = 2
    data=do_preprocessing(data,chat_history3,preprocessing_steps) 
    steps_status = {'step_1': 'done',  'step_2': 'done',  'step_3': 'done',  'step_4': 'active',  'step_5': 'inactive',  'step_6': 'inactive'  }
    python_code=give_pythoncode(4)
    if not graph_list:
        return render_template('step4.html', steps_status=steps_status, graph_options=graph_options, data=data, plot_data_list=plot_data_list, no_of_graph=no_of_graph, python_code=python_code)
    else:
        return render_template('step4.html', steps_status=steps_status, graph_options=graph_options, data=data, plot_data_list=plot_data_list, graph_list=graph_list, no_of_graph=no_of_graph, python_code=python_code) 

@app.route('/Understanding', methods=['POST'])
def step_4_done():
    global data,steps_status,graph_options,plot_data_list,graph_list,no_of_graph
    steps_status = {'step_1': 'done',  'step_2': 'done',  'step_3': 'done',  'step_4': 'active',  'step_5': 'inactive',  'step_6': 'inactive'  }
    python_code=give_pythoncode(4)
    return render_template('step4.html',steps_status=steps_status,graph_options=graph_options,data=data,plot_data_list=plot_data_list,graph_list=graph_list,no_of_graph=no_of_graph,python_code=python_code) 

@app.route('/generate_graph', methods=['POST'])
def generate_graph():
    global data, plot_data_list, graph_list, no_of_graph,chat_history4
    graph_data = request.get_json()
    no_of_graph = 0
    plot_data_list=[]
    chat_history4=[]
    graph_list = []
    if 'allDropdownData' in graph_data:
        graph_list = graph_data['allDropdownData']
    graph_list = [graph for graph in graph_list if graph.get('graph_type') is not None]
    print(f"Received data are: {graph_list}")
    for group in graph_list:
        x_column = group.get('x_column')
        y_column = group.get('y_column')
        graph_type = group.get('graph_type')
        if x_column is None or y_column is None or graph_type is None:
            if not (graph_type == 'Countplot' and x_column is not None and y_column is None):
                continue
        no_of_graph += 1
        if y_column=='None':
            chat_history4.append(f"{graph_type} Graph: {x_column}")
        else:
            chat_history4.append(f"{graph_type} Graph: {x_column} vs {y_column}")
        plot_data_list,chat_history4= do_vizualization(data, x_column, y_column, graph_type, plot_data_list,chat_history4)
    generate_heatmap(data, plot_data_list,chat_history4)
    return jsonify(plot_data_list=plot_data_list)

@app.route('/pre_model_selection', methods=['POST'])
def step_5():
    global steps_status,data
    steps_status = {'step_1': 'done',  'step_2': 'done',  'step_3': 'done',  'step_4': 'done',  'step_5': 'active',  'step_6': 'inactive'  }
    python_code=give_pythoncode(5)
    return render_template('step5.html',steps_status=steps_status,data=data,python_code=python_code)

@app.route('/model_selection', methods=['POST'])
def step_5_analyse():
    global data,steps_status,output_column,model_type,accuracy_table,model_val,important_cols,best_model,model_val,selected_model,chat_history5
    chat_history5=[]
    steps_status = {'step_1': 'done',  'step_2': 'done',  'step_3': 'done',  'step_4': 'done',  'step_5': 'active',  'step_6': 'inactive'  }
    output_column = request.form['selected_option']
    chat_history5.append(f"The selected output column is :{output_column}")
    pca_columns = set(apply_pca(data).columns)
    important_cols = list(pca_columns)
    if output_column in important_cols:
        important_cols.remove(output_column)
    print(f"Important columns: {important_cols}")    
    input_columns_str = "', '".join(important_cols)
    print(f"input_columns_str: {input_columns_str}")
    chat_history5.append(f"The important columns is:\n'{input_columns_str}'")
    data=checking_and_handling_missing_values(data)
    model_type = determine_model_type(data[output_column])
    chat_history5.append(f"Based on the output column selected it is: {model_type}")
    best_model,accuracy_table,model_val = evaluate_top_models(data, output_column,important_cols, model_type)
    chat_history5.append(accuracy_table)
    python_code=give_pythoncode(6)
    return render_template('step5_analyse.html',steps_status=steps_status,output_column=output_column,model_type=model_type,accuracy_table=accuracy_table,python_code=python_code,important_cols=important_cols,best_model=best_model,selected_model=selected_model)

@app.route('/Model_Selection', methods=['POST'])
def step_5_done():
    global steps_status,output_column,model_type,accuracy_table,best_model,important_cols,selected_model
    steps_status = {'step_1': 'done',  'step_2': 'done',  'step_3': 'done',  'step_4': 'done',  'step_5': 'active',  'step_6': 'inactive'  }
    python_code=give_pythoncode(6)
    return render_template('step5_analyse.html',steps_status=steps_status,output_column=output_column,model_type=model_type,accuracy_table=accuracy_table,python_code=python_code,important_cols=important_cols,best_model=best_model,selected_model=selected_model) 

@app.route('/visualization', methods=['POST'])
def step_6():
    global steps_status, accuracy_table, selected_model, model_val,output_column,data,test_data,chat_history6,train_details
    chat_history6=[]
    steps_status = {'step_1': 'done', 'step_2': 'done', 'step_3': 'done', 'step_4': 'done', 'step_5': 'done', 'step_6': 'active'}
    selected_model = request.form['selected_option']
    accuracy_table=[]
    if selected_model in model_val:
        chat_history6.append(f"Selected model from the user :{selected_model}")
        selected_model_metrics = model_val[selected_model]
        filtered_row_df = pd.DataFrame([selected_model_metrics])
        table_message = tabulate(filtered_row_df, headers=filtered_row_df.columns, tablefmt="html", numalign="center", stralign="center",showindex=False)
        accuracy_table = table_message.replace('<table>', '<table class="chat-table" style="border-collapse: collapse;">')
        chat_history6.append(accuracy_table)
    else:
        print("Selected model not found in model_val.")
    if test_filename != None:
        train_details=do_testing(selected_model,data,test_data,output_column,important_cols)
        chat_history6.append(train_details)
    python_code=give_pythoncode(7)
    return render_template('step6.html', steps_status=steps_status, accuracy_table=accuracy_table, python_code=python_code,selected_model=selected_model,train_details=train_details)

@app.route('/Visualization', methods=['POST'])
def step_6_done():
    global steps_status, accuracy_table, selected_model, model_val,output_column,train_details
    steps_status = {'step_1': 'done',  'step_2': 'done',  'step_3': 'done',  'step_4': 'done',  'step_5': 'done',  'step_6': 'active'  }
    python_code=give_pythoncode(7)
    return render_template('step6.html', steps_status=steps_status, accuracy_table=accuracy_table, python_code=python_code,selected_model=selected_model,train_details=train_details)

@app.route('/final', methods=['POST'])
def final():
    global steps_status,chat_history2, chat_history3, chat_history4, chat_history5, chat_history6
    steps_status = {'step_1': 'done',  'step_2': 'done',  'step_3': 'done',  'step_4': 'done',  'step_5': 'done',  'step_6': 'done'  }
    python_code=give_pythoncode(8)
    return render_template('final.html',steps_status=steps_status,chat_one=chat_history2+chat_history3,chat_two=chat_history4,chat_three=chat_history5+chat_history6,python_code=python_code) 

def give_pythoncode(step):
    global data,filename,test_data,test_filename,steps_status,chat_history,headers_code,preprocessing_steps,graph_options,graph_list,no_of_graph,output_column,best_model,plot_data_list,important_cols,model_type,selected_model,accuracy_table,String_col,model_val
    import_statement, function_code, body_code = "", "", ""
    if step >= 2:
        import_statement+="\n#Import Statements\nimport pandas as pd\n"
        body_code+=f"\n# Read train and test data\ndata = pd.read_csv('{filename}')\n"
        if test_filename != None:
            body_code+=f"test = pd.read_csv('{test_filename}')\n"
    if step >= 3:
        body_code+=f"\n# It is found that "
        for key,val in preprocessing_steps.items():
            if val==1:
                body_code+=f"{key}, "
        body_code = body_code[:-2] + f" steps need to be performed on the dataset\n"
        import_statement,function_code,body_code=add_processing_code(import_statement,function_code,body_code,1)
    if step >= 4:
        body_code+=f"\n# By user choice "
        for key,val in preprocessing_steps.items():
            if val==2:
                body_code+=f"{key}, "
        body_code = body_code[:-2] + f" steps are performed on the dataset\n"
        import_statement,function_code,body_code=add_processing_code(import_statement,function_code,body_code,2)
    if step >= 5:
        if no_of_graph>0:
            import_statement+="import seaborn as sns\nimport matplotlib.pyplot as plt\n"
        for group in graph_list:
            x_column = group.get('x_column')
            y_column = group.get('y_column')
            graph_type = group.get('graph_type')
            if x_column is None or y_column is None or graph_type is None:
                if not (graph_type == 'Countplot' and x_column is not None and y_column is None):
                    continue
            import_statement,function_code,body_code=add_graph_code(x_column,y_column,graph_type,import_statement,function_code,body_code)
        body_code += "# A heatmap to visualize the correlation matrix\nnumeric_data = data.select_dtypes(include='number')\nplt.figure(figsize=(10, 8))\nsns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')\nplt.title('Heatmap of Correlation Matrix')\nplt.show()\n\n"
    if step >= 6:
        import_statement += f"from sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import LabelEncoder\n"
        input_columns_str = "', '".join(important_cols)
        body_code += f"output_column= '{output_column}'\nimportant_cols= ['{input_columns_str}']\n\n"
        body_code += "data=data.dropna(subset=important_cols)\ndata=label_encoder(data)\n\n"
        function_code += '\n#Label Encode Function\ndef label_encoder(dataset):\n\tlabel_encoders = {}\n\tencoded_columns = []\n\tfor col in dataset.select_dtypes(include=["object"]).columns:\n\t\tif len(dataset[col].unique()) < 10:\n\t\t\tlabel_encoders[col] = LabelEncoder()\n\t\t\tencoded_col = f"{col}_encoded"\n\t\t\tdataset[encoded_col] = label_encoders[col].fit_transform(dataset[col])\n\t\t\tif "int" in str(dataset[encoded_col].dtype):\n\t\t\t\tdataset[encoded_col] = dataset[encoded_col].astype(int)\n\t\t\telif "float" in str(dataset[encoded_col].dtype):\n\t\t\t\tdataset[encoded_col] = dataset[encoded_col].astype(float)\n\t\t\tencoded_columns.append((col, encoded_col))\n\t\telse:\n\t\t\tdataset.drop(col, axis=1, inplace=True)\n\tfor col, _ in encoded_columns:\n\t\tdataset.drop(col, axis=1, inplace=True)\n\tremaining_string_cols = dataset.select_dtypes(include=["object"]).columns\n\tif not remaining_string_cols.empty:\n\t\tdataset.drop(remaining_string_cols, axis=1, inplace=True)\n\tdataset.columns = dataset.columns.str.replace("_encoded", "")\n\treturn dataset\n\n'
        body_code += f"# Extract features and target variable\nX = data[important_cols]\ny = data[output_column]\n#Test train Split\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\n"
    if step >= 7:
        body_code +=  f"#Based on data set it is {determine_model_type(data[output_column]).capitalize()}\n\n"
        import_statement,function_code,body_code=add_model_code(import_statement,function_code,body_code)
    if step > 7:
        if test_filename != None:
            body_code += f"\ntest_data = test.dropna(subset=important_cols)\ntest_data=label_encoder(test_data)\nX_test_file = test_data[important_cols]\ny_pred_file = model.predict(X_test_file)\ntest_data['Predicted_' + output_column] = y_pred_file\ntest_data.head()\n\n"
    return import_statement+function_code+body_code

def add_processing_code(import_statement,function_code,body_code,value):
    global preprocessing_steps
    for key,val in preprocessing_steps.items():
            if val==value:
                if key == 'Remove_Duplicates':
                    body_code+= f"#Remove Duplicates\ndata.drop_duplicates(inplace=True)\n\n"
                elif key == 'Remove_Outliers':
                    import_statement+=f"from scipy import stats\n"
                    body_code+= "#Define function to remove outliers using IQR method\ndef remove_outliers_iqr(data, column):\n\tQ1 = data[column].quantile(0.15)\n\tQ3 = data[column].quantile(0.85)\n\tIQR = Q3 - Q1\n\tlower_bound = Q1 - 1.5 * IQR\n\tupper_bound = Q3 + 1.5 * IQR\n\tfiltered_entries = (data[column] >= lower_bound) & (data[column] <= upper_bound)\n\treturn data[filtered_entries]\n\n"
                    body_code += "#Remove Outliers\nfor column in data.columns:\n\tif data[column].dtype in ['int64', 'float64']: # Select only numeric columns\n\t\tdata = remove_outliers_iqr(data, column)\n\n"
                elif key == 'Handling_Missing_Values':
                    body_code+= "# Handling Missing Values for Numeric Columns\nnumeric_columns = data.select_dtypes(include=['int64', 'float64']).columns\n\n"
                    body_code+= "for column in numeric_columns:\n\tmean_value = data[column].mean()\n\tdata[column].fillna(mean_value, inplace=True)\n\n"
                    body_code+= "# Handling Missing Values for Non-Numeric Columns\nnon_numeric_columns = data.select_dtypes(exclude=['int64', 'float64']).columns\n\n"
                    body_code+= "for column in non_numeric_columns:\n\tmode_value = data[column].mode()[0]  # Mode for non-numeric columns\n\tdata[column].fillna(mode_value, inplace=True)\n\n"
                elif key == 'Drop_Missing_Values':
                    body_code+= f"#Drop Missing Values\nfor column in data.columns:\n\tif data[column].isnull().any():\n\t\tdata.dropna(subset=[column], inplace=True)\n\n"
                elif key == 'Scale_Columns':
                    import_statement+= f"from sklearn.preprocessing import StandardScaler\n"
                    body_code+= f"#Scale Columns\n# Initialize the scaler\nscaler = StandardScaler()\n# Assuming your DataFrame is named data\nfor column in data.columns:\n\t# Check if the column is numeric\n\tif data[column].dtype in ['int64', 'float64']:\n\t\t# Scale the values of the selected column\n\t\tdata[[column]] = scaler.fit_transform(data[[column]])\n\n"
                elif key == 'Normalize_Columns':
                    import_statement+= f"from sklearn.preprocessing import MinMaxScaler\n"
                    body_code+= f"#Normalize Columns\n# Initialize the scaler\nscaler = MinMaxScaler()\n# Assuming your DataFrame is named data\nfor column in data.columns:\n\t# Check if the column is numeric\n\tif data[column].dtype in ['int64', 'float64']:\n\t\t# Normalize the values of the selected column\n\t\tdata[[column]] = scaler.fit_transform(data[[column]])\n\n"
                elif key == 'Handel_Skewness_of_the_data':
                    import_statement+= f"from scipy import stats\nimport numpy as np\n"
                    body_code+= "# Handel Skewness of the data\nfor column in data.select_dtypes(include=['int64', 'float64']).columns:\n\tskewness = stats.skew(data[column])\n\t# If the skewness is above a certain threshold (e.g., 0.5), perform transformation\n\tif abs(skewness) > 0.5:\n\t\t# Apply a log transformation\n\t\tdata[column] = np.log1p(data[column])\n\n"
    return import_statement,function_code,body_code

def add_graph_code(x_column,y_column,graph_type,import_statement,function_code,body_code):
    if graph_type == 'Line':
        body_code+=f"# Generate the line plot\nsns.lineplot(data=data, x='{x_column}', y='{y_column}')\n# Set plot title and labels\nplt.title('Line Plot')\nplt.xlabel('{x_column}')\nplt.ylabel('{y_column}')\n# Show the plot\nplt.show()\n\n"
    if graph_type == 'Bar':
        body_code+=f"""# Generate the bar plot\nplt.figure(figsize=(10, 6))\nsns.barplot(data=data, x='{x_column}', y='{y_column}', color='purple')\nplt.xlabel('{x_column}')\nplt.ylabel('{y_column}')\nplt.title(f"Bar Plot for '{x_column}' vs '{y_column}'")\nplt.legend(['{y_column}'], title='Legend')\nplt.show()\n\n"""
    if graph_type == 'Histogram':
        body_code += f"# Generate the histogram\nsns.histplot(data=data, x='{x_column}')\n# Set plot title and labels\nplt.title('Histogram')\nplt.xlabel('Values')\nplt.ylabel('Frequency')\n# Show the plot\nplt.show()\n\n"
    if graph_type == 'Pie':
        body_code += f"# Count the frequency of each category in the column\ncategory_counts = data['{x_column}'].value_counts()\n# Plot the pie chart\nplt.figure(figsize=(8, 8))\nplt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)\nplt.title('Pie Chart')\nplt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.\n# Show the plot\nplt.show()\n\n"
    if graph_type == 'Box':
        body_code += f"# Generate the box plot\nsns.boxplot(data=data, x='{x_column}')\n# Set plot title and labels\nplt.title('Box Plot')\nplt.xlabel('{x_column}')\n# Show the plot\nplt.show()\n\n"
    if graph_type == 'Scatter':
        body_code += f"# Generate the scatter plot\nsns.scatterplot(data=data, x='{x_column}', y='{y_column}')\n# Set plot title and labels\nplt.title('Scatter Plot')\nplt.xlabel('{x_column}')\nplt.ylabel('{y_column}')\n# Show the plot\nplt.show()\n\n"
    if graph_type == 'Countplot':
        if y_column == 'None':
            body_code += f"# Generate the count plot\nsns.countplot(data=data, x='{x_column}')\n# Set plot title and labels\nplt.title('Count Plot')\nplt.xlabel('{x_column}')\nplt.ylabel('Count')\n# Show the plot\nplt.show()\n\n"
        else:
            body_code += f"# Generate the count plot with two inputs\nsns.countplot(data=data, x='{x_column}', hue='{y_column}')\n# Set plot title and labels\nplt.title('Count Plot')\nplt.xlabel('{x_column}')\nplt.ylabel('{y_column}')\n# Show the plot\nplt.show()\n\n"
    return import_statement,function_code,body_code

def add_model_code(import_statement,function_code,body_code):
    global selected_model,data,output_column
    if selected_model=='Linear':
        import_statement += "from sklearn.linear_model import LinearRegression\nfrom sklearn.metrics import r2_score\n"
        body_code += "# Initialize the linear regression model\nmodel = LinearRegression()\n# Train the model on the training data\nmodel.fit(X_train, y_train)\n# Make predictions on the testing data\ny_pred = model.predict(X_test)\n# Calculate R^2 score\nr2 = r2_score(y_test, y_pred)\nprint('R^2 score:', r2)\n\n"
    elif selected_model == 'Lasso':
        import_statement += "from sklearn.linear_model import Lasso\nfrom sklearn.metrics import r2_score\n"
        body_code += "# Initialize the Lasso regression model\nmodel = Lasso(alpha=0.1)  # Alpha is the regularization strength, adjust as needed\n# Train the model on the training data\nmodel.fit(X_train, y_train)\n# Make predictions on the testing data\ny_pred = model.predict(X_test)\n# Calculate R^2 score\nr2 = r2_score(y_test, y_pred)\nprint('R^2 score:', r2)\n\n"
    elif selected_model == 'Ridge':
        import_statement += "from sklearn.linear_model import Ridge\nfrom sklearn.metrics import r2_score\n"
        body_code += "# Initialize the Ridge regression model\nmodel = Ridge(alpha=1.0)  # Alpha is the regularization strength, adjust as needed\n# Train the model on the training data\nmodel.fit(X_train, y_train)\n# Make predictions on the testing data\ny_pred = model.predict(X_test)\n# Calculate R^2 score\nr2 = r2_score(y_test, y_pred)\nprint('R^2 score:', r2)\n\n"
    elif selected_model == 'Logistic':
        import_statement += "from sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import accuracy_score\n"
        body_code += "# Initialize the Logistic Regression model\nmodel = LogisticRegression()\n# Train the model on the training data\nmodel.fit(X_train, y_train)\n# Make predictions on the testing data\ny_pred = model.predict(X_test)\n# Calculate accuracy score\naccuracy = accuracy_score(y_test, y_pred)\nprint('Accuracy score:', accuracy)\n\n"
    elif selected_model == 'Naive Bayes':
        import_statement += "from sklearn.naive_bayes import GaussianNB\nfrom sklearn.metrics import accuracy_score\n"
        body_code += "# Initialize the Naive Bayes model\nmodel = GaussianNB()\n# Train the model on the training data\nmodel.fit(X_train, y_train)\n# Make predictions on the testing data\ny_pred = model.predict(X_test)\n# Calculate accuracy score\naccuracy = accuracy_score(y_test, y_pred)\nprint('Accuracy score:', accuracy)\n\n"
    elif selected_model == 'K Nearest Neighbors':
        import_statement += "from sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.metrics import accuracy_score\n"
        body_code += "# Initialize the K Nearest Neighbors model\nmodel = KNeighborsClassifier(n_neighbors=5)  # You can specify the number of neighbors (k) as needed\n# Train the model on the training data\nmodel.fit(X_train, y_train)\n# Make predictions on the testing data\ny_pred = model.predict(X_test)\n# Calculate accuracy score\naccuracy = accuracy_score(y_test, y_pred)\nprint('Accuracy score:', accuracy)\n\n"
    elif selected_model == 'SVM':
        if determine_model_type(data[output_column])== 'classification':
            import_statement += "from sklearn.svm import SVC\nfrom sklearn.metrics import accuracy_score\n"
            body_code += "# Initialize the Support Vector Classifier (SVC) model\nmodel = SVC(kernel='rbf')  # You can specify the kernel as needed, e.g., 'linear', 'rbf', 'poly', etc.\n# Train the model on the training data\nmodel.fit(X_train, y_train)\n# Make predictions on the testing data\ny_pred = model.predict(X_test)\n# Calculate accuracy score\naccuracy = accuracy_score(y_test, y_pred)\nprint('Accuracy score:', accuracy)\n\n"
        else:
            import_statement += "from sklearn.svm import SVR\nfrom sklearn.metrics import r2_score\n"
            body_code += "# Initialize the SVR model\nmodel = SVR(kernel='linear')  # You can specify different kernels such as 'linear', 'rbf', etc.\n# Train the model on the training data\nmodel.fit(X_train, y_train)\n# Make predictions on the testing data\ny_pred = model.predict(X_test)\n# Calculate R^2 score\nr2 = r2_score(y_test, y_pred)\nprint('R^2 score:', r2\n\n)"
    elif selected_model == 'Decision Tree':
        if determine_model_type(data[output_column])== 'classification':
            import_statement += "from sklearn.tree import DecisionTreeClassifier\nfrom sklearn.metrics import accuracy_score\n"
            body_code += "# Initialize the Decision Tree Classifier model\nmodel = DecisionTreeClassifier(random_state=42)  # You can specify other parameters as needed\n# Train the model on the training data\nmodel.fit(X_train, y_train)\n# Make predictions on the testing data\ny_pred = model.predict(X_test)\n# Calculate accuracy score\naccuracy = accuracy_score(y_test, y_pred)\nprint('Accuracy score:', accuracy)\n\n"
        else:
            import_statement += "from sklearn.tree import DecisionTreeRegressor\nfrom sklearn.metrics import r2_score\n"
            body_code += "# Initialize the Decision Tree Regressor model\nmodel = DecisionTreeRegressor(random_state=42)  # You can specify other parameters as needed\n# Train the model on the training data\nmodel.fit(X_train, y_train)\n# Make predictions on the testing data\ny_pred = model.predict(X_test)\n# Calculate R^2 score\nr2 = r2_score(y_test, y_pred)\nprint('R^2 score:', r2)\n\n"
    elif selected_model == 'Random Forest':
        if determine_model_type(data[output_column])== 'classification':
            import_statement += "from sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import accuracy_score\n"
            body_code += "# Initialize the Random Forest Classifier model\nmodel = RandomForestClassifier(n_estimators=100, random_state=42)  # You can specify other parameters as needed\n# Train the model on the training data\nmodel.fit(X_train, y_train)\n# Make predictions on the testing data\ny_pred = model.predict(X_test)\n# Calculate accuracy score\naccuracy = accuracy_score(y_test, y_pred)\nprint('Accuracy score:', accuracy)\n\n"
        else:
            import_statement += "from sklearn.ensemble import RandomForestRegressor\nfrom sklearn.metrics import r2_score\n"
            body_code += "# Initialize the Random Forest Regressor model\nmodel = RandomForestRegressor(n_estimators=100, random_state=42)  # You can specify other parameters as needed\n# Train the model on the training data\nmodel.fit(X_train, y_train)\n# Make predictions on the testing data\ny_pred = model.predict(X_test)\n# Calculate R^2 score\nr2 = r2_score(y_test, y_pred)\nprint('R^2 score:', r2)\n\n"
    return import_statement,function_code,body_code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

