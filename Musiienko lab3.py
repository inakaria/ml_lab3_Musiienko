import pandas as pd
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'D:/windows_10_cmake_Release_Graphviz-11.0.0-win64/Graphviz-11.0.0-win64/bin'

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score
     

print('1. Відкрити та зчитати файл з даними.')
df = pd.read_csv('dataset_2.txt', sep=',', header=None)
df.columns = ['ID', 'Date', 'Column1', 'Column2', 'Column3', 'Column4', 'Column5', 'Target']

print('2. Визначити та вивести кількість записів та кількість полів у завантаженому наборі даних.')
rows, columns = df.shape
print(f'Number of rows: {rows}')
print(f'Number of columns: {columns}')

print('3. Вивести перші 10 записів набору даних.')
print(df.head(10))

print('4. Розділити набір даних на навчальну (тренувальну) та тестову вибірки.')
train_df, test_df = train_test_split(df, test_size=0.25, stratify=df['Target'])
print(train_df.head(10))
print(test_df.head(10))


print('5. Використовуючи  відповідні  функції  бібліотеки  scikit-learn,  збудувати класифікаційну модель дерева прийняття рішень',
      'глибини 5 та навчити її  на  тренувальній  вибірці,  вважаючи,  що  в  наданому  наборі  даних цільова  характеристика', 
      'визначається  останнім  стовпчиком,  а  всі  інші (окрім двох перших) виступають в ролі вихідних аргументів.')
x_train = train_df.iloc[:, 2:-1] # визначає навчальні дані для моделі
y_train = train_df.iloc[:, -1] #  визначає цільову змінну для моделі

train_model_entropy = DecisionTreeClassifier(max_depth=5, criterion='entropy') #max_leafs додати, прибрати max_depth
train_model_entropy.fit(x_train, y_train)

train_model_gini = DecisionTreeClassifier(max_depth=5, criterion='gini')
train_model_gini.fit(x_train, y_train)


print('6. Представити графічно побудоване дерево за допомогою бібліотеки graphviz.')
image1 = export_graphviz(
    train_model_entropy,
    feature_names=x_train.columns,
    class_names=y_train.unique().astype(str).tolist()
    )

graph1 = graphviz.Source(image1)
graph1.render("decision_tree1")

image2 = export_graphviz(
    train_model_gini,
    feature_names=x_train.columns,
    class_names=y_train.unique().astype(str).tolist()
    )

graph2 = graphviz.Source(image2)
graph2.render("decision_tree2")


print('7. Обчислити класифікаційні метрики збудованої моделі для тренувальної та тестової вибірки. Представити результати', 
      'роботи моделі на тестовій вибірці графічно. Порівняти результати, отриманні при застосуванні різних критеріїв розщеплення:', 
      'інформаційний приріст на основі ентропії чи неоднорідності Джині.')

def tree_metrics(model, x_cord, y_cord):
    all_metrics = {'accuracy': 0, 
                   'precision': 0, 
                   'recall': 0, 
                   'F1_score': 0,
                   'MCC': 0, 
                   'BA': 0}

    model_predictions = model.predict(x_cord)
    all_metrics['accuracy'] = accuracy_score(y_cord, model_predictions)
    all_metrics['precision'] = precision_score(y_cord, model_predictions)
    all_metrics['recall'] = recall_score(y_cord, model_predictions)
    all_metrics['F1_score'] = f1_score(y_cord, model_predictions)
    all_metrics['MCC'] = matthews_corrcoef(y_cord, model_predictions)
    all_metrics['BA'] = balanced_accuracy_score(y_cord, model_predictions)

    return all_metrics

# TEST
x_test = test_df.iloc[:, 2:-1]
y_test = test_df.iloc[:, -1]
metrics_test_gini = tree_metrics(train_model_gini, x_test, y_test)
metrics_test_entropy = tree_metrics(train_model_entropy, x_test, y_test)

df_test_test = pd.DataFrame({'Тестова вибірка gini': metrics_test_gini, 'Тестова вибірка entropy': metrics_test_entropy})
df_test_test.plot(kind='bar', figsize=(10, 6), color=['lightblue', "lightgreen"])
plt.title('Порівняння метрик для тестових вибірок')
plt.ylabel('Значення')
plt.xlabel('Метрика')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# TRAIN
x_train = train_df.iloc[:, 2:-1]
y_train = train_df.iloc[:, -1]
metrics_train_gini = tree_metrics(train_model_gini, x_train, y_train)
metrics_train_entropy = tree_metrics(train_model_entropy, x_train, y_train)

df_train_train = pd.DataFrame({'Тренувальна вибірка (gini)': metrics_train_gini, 'Тренувальна вибірка (entropy)': metrics_train_entropy})
df_train_train.plot(kind='bar', figsize=(10, 6), color=['lightblue', "lightgreen"])
plt.title('Порівняння метрик для тренувальних вибірок')
plt.ylabel('Значення')
plt.xlabel('Метрика')
plt.xticks(rotation=45)
plt.legend()
plt.show()
