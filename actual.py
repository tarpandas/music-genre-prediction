import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("music.csv")

x=df.drop(columns=['genre'])
y=df['genre']

le=LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = DecisionTreeClassifier()
model = model.fit(x_train,y_train)

pickle.dump(model, open('data.pkl','wb'))
