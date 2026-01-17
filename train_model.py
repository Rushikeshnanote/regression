import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle, os

df = pd.read_csv("homeprices.csv")
x=df[['Area']]
y=df[['Price']]

model=LinearRegression()
model.fit(x,y)

os.makedirs("model",exist_ok=True)
with open("model/linear_model.pkl","wb") as f:
    pickle.dump(model,f)

print("Model trained succefully.")
