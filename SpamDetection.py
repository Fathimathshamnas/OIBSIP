import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st




data=pd.read_csv(r"C:\Users\shamn\Downloads\spam\spam.csv",encoding='latin1')

data.drop_duplicates(inplace=True)
data['v1']=data['v1'].replace(['ham','spam'],['not spam','spam'])


# print(data.head())
mess=data['v2']
cat=data['v1']
#split data set into training and test daste set
(mess_train,mess_test,cat_train,cat_test)=train_test_split(mess, cat, test_size=0.2)
cv=CountVectorizer(stop_words='english')
features=cv.fit_transform(mess_train)

#creating model
Model=MultinomialNB()
Model.fit(features, cat_train)
#test  our model


features_test=cv.transform(mess_test)
# features_test=mess_test

print(Model.score(features_test, cat_test))

#predict the data in real time
def predict(message):
    input_message=cv.transform([' message']).toarray()

    result=Model.predict(input_message)
    return result


st.header('Spam Detection')


input_mess=st.text_input("Enter messege here")

if st.button('CHECK'):
    output=predict(input_mess)
    st.markdown(output)



#C:\Users\shamn\Downloads\spam\SpamDetection.py


