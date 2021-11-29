import streamlit as st
import numpy as np 
import pandas as pd 
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from PIL import Image

Uni =Image.open("Woxsen_University.jpg")
st.image(Uni)


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
    
def mba():

    uploaded_file = st.file_uploader("Upload CSV file")
    
    if uploaded_file:
        data = pd.read_excel(uploaded_file)
        
        data['Description'] = data['Description'].str.strip()
        data.dropna(axis=0, subset=['Invoice'], inplace=True)
        data['Invoice'] = data['Invoice'].astype('str')
        data = data[~data['Invoice'].str.contains('C')]
        data_plus=data[data["Quantity"]>=0]
        basket = (data[data['Country'] =="United_Kingdom"]
                 .groupby(['Invoice', 'Description'])['Quantity']
                 .sum().unstack().reset_index().fillna(0)
                 .set_index('Invoice'))
    
        basket_sets = basket.applymap(encode_units)
        basket_filter=basket_sets[(basket_sets > 0).sum(axis=1) >=2]
        frq_sets=apriori(basket_filter, min_support=0.03, 
                         use_colnames=True).sort_values('support',ascending=False).reset_index(drop=True)
        frq_sets['length']=frq_sets['itemsets'].apply(lambda x: len(x))
        rules=association_rules(frq_sets, metric='lift', min_threshold=1).sort_values('lift',ascending=False).reset_index(drop=True)
        result=rules.head()
        result.to_csv("sample.csv",index=False)
        new_res = pd.read_csv("sample.csv")
        list(new_res["antecedents"])
        new_res['antecedents'] = new_res['antecedents'].map(lambda x: x.lstrip('frozenset({').rstrip('})'))
        new_res['consequents'] = new_res['consequents'].map(lambda x: x.lstrip('frozenset({').rstrip('})'))
        st.write(new_res)  

if __name__ == "__main__":
    st.title("Market Basket Analysis")
    mba()

