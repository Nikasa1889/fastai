import os
import sys
from niklib.MatifyAPI import MatifyAPI
from requests import Session
from niklib.utils import pad_sequences
import io
import json
import csv
import os
import pandas as pd
from tqdm import tqdm
import pickle

class MattextDataset():
    def __init__(self, mattext_df):
        df = mattext_df
        self.df = df
        #Dict that convert category name to category id
        cat_names = df.catName.unique()
        self.catNameId = {}
        for cat_name in cat_names:
            self.catNameId[cat_name] = df[df.catName == cat_name].catId.iloc[0]
            
    @classmethod
    def from_pickle(cls, fname):
        df = pickle.load( open( fname, "rb" ) )
        return cls(df)

    @classmethod
    def from_matifynet(cls):
        matifyAPI = MatifyAPI(verbose=False)
        categories = matifyAPI.getCategories()
        categories.append([142, "NotFood", [(142, "NotFood", None)]])
        #Request products from all sub category
        allProducts = []
        for categoryId, categoryName, subCategories in tqdm(categories):
            for subCategoryID, subCategoryName, _ in subCategories:
                try:
                    supermarkets = matifyAPI.getProducts (subCategoryID, subCategoryName)
                    for supermarket in supermarkets:
                        for product in supermarket["products"]:
                            allProducts.append((subCategoryID, 
                                                subCategoryName,
                                                product["id"],
                                                product["name"], 
                                                product["description"],
                                                float(product["price"])))
                except Exception as e:
                    print(f'Error while download category {subCategoryName} id {subCategoryID}')
                    print(e)
                    continue
        catId, catName, prodId, prodName, prodDesc, prodPrice = zip(*allProducts)
        product_df = pd.DataFrame({'catId': catId, 
                                  'catName': catName, 
                                  'prodId': prodId,
                                  'prodName': prodName, 
                                  'prodDesc': prodDesc, 
                                  'prodPrice': prodPrice})
        print(f'All downloaded subcategories: {pd.unique(catName)}')
        return cls(product_df)
    
    def save(self, fname):
        pickle.dump(self.df, open( fname, "wb" ) )
            
    def label_dist(self):
        label_dist = self.df[["catName", "prodPrice"]].groupby(by="catName").agg({"catName":"count", "prodPrice":["mean", "std"]})
        label_dist.columns = ["_".join(x) for x in label_dist.columns.ravel()]
        label_dist = label_dist.sort_values(by="catName_count", ascending =False)
        return(label_dist)
    
    def catname2catid(self, cat_name):
        return self.catNameId[cat_name]
        
    def summary(self):
        print(self)
        
    def __repr__(self):
        summary  = f'Total Product: {len(self.df)} \n'
        summary += f'Total Categories: {len(pd.unique(self.df["catId"]))} \n'
        summary += f'Label distribution: \n'
        summary += str(self.label_dist())
        return summary
        
#Error at Chocolate subcategory
#import requests
#productsResponse = requests.get('https://matify.net/listProducts/?categoryId=118&offset=0&len=2000&expired_after=2015-01-01')