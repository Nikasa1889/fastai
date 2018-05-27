from requests import Session
from tabulate import tabulate
import json
import os
if os.getenv('MATIFY_API_ENDPOINT'):
    SERVER_API_ENDPOINT = os.getenv('MATIFY_API_ENDPOINT')
else:
    SERVER_API_ENDPOINT = 'http://matify.net:8000/'

BRANDNAME_TO_STOREID = {'coop mega': 6, 
                        'coop extra': 3 , 
                        'coop marked': 8,
                        'kiwi': 5,
                        'meny': 7,
                        'rema': 12,
                        'matkroken': 11,
                        'coop obs': 9, 
                        'coop prix': 4,
                        'joker': 10, 
                        'bunnpris': 13, 
                        'extra': 14, 
                        'spar': 15
                        }

class MatifyAPI:
    def __init__(self, verbose = True):
        self.verbose = verbose
        self.session = Session()
        self.defaultHeaders = {'Accept':'application/json'}
        
    def getCategories (self):
        categoryResponse = self.session.get(SERVER_API_ENDPOINT + 'listCategories/', 
                                       headers=self.defaultHeaders)
        assert (int(categoryResponse.status_code) == 200), \
                "Error when requesting all categories. Response text: " + categoryResponse.text

        categories = json.loads(categoryResponse.text)
        categoryList = [[category['id'], category['name'], 
                         [[subCategory['id'], subCategory['name'], subCategory['numOfProducts']] 
                          for subCategory in category['sub_categories']]] 
                        for category in categories]
        if self.verbose:
            for categoryId, categoryName, subCategories in categoryList:
                print("----------------------------------------")
                print(categoryName + " (ID: " + str(categoryId) +")")
                print(tabulate(subCategories, headers=["Sub Id", "Name", "Num of Products"]))
        return categoryList

    def getProducts (self, categoryId, categoryName = '', expired_after = '2015-01-01'):
        req = SERVER_API_ENDPOINT + 'listProducts/?categoryId='+ \
                                           str(categoryId)+'&offset=0&len=2000'+ \
                                           '&expired_after='+expired_after
        productsResponse = self.session.get(req, headers=self.defaultHeaders)
        assert (int(productsResponse.status_code) == 200), \
                f'''Error when requesting all products of a catagory. 
                    Request: {req}
                    Response text: {productsResponse.text}'''

        products = json.loads(productsResponse.text)
        if self.verbose:
            print(categoryName + "(ID=" + str(categoryId) + ")" + ": " + str(len(products)))
            print(products)
        return products

    def filterProductWithImage (self, allProducts):
        productWithImages = []
        for categoryName, products in allProducts:
            filteredProducts = [product for product in products if product["image"]]
            productWithImages.append([categoryName, filteredProducts])
            if self.verbose:
                print('%22s : %d images' %(categoryName, len(filteredProducts)))
        return productWithImages
    
    def getStoreId (self, brandName):
        return BRANDNAME_TO_STOREID[brandName];
    
    def uploadCatalog (self, catalogData, brandName, catalogFileName):
        token = 'b2h6ylyfn6pfvoz5wuvc'
        if self.verbose:
            print("Uploading catalog " + catalogFileName)
            
        storeId = self.getStoreId(brandName);
        response = self.session.post( SERVER_API_ENDPOINT + 'upload_products/'+str(storeId),
                                  data = {"token":token, 
                                          "file_name":catalogFileName, 
                                          "data": catalogData})
        #assert (int(response.status_code) == 200), \
        #        "Error when uploading catalog. Response text: " + response.text

        if self.verbose:
            print(response)
            print(response.text)
        return response

