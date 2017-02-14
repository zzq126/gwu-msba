# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 13:17:51 2016

@author: zzq

Programming for Analytics Assignment 03: Web Scraping
Group 2 members:
Ziqing Zhu        G20525987
Junfei ZHeng      G24645892     
Tianweibao Zheng  G49284776
Amit Nayak        G43671734
Soomin Park       G41524771
"""

from bs4 import BeautifulSoup as bs
import requests
import re
import pandas as pd
import time  #in order not to be blocked by the website

#step1
def step1():
    """
    Input: None
    Ouput: Dataframe of the Mexican restaurant information stored in the website

    """
    name=[]
    street = []
    city = []
    state = []
    zip = []
    phnum = []
    numre = []
    rate = []
    pricerange = []
    time.sleep(3) #in order not to be blocked by the website
    
    #scrape information in one page
    def web_scrape(url):  
        r = requests.get(url)
        data = r.text    
        soup = bs(data, "html.parser")
    #select tags containing search results
        for search_result in soup.select('li[class="regular-search-result"]'):
            n = search_result.select('a[data-analytics-label="biz-name"]')
            n = str(n)
            stn = re.search('biz/(.+?)-washington',n) #select information in elements
            if stn is None:
                stnother = re.search('biz/(.+?)-?osq',n).group(1)
                name.append(stnother)
            elif stn is not None:
                stndc = stn.group(1)
                name.append(stndc)
     
            ad = search_result.select("address")
            ad = str(ad)
            street_add = re.search('\n            (.+?)<br>',ad)
            if street_add is None:   #in case of missing information of the website
                street.append('NaN')
                city_add = re.search('<br>(.+?),',ad)
                if city_add is None:
                    city.append('NaN')
                    state.append('NaN')
                    zip.append('NaN')
                elif city_add is not None:
                    city_add = re.search('<br>(.+?),',ad).group(1)
                    city.append(city_add)
                    state_add = re.search(',(.+?) ',ad)
                    if state_add is not None:
                        state_add =state_add.group(1)
                        state.append(state_add)
                        zipcode = re.search(' (\d{5})\n',ad)
                        if zipcode is not None:
                            zipcode = zipcode.group(1)
                            zip.append(zipcode)
                        elif zipcode is None: 
                            zip.append('NaN')
                    elif state_add is None:
                        state.append('NaN')
                        zip.append('NaN')
                elif city_add is None:
                    city.append('NaN')
                    state.append('NaN')
                    zip.append('NaN')
            elif street_add is not None:
                street_add = re.search('\n            (.+?)<br>',ad).group(1)
                street.append(street_add)
                city_add = re.search('<br>(.+?),',ad)
                if city_add is not None:
                    city_add = city_add.group(1)
                    city.append(city_add)
                    state_add = re.search(',(.+?) ',ad)
                    if state_add is not None:
                        state_add = state_add.group(1)
                        state.append(state_add)
                        zipcode = re.search(' (\d{5})\n',ad)
                        if zipcode is None:
                            zip.append('NaN')
                        elif zipcode is not None:
                            zipcode = zipcode.group(1)
                            zip.append(zipcode)
                    elif state_add is None:
                        state.append('NaN')
                elif city_add is None:
                    city.append('NaN')
                    if state_add is not None:
                        state_add = state_add.group(1)
                        state.append(state_add)
                        zipcode = re.search(' (\d{5})\n',ad)
                        if zipcode is None:
                            zip.append('NaN')
                        elif zipcode is not None:
                            zipcode = zipcode.group(1)
                            zip.append(zipcode)
                    if state_add is None:
                        state.append('NaN')
                        if zipcode is None:
                            zip.append('NaN')
                        elif zipcode is not None:
                            zipcode = zipcode.group(1)
                            zip.append(zipcode)
                
            #phone number
            phonen = search_result.find("span",attrs={'class':'biz-phone'})
            phonen = str(phonen)
            pn = re.search('        (.+?)\n    </span>',phonen)
            if pn is None:
                phnum.append('NaN')
            elif pn is not None:
                pn = pn.group(1)
                phnum.append(pn)
            #number of reviews
            review = search_result.find('div',{'class': 'biz-rating biz-rating-large clearfix'})
            if review is None:
                numre.append('NaN')
            elif review is not None:
                review_number = review.getText().split()
                review_number = review_number[0]
                numre.append(review_number)
            
            rating = search_result.find("i",attrs={'class':'star-img'})
            rating = str(rating)
            rate_star = re.search('title="(.+?) star',rating)
            if rate_star is None:
                rate.append('NaN')
            elif rate_star is not None:
                rates = rate_star.group(1)
                r = str(rates)
                rfloat = float(r)
                rate.append(rfloat)
            #price range 
            prange = search_result.find("span",attrs={'class':'price-range'})
            prange = str(prange)
            p = re.search('price-range">(.+?)</span>',prange)
            if p is None:
                pricerange.append('NaN')
            elif p is not None:
                p = p.group(1)
                plen = len(p) *10 # because the price range of the website is $/$$/$$$
                pricerange.append(plen)
        
        return name,street, city, state, zip, phnum, numre, rate, pricerange
    
    
    link = []
    for num in range(0,1000,10): #to obtain all pages' url of search result
        n = str(num)
        u="https://www.yelp.com/search?find_desc=mexican+food&find_loc=Washington%2C+DC&start=%s".replace('%s',n)
        link.append(u)
    for url in link: #loop 
        web_scrape(url)
   
    df = pd.DataFrame()
    df['name'] = name
    df['Street Address'] = street
    df['City'] = city
    df['State'] = state
    df['Zip code'] = zip
    df['Phone Number'] = phnum
    df['Number of review'] = numre
    df['Rating'] = rate
    df['Price range'] = pricerange
    df.to_csv('mexican'+'.csv')
    
    return df

#step1-Chinese Restaurant information    
def step1b():
    """
    Input: None
    Ouput: Dataframe of Chinese restaurant information stored in the website
    """
    name=[]
    street = []
    city = []
    state = []
    zip = []
    phnum = []
    numre = []
    rate = []
    pricerange = []
    time.sleep(3)  #in order not to be blocked by the website
    
    def web_scrape(url):  
        r = requests.get(url)
        data = r.text    
        soup = bs(data, "html.parser")
    
        for search_result in soup.select('li[class="regular-search-result"]'):
            n = search_result.select('a[data-analytics-label="biz-name"]')
            n = str(n)
            stn = re.search('biz/(.+?)-washington',n)
            if stn is None:
                stnother = re.search('biz/(.+?)-?osq',n).group(1) #select information in elements
                name.append(stnother)
            elif stn is not None:
                stndc = stn.group(1)
                name.append(stndc)
    
            ad = search_result.select("address")
            ad = str(ad)
            street_add = re.search('\n            (.+?)<br>',ad) 
            if street_add is None:  #in case of missing information on the website
                street.append('NaN')
                city_add = re.search('<br>(.+?),',ad)
                if city_add is None:
                    city.append('NaN')
                    state.append('NaN')
                    zip.append('NaN')
                elif city_add is not None:
                    city_add = re.search('<br>(.+?),',ad).group(1)
                    city.append(city_add)
                    state_add = re.search(',(.+?) ',ad)
                    if state_add is not None:
                        state_add =state_add.group(1)
                        state.append(state_add)
                        zipcode = re.search(' (\d{5})\n',ad)
                        if zipcode is not None:
                            zipcode = zipcode.group(1)
                            zip.append(zipcode)
                        elif zipcode is None: 
                            zip.append('NaN')
                    elif state_add is None:
                        state.append('NaN')
                        zip.append('NaN')
            elif street_add is not None:
                street_add = re.search('\n            (.+?)<br>',ad).group(1)
                street.append(street_add)
                city_add = re.search('<br>(.+?),',ad)
                if city_add is not None:
                    city_add = city_add.group(1)
                    city.append(city_add)
                    state_add = re.search(',(.+?) ',ad)
                    if state_add is not None:
                        state_add = state_add.group(1)
                        state.append(state_add)
                        zipcode = re.search(' (\d{5})\n',ad)
                        if zipcode is None:
                            zip.append('NaN')
                        elif zipcode is not None:
                            zipcode = zipcode.group(1)
                            zip.append(zipcode)
                    elif state_add is None:
                        state.append('NaN')
                        zip.append('NaN')
                elif city_add is None:
                    city.append('NaN')
                    state_add = re.search(',(.+?) ',ad)
                    if state_add is not None:
                        state_add = state_add.group(1)
                        state.append(state_add)
                        zipcode = re.search(' (\d{5})\n',ad)
                        if zipcode is None:
                            zip.append('NaN')
                        elif zipcode is not None:
                            zipcode = zipcode.group(1)
                            zip.append(zipcode)
                    if state_add is None:
                        state.append('NaN')
                        zipcode = re.search(' (\d{5})\n',ad)
                        if zipcode is None:
                            zip.append('NaN')
                        elif zipcode is not None:
                            zipcode = zipcode.group(1)
                            zip.append(zipcode)
                
            #phone number
            phonen = search_result.find("span",attrs={'class':'biz-phone'})
            phonen = str(phonen)
            pn = re.search('        (.+?)\n    </span>',phonen)
            if pn is None:
                phnum.append('NaN')
            elif pn is not None:
                pn = pn.group(1)
                phnum.append(pn)
            #number of review
            review = search_result.find('div',{'class': 'biz-rating biz-rating-large clearfix'})
            if review is None:
                numre.append('NaN')
            elif review is not None:
                review_number = review.getText().split()
                review_number = review_number[0]
                numre.append(review_number)
            
            rating = search_result.find("i",attrs={'class':'star-img'})
            rating = str(rating)
            rate_star = re.search('title="(.+?) star',rating)
            if rate_star is None:
                rate.append('NaN')
            elif rate_star is not None:
                rates = rate_star.group(1)
                r = str(rates)
                rfloat = float(r)
                rate.append(rfloat)
            #price range
            prange = search_result.find("span",attrs={'class':'price-range'})
            prange = str(prange)
            p = re.search('price-range">(.+?)</span>',prange)
            if p is None:
                pricerange.append('NaN')
            elif p is not None:
                p = p.group(1)
                plen = len(p)*10  #price range is expressed by $/$$/$$$
                pricerange.append(plen)
        
        return name,street, city, state, zip, phnum, numre, rate, pricerange
    
    
    link = []
    for num in range(0,750,10): #to obtain all pages' url of search result
        n = str(num)
        u="https://www.yelp.com/search?find_desc=chinese+food&find_loc=Washington,+DC&start=%s".replace('%s',n)
        link.append(u)
    for url in link: #loop
        web_scrape(url)

    
    df = pd.DataFrame()
    df['name'] = name
    df['Street Address'] = street
    df['City'] = city
    df['State'] = state
    df['Zip code'] = zip
    df['Phone Number'] = phnum
    df['Number of review'] = numre
    df['Rating'] = rate
    df['Price range'] = pricerange
    df.to_csv('chinese'+'.csv')
    
    return df
#step2   
def step2():
    """
    Input: None
    Output: a pdf file containing histogram of ratings in all Mexican and Chinese restaurants
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    plt.figure(1)
    data1 = pd.read_csv('mexican.csv')
    df1 = data1.dropna()
    df1 = data1.reset_index()
    df1.hist(column=9,bins=5)
    plt.ylabel('Frequency')
    plt.xlabel('Rating')
    plt.suptitle('Mexican Food Rating Histogram')
    plt.savefig('hisRatingMexican.pdf')
    plt.show()
    
    plt.figure(2)
    data2 = pd.read_csv('chinese.csv')
    df2 = data2.dropna()
    df2 = data2.reset_index()
    df2.hist(column=9,bins=5)
    plt.ylabel('Frequency')
    plt.xlabel('Rating')
    plt.suptitle('Chinese Food Rating Histogram')
    plt.savefig('histRatingChinese.pdf')
    plt.show()
    
    return
    
#step 3 
def step3():
    """
    Input: None
    Output: 4 pdf files and 4 plots of relationship between rating and price range/number of 
    reviews in all Mexican and Chinese restaurants
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

#Plot the relationship between Rating (Y) and Number of reviews (X). Label it and save it as MXY2.pdf
    plt.figure(1)
    data = pd.read_csv("mexican.csv")
    data = data.dropna()

    x = data['Number of review']
    y = data['Rating']
    fit = np.polyfit(x,y,1) 
    fit_fn = np.poly1d(fit) 
    plt.plot(x,y, 'yo', x, fit_fn(x), '--k') 
    plt.suptitle('relationship between Rating (Y) and Number of reviews (X)') 
    plt.xlabel('Number of Reviews')
    plt.ylabel('Rating')
    plt.savefig('MXY2.pdf')
    plt.show()

#Plot the relationship between Rating (Y) and Price Range (X). Label it and save it asMXY1.pdf
    plt.figure(2)
    data = pd.read_csv("mexican.csv")
    data = data.dropna()

    x = data['Price range']
    y = data['Rating']
    fit = np.polyfit(x,y,1) 
    fit_fn = np.poly1d(fit) 
    plt.plot(x,y, 'yo', x, fit_fn(x), '--k') 
    plt.suptitle( 'The relationship between Rating(Y) and Price Range (X)')
    plt.xlabel('Price range')
    plt.ylabel('Rating')
    plt.savefig('MXY1.pdf')
    plt.show()

#Plot the relationship between Rating (Y) and Number of reviews (X). Label it and save it as CXY2.pdf
    plt.figure(3)
    data = pd.read_csv("chinese.csv")
    data = data.dropna()

    x = data['Number of review']
    y = data['Rating']
    fit = np.polyfit(x,y,1) 
    fit_fn = np.poly1d(fit) 
    plt.plot(x,y, 'yo', x, fit_fn(x), '--k') 
    plt.suptitle('The relationship between Rating(Y) and Number of review (X)')
    plt.xlabel('Number of Reviews')
    plt.ylabel('Rating')
    plt.savefig('CXY2.pdf')
    plt.show()

#Plot the relationship between Rating (Y) and Price Range (X) CXY1.pdf
    plt.figure(4)
    data = pd.read_csv("chinese.csv")
    data = data.dropna()

    x = data['Price range']
    y = data['Rating']
    fit = np.polyfit(x,y,1) 
    fit_fn = np.poly1d(fit) 
    plt.plot(x,y, 'yo', x, fit_fn(x), '--k') 
    plt.suptitle('The relationship between Rating(Y) and Price Range (X)')
    plt.xlabel('Price range')
    plt.ylabel('Rating')
    plt.savefig('CXY1.pdf')
    plt.show()
    
    return

#step 4
from mpl_toolkits.mplot3d import *
def step4():
    """
    Input: None
    Output: coefficients of regression model and a pdf file containing the 3d plot
    """
    import pandas as pd
    import numpy as np
    from sklearn import linear_model
    from sklearn.metrics import r2_score

#Combine mexican.csv and chinese.csv into one one.
    test1 = pd.read_csv("mexican.csv")
    test2 = pd.read_csv("chinese.csv")
    test3 = pd.concat([test1, test2], axis=0)
    test3= test3.reset_index() 
    test3 = test3.dropna()
    
# Regress Rating (Y) on Price Range (X) and Number of reviews (X).
    Xtotal = pd.concat([test3["Price range"], test3["Number of review"]], axis=1)
    Xtotal = Xtotal.as_matrix()
    X = np.array([np.concatenate((v,[1])) for v in Xtotal])
    model = linear_model.LinearRegression(fit_intercept = True)
    yarr = test3["Rating"]
    fit = model.fit(X,yarr)
    pred = model.predict(X)
    
#Print out the coefficients and the R2.    
    print("Intercept : ",fit.intercept_)
    print("Slope : ", fit.coef_)
    pred = model.predict(X)
    r2 = r2_score(yarr,pred) 
    print ('R-squared: %.2f' % (r2))

## plot a 3d scatter plot
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm      
    fig = plt.figure()
    ax = fig.gca(projection='3d')               # to work in 3d
    plt.hold(True)
    x_max = max(test3["Price range"])    
    y_max = max(test3["Number of review"])   
    
    b0 = float(fit.intercept_)
    b1 = float(fit.coef_[0])
    b2 = float(fit.coef_[1])   
    
    x_surf=np.linspace(0, x_max, 100)                # generate a mesh
    y_surf=np.linspace(0, y_max, 100)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
    z_surf = b0 + b1*x_surf +b2*y_surf         # ex. function, which depends on x and y
    ax.plot_surface(x_surf, y_surf, z_surf, cmap=cm.hot, alpha=0.2);    # plot a 3d surface plot
    
    x=test3["Price range"]
    y=test3["Number of review"]
    z=test3["Rating"]
    ax.scatter(x, y, z);                        # plot a 3d scatter plot


    ax.set_xlabel('fit.coef_[0]')
    ax.set_ylabel('fit.coef_[1]')
    ax.set_zlabel('fit.intercept_')
    plt.suptitle('regression')
    plt.xlabel('Price range')
    plt.ylabel('Number of review')
    plt.savefig('regression.pdf')
    plt.show()
    return
