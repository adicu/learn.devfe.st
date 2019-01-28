---
layout: curriculum
title: Devfest Data Science Track 2019
source: "https://github.com/Siaan/data-science-2019"
---

To run this without any environment setup, go [here](https://mybinder.org/v2/gh/Siaan/data-science-2019/master?filepath=yelp_pymc3.ipynb) (this might take a minute or so to load).

# Setting up the Yelp API

In order to interact with the API, you need to make a Yelp Fusion account to get the API key. This allows you to then use the functionality of the API.

<img src='./Yelp_Fusion_Walkthrough_1.png'>
<img src='./Yelp_Fusion_Walkthrough_2.png'>
<img src='./Yelp_Fusion_Walkthrough_3.png'>
<img src='./Yelp_Fusion_Walkthrough_4.png'>

Once you have your API_KEY, you'll want to put it down below:


```python
API_KEY = "" # PUT YOUR API_KEY here
```

# Introduction

Jim and Pam are very hungry. After a long day of work at The Office of University Life, they decided to figure out a place to eat. Too bad they didn't have Yelp installed, they had something better. They were Data Scientists!

<img src="wow.jpeg" width=500px>

I know, WOW INDEED. A Data Scientist is many things: primarily they are multi-discliplinary wizards that extract insights from data by leveraging a wide variety of tools and skills. These insights are typically serve to answer lucrative or interesting questions like:

- How can we improve productivity in the Office of University Life?
- How can we rig an election?
- What is the spatial distribution hot dog carts in NYC throughout the day?

*Data Science is recognized as one of the hottest fields there is being named the "Sexiest Job of the 21st Century" in 2012, but since then it has become somewhat of a buzzword with negative connotations. With widespread access to user data and large scale hacks compromising private data on millions of people, data is constantly being exploited to target users. With increasing ethical dilemmas about the misuse of data, new laws like the EU's [GDPR](https://eugdpr.org/) are being established to prevent exploitation of users data.*


```python
from IPython.display import YouTubeVideo

YouTubeVideo('xC-c7E5PK0Y', width=800, height=500)
```


<iframe
    width="800"
    height="500"
    src="https://www.youtube.com/embed/xC-c7E5PK0Y"
    frameborder="0"
    allowfullscreen
></iframe>
        



Ok back to Jim and Pam, they are data scientists and they need to solve the problem of soothing their aching bellies by fufilling their yearning for the sweet taste of food. So here they go.

## Structure of this First Part

- Using requests and an API
- Modifying Sample Code
- Getting Data
- Making Data Easier to work with

# Imports

[API](https://en.wikipedia.org/wiki/Application_programming_interface) or Application Programming Interface is like a standard list of possible interactions between the user and a service. For example:

Burger King:
- Get Burger
- Pay for Meal
- Calculate Tip

From [Yelp's Github Repository](https://github.com/Yelp/yelp-fusion/blob/master/fusion/python/sample.py), we can use their API to define functions that we can then use to get data from Yelp!


<center> These code snippets were modified from the sample code provided by Yelp. </center>


```python
import json
import pprint
import requests
import sys
import urllib

from urllib.error import HTTPError
from urllib.parse import quote
from urllib.parse import urlencode

import pandas as pd

# API constants, you shouldn't have to change these.
API_HOST = 'https://api.yelp.com'
SEARCH_PATH = '/v3/businesses/search'
BUSINESS_PATH = '/v3/businesses/'  # Business ID will come after slash.
REVIEW_PATH = '/v3/businesses/{}/reviews'

# Defaults for our simple example.
DEFAULT_TERM = 'dinner'
DEFAULT_LOCATION = 'San Francisco, CA'
SEARCH_LIMIT = 10

def request(host, path, api_key, url_params=None):
    """Given your API_KEY, send a GET request to the API.
    Args:
        host (str): The domain host of the API.
        path (str): The path of the API after the domain.
        API_KEY (str): Your API Key.
        url_params (dict): An optional set of query parameters in the request.
    Returns:
        dict: The JSON response from the request.
    Raises:
        HTTPError: An error occurs from the HTTP request.
    """
    url_params = url_params or {}
    url = '{0}{1}'.format(host, quote(path.encode('utf8')))
    headers = {
        'Authorization': 'Bearer %s' % api_key,
    }

    print(u'Querying {0} ...'.format(url))

    response = requests.request('GET', url, headers=headers, params=url_params)

    return response.json()


def search(api_key, term, location, limit=SEARCH_LIMIT):
    """Query the Search API by a search term and location.
    
    Can add additional terms:
    
        sort_by (string) Optional. Suggestion to the search algorithm that the results be sorted by one of the these
            modes: best_match, rating, review_count or distance. The default is best_match. Note that specifying the
            sort_by is a suggestion (not strictly enforced) to Yelp's search, which considers multiple input parameters
            to return the most relevant results. For example, the rating sort is not strictly sorted by the rating value,
            but by an adjusted rating value that takes into account the number of ratings, similar to a Bayesian average. 
            This is to prevent skewing results to businesses with a single review.
            
        price (string) Optional. Pricing levels to filter the search result with: 1 = $, 2 = $$, 3 = $$$, 4 = $$$$. 
            The price filter can be a list of comma delimited pricing levels. For example, "1, 2, 3" will filter the
            results to show the ones that are $, $$, or $$$.
            
        open_now (boolean) Optional. Default to false. When set to true, only return the businesses open now.
            Notice that open_at and open_now cannot be used together.
    
    Args:
        term (str): The search term passed to the API.
        location (str): The search location passed to the API.
    Returns:
        dict: The JSON response from the request.
    """

    url_params = {
        'term': term.replace(' ', '+'),
        'location': location.replace(' ', '+'),
        'limit': limit
    }
    return request(API_HOST, SEARCH_PATH, api_key, url_params=url_params)


def get_business(API_KEY, business_id):
    """Query the Business API by a business ID.
    Args:
        business_id (str): The ID of the business to query.
    Returns:
        dict: The JSON response from the request.
    """
    business_path = BUSINESS_PATH + business_id

    return request(API_HOST, business_path, API_KEY)

def query_api(term, location):
    """Queries the API by the input values from the user.
    Args:
        term (str): The search term to query.
        location (str): The location of the business to query.
    """
    response = search(API_KEY, term, location)

    businesses = response.get('businesses')

    if not businesses:
        print(u'No businesses for {0} in {1} found.'.format(term, location))
        return

    business_id = businesses[0]['id']

    print(u'{0} businesses found, querying business info ' \
        'for the top result "{1}" ...'.format(
            len(businesses), business_id))
    response = get_business(API_KEY, business_id)

    print(u'Result for business "{0}" found:'.format(business_id))
    pprint.pprint(response, indent=2)
    return response
```

## Basic Query

Let's first begin with a basic query, reading the documentation of the function we see that we simply need to input our search string followed by a location in order to get data our top result. We that the max number returned is limited by the parameter SEARCH_LIMIT.


```python
response = query_api("dinner", 'Morningside Heights, NY')
```

    Querying https://api.yelp.com/v3/businesses/search ...
    10 businesses found, querying business info for the top result "E2mNgb479B3BCfwi2G_KdQ" ...
    Querying https://api.yelp.com/v3/businesses/E2mNgb479B3BCfwi2G_KdQ ...
    Result for business "E2mNgb479B3BCfwi2G_KdQ" found:
    { 'alias': 'flat-top-new-york',
      'categories': [ {'alias': 'newamerican', 'title': 'American (New)'},
                      {'alias': 'cafes', 'title': 'Cafes'},
                      {'alias': 'breakfast_brunch', 'title': 'Breakfast & Brunch'}],
      'coordinates': {'latitude': 40.810041, 'longitude': -73.958693},
      'display_phone': '(646) 820-7735',
      'hours': [ { 'hours_type': 'REGULAR',
                   'is_open_now': True,
                   'open': [ { 'day': 0,
                               'end': '1530',
                               'is_overnight': False,
                               'start': '1130'},
                             { 'day': 0,
                               'end': '2200',
                               'is_overnight': False,
                               'start': '1730'},
                             { 'day': 1,
                               'end': '1530',
                               'is_overnight': False,
                               'start': '1130'},
                             { 'day': 1,
                               'end': '2200',
                               'is_overnight': False,
                               'start': '1730'},
                             { 'day': 2,
                               'end': '1530',
                               'is_overnight': False,
                               'start': '1130'},
                             { 'day': 2,
                               'end': '2200',
                               'is_overnight': False,
                               'start': '1730'},
                             { 'day': 3,
                               'end': '1530',
                               'is_overnight': False,
                               'start': '1130'},
                             { 'day': 3,
                               'end': '2200',
                               'is_overnight': False,
                               'start': '1730'},
                             { 'day': 4,
                               'end': '1530',
                               'is_overnight': False,
                               'start': '1130'},
                             { 'day': 4,
                               'end': '2300',
                               'is_overnight': False,
                               'start': '1730'},
                             { 'day': 5,
                               'end': '1530',
                               'is_overnight': False,
                               'start': '1030'},
                             { 'day': 5,
                               'end': '2300',
                               'is_overnight': False,
                               'start': '1730'},
                             { 'day': 6,
                               'end': '1530',
                               'is_overnight': False,
                               'start': '1030'}]}],
      'id': 'E2mNgb479B3BCfwi2G_KdQ',
      'image_url': 'https://s3-media1.fl.yelpcdn.com/bphoto/y2YlqtpKU2tciozvngNbsg/o.jpg',
      'is_claimed': True,
      'is_closed': False,
      'location': { 'address1': '1241 Amsterdam Ave',
                    'address2': '',
                    'address3': '',
                    'city': 'New York',
                    'country': 'US',
                    'cross_streets': 'Morningside Dr & 122nd St',
                    'display_address': ['1241 Amsterdam Ave', 'New York, NY 10027'],
                    'state': 'NY',
                    'zip_code': '10027'},
      'name': 'Flat Top',
      'phone': '+16468207735',
      'photos': [ 'https://s3-media1.fl.yelpcdn.com/bphoto/y2YlqtpKU2tciozvngNbsg/o.jpg',
                  'https://s3-media1.fl.yelpcdn.com/bphoto/K2h2sJ37uVh_Bxx9VyMZ3w/o.jpg',
                  'https://s3-media3.fl.yelpcdn.com/bphoto/MCL75jklkBZgrDkNquJ5Cg/o.jpg'],
      'price': '$$',
      'rating': 4.0,
      'review_count': 405,
      'transactions': ['restaurant_reservation', 'pickup'],
      'url': 'https://www.yelp.com/biz/flat-top-new-york?adjust_creative=dLQC-YFBm9Sjlrw_1z9OTQ&utm_campaign=yelp_api_v3&utm_medium=api_v3_business_lookup&utm_source=dLQC-YFBm9Sjlrw_1z9OTQ'}


This output is intimidating, but there is a great amount of data here to be worked with. The data that is returned in JSON format. This just means that the data is stored in a standard format that we can explore.


```python
response.keys()
```




    dict_keys(['id', 'alias', 'name', 'image_url', 'is_claimed', 'is_closed', 'url', 'phone', 'display_phone', 'review_count', 'categories', 'rating', 'location', 'coordinates', 'photos', 'price', 'hours', 'transactions'])



For us in Python, this JSON format can simply be interpreted as a dictionary where information such as the rating, location, photos, price, etc. of a restaurant can be stored in key, value pairs. Let's say we want to look at the location. We can simply do the ``` response['location'] ```


```python
response['location']
```




    {'address1': '1241 Amsterdam Ave',
     'address2': '',
     'address3': '',
     'city': 'New York',
     'zip_code': '10027',
     'country': 'US',
     'state': 'NY',
     'display_address': ['1241 Amsterdam Ave', 'New York, NY 10027'],
     'cross_streets': 'Morningside Dr & 122nd St'}



And to get the address from this stage we can do ```response['location']['address1']```


```python
response['location']['address1']
```




    '1241 Amsterdam Ave'



Let's say that Jim and Pam want to see images of the place. We can actually use a function, ```urllib.request.urlopen```, to write it as an image to a file.


```python
print(response['image_url'])
from IPython.display import Image

Image(url=response['image_url'])
```

    https://s3-media1.fl.yelpcdn.com/bphoto/y2YlqtpKU2tciozvngNbsg/o.jpg





<img src="https://s3-media1.fl.yelpcdn.com/bphoto/y2YlqtpKU2tciozvngNbsg/o.jpg"/>



Jim and Pam approve, but lets see if we can check out more options.

# Getting Multiple Responses

Let's start getting serious: Looking at these restaurants one at a time is infuriating. What is really nice is that Yelp has already provided us with code that actually works, so let's take advantage of that and adapt "query_api" into a function that can grab many restaurants for us.


```python
def query_api_limits(term, location, num_limit = 10):
    """Queries the API by the input values from the user. Will get at most num_limit businesses.
    Args:
        term (str): The search term to query.
        location (str): The location of the business to query.
        num_limit (int): Max number of businesses.
    """
    response = search(API_KEY, term, location, limit=num_limit)

    businesses = response.get('businesses')

    if not businesses:
        print(u'No businesses for {0} in {1} found.'.format(term, location))
        return
    
    #Grab a bunch of business ids
    business_ids = [i['id'] for i in businesses]
    first_business_id = business_ids[0]
    
    print(u'{0} businesses found, querying business info ' \
        'for the top result "{1}" ...'.format(
            len(businesses), first_business_id))
    
    #Use get_business in order to graph informatino on each of the businesses with the Business ID
    responses = [get_business(API_KEY, business_id) for business_id in business_ids]
    
    return responses
```

Basically what we changed here was to grab a bunch of the business ids and then we can use those business ids in order to get data about each of the individual buisnesses associated with the business_id with the call to ```get_business(API_KEY, business_id)```.

The function above is adapted to our liking now, we see that we can actually return a list of businesses. Let's see if we can try that out.


```python
responses = query_api_limits("", 'Morningside Heights, NY')
```

    Querying https://api.yelp.com/v3/businesses/search ...
    10 businesses found, querying business info for the top result "JV5oa5-KGdiWnqrKPoxSug" ...
    Querying https://api.yelp.com/v3/businesses/JV5oa5-KGdiWnqrKPoxSug ...
    Querying https://api.yelp.com/v3/businesses/QHYqNQhQ8NVK0RX68Y4PuQ ...
    Querying https://api.yelp.com/v3/businesses/8lLs3dsSN-Am2_EtNfbXqA ...
    Querying https://api.yelp.com/v3/businesses/vZ5-JXlJS75k8wmPNS5U5w ...
    Querying https://api.yelp.com/v3/businesses/8Qe6g3Dv5NXN1Zq-egJk9w ...
    Querying https://api.yelp.com/v3/businesses/7tTVuBwJ4LLtQ3-HHW80_A ...
    Querying https://api.yelp.com/v3/businesses/E2mNgb479B3BCfwi2G_KdQ ...
    Querying https://api.yelp.com/v3/businesses/80fvd_DsoW5XhwQIMBGvtg ...
    Querying https://api.yelp.com/v3/businesses/A46G2OAvLxFswiONB50Rrg ...
    Querying https://api.yelp.com/v3/businesses/J9xVQScnr0lYWl61_mLXMA ...


Seemed like it works! But now we have alot of data to work with. It would be great if we could have data structures made for storing this kind of information. Therefore we are going to introduce something really quite powerful...

<img src="panda.gif">

OOOOO its about to get real. We are gonna introduce PANDAS. Pandas is a package that most data scientists like to use. It's a really great way for storing data and performing data analysis and is widely used by data scientists. Here we are going to discuss the one of the key data structure in Pandas: the DataFrame.

## Basics of the DataFrame

[Here](https://pandas.pydata.org/pandas-docs/stable/10min.html) is a very short introduction on how to use Pandas. I would also recommend taking a look at this [resource](https://jakevdp.github.io/PythonDataScienceHandbook/03.00-introduction-to-pandas.html) as well before you move on.

## Using the DataFrame to store our Data

Alright, now Jim and Pam are going to store their data into a DataFrame, let's write some simple functions to do that:


```python
def create_new_entry(metadata):
    """
    Reads in the metadata from a response from query_api and stores individual entries into a pandas DataFrame
    
    Args:
        param1: metadata (dict) stores the response from query_api
        
    Returns:
        pandas.DataFrame with one entry
    """
    new_entry = dict()
    new_entry['id']=metadata['id']
    new_entry['name']=metadata['name']
    new_entry['review_count'] =metadata['review_count']
    new_entry['rating']=metadata['rating']
    new_entry['latitude']=metadata['coordinates']['latitude']
    new_entry['longitude']=metadata['coordinates']['longitude']
    new_entry['open']=metadata['hours'][0]['is_open_now']
    return pd.DataFrame(new_entry,index=[0])

def creating_data_frame(responses):
    """
    Reads in a list of responses from query_api and stores the entries all in a pandas DataFrame using create_new_entry
    
    Args:
        param1: responses (list) stories responses from query_api
        
    Returns:
        pandas.DataFrame with all responses as entries
    """
    output= pd.concat([create_new_entry(response) for response in responses])
    output = output.reset_index(drop=True)
    return output

def create_category_entry(metadata):
    """
    Method to record the categories of restaurant the response from query_api in a pandas.DataFrame
    
    Args:
        param1: metadata (dict) stores the response from query_api
        
    Returns:
        pandas.DataFrame with entries with the 'id' of the restaurant and the associated categorization
    """
    new_entry=dict()
    categories=dict()
    new_entry['id']=metadata['id']
    for i in range(len(metadata['categories'])):
        categories[i]=metadata['categories'][i]
    new_entry['categories']=categories
    return pd.DataFrame(new_entry)

def create_category_data_frame(responses):
    """
    Method to record the categories of restaurants the responses from query_api in a pandas.DataFrame
    
    Args:
        param1: responses (list) stories responses from query_api
        
    Returns:
        pandas.DataFrame with entries with the 'id's of the restaurants and their associated categorizations
    """
    output = pd.concat([create_category_entry(response) for response in responses])
    output = output.reset_index(drop=True)
    return output
```


```python
restaurant_data_frame = creating_data_frame(responses)
```


```python
category_data_frame= create_category_data_frame(responses)
```

Here we are going to store some of the information about each business in a dataframe.


```python
restaurant_data_frame.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>review_count</th>
      <th>rating</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>open</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>JV5oa5-KGdiWnqrKPoxSug</td>
      <td>Absolute Bagels</td>
      <td>1236</td>
      <td>4.5</td>
      <td>40.802510</td>
      <td>-73.967450</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>QHYqNQhQ8NVK0RX68Y4PuQ</td>
      <td>Levain Bakery</td>
      <td>637</td>
      <td>4.5</td>
      <td>40.804974</td>
      <td>-73.955151</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8lLs3dsSN-Am2_EtNfbXqA</td>
      <td>Community Food &amp; Juice</td>
      <td>846</td>
      <td>3.5</td>
      <td>40.805798</td>
      <td>-73.965675</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>vZ5-JXlJS75k8wmPNS5U5w</td>
      <td>Max Soha</td>
      <td>387</td>
      <td>4.0</td>
      <td>40.811302</td>
      <td>-73.958183</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8Qe6g3Dv5NXN1Zq-egJk9w</td>
      <td>Hungarian Pastry Shop</td>
      <td>599</td>
      <td>3.5</td>
      <td>40.803580</td>
      <td>-73.963650</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



And here we are going to store the information with the restaurant id and the categories themselves


```python
category_data_frame.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>categories</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>JV5oa5-KGdiWnqrKPoxSug</td>
      <td>{'alias': 'bakeries', 'title': 'Bakeries'}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JV5oa5-KGdiWnqrKPoxSug</td>
      <td>{'alias': 'bagels', 'title': 'Bagels'}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>QHYqNQhQ8NVK0RX68Y4PuQ</td>
      <td>{'alias': 'bakeries', 'title': 'Bakeries'}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8lLs3dsSN-Am2_EtNfbXqA</td>
      <td>{'alias': 'newamerican', 'title': 'American (N...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8lLs3dsSN-Am2_EtNfbXqA</td>
      <td>{'alias': 'breakfast_brunch', 'title': 'Breakf...</td>
    </tr>
  </tbody>
</table>
</div>



Let's quickly take a look at restaurant ids.


```python
location_ids = category_data_frame['id'].unique()
```


```python
location_ids
```




    array(['JV5oa5-KGdiWnqrKPoxSug', 'QHYqNQhQ8NVK0RX68Y4PuQ',
           '8lLs3dsSN-Am2_EtNfbXqA', 'vZ5-JXlJS75k8wmPNS5U5w',
           '8Qe6g3Dv5NXN1Zq-egJk9w', '7tTVuBwJ4LLtQ3-HHW80_A',
           'E2mNgb479B3BCfwi2G_KdQ', '80fvd_DsoW5XhwQIMBGvtg',
           'A46G2OAvLxFswiONB50Rrg', 'J9xVQScnr0lYWl61_mLXMA'], dtype=object)



Cool! We have alot of the relevant data we need in order to make an informed decision on where to eat. Now let's say that Jim really wanted a way to have the categories stored in a list so anyone can access this category list uniquely. What is very nice about the location_id is that assuming it is unique, allows us to understand quickly get data on the specific restaurant from either the ```restaurant_data_frame``` or the ```category_data_frame``` since they share the id row in common.

We are going to now take the ```category_data_frame``` and store it as a dictionary where the key is the location_id and the value is a list of the labels that was categorized.


```python
def collate_data(df,location_ids):
    """
    Method to take the category_DataFrame and store it as a dictionary where the key is the location_id 
    and the value is a list of the labels that was categorized.
    
    Args: 
        param1: df (pandas.DataFrame) the input category dataFrame
        param2: location_ids (list, array) unique identifier for the restaurant
        
    Returns
        Dictionary: a dictionary where the key is the location_id and the value
        is a list of the labels that was categorized.
    """
    
    output=dict()

    for loc in location_ids:
        aliases = []
        for i in df[df['id']==loc]['categories']:
            aliases.append(i['alias'])
        output[loc]=aliases

    return output
```


```python
collate_data(category_data_frame, location_ids)
```




    {'JV5oa5-KGdiWnqrKPoxSug': ['bakeries', 'bagels'],
     'QHYqNQhQ8NVK0RX68Y4PuQ': ['bakeries'],
     '8lLs3dsSN-Am2_EtNfbXqA': ['newamerican', 'breakfast_brunch', 'bars'],
     'vZ5-JXlJS75k8wmPNS5U5w': ['italian'],
     '8Qe6g3Dv5NXN1Zq-egJk9w': ['bakeries', 'coffee'],
     '7tTVuBwJ4LLtQ3-HHW80_A': ['parks'],
     'E2mNgb479B3BCfwi2G_KdQ': ['newamerican', 'cafes', 'breakfast_brunch'],
     '80fvd_DsoW5XhwQIMBGvtg': ['churches', 'landmarks'],
     'A46G2OAvLxFswiONB50Rrg': ['pizza'],
     'J9xVQScnr0lYWl61_mLXMA': ['italian', 'breakfast_brunch', 'cocktailbars']}



Jim and Pam had the opportunity to explore their data and now they decided that they want to give restaurant 'QHYqNQhQ8NVK0RX68Y4PuQ' a chance, therefore from now on, we are going to consider this as their_restaurant_id.


```python
their_restaurant_id='QHYqNQhQ8NVK0RX68Y4PuQ'
```

Now Jim is worried about how good the restaurant really is. He saw on the Yelp page that the rating was a 4.5 but looking at some of the ratings of the three reviews he read he saw a 5 and 2 and a 3.


```python
def get_reviews(business_id,api_key=API_KEY,limit=3):
    """Query the Review API by a business ID.
    Args:
        business_id (str): The ID of the business to query.
    Returns:
        dict: The JSON response from the request.
    """
    review_path = REVIEW_PATH.format(business_id)
    
    url_params = {
        'limit': limit
    }
    
    return request(API_HOST,review_path, api_key, url_params=url_params)
```


```python
reviews = get_reviews(their_restaurant_id)['reviews']
```

    Querying https://api.yelp.com/v3/businesses/QHYqNQhQ8NVK0RX68Y4PuQ/reviews ...


Looking through the returned JSON files we get the following ratings.


```python
reviews[0]['rating']
```




    5




```python
reviews[1]['rating']
```




    2




```python
reviews[2]['rating']
```




    3



# Hold up
We are about to go into a specific analysis (with Bayesian Inference), but the machinery that we established above can be used to get some really great data and get some really great analysis. Some ideas:
- Using the categories to try to see how specific restaurants are distributed around Columbia. If we can do this overtime, we may even be able to see a changing social demographic reflected in what we eat! (We really are what we eat) (Humanities, social demographic change)
- Trying to run Sentiment Analysis on these reviews which can be accessed through the URL. Using a webscraper like BeautifulSoup, one could probably do some really cool analysis (NLP)
- Use the image urls to pull images and classify them to determine what kind of food they serve. (Convolutional Neural Nets)
- A data visualization that shows the restaurants opening and closing over the course of day, maybe even foot traffic over a period of a year. (Predictive Analytics) 

__How can Jim quantify this uneasiness? What should the rating probably be?__

<img src='math.png'>

Let's try to use some math/Bayesian Inference.

## Bayesian Inference

Bayesian Inference is a rich and nuanced topic to discuss especially since it is technically a philosophy. There is a whole philisophical competition between [Bayesian](https://rationalwiki.org/wiki/Bayesian) and Frequentist reasoning which is really interesting, but we would like to apply Bayesian Inference to help us guess what the likelihood of a restaurant being a certain rating is.

These were some really great resources that Jim and Pam took direction from:
- https://towardsdatascience.com/estimating-probabilities-with-bayesian-modeling-in-python-7144be007815
- https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers 

Here is a simple example of what we are trying to understand with Bayesian Inference. Say Pam's friend, Micheal, said that this sushi place was really good. She will now have some prior belief in how good this sushi place is. But, when she reads the reviews, she notices not all of them are that good. But say she trusts Micheal a whole lot, so she still believes the place is good.

Now along with Bayesian Inference, Pam had a prior belief, which was then modified by an observation, which resulted in a posterior belief. Let's see if we can try to simulate that.

## Let's begin with some imports


```python
import pandas as pd
import numpy as np

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 22
%matplotlib inline

# from matplotlib import MatplotlibDeprecationWarning

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# warnings.filterwarnings('ignore', category=MatplotlibDeprecationWarning)

import pymc3 as pm

# Helper functions
from utils import draw_pdf_contours, Dirichlet, plot_points, annotate_plot, add_legend, display_probs
```

    WARNING (theano.configdefaults): install mkl with `conda install mkl-service`: No module named 'mkl'


Jim decides to approach the problem like this to make it simpler for himself. He observed the ratings 5, 3, 2 each once and knows that the average should be 4.5 according to the rating of the restaurant. Now the ratings of 3 and 2 hurt how he feels about the 4.5 rating, but can we quantify that? (For now we are going to pretend the ratings of 4 and 1 don't matter)

The average rating of 4.5 reflects our prior beliefs and we can actually tune that in the hyperparameters. Since the average is so high, we strongly believe that we should see alot of 5 ratings in general therefore we phrase the problem accordingly.


```python
# observations
ratings = ['5','4', '3', '2','1']
c = np.array([1, 0,1, 1,0])

# hyperparameters
jims_parameters = np.array([16, 1, 1, 1,1])
```

We chose the hyperparameters 8,1,1 because if we imagine that they were the number of observations for observing the ratings 5,3,2 respectively then we have an expected value/average of 4.5.


```python
(16*5+4+3+2+1)/(4+16)
```




    4.5




```python
display_probs(dict(zip(ratings, (jims_parameters + c) / (c.sum() + jims_parameters.sum()))))
```

    Rating: 5        Prevalence: 73.91%.
    Rating: 4        Prevalence: 4.35%.
    Rating: 3        Prevalence: 8.70%.
    Rating: 2        Prevalence: 8.70%.
    Rating: 1        Prevalence: 4.35%.


The expected value for Jim's rating (with our new observations now) should be:


```python
5*0.7391+4*.0435+3*.0870+2*.0870+1*.0435
```




    4.348



Wow, we were able to quantify how our observations affected Jim's prior beliefs. Now Pam is much more of a believer and her hyperparameters are larger, signaling a stronger belief in the original rating, but with the same expected value.


```python
pams_parameters=np.array([32, 2, 2, 2,2])
```


```python
display_probs(dict(zip(ratings, (pams_parameters + c) / (c.sum() + pams_parameters.sum()))))
```

    Rating: 5        Prevalence: 76.74%.
    Rating: 4        Prevalence: 4.65%.
    Rating: 3        Prevalence: 6.98%.
    Rating: 2        Prevalence: 6.98%.
    Rating: 1        Prevalence: 4.65%.


These are our expected value for Pam's rating (with our new observations now) should be:


```python
5*.7674+4*.0465+3*.0698+2*.0698+1*.0465
```




    4.418499999999999



We see that that Pam is less shaken by negative reviews and her expected value for the rating is less affect in comparison to Jim and we can visualize that as well:


```python
alpha_list=[jims_parameters,pams_parameters]
```


```python
values = []
for alpha_new in alpha_list:
    values.append((alpha_new + c) / (c.sum() + alpha_new.sum()))

value_df = pd.DataFrame(values, columns = ratings)
value_df['alphas'] = ["Jim","Pam"] #[str(x) for x in alpha_list]
value_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>5</th>
      <th>4</th>
      <th>3</th>
      <th>2</th>
      <th>1</th>
      <th>alphas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.739130</td>
      <td>0.043478</td>
      <td>0.086957</td>
      <td>0.086957</td>
      <td>0.043478</td>
      <td>Jim</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.767442</td>
      <td>0.046512</td>
      <td>0.069767</td>
      <td>0.069767</td>
      <td>0.046512</td>
      <td>Pam</td>
    </tr>
  </tbody>
</table>
</div>




```python
melted = pd.melt(value_df, id_vars = 'alphas', value_name='prevalence',
        var_name = 'ratings')

plt.figure(figsize = (8, 6))
sns.barplot(x = 'alphas', y = 'prevalence', hue = 'ratings', data = melted,
            edgecolor = 'k', linewidth = 1.5);
plt.xticks(size = 14); plt.yticks(size = 14)
plt.title('Expected Value');
```


![png](output_89_0.png)


Therefore we see that Pam is a lot less shaken by what she sees in terms of reviews.

# More complicated Analysis

Our overall system has five discrete choices each with an unknown probability and 3 total observations is a multinomial distribution. A multinomial distribution is the generalization of the binomial distribution where there are more than just two outcomes and can be characterized byby k, the number of outcomes, n, the number of trials, and p, a vector of probabilities for each of the outcomes. Our objective is to find p, the probability of seeing each rating in our observations.

"In Bayesian statistics, the parameter vector for a multinomial is drawn from a [Dirichlet Distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution), which forms the prior distribution for the parameter." ([Medium Article](https://towardsdatascience.com/estimating-probabilities-with-bayesian-modeling-in-python-7144be007815)) Similiar to the multinomial, the Dirichlet Distribution is characterized by, k, the number of outcomes, and alpha, a vector of positive real values called the concentration parameter. Alpha is called a hyperparameter because it is a parameter of the prior.

Let's talk more about the hyperparameters and Jim and Pam's prior beliefs. The way we previously thought about the hyperparameters were as psuedocounts, basically as events that we maybe already have seen. If we have seen much more data, a couple of stray values may not shake up our confidence that much.


## Bayesian Inference in Python with PyMC3

Let's try to get a range of estimates, we use Bayesian inference in order to construct a model of the Jim's feelings as he sees these observations come in, then sample from the posterior to approximate the posterior (the posterior being the end distribution). This is implemented through Markov Chain Monte Carlo/ No-U-Turn Sampler in PyMC3.

Let's first start by setting up the model:


```python
with pm.Model() as jims_model:
    # Parameters of the Multinomial are from a Dirichlet
    parameters = pm.Dirichlet('parameters', a=jims_parameters, shape=5)
    # Observed data is from a Multinomial distribution
    observed_data = pm.Multinomial(
        'observed_data', n=3, p=parameters, shape=3, observed=c)
```


```python
jims_model
```




$$
            \begin{array}{rcl}
            parameters &\sim & \text{Dirichlet}(\mathit{a}=array)\\observed_data &\sim & \text{Multinomial}(\mathit{n}=3, \mathit{p}=f(\text{parameters},~f(f(\text{parameters}))))
            \end{array}
            $$



## Sampling from the Model

Now for sampling from the model, this code draws 1000 samples from the posterior distribution with 2 different chains.


```python
with jims_model:
    # Sample from the posterior
    jims_trace = pm.sample(draws=1000, chains=2, tune=500, 
                      discard_tuned_samples=True)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 4 jobs)
    NUTS: [parameters]
    Sampling 2 chains: 100%|██████████| 3000/3000 [00:01<00:00, 1580.75draws/s]



```python
prop_cycle = plt.rcParams['axes.prop_cycle']
cs = [x['color'] for x in list(prop_cycle)]

ax = pm.traceplot(jims_trace, varnames = ['parameters'], figsize = (20, 8), combined = True);
ax[0][0].set_title('Posterior Probability Distribution'); ax[0][1].set_title('Trace Samples');
ax[0][0].set_xlabel('Probability'); ax[0][0].set_ylabel('Density');
ax[0][1].set_xlabel('Sample number');
add_legend(ax[0][0])
add_legend(ax[0][1])
```


![png](output_100_0.png)


This first plot is a kernel density estimate (KDE) for the sampled values which is just a probability density function (PDF) of the event probabilities. We see that the PDF for getting a rating 5 is pretty high. Let's quickly take a look at how Pam may feel now.


```python
#Pams Model
with pm.Model() as pams_model:
    # Parameters of the Multinomial are from a Dirichlet
    parameters = pm.Dirichlet('parameters', a=pams_parameters, shape=5)
    # Observed data is from a Multinomial distribution
    observed_data = pm.Multinomial(
        'observed_data', n=3, p=parameters, shape=3, observed=c)
    
with pams_model:
    # Sample from the posterior
    pams_trace = pm.sample(draws=1000, chains=2, tune=500, 
                      discard_tuned_samples=True)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 4 jobs)
    NUTS: [parameters]
    Sampling 2 chains: 100%|██████████| 3000/3000 [00:01<00:00, 2162.66draws/s]



```python
prop_cycle = plt.rcParams['axes.prop_cycle']
cs = [x['color'] for x in list(prop_cycle)]

ax = pm.traceplot(pams_trace, varnames = ['parameters'], figsize = (20, 8), combined = True);
ax[0][0].set_title('Posterior Probability Distribution'); ax[0][1].set_title('Trace Samples');
ax[0][0].set_xlabel('Probability'); ax[0][0].set_ylabel('Density');
ax[0][1].set_xlabel('Sample number');
add_legend(ax[0][0])
add_legend(ax[0][1])
```


![png](output_103_0.png)


We see here that that just by looking at the KDE that Pam is pretty confident that this restaurant is going to get a rating of 5.

Although we have a rough sense for what the probabilities are now, there are better ways to view the uncertainties within the samples. Starting with Jim.

# Jim's Uncertainty


```python
ax = pm.plot_posterior(jims_trace, varnames = ['parameters'], 
                       figsize = (20, 10), edgecolor = 'k');

plt.rcParams['font.size'] = 22
for i, a in enumerate(ratings):
    ax[i].set_title(a);
```


![png](output_107_0.png)


# Pam's Uncertainty


```python
ax = pm.plot_posterior(pams_trace, varnames = ['parameters'], 
                       figsize = (20, 10), edgecolor = 'k');

plt.rcParams['font.size'] = 22
for i, a in enumerate(ratings):
    ax[i].set_title(a);
```


![png](output_109_0.png)


These histrograms indicate how many times each probability was sampled from the MCMC process. Now we have point estimates for the probabilities, the mean, and the Bayesian equivalent of a confidence interval which is the 95% highest probability density known as the [credible interval](https://en.wikipedia.org/wiki/Credible_interval).

__OPTIONS OPTIONS OPTIONS__:

There is so much we can consider now, imagine performing this kind of analysis over time, building up a history of observations as Jim and Pam go visit more places, so we can better predict which places have the best rating. Now imagine tracking everyone and how they are all responding, visualizing all of these interactions, and then building it into an app. The options from here on out are endless, but here we have demonstrated a couple cool tools that hopefully will inspire you :).
