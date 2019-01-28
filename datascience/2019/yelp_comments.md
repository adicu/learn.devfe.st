---
layout: curriculum
title: Devfest Data Science Track 2019
source: "https://github.com/Siaan/data-science-2019"
---

To run this without any environment setup, go [here](https://mybinder.org/v2/gh/Siaan/data-science-2019/master?filepath=yelp_comments.ipynb) (this might take a minute or so to load).

Requirements for this track: Basic knowledge of Python.

# What do I need to get started?

But before we even get started, we have to set our environment up. This guide was written in Python 3.6. If you haven't already, download the latest version of Python (https://www.anaconda.com/download) and Pip. Once you have Python and Pip installed.


Once you have your notebook up and running, you can download all the data (yelp.csv) from GitHub. Make sure you have the data in the same directory as your notebook and then we’re good to go!

# A Quick Note on Jupyter

For those of you who are unfamiliar with Jupyter notebooks, I’ve provided a brief review of which functions will be particularly useful to move along with this tutorial.

In the image below, you’ll see three buttons labeled 1-3 that will be important for you to get a grasp of -- the save button (1), add cell button (2), and run cell button (3).


```python
#ignore this code; only used for image
from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "https://camo.githubusercontent.com/722c3d4565c0378b0b2902181281a76dbcf2506d/68747470733a2f2f7777772e7477696c696f2e636f6d2f626c6f672f77702d636f6e74656e742f75706c6f6164732f323031372f30392f717769674b704f7370683332416377524e4247415079506638383565736f346e534f756e677a4845614a35635a6365454836523941774e395a5169315558324b3444574b324e7676515941356e61704f497a2d7063666736597a644371534e475155507639625231706f4a365064336e5572546f5a314450337752485a6869455f446246624c737a2e706e67")
```




<img src="https://camo.githubusercontent.com/722c3d4565c0378b0b2902181281a76dbcf2506d/68747470733a2f2f7777772e7477696c696f2e636f6d2f626c6f672f77702d636f6e74656e742f75706c6f6164732f323031372f30392f717769674b704f7370683332416377524e4247415079506638383565736f346e534f756e677a4845614a35635a6365454836523941774e395a5169315558324b3444574b324e7676515941356e61704f497a2d7063666736597a644371534e475155507639625231706f4a365064336e5572546f5a314450337752485a6869455f446246624c737a2e706e67"/>



The first button is the button you’ll use to save your work as you go along (1). Feel free to choose when to save your work.

Next, we have the “add cell” button (2). Cells are blocks of code that you can run together. These are the building blocks of jupyter notebook because it provides the option of running code incrementally without having to to run all your code at once. Throughout this tutorial, you’ll see lines of code blocked off -- each one should correspond to a cell.

Lastly, there’s the “run cell” button (3). Jupyter Notebook doesn’t automatically run it your code for you; you have to tell it when by clicking this button. As with add button, once you’ve written each block of code in this tutorial onto your cell, you should then run it to see the output (if any). If any output is expected, note that it will also be shown in this tutorial so you know what to expect. Make sure to run your code as you go along because many blocks of code in this tutorial rely on previous cells.

# Background

You've likely heard the phrase 'data science' at some point of your life. Whether that be in the news, in a statistics or computer science course, or during your walk over to ferris for lunch. To demystify the term, let's first ask ourselves what do we mean by data?

Data is another ambiguous term, but more so because it can encompass so much. Anything that can be collected or transcribed can be data, whether it's numerical, text, images, sounds, anything!

# What is Data Science?

Data Science is where computer science and statistics intersect to solve problems involving sets of data. This can be simple statistical analysis to compute means, medians, standard deviations for a numerical dataset, but it can also mean creating robust algorithms.

In other words, it's taking techniques developed in the areas of statistics and math and using them to learn from some sort of data source.

# Is data science the same as machine learning?

Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to progressively improve their performance on a specific task. While they do have overlap, they are not the same! The ultimate goal of data science is to use data for some sort of insight, and that can often include learning how to do prediction from historical data. But it's not the full picture. Visualization, data acquisition and storage are just as important as using machine learning to "predict the future."

# Why is Data Science important?

Data Science has so much potential! By using data in creative and innovative ways, we can gain a lot of insight on the world, whether that be in economics, biology, sociology, math - any topic you can think of, data science has its own unique application.


```python
#start of check point 1 
```


```python
#end of check point 1
```


```python
#start of check point 2
```

# Data 
Our data contains 10,000 reviews, with the following information for each one:

| Data          | Info                             |
|---------------|----------------------------------|
|   business_id | ID of the business being reviewed|
|   date        | Day of the review was posted.    |
|   review_id   | ID for the posted review.        |
|   stars       | 1-5 rating for the business.     |
|   text        | Review text                      |
|   type        | Type of text.                    |
|   user_id     | User's ID.                       |
|   comments    | Cool, Useful, Funny.             |


# Importing the dataset

Firstly, let’s import the necessary Python libraries. NLTK is pretty much the standard library in Python library for text processing, which has many useful features. 
Today, we will just use NLTK for stopword removal.

In your terminal: conda install -c anaconda nltk


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
```


```python
#Should return True if package was successfully downloaded
import nltk 
nltk.download('stopwords')


```

    [nltk_data] Downloading package stopwords to /home/alan/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!





    True



A Data frame is a two-dimensional data structure, i.e., data is aligned in a tabular fashion in rows and columns.
Import the Yelp Reviews CSV file and store it in a Pandas dataframe called yelp. A Comma Separated Values (CSV) file is a plain text file that contains a list of data. These files are often used for exchanging data between different applications.


```python
yelp = pd.read_csv('yelp.csv')
```

Let’s get some basic information about the data. The .shape method tells us the number of rows and columns in the dataframe.


```python
yelp.shape
```




    (10000, 10)



We can learn more using .head(), and .describe().



head() - This function returns the first n rows for the object based on position. It is useful for quickly testing if your object has the right type of data in it.

describe() - Generates descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.


```python
yelp.head()
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
      <th>business_id</th>
      <th>date</th>
      <th>review_id</th>
      <th>stars</th>
      <th>text</th>
      <th>type</th>
      <th>user_id</th>
      <th>cool</th>
      <th>useful</th>
      <th>funny</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9yKzy9PApeiPPOUJEtnvkg</td>
      <td>2011-01-26</td>
      <td>fWKvX83p0-ka4JS3dc6E5A</td>
      <td>5</td>
      <td>My wife took me here on my birthday for breakf...</td>
      <td>review</td>
      <td>rLtl8ZkDX5vH5nAx9C3q5Q</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ZRJwVLyzEJq1VAihDhYiow</td>
      <td>2011-07-27</td>
      <td>IjZ33sJrzXqU-0X6U8NwyA</td>
      <td>5</td>
      <td>I have no idea why some people give bad review...</td>
      <td>review</td>
      <td>0a2KyEL0d3Yb1V6aivbIuQ</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6oRAC4uyJCsJl1X0WZpVSA</td>
      <td>2012-06-14</td>
      <td>IESLBzqUCLdSzSqm0eCSxQ</td>
      <td>4</td>
      <td>love the gyro plate. Rice is so good and I als...</td>
      <td>review</td>
      <td>0hT2KtfLiobPvh6cDC8JQg</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>_1QQZuf4zZOyFCvXc0o6Vg</td>
      <td>2010-05-27</td>
      <td>G-WvGaISbqqaMHlNnByodA</td>
      <td>5</td>
      <td>Rosie, Dakota, and I LOVE Chaparral Dog Park!!...</td>
      <td>review</td>
      <td>uZetl9T0NcROGOyFfughhg</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6ozycU1RpktNG2-1BroVtw</td>
      <td>2012-01-05</td>
      <td>1uJFq2r5QfJG_6ExMRCaGw</td>
      <td>5</td>
      <td>General Manager Scott Petello is a good egg!!!...</td>
      <td>review</td>
      <td>vYmM4KTsC8ZfQBg-j5MWkw</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



To get an insight on the length of each review, we can create a new column in yelp called text length. This column will store the number of characters in each review.


```python
yelp['text length'] = yelp['text'].str.len()
```

We can now see the text length column in our dataframe. Here I used Jupyter notebook's pretty-printing (default print format) of dataframes, because it shows you the beginning and end, which can be useful when you have sorted data.


```python
yelp
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
      <th>business_id</th>
      <th>date</th>
      <th>review_id</th>
      <th>stars</th>
      <th>text</th>
      <th>type</th>
      <th>user_id</th>
      <th>cool</th>
      <th>useful</th>
      <th>funny</th>
      <th>text length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9yKzy9PApeiPPOUJEtnvkg</td>
      <td>2011-01-26</td>
      <td>fWKvX83p0-ka4JS3dc6E5A</td>
      <td>5</td>
      <td>My wife took me here on my birthday for breakf...</td>
      <td>review</td>
      <td>rLtl8ZkDX5vH5nAx9C3q5Q</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>889</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ZRJwVLyzEJq1VAihDhYiow</td>
      <td>2011-07-27</td>
      <td>IjZ33sJrzXqU-0X6U8NwyA</td>
      <td>5</td>
      <td>I have no idea why some people give bad review...</td>
      <td>review</td>
      <td>0a2KyEL0d3Yb1V6aivbIuQ</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1345</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6oRAC4uyJCsJl1X0WZpVSA</td>
      <td>2012-06-14</td>
      <td>IESLBzqUCLdSzSqm0eCSxQ</td>
      <td>4</td>
      <td>love the gyro plate. Rice is so good and I als...</td>
      <td>review</td>
      <td>0hT2KtfLiobPvh6cDC8JQg</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>76</td>
    </tr>
    <tr>
      <th>3</th>
      <td>_1QQZuf4zZOyFCvXc0o6Vg</td>
      <td>2010-05-27</td>
      <td>G-WvGaISbqqaMHlNnByodA</td>
      <td>5</td>
      <td>Rosie, Dakota, and I LOVE Chaparral Dog Park!!...</td>
      <td>review</td>
      <td>uZetl9T0NcROGOyFfughhg</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>419</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6ozycU1RpktNG2-1BroVtw</td>
      <td>2012-01-05</td>
      <td>1uJFq2r5QfJG_6ExMRCaGw</td>
      <td>5</td>
      <td>General Manager Scott Petello is a good egg!!!...</td>
      <td>review</td>
      <td>vYmM4KTsC8ZfQBg-j5MWkw</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>469</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-yxfBYGB6SEqszmxJxd97A</td>
      <td>2007-12-13</td>
      <td>m2CKSsepBCoRYWxiRUsxAg</td>
      <td>4</td>
      <td>Quiessence is, simply put, beautiful.  Full wi...</td>
      <td>review</td>
      <td>sqYN3lNgvPbPCTRsMFu27g</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>2094</td>
    </tr>
    <tr>
      <th>6</th>
      <td>zp713qNhx8d9KCJJnrw1xA</td>
      <td>2010-02-12</td>
      <td>riFQ3vxNpP4rWLk_CSri2A</td>
      <td>5</td>
      <td>Drop what you're doing and drive here. After I...</td>
      <td>review</td>
      <td>wFweIWhv2fREZV_dYkz_1g</td>
      <td>7</td>
      <td>7</td>
      <td>4</td>
      <td>1565</td>
    </tr>
    <tr>
      <th>7</th>
      <td>hW0Ne_HTHEAgGF1rAdmR-g</td>
      <td>2012-07-12</td>
      <td>JL7GXJ9u4YMx7Rzs05NfiQ</td>
      <td>4</td>
      <td>Luckily, I didn't have to travel far to make m...</td>
      <td>review</td>
      <td>1ieuYcKS7zeAv_U15AB13A</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>274</td>
    </tr>
    <tr>
      <th>8</th>
      <td>wNUea3IXZWD63bbOQaOH-g</td>
      <td>2012-08-17</td>
      <td>XtnfnYmnJYi71yIuGsXIUA</td>
      <td>4</td>
      <td>Definitely come for Happy hour! Prices are ama...</td>
      <td>review</td>
      <td>Vh_DlizgGhSqQh4qfZ2h6A</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>349</td>
    </tr>
    <tr>
      <th>9</th>
      <td>nMHhuYan8e3cONo3PornJA</td>
      <td>2010-08-11</td>
      <td>jJAIXA46pU1swYyRCdfXtQ</td>
      <td>5</td>
      <td>Nobuo shows his unique talents with everything...</td>
      <td>review</td>
      <td>sUNkXg8-KFtCMQDV6zRzQg</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>186</td>
    </tr>
    <tr>
      <th>10</th>
      <td>AsSCv0q_BWqIe3mX2JqsOQ</td>
      <td>2010-06-16</td>
      <td>E11jzpKz9Kw5K7fuARWfRw</td>
      <td>5</td>
      <td>The oldish man who owns the store is as sweet ...</td>
      <td>review</td>
      <td>-OMlS6yWkYjVldNhC31wYg</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>298</td>
    </tr>
    <tr>
      <th>11</th>
      <td>e9nN4XxjdHj4qtKCOPq_vg</td>
      <td>2011-10-21</td>
      <td>3rPt0LxF7rgmEUrznoH22w</td>
      <td>5</td>
      <td>Wonderful Vietnamese sandwich shoppe. Their ba...</td>
      <td>review</td>
      <td>C1rHp3dmepNea7XiouwB6Q</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>321</td>
    </tr>
    <tr>
      <th>12</th>
      <td>h53YuCiIDfEFSJCQpk8v1g</td>
      <td>2010-01-11</td>
      <td>cGnKNX3I9rthE0-TH24-qA</td>
      <td>5</td>
      <td>They have a limited time thing going on right ...</td>
      <td>review</td>
      <td>UPtysDF6cUDUxq2KY-6Dcg</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>433</td>
    </tr>
    <tr>
      <th>13</th>
      <td>WGNIYMeXPyoWav1APUq7jA</td>
      <td>2011-12-23</td>
      <td>FvEEw1_OsrYdvwLV5Hrliw</td>
      <td>4</td>
      <td>Good tattoo shop. Clean space, multiple artist...</td>
      <td>review</td>
      <td>Xm8HXE1JHqscXe5BKf0GFQ</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>593</td>
    </tr>
    <tr>
      <th>14</th>
      <td>yc5AH9H71xJidA_J2mChLA</td>
      <td>2010-05-20</td>
      <td>pfUwBKYYmUXeiwrhDluQcw</td>
      <td>4</td>
      <td>I'm 2 weeks new to Phoenix. I looked up Irish ...</td>
      <td>review</td>
      <td>JOG-4G4e8ae3lx_szHtR8g</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1206</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Vb9FPCEL6Ly24PNxLBaAFw</td>
      <td>2011-03-20</td>
      <td>HvqmdqWcerVWO3Gs6zbrOw</td>
      <td>2</td>
      <td>Was it worth the 21$ for a salad and small piz...</td>
      <td>review</td>
      <td>ylWOj2y7TV2e3yYeWhu2QA</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>705</td>
    </tr>
    <tr>
      <th>16</th>
      <td>supigcPNO9IKo6olaTNV-g</td>
      <td>2008-10-12</td>
      <td>HXP_0Ul-FCmA4f-k9CqvaQ</td>
      <td>3</td>
      <td>We went here on a Saturday afternoon and this ...</td>
      <td>review</td>
      <td>SBbftLzfYYKItOMFwOTIJg</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>1469</td>
    </tr>
    <tr>
      <th>17</th>
      <td>O510Re68mOy9dU490JTKCg</td>
      <td>2010-05-03</td>
      <td>j4SIzrIy0WrmW4yr4--Khg</td>
      <td>5</td>
      <td>okay this is the best place EVER! i grew up sh...</td>
      <td>review</td>
      <td>u1KWcbPMvXFEEYkZZ0Yktg</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>363</td>
    </tr>
    <tr>
      <th>18</th>
      <td>b5cEoKR8iQliq-yT2_O0LQ</td>
      <td>2009-03-06</td>
      <td>v0cTd3PNpYCkTyGKSpOfGA</td>
      <td>3</td>
      <td>I met a friend for lunch yesterday. \n\nLoved ...</td>
      <td>review</td>
      <td>UsULgP4bKA8RMzs8dQzcsA</td>
      <td>5</td>
      <td>6</td>
      <td>4</td>
      <td>1161</td>
    </tr>
    <tr>
      <th>19</th>
      <td>4JzzbSbK9wmlOBJZWYfuCg</td>
      <td>2011-11-17</td>
      <td>a0lCu-j2Sk_kHQsZi_eNgw</td>
      <td>4</td>
      <td>They've gotten better and better for me in the...</td>
      <td>review</td>
      <td>nDBly08j5URmrHQ2JCbyiw</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>726</td>
    </tr>
    <tr>
      <th>20</th>
      <td>8FNO4D3eozpIjj0k3q5Zbg</td>
      <td>2008-10-08</td>
      <td>MuqugTuR5DdIPcZ2IVP3aQ</td>
      <td>3</td>
      <td>DVAP....\n\nYou have to go at least once in yo...</td>
      <td>review</td>
      <td>C6IOtaaYdLIT5fWd7ZYIuA</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>565</td>
    </tr>
    <tr>
      <th>21</th>
      <td>tdcjXyFLMKAsvRhURNOkCg</td>
      <td>2011-06-28</td>
      <td>LmuKVFh03Uz318VKnUWrxA</td>
      <td>5</td>
      <td>This place shouldn't even be reviewed - becaus...</td>
      <td>review</td>
      <td>YN3ZLOdg8kpnfbVcIhuEZA</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>104</td>
    </tr>
    <tr>
      <th>22</th>
      <td>eFA9dqXT5EA_TrMgbo03QQ</td>
      <td>2011-07-13</td>
      <td>CQYc8hgKxV4enApDkx0IhA</td>
      <td>5</td>
      <td>first time my friend and I went there... it wa...</td>
      <td>review</td>
      <td>6lg55RIP23VhjYEBXJ8Njw</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>148</td>
    </tr>
    <tr>
      <th>23</th>
      <td>IJ0o6b8bJFAbG6MjGfBebQ</td>
      <td>2010-09-05</td>
      <td>Dx9sfFU6Zn0GYOckijom-g</td>
      <td>1</td>
      <td>U can go there n check the car out. If u wanna...</td>
      <td>review</td>
      <td>zRlQEDYd_HKp0VS3hnAffA</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>594</td>
    </tr>
    <tr>
      <th>24</th>
      <td>JhupPnWfNlMJivnWB5druA</td>
      <td>2011-05-22</td>
      <td>cFtQnKzn2VDpBedy_TxlvA</td>
      <td>5</td>
      <td>I love this place! I have been coming here for...</td>
      <td>review</td>
      <td>13xj6FSvYO0rZVRv5XZp4w</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>294</td>
    </tr>
    <tr>
      <th>25</th>
      <td>wzP2yNpV5p04nh0injjymA</td>
      <td>2010-05-26</td>
      <td>ChBeixVZerfFkeO0McdlbA</td>
      <td>4</td>
      <td>This place is great.  A nice little ole' fashi...</td>
      <td>review</td>
      <td>rLtl8ZkDX5vH5nAx9C3q5Q</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1012</td>
    </tr>
    <tr>
      <th>26</th>
      <td>qjmCVYkwP-HDa35jwYucbQ</td>
      <td>2013-01-03</td>
      <td>kZ4TzrVX6qeF0OvrVTGVEw</td>
      <td>5</td>
      <td>I love love LOVE this place. My boss (who is i...</td>
      <td>review</td>
      <td>fpItLlgimq0nRltWOkuJJw</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>921</td>
    </tr>
    <tr>
      <th>27</th>
      <td>wct7rZKyZqZftzmAU-vhWQ</td>
      <td>2008-03-21</td>
      <td>B5h25WK28rJjx4KHm4gr7g</td>
      <td>4</td>
      <td>Not that my review will mean much given the mo...</td>
      <td>review</td>
      <td>RRTraCQw77EU4yZh0BBTag</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>550</td>
    </tr>
    <tr>
      <th>28</th>
      <td>vz2zQQSjy-NnnKLZzjjoxA</td>
      <td>2011-03-30</td>
      <td>Y_ERKao0J5WsRiCtlKSNSA</td>
      <td>4</td>
      <td>Came here for breakfast yesterday, it had been...</td>
      <td>review</td>
      <td>EP3cGJvYiuOwumerwADplg</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1011</td>
    </tr>
    <tr>
      <th>29</th>
      <td>i213sY5rhkfCO8cD-FPr1A</td>
      <td>2012-07-12</td>
      <td>hre97jjSwon4bn1muHKOJg</td>
      <td>4</td>
      <td>Always reliably good.  Great beer selection as...</td>
      <td>review</td>
      <td>kpbhy1zPewGDmdNfNqQp-g</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>225</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9970</th>
      <td>R6aazv8FB-6BeanY3ag8kw</td>
      <td>2009-09-26</td>
      <td>gP17ykqduf3AlewSaRb61w</td>
      <td>5</td>
      <td>This place is super cute lunch joint.  I had t...</td>
      <td>review</td>
      <td>mtoKqaQjGPWEc5YZbrYV9w</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>432</td>
    </tr>
    <tr>
      <th>9971</th>
      <td>JOZqBKIOB8WEBAWm7v1JFA</td>
      <td>2008-07-22</td>
      <td>QI9rfeWrZnvK5ojz8cEoRg</td>
      <td>5</td>
      <td>The staff is great, the food is great, even th...</td>
      <td>review</td>
      <td>uBAMd01ZtGXaHrRD6THNzg</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>318</td>
    </tr>
    <tr>
      <th>9972</th>
      <td>OllL0G9Kh_k1lx-2vrFDXQ</td>
      <td>2012-10-23</td>
      <td>U23UfuxN9DpAU0Dslc5KjQ</td>
      <td>4</td>
      <td>Yay, even though I miss living in Coronado I a...</td>
      <td>review</td>
      <td>Gh1EXuS42DY3rV_MzFpJpg</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>411</td>
    </tr>
    <tr>
      <th>9973</th>
      <td>XHr5mXFgobOHoxbPJxmYdg</td>
      <td>2009-09-28</td>
      <td>udMiWjeG0OGcb4nNddDkBg</td>
      <td>5</td>
      <td>Wow!  Went on a Sunday around 11am - busy but ...</td>
      <td>review</td>
      <td>yRYNx24kUDRRBfJu1Rcojg</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>353</td>
    </tr>
    <tr>
      <th>9974</th>
      <td>cdacUBBL2tDbDnB1EfhpQw</td>
      <td>2009-12-16</td>
      <td>bVU-_x9ijxjEImNluy84OA</td>
      <td>2</td>
      <td>If Cowboy Ciao is the best restaurant in Scott...</td>
      <td>review</td>
      <td>V9Uqt00HXwXT6mzsVCjMAw</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>473</td>
    </tr>
    <tr>
      <th>9975</th>
      <td>EWMwV5V9BxNs_U6nNVMeqw</td>
      <td>2007-10-20</td>
      <td>g4LsVAoafmUDHiS-_yN4tA</td>
      <td>5</td>
      <td>When I lived in Phoenix, I was a regular at Fe...</td>
      <td>review</td>
      <td>TLj3XaclA7V4ldJ5yNP-9Q</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1015</td>
    </tr>
    <tr>
      <th>9976</th>
      <td>iDYzGVIF1TDWdjHNgNjCVw</td>
      <td>2009-09-11</td>
      <td>bKjMcpNj0xSu2UI2EFQn1g</td>
      <td>3</td>
      <td>I was looking for chile rellenos and this plac...</td>
      <td>review</td>
      <td>2tUCLMHQKz4kA1VlRB_w0Q</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>465</td>
    </tr>
    <tr>
      <th>9977</th>
      <td>iDYzGVIF1TDWdjHNgNjCVw</td>
      <td>2012-10-30</td>
      <td>qaNZyCUJA6Yp0mvPBCknPQ</td>
      <td>5</td>
      <td>Why did I wait so long to try this neighborhoo...</td>
      <td>review</td>
      <td>Id-8-NMEKxeXBR44eUdDeA</td>
      <td>3</td>
      <td>6</td>
      <td>3</td>
      <td>2918</td>
    </tr>
    <tr>
      <th>9978</th>
      <td>9Y3aQAVITkEJYe5vLZr13w</td>
      <td>2010-04-01</td>
      <td>ZoTUU6EJ1OBNr7mhqxHBLw</td>
      <td>5</td>
      <td>This is the place for a fabulos breakfast!! I ...</td>
      <td>review</td>
      <td>vasHsAZEgLZGJDTlIweUYQ</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>493</td>
    </tr>
    <tr>
      <th>9979</th>
      <td>GV1P1x9eRb4iZHCxj5_IjA</td>
      <td>2012-12-07</td>
      <td>eVUs1C4yaVJNrc7SGTAheg</td>
      <td>5</td>
      <td>Highly recommend. This is my second time here ...</td>
      <td>review</td>
      <td>bJFdmJJxfXgCYA5DMmyeqQ</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>244</td>
    </tr>
    <tr>
      <th>9980</th>
      <td>GHYOl_cnERMOhkCK_mGAlA</td>
      <td>2011-07-03</td>
      <td>Q-y3jSqccdytKxAyo1J0Xg</td>
      <td>5</td>
      <td>5 stars for the great $5 happy hour specials. ...</td>
      <td>review</td>
      <td>xZvRLPJ1ixhFVomkXSfXAw</td>
      <td>6</td>
      <td>6</td>
      <td>4</td>
      <td>393</td>
    </tr>
    <tr>
      <th>9981</th>
      <td>AX8lx9wHNYT45lyd7pxaYw</td>
      <td>2008-11-27</td>
      <td>IyunTh7jnG7v3EYwfF3hPw</td>
      <td>5</td>
      <td>We brought the entire family to Giuseppe's las...</td>
      <td>review</td>
      <td>fczQCSmaWF78toLEmb0Zsw</td>
      <td>10</td>
      <td>9</td>
      <td>5</td>
      <td>885</td>
    </tr>
    <tr>
      <th>9982</th>
      <td>KV-yJLmlODfUG1Mkds6kYw</td>
      <td>2012-02-25</td>
      <td>rIgZgxJPWTacq3mV6DfWfg</td>
      <td>4</td>
      <td>Best corned beef sandwich I've had anywhere at...</td>
      <td>review</td>
      <td>J-oVr0th2Y7ltPPOwy0Z8Q</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>240</td>
    </tr>
    <tr>
      <th>9983</th>
      <td>24V8QQWO6VaVggHdxjQQ_A</td>
      <td>2010-06-06</td>
      <td>PqiIeFOiVr-tj_FtHGAH2g</td>
      <td>3</td>
      <td>3.5 stars. \n\nWe decided to check this place ...</td>
      <td>review</td>
      <td>LaEj3VpQh7bgpAZLzSRRrw</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>861</td>
    </tr>
    <tr>
      <th>9984</th>
      <td>wepFVY82q_tuDzG6lQjHWw</td>
      <td>2012-02-12</td>
      <td>spusZYROtBKw_5tv3gYm4Q</td>
      <td>1</td>
      <td>Went last night to Whore Foods to get basics t...</td>
      <td>review</td>
      <td>W7zmm1uzlyUkEqpSG7PlBw</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1673</td>
    </tr>
    <tr>
      <th>9985</th>
      <td>EMGkbiCMfMTflQux-_JY7Q</td>
      <td>2012-10-17</td>
      <td>wB-f0xfx7WIyrOsRJMkDOg</td>
      <td>4</td>
      <td>Awesome food! Little pricey but delicious. Lov...</td>
      <td>review</td>
      <td>9MJAacmjxtctbI3xncsK5Q</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>68</td>
    </tr>
    <tr>
      <th>9986</th>
      <td>oCA2OZcd_Jo_ggVmUx3WVw</td>
      <td>2012-03-31</td>
      <td>ijPZPKKWDqdWOIqYkUsJJw</td>
      <td>4</td>
      <td>I came here in December and look forward to my...</td>
      <td>review</td>
      <td>yzwPJdn6yd2ccZqfy4LhUA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>647</td>
    </tr>
    <tr>
      <th>9987</th>
      <td>r-a-Cn9hxdEnYTtVTB5bMQ</td>
      <td>2012-04-07</td>
      <td>j9HwZZoBBmJgOlqDSuJcxg</td>
      <td>1</td>
      <td>The food is delicious.  The service:  discrimi...</td>
      <td>review</td>
      <td>toPtsUtYoRB-5-ThrOy2Fg</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>200</td>
    </tr>
    <tr>
      <th>9988</th>
      <td>xY1sPHTA2RGVFlh5tZhs9g</td>
      <td>2012-06-02</td>
      <td>TM8hdYqs5Zi1jO5Yrq6E0g</td>
      <td>4</td>
      <td>For our first time we had a great time! Our se...</td>
      <td>review</td>
      <td>GvaNZY4poCcd3H4WxHjrLQ</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>496</td>
    </tr>
    <tr>
      <th>9989</th>
      <td>mQUC-ATrFuMQSaDQb93Pug</td>
      <td>2011-10-01</td>
      <td>ta2P9joJqeFB8BzFp-AzjA</td>
      <td>5</td>
      <td>Great food and service! Country food at its best!</td>
      <td>review</td>
      <td>fKaO8fR1IAcfvZb6cBrs2w</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>49</td>
    </tr>
    <tr>
      <th>9990</th>
      <td>R8VwdLyvsp9iybNqRvm94g</td>
      <td>2011-10-03</td>
      <td>pcEeHdAJPoFNF23es0kKWg</td>
      <td>5</td>
      <td>Yes I do rock the hipster joints.  I dig this ...</td>
      <td>review</td>
      <td>b92Y3tyWTQQZ5FLifex62Q</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>263</td>
    </tr>
    <tr>
      <th>9991</th>
      <td>WJ5mq4EiWYAA4Vif0xDfdg</td>
      <td>2011-12-05</td>
      <td>EuHX-39FR7tyyG1ElvN1Jw</td>
      <td>5</td>
      <td>Only 4 stars? \n\n(A few notes: The folks that...</td>
      <td>review</td>
      <td>hTau-iNZFwoNsPCaiIUTEA</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>908</td>
    </tr>
    <tr>
      <th>9992</th>
      <td>f96lWMIAUhYIYy9gOktivQ</td>
      <td>2009-03-10</td>
      <td>YF17z7HWlMj6aezZc-pVEw</td>
      <td>5</td>
      <td>I'm not normally one to jump at reviewing a ch...</td>
      <td>review</td>
      <td>W_QXYA7A0IhMrvbckz7eVg</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>1326</td>
    </tr>
    <tr>
      <th>9993</th>
      <td>maB4VHseFUY2TmPtAQnB9Q</td>
      <td>2011-06-27</td>
      <td>SNnyYHI9rw9TTltVX3TF-A</td>
      <td>4</td>
      <td>Judging by some of the reviews, maybe I went o...</td>
      <td>review</td>
      <td>T46gxPbJMWmlLyr7GxQLyQ</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>426</td>
    </tr>
    <tr>
      <th>9994</th>
      <td>L3BSpFvxcNf3T_teitgt6A</td>
      <td>2012-03-19</td>
      <td>0nxb1gIGFgk3WbC5zwhKZg</td>
      <td>5</td>
      <td>Let's see...what is there NOT to like about Su...</td>
      <td>review</td>
      <td>OzOZv-Knlw3oz9K5Kh5S6A</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1968</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>VY_tvNUCCXGXQeSvJl757Q</td>
      <td>2012-07-28</td>
      <td>Ubyfp2RSDYW0g7Mbr8N3iA</td>
      <td>3</td>
      <td>First visit...Had lunch here today - used my G...</td>
      <td>review</td>
      <td>_eqQoPtQ3e3UxLE4faT6ow</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>668</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>EKzMHI1tip8rC1-ZAy64yg</td>
      <td>2012-01-18</td>
      <td>2XyIOQKbVFb6uXQdJ0RzlQ</td>
      <td>4</td>
      <td>Should be called house of deliciousness!\n\nI ...</td>
      <td>review</td>
      <td>ROru4uk5SaYc3rg8IU7SQw</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>881</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>53YGfwmbW73JhFiemNeyzQ</td>
      <td>2010-11-16</td>
      <td>jyznYkIbpqVmlsZxSDSypA</td>
      <td>4</td>
      <td>I recently visited Olive and Ivy for business ...</td>
      <td>review</td>
      <td>gGbN1aKQHMgfQZkqlsuwzg</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1425</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>9SKdOoDHcFoxK5ZtsgHJoA</td>
      <td>2012-12-02</td>
      <td>5UKq9WQE1qQbJ0DJbc-B6Q</td>
      <td>2</td>
      <td>My nephew just moved to Scottsdale recently so...</td>
      <td>review</td>
      <td>0lyVoNazXa20WzUyZPLaQQ</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>880</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>pF7uRzygyZsltbmVpjIyvw</td>
      <td>2010-10-16</td>
      <td>vWSmOhg2ID1MNZHaWapGbA</td>
      <td>5</td>
      <td>4-5 locations.. all 4.5 star average.. I think...</td>
      <td>review</td>
      <td>KSBFytcdjPKZgXKQnYQdkA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>461</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 11 columns</p>
</div>



# Exploring the dataset and Creating Visualizations

Let’s visualise the data a little more by plotting some graphs with the Seaborn library.
Seaborn’s FacetGrid allows us to create a grid of histograms placed side by side. We can use FacetGrid to see if there’s any relationship between our newly created text length feature and the stars rating.


```python
#sns.set();
#g = sns.distplot(yelp)
#g.map(plt.hist, 'text length', bins=50)
```

Seems like overall, the distribution of text length is similar across all five ratings. However, the number of text reviews seems to be skewed a lot higher towards the 4-star and 5-star ratings (that is - there are more 4 and 5-star ratings). This may cause some issues later on in the process.

Next, let’s create a box plot of the text length for each star rating. The advantage of a box plot is in comparing distributions across many different groups all at once.


```python
sns.boxplot(x='stars', y='text length', data=yelp)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9459243048>




![png](output_48_1.png)


From the plot, looks like the 1-star and 2-star ratings have much longer text, but there are many outliers (which can be seen as points above the boxes). An outlier can distort results, such as dragging the mean in a certain direction, and can lead to faulty conclusions being made. Because of this, maybe text length won’t be such a useful feature to consider after all.


Correlation is used to test relationships between quantitative variables or categorical variables. In other words, it’s a measure of how things are related. Let’s group the data by the star rating, and see if we can find a correlation among the user comments: cool, useful, and funny. We can use the .corr() method from Pandas to find any correlations in the dataframe.

"groupby" operation involves some combination of splitting the object, applying a function, and combining the results. This can be used to group large amounts of data and compute operations on these groups.


```python
stars = yelp.groupby('stars').mean()
stars.corr()
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
      <th>cool</th>
      <th>useful</th>
      <th>funny</th>
      <th>text length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cool</th>
      <td>1.000000</td>
      <td>-0.743329</td>
      <td>-0.944939</td>
      <td>-0.857664</td>
    </tr>
    <tr>
      <th>useful</th>
      <td>-0.743329</td>
      <td>1.000000</td>
      <td>0.894506</td>
      <td>0.699881</td>
    </tr>
    <tr>
      <th>funny</th>
      <td>-0.944939</td>
      <td>0.894506</td>
      <td>1.000000</td>
      <td>0.843461</td>
    </tr>
    <tr>
      <th>text length</th>
      <td>-0.857664</td>
      <td>0.699881</td>
      <td>0.843461</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



To visualise these correlations, we can use Seaborn’s heatmap. A heatmap is a graphical representation of data where the individual values contained in a matrix are represented as colors.


```python
sns.heatmap(data=stars.corr(), annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9458901828>




![png](output_54_1.png)


Looking at the map, funny is strongly correlated with useful, and useful seems strongly correlated with text length. We can also see a negative correlation between cool and the other three features.

# Independent and dependent variables

Our task is to predict if a review is either bad or good, so let’s just grab reviews that are either 1 or 5 stars from the yelp dataframe. We can store the resulting reviews in a new dataframe called yelp_class.



```python
yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]
yelp_class.shape
```




    (4086, 11)



We can see from .shape that yelp_class only has 4086 reviews, compared to the 10,000 reviews in the original dataset. This is because we aren’t taking into account the reviews rated 2, 3, and 4 stars.

In machine learning and statistics, classification is a supervised learning approach in which the computer program learns from the data input given to it and then uses this learning to classify new observation. This data set may simply be bi-class (like identifying whether the person is male or female or that the mail is spam or non-spam) or it may be multi-class too.

Next, let’s create the X and y for our classification task. X will be the text column of yelp_class, and y will be the stars column.



```python
X = yelp_class['text']
y = yelp_class['stars']
```

# Text pre-processing

The main issue with our data is that it is all in plain-text format.


```python
X[0]
```




    'My wife took me here on my birthday for breakfast and it was excellent.  The weather was perfect which made sitting outside overlooking their grounds an absolute pleasure.  Our waitress was excellent and our food arrived quickly on the semi-busy Saturday morning.  It looked like the place fills up pretty quickly so the earlier you get here the better.\n\nDo yourself a favor and get their Bloody Mary.  It was phenomenal and simply the best I\'ve ever had.  I\'m pretty sure they only use ingredients from their garden and blend them fresh when you order it.  It was amazing.\n\nWhile EVERYTHING on the menu looks excellent, I had the white truffle scrambled eggs vegetable skillet and it was tasty and delicious.  It came with 2 pieces of their griddled bread with was amazing and it absolutely made the meal complete.  It was the best "toast" I\'ve ever had.\n\nAnyway, I can\'t wait to go back!'



The classification algorithm will need some sort of feature vector in order to perform the classification task. The simplest way to convert a corpus to a vector format is the bag-of-words approach, where each unique word in a text will be represented by one number.

A feature vector is just a vector that contains information describing an object's important characteristics. In image processing, features can take many forms. For example: a simple feature representation of an image is the raw intensity value of each pixel.

First, let’s write a function that will split a message into its individual words, and return a list. We will also remove the very common words (such as “the”, “a”, “an”, etc.), also known as stopwords. To do this, we can take advantage of the NLTK library. The function below removes punctuation, stopwords, and returns a list of the remaining words, or tokens.



```python
import string

def text_process(text):
   
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
```

To check if the function works, let’s pass in some random text and see if it gets processed correctly.



```python
sample_text = "Hey there! This is a sample review, which happens to contain punctuations."
```


```python
text_process(sample_text)
```




    ['Hey', 'sample', 'review', 'happens', 'contain', 'punctuations']



# Vectorisation

There are several Python libraries which provide solid implementations of a range of machine learning algorithms. One of the best known is Scikit-Learn, a package that provides efficient versions of a large number of common algorithms.

At the moment, we have our reviews as lists of tokens (a list of words). To enable Scikit-learn algorithms to work on our text, we need to convert each review into a vector.

We can use Scikit-learn’s CountVectorizer to convert the text collection into a matrix of token counts. You can imagine this resulting matrix as a 2-D matrix, where each row is a unique word, and each column is a review.

Let’s import CountVectorizer and fit an instance to our review text (stored in X), passing in our text_process function as the analyser.


```python
from sklearn.feature_extraction.text import CountVectorizer
```


```python
bow_transformer = CountVectorizer(analyzer=text_process).fit(X)
```

Since there are many reviews, we can expect a lot of zero counts for the presence of a word in the collection. Because of this, Scikit-learn will output a sparse matrix (a matrix that is comprised of mostly zero values.)

Now, we can look at the size of the vocabulary stored in the vectoriser (based on X) like this:



```python
len(bow_transformer.vocabulary_)
```




    26435



To illustrate how the vectoriser works, let’s try a random review and get its bag-of-word counts as a vector. Here’s the twenty-fifth review as plain-text:


```python
review_25 = X[24]
review_25
```




    "I love this place! I have been coming here for ages.\nMy favorites: Elsa's Chicken sandwich, any of their burgers, dragon chicken wings, china's little chicken sandwich, and the hot pepper chicken sandwich. The atmosphere is always fun and the art they display is very abstract but totally cool!"



Now let’s see our review represented as a vector:


```python
bow_25 = bow_transformer.transform([review_25])
bow_25
```




    <1x26435 sparse matrix of type '<class 'numpy.int64'>'
    	with 24 stored elements in Compressed Sparse Row format>



This means that there are 24 unique words in the review (after removing stopwords). Two of them appear thrice, and the rest appear only once. Let’s go ahead and check which ones appear thrice:



```python
print(bow_transformer.get_feature_names()[11443])
print(bow_transformer.get_feature_names()[22077])
```

    chicken
    sandwich


Now that we’ve seen how the vectorisation process works, we can transform our X series into a sparse matrix. To do this, let’s use the .transform() method on our bag-of-words transformed object.



```python
X = bow_transformer.transform(X)
```

We can check out the shape of our new X.


```python
print('Shape of Sparse Matrix: ', X.shape)
print('Amount of Non-Zero occurrences: ', X.nnz)
# Percentage of non-zero values
density = (100.0 * X.nnz / (X.shape[0] * X.shape[1]))
print("Density: {}".format((density)))
```

    Shape of Sparse Matrix:  (4086, 26435)
    Amount of Non-Zero occurrences:  222391
    Density: 0.2058920276658241


# Training data and test data

The data we use is usually split into training data and test data. The training set contains a known output and the model learns on this data in order to be generalized to other data later on. We have the test dataset (or subset) in order to test our model’s prediction on this subset.

As we have finished processing the review text in X, It’s time to split our X and y into a training and a test set using train_test_split from Scikit-learn. We will use 30% of the dataset for testing.


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```

# Training our model

What is Naive Bayes algorithm?

It is a classification technique based on Bayes’ Theorem with an assumption of independence among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. For example, a fruit may be considered to be an apple if it is red, round, and about 3 inches in diameter. Even if these features depend on each other or upon the existence of the other features, all of these properties independently contribute to the probability that this fruit is an apple and that is why it is known as ‘Naive’.

Multinomial (consisting of several terms) Naive Bayes is a specialised version of Naive Bayes designed more for text documents. Let’s build a Multinomial Naive Bayes model and fit it to our training set (X_train and y_train).


```python
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)
```




    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)



# Testing and evaluating our model

Our model has now been trained! It’s time to see how well it predicts the ratings of previously unseen reviews (reviews from the test set). First, let’s store the predictions as a separate numpy array called preds.



```python
preds = nb.predict(X_test)
```

Next, let’s evaluate our predictions against the actual ratings (stored in y_test) using confusion_matrix and classification_report from Scikit-learn.



```python
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, preds))
print('\n')
print(classification_report(y_test, preds))
```

    [[157  71]
     [ 24 974]]
    
    
                  precision    recall  f1-score   support
    
               1       0.87      0.69      0.77       228
               5       0.93      0.98      0.95       998
    
       micro avg       0.92      0.92      0.92      1226
       macro avg       0.90      0.83      0.86      1226
    weighted avg       0.92      0.92      0.92      1226
    


Looks like our model has achieved 92% accuracy! This means that our model can predict whether a user liked a local business or not, based on what they typed!

# Data Bias

Although our model achieved quite a high accuracy, there are some issues with bias caused by the dataset.
Let’s take some singular reviews, and see what rating our model predicts for each one.

Machine bias is the effect of erroneous assumptions in machine learning processes. Bias reflects problems related to the gathering or use of data, where systems draw improper conclusions about data sets. 

# Predicting a single positive review


```python
positive_review = yelp_class['text'][59]
positive_review
```




    "This restaurant is incredible, and has the best pasta carbonara and the best tiramisu I've had in my life. All the food is wonderful, though. The calamari is not fried. The bread served with dinner comes right out of the oven, and the tomatoes are the freshest I've tasted outside of my mom's own garden. This is great attention to detail.\n\nI can no longer eat at any other Italian restaurant without feeling slighted. This is the first place I want take out-of-town visitors I'm looking to impress.\n\nThe owner, Jon, is helpful, friendly, and really cares about providing a positive dining experience. He's spot on with his wine recommendations, and he organizes wine tasting events which you can find out about by joining the mailing list or Facebook page."



Seems like someone had the time of their life at this place, right? We can expect our model to predict a rating of 5 for this review.


```python
positive_review_transformed = bow_transformer.transform([positive_review])
nb.predict(positive_review_transformed)[0]
```




    5



Our model thinks this review is positive, just as we expected.

# Predicting a single negative review


```python
negative_review = yelp_class['text'][281]
negative_review

```




    'Still quite poor both in service and food. maybe I made a mistake and ordered Sichuan Gong Bao ji ding for what seemed like people from canton district. Unfortunately to get the good service U have to speak Mandarin/Cantonese. I do speak a smattering but try not to use it as I never feel confident about the intonation. \n\nThe dish came out with zichini and bell peppers (what!??)  Where is the peanuts the dried fried red peppers and the large pieces of scallion. On pointing this out all I got was " Oh you like peanuts.. ok I will put some on" and she then proceeded to get some peanuts and sprinkle it on the chicken.\n\nWell at that point I was happy that atleast the chicken pieces were present else she would probably end up sprinkling raw chicken pieces on it like the raw peanuts she dumped on top of the food. \n\nWell then  I spoke a few chinese words and the scowl turned into a smile and she then became a bit more friendlier. \n\nUnfortunately I do not condone this type of behavior. It is all in poor taste...'




```python
negative_review_transformed = bow_transformer.transform([negative_review])
nb.predict(negative_review_transformed)[0]
```




    1



Our model is right again!

# Where does the model go wrong?

Here’s another negative review. Let’s see if the model predicts this one correctly.


```python
another_negative_review = yelp_class['text'][140]
another_negative_review
```




    "Other than the really great happy hour prices, its hit or miss with this place. More often a miss. :(\n\nThe food is less than average, the drinks NOT strong ( at least they are inexpensive) , but the service is truly hit or miss.\n\nI'll pass."




```python
negative_review_transformed = bow_transformer.transform([another_negative_review])
nb.predict(negative_review_transformed)[0]
```




    5



# Why the incorrect prediction? 

One explanation as to why this may be the case is that our initial dataset had a much higher number of 5-star reviews than 1-star reviews. This means that the model is more biased towards positive reviews compared to negative ones.

Introduction adapted from: https://github.com/adicu/data-science/blob/master/All%20Levels.ipynb
