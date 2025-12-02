# Final Project: Instagram Post Performance Analysis


## The Data

This dataset includes Instagram post-level metrics for reach and
engagement. Key numeric columns are Impressions, Likes, Comments, Saves,
Shares, Profile Visits, Follows, and the breakdown of impressions by
source (From Home, From Hashtags, From Explore, From Other). There are
also two text columns: Caption and Hashtags. The goal of the analysis is
clear and testable: to see which content and distribution features
predict post reach and engagement, and whether the text in captions or
hashtags is associated with higher engagement.

### Read

Load the datasets.

``` python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
```

``` python
ins = pd.read_csv("~/Desktop/Data Wrangling/project/Instagram Data.csv", encoding="latin1")                
```

``` python
ins.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | Impressions | From Home | From Hashtags | From Explore | From Other | Saves | Comments | Shares | Likes | Profile Visits | Follows | Caption | Hashtags |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 3920 | 2586 | 1028 | 619 | 56 | 98 | 9 | 5 | 162 | 35 | 2 | Here are some of the most important data visua... | \#finance #money #business #investing #investme... |
| 1 | 5394 | 2727 | 1838 | 1174 | 78 | 194 | 7 | 14 | 224 | 48 | 10 | Here are some of the best data science project... | \#healthcare #health #covid #data #datascience ... |
| 2 | 4021 | 2085 | 1188 | 0 | 533 | 41 | 11 | 1 | 131 | 62 | 12 | Learn how to train a machine learning model an... | \#data #datascience #dataanalysis #dataanalytic... |
| 3 | 4528 | 2700 | 621 | 932 | 73 | 172 | 10 | 7 | 213 | 23 | 8 | Here’s how you can write a Python program to d... | \#python #pythonprogramming #pythonprojects #py... |
| 4 | 2518 | 1704 | 255 | 279 | 37 | 96 | 5 | 4 | 123 | 8 | 0 | Plotting annotations while visualizing your da... | \#datavisualization #datascience #data #dataana... |

</div>

``` python
ins.columns
```

    Index(['Impressions', 'From Home', 'From Hashtags', 'From Explore',
           'From Other', 'Saves', 'Comments', 'Shares', 'Likes', 'Profile Visits',
           'Follows', 'Caption', 'Hashtags'],
          dtype='object')

### Visualization 1

To understand the overall structure of the dataset before building
models, I first visualize the distributions of impressions and
engagement. These plots give an initial idea of how skewed the reach
metrics are and whether user interactions differ systematically across
posts.

``` python
# Distribution of Impressions
plt.figure(figsize=(8,5))
sns.histplot(ins['Impressions'], bins=50)
plt.title("Distribution of Impressions")
plt.xlabel("Impressions")
plt.ylabel("Number of Posts")
plt.show()
```

![](readme_files/figure-commonmark/cell-6-output-1.png)

``` python
# Distribution of composite engagement metric
ins['engagement'] = ins[['Likes','Comments','Saves','Shares','Profile Visits','Follows']].sum(axis=1)

plt.figure(figsize=(8,5))
sns.histplot(ins['engagement'], bins=50)
plt.title("Distribution of Engagement (sum of Likes, Comments, Saves, Shares, Profile Visits, Follows)")
plt.xlabel("Engagement")
plt.ylabel("Number of Posts")
plt.show()
```

![](readme_files/figure-commonmark/cell-7-output-1.png)

``` python
# Total impressions by source to see which source drives the most engagement
source = ['From Home','From Hashtags','From Explore','From Other']
# Sum impressions for each source, convert to DataFrame, and rename columns for plotting
total = ins[source].sum().rename('Total Impressions').reset_index().rename(columns={'index':'Source'})

plt.figure(figsize=(8,5))
sns.barplot(data=total, x='Source', y='Total Impressions', hue='Source')
plt.title("Total Impressions by Source")
plt.ylabel("Total Impressions")
plt.show()
```

![](readme_files/figure-commonmark/cell-8-output-1.png)

### Feature Engineering

To better characterize the properties of each post, I created several
new variables, including caption length, hashtag counts, and indicators
that detect the presence and frequency of the phrase “data science” in
both captions and hashtags.

caption_length

``` python
# Count characters in caption to further check whether longer captions increase engagement
ins['caption_length'] = ins['Caption'].astype(str).str.len()
```

hashtags_count

``` python
# Count hashtags to see how heavily tags are used
def count_hashtags(s):
    if str(s).strip() == "":
        return 0 # Return 0 for empty values because no hashtags exist
    tokens = str(s).split() # Split by whitespace as hashtags are separated by spaces
    return len(tokens) # Count all tokens since every token is a hashtag in this format
ins['hashtags_count'] = ins['Hashtags'].apply(count_hashtags)
```

pct_from_hashtags

``` python
# Calculate the fraction of impressions from hashtags, replace total impressions of 0 with NaN to avoid division by zero, then fill resulting NaNs with 0 for clean data
ins['pct_from_hashtags'] = (ins['From Hashtags'] / ins['Impressions'].replace(0, np.nan)).fillna(0)  
```

high_engagement

``` python
# Create a binary indicator for engagement above the median for use in logistic regression
ins['high_engagement'] = (ins['engagement'] > ins['engagement'].median()).astype(int)
```

data_science_in_caption

``` python
# Create a binary variable detecting presence of "data science" in Caption
ins['data_science_caption'] = ins['Caption'].astype(str).str.contains('[Dd]ata\\s?[Ss]cience', regex=True)

# Convert True/False into numeric 1/0 for modeling
ins['data_science_caption'] = ins['data_science_caption'].apply(lambda x: 1 if x else 0)

# Count number of times "data science" appears in caption
ins['data_science_count_caption'] = ins['Caption'].astype(str).str.count('[Dd]ata\\s?[Ss]cience')

# View results
ins[['Caption', 'data_science_caption', 'data_science_count_caption']].head()         
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | Caption | data_science_caption | data_science_count_caption |
|----|----|----|----|
| 0 | Here are some of the most important data visua... | 0 | 0 |
| 1 | Here are some of the best data science project... | 1 | 2 |
| 2 | Learn how to train a machine learning model an... | 0 | 0 |
| 3 | Here’s how you can write a Python program to d... | 0 | 0 |
| 4 | Plotting annotations while visualizing your da... | 0 | 0 |

</div>

data_science_in_hashtags

``` python
# Create a binary variable detecting presence of "data science" in hashtags
ins['data_science_hashtags'] = ins['Hashtags'].astype(str).str.contains('[Dd]ata\\s?[Ss]cience', regex=True)

# Convert True/False into numeric 1/0 for modeling
ins['data_science_hashtags'] = ins['data_science_hashtags'].apply(lambda x: 1 if x else 0)

# Count number of times "data science" appears in hashtags
ins['data_science_count_hashtags'] = ins['Hashtags'].astype(str).str.count('[Dd]ata\\s?[Ss]cience')

# View result
ins[['Hashtags', 'data_science_hashtags', 'data_science_count_hashtags']].head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | Hashtags | data_science_hashtags | data_science_count_hashtags |
|----|----|----|----|
| 0 | \#finance #money #business #investing #investme... | 1 | 1 |
| 1 | \#healthcare #health #covid #data #datascience ... | 1 | 1 |
| 2 | \#data #datascience #dataanalysis #dataanalytic... | 1 | 2 |
| 3 | \#python #pythonprogramming #pythonprojects #py... | 0 | 0 |
| 4 | \#datavisualization #datascience #data #dataana... | 1 | 1 |

</div>

### Subsetting

Before moving to regression models, I examine how posts differ when
grouped by hashtag usage and engagement levels. These comparisons give a
first look at possible patterns and help set expectations for the
modeling stage.

``` python
# Subset posts that used at least one hashtag to compare hashtagged vs non-hashtagged
hashtag_posts = ins[ins['hashtags_count'] > 0]
hashtag_posts.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | Impressions | From Home | From Hashtags | From Explore | From Other | Saves | Comments | Shares | Likes | Profile Visits | ... | Hashtags | engagement | caption_length | hashtags_count | pct_from_hashtags | high_engagement | data_science_caption | data_science_count_caption | data_science_hashtags | data_science_count_hashtags |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 3920 | 2586 | 1028 | 619 | 56 | 98 | 9 | 5 | 162 | 35 | ... | \#finance #money #business #investing #investme... | 311 | 112 | 22 | 0.262245 | 0 | 0 | 0 | 1 | 1 |
| 1 | 5394 | 2727 | 1838 | 1174 | 78 | 194 | 7 | 14 | 224 | 48 | ... | \#healthcare #health #covid #data #datascience ... | 497 | 187 | 18 | 0.340749 | 1 | 1 | 2 | 1 | 1 |
| 2 | 4021 | 2085 | 1188 | 0 | 533 | 41 | 11 | 1 | 131 | 62 | ... | \#data #datascience #dataanalysis #dataanalytic... | 258 | 117 | 18 | 0.295449 | 0 | 0 | 0 | 1 | 2 |
| 3 | 4528 | 2700 | 621 | 932 | 73 | 172 | 10 | 7 | 213 | 23 | ... | \#python #pythonprogramming #pythonprojects #py... | 433 | 202 | 11 | 0.137147 | 1 | 0 | 0 | 0 | 0 |
| 4 | 2518 | 1704 | 255 | 279 | 37 | 96 | 5 | 4 | 123 | 8 | ... | \#datavisualization #datascience #data #dataana... | 236 | 178 | 29 | 0.101271 | 0 | 0 | 0 | 1 | 1 |

<p>5 rows × 22 columns</p>
</div>

``` python
# Subset high engagement posts to identify characteristics of top-performing posts
high_engagement_posts = ins[ins['high_engagement'] == 1]
high_engagement_posts.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | Impressions | From Home | From Hashtags | From Explore | From Other | Saves | Comments | Shares | Likes | Profile Visits | ... | Hashtags | engagement | caption_length | hashtags_count | pct_from_hashtags | high_engagement | data_science_caption | data_science_count_caption | data_science_hashtags | data_science_count_hashtags |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 1 | 5394 | 2727 | 1838 | 1174 | 78 | 194 | 7 | 14 | 224 | 48 | ... | \#healthcare #health #covid #data #datascience ... | 497 | 187 | 18 | 0.340749 | 1 | 1 | 2 | 1 | 1 |
| 3 | 4528 | 2700 | 621 | 932 | 73 | 172 | 10 | 7 | 213 | 23 | ... | \#python #pythonprogramming #pythonprojects #py... | 433 | 202 | 11 | 0.137147 | 1 | 0 | 0 | 0 | 0 |
| 8 | 3749 | 2384 | 857 | 248 | 49 | 155 | 6 | 8 | 159 | 36 | ... | \#dataanalytics #datascience #data #machinelear... | 368 | 162 | 30 | 0.228594 | 1 | 0 | 0 | 1 | 1 |
| 9 | 4115 | 2609 | 1104 | 178 | 46 | 122 | 6 | 3 | 191 | 31 | ... | \#python #pythonprogramming #pythonprojects #py... | 359 | 79 | 11 | 0.268287 | 1 | 0 | 0 | 0 | 0 |
| 14 | 9453 | 2525 | 5799 | 208 | 794 | 100 | 6 | 10 | 294 | 181 | ... | \#data #datascience #dataanalysis #dataanalytic... | 633 | 60 | 19 | 0.613456 | 1 | 0 | 0 | 1 | 1 |

<p>5 rows × 22 columns</p>
</div>

### Summarizing

``` python
# Summarize impressions, engagement, likes, and comments grouped by whether posts have hashtags
summary_by_hashtag = ins.groupby(ins['hashtags_count']>0)[['Impressions','engagement','Likes','Comments']].agg(['mean','median','count'])
summary_by_hashtag
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead tr th {
        text-align: left;
    }
&#10;    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>

|  | Impressions |  |  | engagement |  |  | Likes |  |  | Comments |  |  |
|----|----|----|----|----|----|----|----|----|----|----|----|----|
|  | mean | median | count | mean | median | count | mean | median | count | mean | median | count |
| hashtags_count |  |  |  |  |  |  |  |  |  |  |  |  |
| True | 5703.991597 | 4289.0 | 119 | 414.495798 | 314.0 | 119 | 173.781513 | 151.0 | 119 | 6.663866 | 6.0 | 119 |

</div>

``` python
# Calculate total impressions from each source to see which source contributes the most to overall impressions
summary_by_source = pd.DataFrame({
    'from_home_total': ins['From Home'].sum(),
    'from_hashtags_total': ins['From Hashtags'].sum(),
    'from_explore_total': ins['From Explore'].sum(),
    'from_other_total': ins['From Other'].sum()
}, index=[0])
summary_by_source
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | from_home_total | from_hashtags_total | from_explore_total | from_other_total |
|-----|-----------------|---------------------|--------------------|------------------|
| 0   | 294619          | 224614              | 128294             | 20360            |

</div>

### Visualization 2

To further check the newly created variables, I created a second set of
visualizations. These plots show how hashtag use and “data
science”-related terms relate to post performance, providing an initial
look at how the engineered textual features might affect reach and
interaction.

``` python
# Scatterplot of hashtag count vs engagement to check whether more hashtags is linked to higher engagement
plt.figure(figsize=(8,5))
sns.scatterplot(data=ins, x='hashtags_count', y='engagement')
plt.title("Hashtag Count vs Engagement")
plt.xlabel("Number of Hashtags")
plt.ylabel("Engagement")
plt.show()
```

![](readme_files/figure-commonmark/cell-19-output-1.png)

``` python
# Boxplot to compare impressions for posts that mention “data science” in captions vs those do not
plt.figure(figsize=(8,5))
sns.boxplot(data=ins, x='data_science_caption', y='Impressions')
plt.title("Impressions by Presence of 'Data Science' in Caption")
plt.xlabel("Data Science Mention in Caption")
plt.ylabel("Impressions")
plt.show()
```

![](readme_files/figure-commonmark/cell-20-output-1.png)

``` python
# Scatterplot to show whether count of “data science” in caption is associated with higher engagement
plt.figure(figsize=(8,5))
sns.scatterplot(data=ins, x='data_science_count_caption', y='engagement')
plt.title("Data Science Count in Caption vs Engagement")
plt.xlabel("Data Science Count in Caption")
plt.ylabel("Engagement")
plt.show()
```

![](readme_files/figure-commonmark/cell-21-output-1.png)

``` python
# Boxplot to compare impressions for posts that mention “data science” in hashtags vs those do not
plt.figure(figsize=(8,5))
sns.boxplot(data=ins, x='data_science_hashtags', y='Impressions')
plt.title("Impressions by Presence of 'Data Science' in Hashtags")
plt.xlabel("Data Science Mention in Hashtags")
plt.ylabel("Impressions")
plt.show()
```

![](readme_files/figure-commonmark/cell-22-output-1.png)

``` python
# Scatterplot to show whether count of “data science” in hashtags is associated with higher engagement
plt.figure(figsize=(8,5))
sns.scatterplot(data=ins, x='data_science_count_hashtags', y='engagement')
plt.title("Data Science Count in Hashtags vs Engagement")
plt.xlabel("Data Science Count in Hashtags")
plt.ylabel("Engagement")
plt.show()
```

![](readme_files/figure-commonmark/cell-23-output-1.png)

### Model

Finally, I estimate three models: OLS, Logit, and Poisson, to analyze
how text features, hashtag behavior, and “data science”–related terms
predict different outcomes. Each model corresponds to a different type
of dependent variable: continuous reach (OLS), binary engagement
(Logit), and count-based interactions (Poisson).

OLS model

``` python
# Create log-transformed outcome variable in the DataFrame to reduce skew
ins['log_engagement'] = np.log(ins['engagement'] + 1)  

# Define formula for predictors
ols_formula = (
    "log_engagement ~ caption_length + hashtags_count + pct_from_hashtags + "
    "data_science_caption + data_science_hashtags + data_science_count_caption + "
    "data_science_count_hashtags + "
    "Q('From Home') + Q('From Hashtags') + Q('From Explore') + Q('From Other')"
)

# Build ols model
ols_model = sm.formula.ols(formula=ols_formula, data=ins).fit()

ols_model.summary()
```

|                   |                  |                     |          |
|-------------------|------------------|---------------------|----------|
| Dep. Variable:    | log_engagement   | R-squared:          | 0.783    |
| Model:            | OLS              | Adj. R-squared:     | 0.761    |
| Method:           | Least Squares    | F-statistic:        | 35.17    |
| Date:             | Tue, 02 Dec 2025 | Prob (F-statistic): | 1.43e-30 |
| Time:             | 16:13:41         | Log-Likelihood:     | -11.223  |
| No. Observations: | 119              | AIC:                | 46.45    |
| Df Residuals:     | 107              | BIC:                | 79.79    |
| Df Model:         | 11               |                     |          |
| Covariance Type:  | nonrobust        |                     |          |

OLS Regression Results

|  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
|  | coef | std err | t | P\>\|t\| | \[0.025 | 0.975\] |
| Intercept | 5.9703 | 0.170 | 35.024 | 0.000 | 5.632 | 6.308 |
| caption_length | -0.0003 | 0.000 | -1.409 | 0.162 | -0.001 | 0.000 |
| hashtags_count | -0.0283 | 0.007 | -4.219 | 0.000 | -0.042 | -0.015 |
| pct_from_hashtags | -1.2894 | 0.343 | -3.760 | 0.000 | -1.969 | -0.609 |
| data_science_caption | -0.0113 | 0.127 | -0.089 | 0.929 | -0.262 | 0.239 |
| data_science_hashtags | -0.0494 | 0.103 | -0.478 | 0.634 | -0.254 | 0.156 |
| data_science_count_caption | 0.1271 | 0.074 | 1.710 | 0.090 | -0.020 | 0.274 |
| data_science_count_hashtags | 0.0638 | 0.032 | 2.001 | 0.048 | 0.001 | 0.127 |
| Q('From Home') | 0.0001 | 3.12e-05 | 4.434 | 0.000 | 7.65e-05 | 0.000 |
| Q('From Hashtags') | 0.0002 | 2.72e-05 | 8.567 | 0.000 | 0.000 | 0.000 |
| Q('From Explore') | -4.502e-06 | 2.02e-05 | -0.222 | 0.824 | -4.46e-05 | 3.56e-05 |
| Q('From Other') | -8.481e-05 | 0.000 | -0.685 | 0.495 | -0.000 | 0.000 |

|                |        |                   |          |
|----------------|--------|-------------------|----------|
| Omnibus:       | 1.056  | Durbin-Watson:    | 1.933    |
| Prob(Omnibus): | 0.590  | Jarque-Bera (JB): | 1.080    |
| Skew:          | -0.220 | Prob(JB):         | 0.583    |
| Kurtosis:      | 2.843  | Cond. No.         | 5.82e+04 |

<br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 5.82e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.

The OLS model using log-transformed engagement has an R-squared of
0.783, showing strong explanatory power. The results indicate that
adding more hashtags is linked to lower engagement, and a higher share
of impressions from hashtags also predicts lower engagement. On the
other hand, mentions of “data science” within hashtags are positively
associated with engagement. Impressions from Home and from Hashtags are
both significant positive predictors, with Home impressions having the
largest impact. Overall, the model suggests that just adding hashtags
does not help much, while using targeted hashtag content and getting
more exposure from key channels, especially the Home feed, is more
important for higher engagement.

Logit model

``` python
# Logistic regression predicting high engagement
logit_formula = (
    "high_engagement ~ caption_length + hashtags_count + pct_from_hashtags + "
    "data_science_caption + data_science_hashtags + data_science_count_caption + "
    "data_science_count_hashtags + "
    "Q('From Home') + Q('From Hashtags') + Q('From Explore') + Q('From Other')"
)

logit_model = sm.formula.logit(formula=logit_formula, data=ins).fit()
logit_model.summary()
```

    Optimization terminated successfully.
             Current function value: 0.251897
             Iterations 10

|                  |                  |                   |           |
|------------------|------------------|-------------------|-----------|
| Dep. Variable:   | high_engagement  | No. Observations: | 119       |
| Model:           | Logit            | Df Residuals:     | 107       |
| Method:          | MLE              | Df Model:         | 11        |
| Date:            | Tue, 02 Dec 2025 | Pseudo R-squ.:    | 0.6366    |
| Time:            | 16:13:41         | Log-Likelihood:   | -29.976   |
| converged:       | True             | LL-Null:          | -82.480   |
| Covariance Type: | nonrobust        | LLR p-value:      | 1.810e-17 |

Logit Regression Results

|                             |          |         |        |          |           |         |
|-----------------------------|----------|---------|--------|----------|-----------|---------|
|                             | coef     | std err | z      | P\>\|z\| | \[0.025   | 0.975\] |
| Intercept                   | -16.2969 | 6.489   | -2.512 | 0.012    | -29.015   | -3.579  |
| caption_length              | -0.0021  | 0.003   | -0.817 | 0.414    | -0.007    | 0.003   |
| hashtags_count              | -0.0844  | 0.088   | -0.957 | 0.338    | -0.257    | 0.088   |
| pct_from_hashtags           | -2.9330  | 12.833  | -0.229 | 0.819    | -28.085   | 22.219  |
| data_science_caption        | 0.0068   | 1.950   | 0.004  | 0.997    | -3.814    | 3.828   |
| data_science_hashtags       | 0.8738   | 1.456   | 0.600  | 0.548    | -1.979    | 3.727   |
| data_science_count_caption  | 1.1364   | 1.305   | 0.871  | 0.384    | -1.422    | 3.694   |
| data_science_count_hashtags | -0.8919  | 0.795   | -1.122 | 0.262    | -2.450    | 0.666   |
| Q('From Home')              | 0.0062   | 0.002   | 3.294  | 0.001    | 0.003     | 0.010   |
| Q('From Hashtags')          | 0.0023   | 0.002   | 1.290  | 0.197    | -0.001    | 0.006   |
| Q('From Explore')           | 0.0020   | 0.001   | 1.934  | 0.053    | -2.69e-05 | 0.004   |
| Q('From Other')             | 0.0042   | 0.003   | 1.530  | 0.126    | -0.001    | 0.010   |

<br/><br/>Possibly complete quasi-separation: A fraction 0.15 of observations can be<br/>perfectly predicted. This might indicate that there is complete<br/>quasi-separation. In this case some parameters will not be identified.

The logistic regression predicting whether a post gets above-median
engagement has a pseudo R-squared of 0.6366, showing reasonably good
classification. The only significant predictor is impressions from Home,
with each extra Home impression increasing the odds of high engagement
by about 0.62%. Textual features such as caption length, number of
hashtags, and the data-science indicators are not significant,
suggesting that exposure through the Home feed is the strongest factor
for whether a post performs above the median.

Poisson model

``` python
# Poisson model predicting engagement counts
poisson_formula = (
    "engagement ~ caption_length + hashtags_count + pct_from_hashtags + "
    "data_science_caption + data_science_hashtags + data_science_count_caption + "
    "data_science_count_hashtags + "
    "Q('From Home') + Q('From Hashtags') + Q('From Explore') + Q('From Other')"
)

poisson_model = sm.formula.poisson(formula=poisson_formula, data=ins).fit()

poisson_model.summary()
```

    Warning: Maximum number of iterations has been exceeded.
             Current function value: 21.360429
             Iterations: 35

The Poisson regression on raw engagement counts shows strong explanatory
power, with a pseudo R-squared of 0.7739. Caption length and the number
of hashtags both have small but significant negative effects on
engagement, and a higher share of impressions coming from hashtags is
also linked to lower engagement. In contrast, mentioning the target
phrase, especially when mentioned multiple times, tends to increase
engagement in both captions and hashtags. All four impression-source
variables (Home, Hashtags, Explore, Other) are positive predictors, and
increases in Home impressions appear to have the largest practical
effect. Overall, the Poisson results are consistent with the OLS
findings: more general hashtags do not necessarily help, while
topic-specific mentions and stronger visibility through key sources,
especially the Home feed, are associated with higher engagement.
