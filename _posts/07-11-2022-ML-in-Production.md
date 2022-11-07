<div id="container" style="position:relative;">
<div style="float:left"><h1>  Machine Learning in Production </h1></div>
<div style="position:relative; float:right"><img style="height:65px" src ="https://drive.google.com/uc?export=view&id=1EnB0x-fdqMp6I5iMoEBBEuxB_s7AmE2k" />
</div>
</div>

We have spent a lot of time in this course working, as many data scientists do, in our own local development enviroments (our laptops). Within large mature organizations leveraging machine learning, this may be representative of a smaller proportion of the work put into data science development. Most go through the process of first doing exploratory and prototyping work in ML, with the goal of eventually incorporating said models into their service offerings or products. 

In this case we are referring to the difference between doing data science _as an activity_ versus _developing it as product._

There is no value in building models if the outputs of those models cannot be used. While many companies may apply data science techniques only to develop insights used to inform business decision-making (this being closer to the practice of ordinary data analytics, only with larger data sets and the addition of machine learning) others seek to develop data science infrastructure and systems such that it may be benefit others throughout the organization and be incorporated into other systems and software.

For example:
- Companies such as Amazon and Netflix heavily rely upon recommendation for suggesting products and content
- Uber uses machine learning to predict arrival times of drivers in near-real time
- Facebook leverages computer vision techniques to auto-label and tag photos

These are all examples of turning ML in product, where the outputs of machine learning models are incorporated into features within a product and enhance the experience for the user. 

This is not always the case, as sometimes 'productionalization' (the process of putting machine learning into production) may mean only serving up the outputs internally within an organization. For example, a customer scoring model might have its outputs made available to different teams and arms of the organization with no connection to a data science team which developed it, but those outputs might serve a very valuable purpose for marketing and sales to inform their decisions about their targeting and personalization strategies.

So in the remainder of this lecture we will address the following questions:
- What is meant by putting ML "into production"?
- How do we make the outputs of our models available for others to consume?
- How does machine learning fit into the broader [software development lifecycle (SDLC)](https://hackernoon.com/software-development-lifecycle-sdlc-a-simple-explanation-78d77c466355)?

### What is meant by Production?

In order to understand what is meant by "ML in prod" we must take a step back and how organizations build software and systems thereof. In this context, an environment (technically, a _deployment environment_, deployment referring to a bundled package of software) refers to a system where software runs. This may refer to a number of servers and other associated systems collectively, or may also refer to a single server or collection of servers in a cluster, often in a more colloquial sense.

The **Development Environment** (or "dev") is the lowest environment and where new software or new features of existing software are developed. Development environments may vary significantly in terms of the quality of data and processes compared with the other environments. They serve as "sandbox" infrastructure for doing development work and unit testing the software as it is created. As such, it is usually acceptable for systems and processes to fail within the development environment, and the permissions are typically much more open within a dev environment and user base larger than other environments as there is little risk of "breaking anything."

The **Testing Environment** (or "test") is the next environment above development. Code which has passed unit tests is moved into this environment where more extensive testing takes place (_e.g._ functional, integration, and performance testing).

The **Staging Environment** (or "stage") is the final environment between initial development work and production. This environment should be as close to production as possible, as code is tested here before being finally deployed into prod. As such, it typically has more restrictions on its user base and permissions. Final testing is done in the staging environment before code is considered to be "production-ready": user acceptance testing (UAT) and other forms of _production acceptance testing_ or PAT.

The **Production Environment** (or "prod") are the systems running the software with which users and other systems interact. What is live in production is the gold standard and is the "known good" codebase. Testing is not done in prod and if bugs arise, they are fixed in one of the lower environments before being patched in the production systems.

![envs_code_promo](https://drive.google.com/uc?export=view&id=1nDR5Uqu8zp-q0iQZESpxNztDPKFLAW6o)





Moving code up through the environments as it is successively tested and approved is referred to as the _code promotion_ process - or you would say code is being promoted from one environment to the next highest one, or deployed in a given environment. In organizations with mature software development infrastructure and processes in place, these tasks may also be automated to varying degrees.

Also, it should be noted that this four environment setup, while typical, is not universal; differing configurations consisting of greater or fewer environments can and do exist. For example, there may also be dedicated environments for user acceptance (UAT) and production acceptance testing (PAT) separate from test or stage, or conversely, testing and staging code may be done in a single intermediary environment between dev and prod.

### Building a Web Service

While we will not go about deploying a productionized model at scale, we will create a local version of a model serving service to demonstrate the approach and what one would look like. Inside of an organization, this service would be developed in a lower environment and then gradually promoted to production, such that other teams or products could benefit from model outputs.

For this task we will be using [Flask](https://palletsprojects.com/p/flask/), a lightweight web framework built entirely in python. Flask runs a micro web server and will allow us to build a very simple web application (in this case, a [REST API](https://www.smashingmagazine.com/2018/01/understanding-using-rest-api/) to serve up model predictions from out trained model) with little or no front-end development required.

Let's create a new directory work in for our model API, and create our first python script to run an example webserver, which we'll call `helloworld.py`:

```python
from flask import Flask

app = Flask("helloworld")

@app.route('/')
def hello():
     return 'Hello World!'

if __name__ == "__main__":
    app.run()

```

As you can see, Flask requires creating an app with the app name as a parameter, and then wrapping a function returning output in the `.route` decorator which takes the URL path from root as its argument.

Save the file, and then in the terminal run `python helloworld.py` you should get the following output below if the server is running correctly:
```
 * Serving Flask app "helloworld" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

We can then copy and paste the URL above into a web browser (or equivalently, enter `localhost:5000`) and we'll see the content rendered by Flask from the python code we wrote above. It's that simple. Now we're running our very own tiny webserver!

When you're done, to stop the server process, return focus to the terminal and hit CTRL+C.

#### Exercise 1
Using the above, and doing some research in the [Flask documentation](http://flask.palletsprojects.com/en/1.1.x/quickstart/#a-minimal-application), create a new file, `helloworld2.py`, and add a different path which displays a message of your choosing. Try experimenting with returning different python objects to see what the results are. (Note: you will have to stop and restart the Flask server process for each change you make)

So we now have a super simple webserver that can return data to the user via the browser. But we wish to build a machine learning service! Fortunately, we can also interact with our Flask web application programmatically as well, by using a command line tool like `curl`:

```bash
curl -L http://localhost:5000/
```

(**Note**: If you are using git bash on Windows, you will need to add the `--silent` argument in the `curl` command to suppress other output)

Alternatively, we can also do so using python and the `requests` package:


```python
import requests

response = requests.get('http://localhost:5000/')
response.__dict__
```




    {'_content': b'Hello World!',
     '_content_consumed': True,
     '_next': None,
     'status_code': 200,
     'headers': {'Content-Type': 'text/html; charset=utf-8', 'Content-Length': '12', 'Server': 'Werkzeug/0.16.0 Python/3.7.4', 'Date': 'Mon, 09 Mar 2020 13:51:58 GMT'},
     'raw': <urllib3.response.HTTPResponse at 0x10f43fe10>,
     'url': 'http://127.0.0.1:5000/',
     'encoding': 'utf-8',
     'history': [],
     'reason': 'OK',
     'cookies': <RequestsCookieJar[]>,
     'elapsed': datetime.timedelta(microseconds=1930),
     'request': <PreparedRequest [GET]>,
     'connection': <requests.adapters.HTTPAdapter at 0x10f41e6d0>}



Our data is contained in the response object in `_content`. From here we will build up a general API which can return results from a trained model when given input data.

### Sending Data to the Server

Most of the time when we are using a web browser, you are getting results from the server. However, we can also send data to the server, and have logic carried out on the backend and return results. A common of example of this is submitting a form on a website: that information is sent to the server, where backend logic would save that information in the database - _e.g._ for setting up a new user account.

These are examples of different [HTTP request types](https://en.wikipedia.org/wiki/Hypertext_Transfer_Protocol#Request_methods). We can specify the request method for each route (URL path) in Flask, depending upon whether we want to be able to just receive data from the server (`GET`) or send data to the server and receive a response based upon it (`POST`).

We can update our simple webapp to use a `POST` method, and for our simplest case, return the value that was sent. Flask takes the list of valid methods for a given route as a list. Stop the flask server process in your terminal if it is still running (with CTRL+C) and create a new python file, `helloworld_postapp.py`: 

```python
from flask import Flask

app = Flask("helloworldpostapp")

@app.route('/', methods=['POST'])
def hello():
     return 'Hello World!'

if __name__ == "__main__":
    app.run()

```

We can again run the new flask app with `python helloworld_postapp.py`.

However, now if we open a new terminal and make a request to the server, we will get an error as the default request type is `GET`:

`curl -L http://localhost:5000`
```
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
<title>405 Method Not Allowed</title>
<h1>Method Not Allowed</h1>
<p>The method is not allowed for the requested URL.</p>
```

We now need to specify the we are making a `POST` request and sending data to the server. We can do this in `curl` using the `-X` option, and the data we wish to send after the `-d` option. This data should be send as JSON, and since we're using the terminal, we also need to escape the quotes and any other special characters:

`curl -X POST http://localhost:5000 -d "{\"mykey\":\"Hello World\"}"`

However, we are not actually using the information being sent to the server yet. Now to take out data that comes from the user making the `POST` request, we add the `request` object from Flask and extract the data from it (here we are using `force=True` such that we don't need to set standard request parameters such as User-Agent that are often as required, as by a web browser). Data that is sent back from Flask is almost always wrapped in the `jsonify` function.

Stop the server process again, and create a new python file `helloworld_postapp2.py`:

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['POST'])
def hello():
    input_json = request.get_json(force=True)
    print(f'Data sent in request:{input_json}')
    return jsonify(input_json)

if __name__ == "__main__":
    app.run()
```

Now we can run the app with `python helloworld_postapp2.py` and test in a separate terminal with curl - we can send data to the server and our python code will return it back to the user!

`curl -X POST http://localhost:5000 -d "{\"mykey\":\"Hello World\"}"`

#### Exercise 2

Modify the code above to take a dictionary with a single key, `mylist`, with an associated list of integers as input in the POST request, and return the sum of the integers as the response.

You can send data to your application using `curl` as below:  
`curl -X POST http://localhost:5000/ -d "{\"mylist\":[1,2,3,4,5]}"`

### Creating a Simple Prediction Service

Before we get going, we will train up a very a simple model that we wish to productionize. For this, we will return back to the [sentiment labelled sentences dataset](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences) we saw previously. Here we will train a model on the Yelp reviews and their sentiment, then deploy it and use our API to pass in new reviews and get sentiment scores returned.

As before, we first read in the data:


```python
import pandas as pd

yelp_df = pd.read_csv('yelp_labelled.txt', sep='\t', header=None)
yelp_df.columns = ['review', 'positive']
yelp_df.head()
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
      <th>review</th>
      <th>positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wow... Loved this place.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Crust is not good.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Not tasty and the texture was just nasty.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Stopped by during the late May bank holiday of...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The selection on the menu was great and so wer...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Now we will convert to a document-term matrix and train a simple logistic regression model. Here we are not bothering to tune hyperparameters nor worry about overfitting as this is just for demonstrative purposes.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# Create the document-term matrix
vect = CountVectorizer()
vect.fit(yelp_df['review'])
X = vect.transform(yelp_df['review'])

# Get the outcome variable
y = yelp_df['positive']

logit = LogisticRegression()
logit.fit(X, y)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)



Great! Now we have a simple model. We can pass new data into this model to make predictions about whether the sentiment is positive or not based on the vocabulary found in the Yelp reviews:


```python
new_review = ["Absolutely love this place! The best ever!"]

X_new = vect.transform(new_review)
logit.predict_proba(X_new)
```




    array([[0.0848211, 0.9151789]])



The model believes this is positive sentiment, with a score of ~92%. Now let's take the ability of our model to make new predictions out of our local machine and deploy this model to our web service!

Our model serving API will take in raw text, so every review passed into it will also have to go through the preprocessing step of the CountVectorizer as well. So we will pickle both our CountVectorizer and the fitted model.

*Note: make a "flask" folder in the same place as your notebook where we will store the files.*


```python
import joblib

joblib.dump(vect, "count_vectorizer.pkl")
joblib.dump(logit, "sentiment_logit.pkl")
```




    ['sentiment_logit.pkl']



Now we have a nice saved versions of both the preprocessing step and the model which are both already fitted. We are now ready to "productionize" our model, which in our prototype application case means putting the files into the correct directory along with the application and adding some python to call them and return predictions based upon input.

Create a new python file called `logit.py` for our new prediction service:

```python
from flask import Flask, request, jsonify
import joblib

app = Flask("review_sentiment_api")

@app.route('/predict/', methods=['POST'])
def predict_sentiment():
    input_json = request.get_json(force=True)
    print(f'Data sent in request:{input_json}')

    # Read the review data and apply preprocessing (vectorization)
    review_text = [input_json['review']]
    print('Vectorizing data')
    countvectorizer = joblib.load("count_vectorizer.pkl")    
    X_new = countvectorizer.transform(review_text)
    
    # Score the model
    print('Scoring...')
    sentiment_logit = joblib.load("sentiment_logit.pkl")
    sentiment_score = sentiment_logit.predict_proba(X_new)[0]
    print(sentiment_score)
    negative_score = sentiment_score[0]
    positive_score = sentiment_score[1]

    return jsonify({"positive":positive_score, "negative":negative_score})

if __name__ == "__main__":
    app.run()
```

Now we can test our model service, again sending the data using `curl`, and we will get a response of the predicted sentiment values from the logistic regression model for any new text we want to score!

Run the file in one terminal with `python logit.py`, then test in a separate terminal using `curl`:

`curl -X POST http://localhost:5000/predict/ -d "{\"review\":\"Thanks, I hate it\"}"`

### Creating a Pipeline for Preprocessing and Prediction

In this case we had to fit and pickle separate objects (Estimators) to preprocess the data. These would both need to be transferred to the server where we would be running our model serving application. This is a good case for using a pipeline, as we can then fit and pickle a single sklearn object to use in serving results from our API:


```python
from sklearn.pipeline import make_pipeline

sentiment_pipeline = make_pipeline(vect, logit)
sentiment_pipeline.predict_proba({'This is the worst'})
```




    array([[0.66045534, 0.33954466]])




```python
# Pickle our new pipeline for both the vectorizer and the sentiment model
joblib.dump(sentiment_pipeline, "sentiment_pipeline.pkl")
```




    ['sentiment_pipeline.pkl']



Now we can update our code to use the pipeline object, which significantly simplifies it, as the vectorizer and model predictions are now all done in a single step, and we can call the `predict_proba` method directly on the review text:

```python
def predict_sentiment():
    input_json = request.get_json(force=True)
    print(f'Data sent in request:{input_json}')

    # Read the review data and apply preprocessing (vectorization)
    review_text = [input_json['review']]
    
    # Process in the pipeline and make predictions
    sentiment_logit = joblib.load("sentiment_pipeline.pkl")
    sentiment_score = sentiment_logit.predict_proba(review_text)[0]
    
    print(sentiment_score)
    negative_score = sentiment_score[0]
    positive_score = sentiment_score[1]

    return jsonify({"positive":positive_score, "negative":negative_score})
```

Update your `logit.py` file with the above and create a new file, `logit_with_pipeline.py`. Start the flask server with `python logit_with_pipeline.py`, and again test with `curl` in a separate terminal:

`curl -X POST http://localhost:5000/predict/ -d "{\"review\":\"Thanks, I hate it\"}"`

### Scaling Machine Learning Services

So what have we done so far?
1. Created a simple model serving API to provide results to a user
2. Trained a preprocessing and model prediction pipeline
3. Saved our pipeline and integrated into our ML service

This mirrors the steps one would follow when putting a model into a "real" production environment, where, after the infrastructure is built, model prototypes are often built offline and then gradually promoted until live, or are handed over to a team of machine learning engineers to do the same. Note that even here, there is nothing stopping us from taking our simple code and saved pipeline off our local machine and transferring to a cloud provider such as AWS, in which case with the correct permissions on an EC2 instance we could make our model API open to anyone with a an internet connection (!) A more sophisticated ML API would also have other parameters it could retain, such as metadata on different models available, model versioning, and even URLs to retrain models based upon a specified dataset!

In practice, scaling machine learning services needs to address many of the same challenges faced by other systems (_e.g._ load-balancing to account for variable demand, version control and data audit, security and permissions management, etc.). There are solutions developed by the large cloud computing providers to do exactly that, most notably:
* Model serving in [Amazon Sagemaker](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-hosting.html)
* Deploying models in [Google Cloud ML Engine](https://cloud.google.com/ml-engine/docs/deploying-models)
* Model deployment with [Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-deploy-and-where)

There also exist numerous products and companies being created to address this growing space, for [Algorithmia](https://algorithmia.com/product) specializes specifically in MLOps, and open platforms such as the open source [MLFlow](https://mlflow.org/) by Databricks and [Kubeflow](https://www.kubeflow.org/) by Google seek to address the complete end-to-end machine learning lifecycle.


### Machine Learning and DevOps (MLOps)


As machine learning increasingly becomes a more common component of software, there is also a lot of thought that needs to be given as to how to integrate it into traditional development processes. Enter the growing field of Development + Operations ([DevOps](https://www.atlassian.com/devops)) + ML = [MLOps](https://en.wikipedia.org/wiki/MLOps).

![devops](https://drive.google.com/uc?export=view&id=1c2T9YSz3aVoX4LK4qNldCpHaRG4Qd4tz)



Machine learning poses a number of interesting problems not faced in traditional software development. To name a few:
- How do we evaluate whether systems are ready for production when their behavior is data-dependent and not entirely predictable?
- Unlike traditional software, the performance of models may degrade over time: how do we determine the frequency for retraining and deploying new versions of models?
- Machine learning systems are often much more complicated than other applications in terms of their dependencies - how do we identify and minimize [hidden technical debt](https://hub.packtpub.com/technical-and-hidden-debts-in-machine-learning-google-engineers-give-their-perspective/) that arises?

These, and many other complex challenges, are factors which must be considered and addressed in the design and implementation of machine learning systems.

### Phases of the MLOps Lifecycle

When moving from beyond their laptop to the greater umbrella of machine learning operations, the data scientist will be exposed to a much broader ecosystem of tools, platforms, and systems to both deploy and operate models in a production environment. The MLOps lifecycle is far more than just taking a trained model and the associated code and putting it onto a server.

Though there are many varying frameworks describing the MLOps lifecycle (and furthermore, differences in the lifecycle of models within different organizations, or even for different individual models), they generally encompass the areas identified below:

<img src="https://drive.google.com/uc?export=view&id=1PymHwG4erNxiwDBmWWEJHcjPihb8N88d"/>

- **Development:** Building a prototype or initial model to be put into production. This includes identifying both the required data and output(s), as well as acquiring the necessary data and training a prototype model.


- **Validation:** Includes testing (unit tests, functional testing, performance testing, etc.) as well as expected performance and behavior from a model standpoint given particular unseen data as input. This is not only done prior to productionizing a model, but additionally may also be performed on a continual basis in automated way after it has been deployed. 


- **Deployment:** Requires identifying dependencies required for a model, and packaging these with it in such a way that they can easily be moved to a production environment for the model to run. For most organizations, this now almost certainly will involve leveraging a [container framework](https://en.wikipedia.org/wiki/Docker_(software)). Pipelines may also need to be developed to make the necessary data inputs available in the production environment. Finally, "soft launching" and/or experimentation via A/B testing are integral to taking a new model live while mitigating possible unforeseen negative impacts.


- **Storage and Versioning:** Models need to be stored in a centralized location, as well as use a standard format or formats for ease and consistency in deployment and operation. A version control system for different deployed models should be in place for auditing and governance purposes, as well as the ability to "roll back" changes should this be required. A model management system of this kind also facilitates greater collaboration and efficiencies for development and operations teams.


- **Serving and Integration**: The outputs of the model must be made available, either within the organization, externally through integrations (*e.g.* in a user-facing app or web page), or both, depending upon the model's purpose. This may include APIs for internal use within an organization which allow other developers or business users to make requests while being unaware of the model's internal workings (as we've seen in this lab).


- **Monitoring and Observability**: Once a model is deployed, it must be available for monitoring in order to continually assess performance, as well as have the necessary code and infrastructure in place to diagnose the root cause of incidents should there be failures or unexpected behaviors - for example, being able to ascertain what data is being used to make individual predictions or confirm dependencies on other systems. Model performance should be monitored in real-time or near real-time to assess performance on a continual basis, and be tied back to organizational KPIs.


- **Retraining**: Model performance may deteriorate over time due to [concept drift](https://en.wikipedia.org/wiki/Concept_drift) and other factors (*e.g.* qualitative changes in data or other dependent systems) and necessitate retraining and deploying new models. The cadence required for this is informed by performance monitoring as above. Retraining can implemented as part of the MLOps infrastructure, and done in an automated or "online" manner with varying levels of complexity and sophistication depending on need and the technical maturity of the organization.

### Conclusion


Here we have covered a number of concepts related to productionizing machine learning models, as well as building a simple prototype of a model serving API on our local machine.

This is only scratching the surface, as the topic of how data science and machine learning fit into the broader software development lifecycle (SDLC) is a deep and complex subject. Best practices and optimal approaches of how to best integrate machine learning into core capabilities and products will differ from organization to organization. This is a challenging and complex problem which will only grow as the application of machine models continually becomes more and more commonplace.

---

<div id="container" style="position:relative;">
<div style="position:relative; float:right"><img style="height:25px""width: 50px" src ="https://drive.google.com/uc?export=view&id=14VoXUJftgptWtdNhtNYVm6cjVmEWpki1" />
</div>
</div>

