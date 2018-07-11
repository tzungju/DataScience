import random

def read_file(path):
    """
    read and return all data in a file
    """
    with open(path, 'r') as f:
        return f.read()

def load_data():
    """
    load and return the data in features and labels lists
    each item in features contains the raw email text
    each item in labels is either 1(spam) or 0(ham) and identifies corresponding item in features
    """
    # load all data from file
    data_path = "data/SpamDetectionData.txt"
    all_data = read_file(data_path)
    
    # split the data into lines, each line is a single sample
    all_lines = all_data.split('\n')

    # each line in the file is a sample and has the following format
    # it begins with either "Spam," or "Ham,", and follows by the actual text of the email
    # e.g. Spam,<p>His honeyed and land....
    
    # extract the feature (email text) and label (spam or ham) from each line
    features = []
    labels = []
    for line in all_lines:
        if line[0:4] == 'Spam':
            labels.append(1)
            features.append(line[5:])
            pass
        elif line[0:3] == 'Ham':
            labels.append(0)
            features.append(line[4:])
            pass
        else:
            # ignore markers, empty lines and other lines that aren't valid sample
            # print('ignore: "{}"'.format(line));
            pass
    
    return features, labels
    
features, labels = load_data()

print("total no. of samples: {}".format(len(labels)))
print("total no. of spam samples: {}".format(labels.count(1)))
print("total no. of ham samples: {}".format(labels.count(0)))

print("\nPrint a random sample for inspection:")
random_idx = random.randint(0, len(labels))
print("example feature: {}".format(features[random_idx][0:]))
print("example label: {} ({})".format(labels[random_idx], 'spam' if labels[random_idx] else 'ham'))

# plot word cloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

words_list = [index for index, x in enumerate(labels) if x == 1]
spam_words = ' '.join([features[i] for i in words_list])
spam_wc = WordCloud(width = 512,height = 512).generate(spam_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(spam_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()

# ## Preprocess Data - Split data randomly into training and test subsets
from sklearn.model_selection import train_test_split

# load features and labels
features, labels = load_data()

# split data into training / test sets
features_train, features_test, labels_train, labels_test = train_test_split(
    features, 
    labels, 
    test_size=0.2,   # use 10% for testing
    random_state=42)

print("no. of train features: {}".format(len(features_train)))
print("no. of train labels: {}".format(len(labels_train)))
print("no. of test features: {}".format(len(features_test)))
print("no. of test labels: {}".format(len(labels_test)))


# ## Preprocess Data - Vectorize text data
# Use [TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) to vectorize text input
from sklearn.feature_extraction.text import TfidfVectorizer

# vectorize email text into tfidf matrix
# TfidfVectorizer converts collection of raw documents to a matrix of TF-IDF features.
# It's equivalent to CountVectorizer followed by TfidfTransformer.
vectorizer = TfidfVectorizer(
    input='content',     # input is actual text
    lowercase=True,      # convert to lower case before tokenizing
    stop_words='english' # remove stop words
)
features_train_transformed = vectorizer.fit_transform(features_train)
features_test_transformed  = vectorizer.transform(features_test)


# ## Train a Naive Bayes Classifier
# Use [MultinomialNB](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) to train a Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
import pickle

def save(vectorizer, classifier):
    '''
    save classifier to disk
    '''
    with open('model.pkl', 'wb') as file:
        pickle.dump((vectorizer, classifier), file)
        
def load():
    '''
    load classifier from disk
    '''
    with open('model.pkl', 'rb') as file:
      vectorizer, classifier = pickle.load(file)
    return vectorizer, classifier

# train a classifier
classifier = MultinomialNB()
classifier.fit(features_train_transformed, labels_train)

# save the trained model
save(vectorizer, classifier)

# score the classifier accuracy
print("classifier accuracy {:.2f}%".format(classifier.score(features_test_transformed, labels_test) * 100))

# # Calculate F1 Score
# Calculate [F1 score](https://en.wikipedia.org/wiki/F1_score) using [sklearn metrics.f1_score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
import numpy as np
from sklearn import metrics
prediction = classifier.predict(features_test_transformed)
fscore = metrics.f1_score(labels_test, prediction, average='macro')
print("F score {:.2f}".format(fscore))

# ## Using the trained classifier for prediction
vectorizer, classifer = load()

print('\nPerform a test')                    
#email_input = 'enter your email here'
email_input = ['<p>Sick sea he uses might where each sooth would by he and dear friend then. Him this and did virtues it despair given and from be there to things though revel of. Felt charms waste said below breast. Nor haply scorching scorching in sighed vile me he maidens maddest. Alas of deeds monks. Dote my and was sight though. Seemed her feels he childe which care hill.</p><p>Of her was of deigned for vexed given. A along plain. Pile that could can stalked made talethis to of his suffice had. Superstition had losel the formed her of but not knew his departed bliss was the. Riot spent only tear childe. Ere in a disporting more. Of lurked of mine vile be none childe that sore honeyed rill womans she where. She time all upon loathed to known. Seek atonement hall sore where ear. Ofttimes rake domestic dear the monks one thence come friends. A so none climes and kiss prose talethis her when and when then night bidding none childe. Will fame deemed relief delphis he whateer. Soon love scorching low of lone mine ee haply. Than oft lurked worse perchance and gild earth. Are did the losel of none would ofttimes his and. His in this basked such one at so was himnot native. Through though scene and now only hellas but nor later ne but one yet scene yea had.</p>']
email_input_transformed = vectorizer.transform(email_input)
prediction = classifer.predict(email_input_transformed)

print('EMAIL:', email_input)
print('The email is', 'SPAM' if prediction else 'HAM')


