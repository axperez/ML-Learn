'''
Program: Naive Bayes Classifier for Spam/Ham Emails
'''
# %%
import pandas
import numpy as np

def process_email(text):
    return list(set(text.split()))

def model_emails(emails):
    model = {}
    total = {'spam': 0, 'ham': 0}

    for index, email in emails.iterrows():
        for word in email['words']:
            if word not in model:
                model[word] = {'spam': 0, 'ham': 0}
            if email['spam']:
                model[word]['spam'] += 1
                total['spam'] += 1
            else:
                model[word]['ham'] += 1
                total['ham'] += 1

    return model, total

def prob_naive_bayes(word, model):
    count_spam_with_word = model[word]['spam']
    count_ham_with_word = model[word]['ham']
    return 1.0*(count_spam_with_word)/(count_spam_with_word+count_ham_with_word)

def predict_naive_bayes(email, model, total):
    words = process_email(email)
    spam_probs = []
    ham_probs = []
    for word in words:
        spam_probs.append(1.0*model[word]['spam']/total['spam'])
        ham_probs.append(1.0*model[word]['ham']/total['ham'])
    spam_probs.append(total['spam']/(total['spam'] + total['ham']))
    ham_probs.append(total['ham']/(total['spam'] + total['ham']))
    prod_spams = np.float(np.prod(spam_probs))
    prod_hams = np.float(np.prod(ham_probs))
   
    return 1.0*prod_spams/(prod_spams+prod_hams)

def predict_naive_bayes_alternate(email, model, total):
    words = process_email(email)
    spam_probs = []
    ham_probs = []
    for word in words:
        spam_pr = prob_naive_bayes(word, model)
        spam_probs.append(spam_pr)
        ham_probs.append(1-spam_pr)
    
    prod_spams = np.float(np.prod(spam_probs))
    prod_hams = np.float(np.prod(ham_probs))
    
    return 1.0*prod_spams/(prod_spams+prod_hams)

if __name__ == "__main__":
    emails = pandas.read_csv('data/emails.csv')
    emails['words'] = emails['text'].apply(process_email)
    model, total = model_emails(emails)
    predict_naive_bayes("hi mom how are you", model, total)

# %%
