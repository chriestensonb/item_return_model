# item_return_model

### Goal
- This repo contains coed that trains a model to predict when an item will be returned, and uses that model to infer in items in a test set will be returned.
- Data not included in repo

### How to Use
- Clone this repo from you terminal: `https://github.com/chriestensonb/item_return_model.git`
- Create a virtual environment: `python3 -m venv venv`
- Activate your virtual environment: `source venv/bin/activate` (`venv/Scripts/activate` for Windows)
- Install dependencies: `pip install -r requirements.txt`
- Place the train and test files in your favorite directory `<my_dir>`
- Run `python main.py <my_dir>`
    - If no directory is provided it assumes the data is in the present working directory.

### Assumptions made in training and inference
1. The product catalog does not change over time, so all products are in both the train and test sets. This could be removed if we had a model predicting return rates by product.
2. An entire order's data is available for inference not just the single line item.

### Caveat
1. This is the final code and so many things like EDA, Feature Selection and Hyperparameter tuning have been ommitted.

### What I did
0. Create an MSRP feature and an ProductID feature which hashes the feautres that uniquely determine a product.

1. Initial EDA shows that most of the obvious features did not hold enough predictive power to create a reasonable model.

2. Derived a few non-obvious features
  - A flag for orders that have multiple of the same product, assuming that this usually indicates a return will happen.
  - The probability that a product will be returned based on the training data.  Mapped this to the test data.
  - Derived date features from the order data, assuming that most returns happen on product purchased around holidays or weekends.

3. Used MLFlow for experiment tracking and model selection

4. Implemented final code in this repo, using black to standardize formatting and flake8 to adhere to pep8 standards.
