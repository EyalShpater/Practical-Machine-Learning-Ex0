# Practical Machine Learning
# Homework No. 0

# Student1 name: ...
# Student1 ID: ...
# Student2 name: ...
# Student2 ID: ...

# Needed libraries
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

###############################################################
############ Do not change this section  ######################
###############################################################
model = SentenceTransformer('average_word_embeddings_glove.6B.300d')


# def embed_text(text_list):
#     """
#         Embeds a list of text samples into a vector space.
#         ## Do not change the implementation of  this function
#         Args:
#         text_list (list): A list of text samples to be embedded.
#
#         Returns:
#         numpy.ndarray: A matrix of embedded vectors representing the input text samples.
#         """
#     embedded_text = model.encode(text_list)
#     return embedded_text


###############################################################



# Loading Dataset using Pandas
def load_dataset(file_path):
    """
    Loads a dataset from a CSV file.

    Args:
    file_path (str): The file path to the CSV file containing the dataset.

    Returns:
    pandas.DataFrame: A DataFrame representing the loaded dataset.

    Notes:
    The function reads the CSV file using Pandas and removes columns containing NaN values.
    Ensure the 'file_path' points to a valid CSV file with the appropriate encoding.
    """

    try:
        df = pd.read_csv(file_path, encoding='latin1')
        df_no_missing = df.dropna(axis=1)
        return df_no_missing
    except Exception as e:
        print(f"Error loading the dataset: {e}")
        return None


def explore_email_dataset(dataset):
    """
    Explore the email dataset to gather information and statistics.

    Args:
    dataset (pandas.DataFrame): The input dataset containing 'text' and 'label' columns.

    Returns:
    tuple: A tuple containing the following calculated statistics:
        - num_samples (int): Total number of samples in the dataset.
        - spam_count (int): Number of samples labeled as 'spam'.
        - ham_count (int): Number of samples labeled as 'ham'.
        - overall_average_length (float): Average length of all email messages.
        - average_spam_length (float): Average length of 'spam' email messages.
        - average_ham_length (float): Average length of 'ham' email messages.

    """

    print(f"Printing the first 10 rows of the dataset:\n {dataset.head(10)}")

    # Count of spam and ham samples
    num_samples = len(dataset)
    spam_count = len(dataset[dataset['label'] == 'spam'])
    ham_count = len(dataset[dataset['label'] == 'ham'])

    # Calculate average email length for spam and ham
    spam_lengths = (dataset[dataset['label'] == 'spam']['text'].apply(lambda text: len(text))).sum()
    ham_lengths = (dataset[dataset['label'] == 'ham']['text'].apply(lambda text: len(text))).sum()
    average_spam_length = spam_lengths / spam_count if spam_count != 0 else 0
    average_ham_length = ham_lengths / ham_count if ham_count != 0 else 0

    # Overall average email length
    overall_lengths = spam_lengths + ham_lengths
    overall_average_length = overall_lengths / num_samples

    return {
        "num_samples": num_samples,
        "spam_count": spam_count,
        "ham_count": ham_count,
        "overall_average_length": overall_average_length,
        "average_spam_length": average_spam_length,
        "average_ham_length": average_ham_length
    }


# # Convert text to vectors
# def text2vector(dataset):
#     """
#     Add a column named 'vector' that contains a vector representation of the text data in the 'text' column using the
#     provided embed_text function.
#
#     Args:
#     dataset (pandas.DataFrame): The input dataset containing a 'text' column to be converted.
#
#     Returns:
#     pandas.DataFrame: The input dataset with an additional 'vector' column containing vector representations of text.
#
#     Notes:
#     The 'dataset' argument should be a pandas DataFrame containing a 'text' column.
#     """
#     dataset['vector'] = dataset['text'].apply(...)  # embed the text column to obtain a vector representation
#     return dataset
#
#
# # Partition Data into Train and Test Sets
# def train_test_split(dataset, test_size=0.2, seed=42):
#     """
#     Splits a dataset into training and testing subsets.
#
#     Args:
#     dataset (pandas.DataFrame): The input dataset to be split into train and test sets.
#     test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
#     seed (int): The seed used by the random number generator (default is 42) for reproducibility.
#
#     Returns:
#     tuple: A tuple containing two pandas DataFrames (train_data, test_data) representing the training and testing subsets.
#     """
#
#     ...
#     return train_data, test_data
#
#
# # Implementing distance functions
# def calculate_distance(vector1, vector2, distance_type='euclidean'):
#     """
#     Calculates the distance between two vectors based on specified distance metrics.
#
#     Args:
#     vector1 (numpy.ndarray): The first input vector.
#     vector2 (numpy.ndarray): The second input vector.
#     distance_type (str): Type of distance metric to use ('euclidean', 'manhattan', or 'cosine').
#
#     Returns:
#     float: The calculated distance between the input vectors.
#
#     Examples:
#     >>> vec1 = np.array([1, 2, 3])
#     >>> vec2 = np.array([4, 5, 6])
#     >>> euclidean_dist = calculate_distance(vec1, vec2, distance_type='euclidean')
#     """
#
#     if distance_type == 'euclidean':
#         distance = ... # hint:use np.linalg.norm
#     elif distance_type == 'manhattan':
#         distance = ...
#     elif distance_type == 'cosine':
#         dot_product = np.dot(vector1, vector2)
#         vector1norm = ...
#         vector2norm = ...
#         cosine_similarity = ... / ...
#         distance = 1 - cosine_similarity
#     else:
#         raise ValueError("Invalid distance type. Choose 'euclidean', 'manhattan', or 'cosine'.")
#     return distance
#
#
# def test_calculate_distance():
#     """
#     Test function for calculating distance matrix based on different distance metrics.
#
#     Notes:
#     This function uses embeddings obtained from text sentences to calculate distance matrices
#     using different distance metrics (euclidean, cosine, manhattan) with the `calculate_distance` function.
#
#     """
#     sentences = ['I love my big dog.',
#                  'I love my cat.',
#                  'I hate animals.',
#                  'I am a student in the Academic College of Tel Aviv-Yaffa.']
#     embeddings = embed_text(sentences)
#
#     def calculate_distance_matrix(vectors_list, distance_type='euclidean'):
#         """
#         Calculate the distance matrix based on specified distance metric.
#
#         Args:
#         vectors_list (list): List of vectors to calculate distances between.
#         distance_type (str): Type of distance metric to use ('euclidean', 'cosine', 'manhattan').
#
#         Returns:
#         numpy.ndarray: A matrix containing distances between vectors based on the specified metric.
#
#         Note: In Python, functions can be nested within other functions, allowing for the creation of inner functions
#         within the scope of outer functions. This nesting capability enables the encapsulation of functionality and the
#         organization of code into more manageable and modular structures.
#         """
#
#         # Calculating the distance matrix containing the distance between every two sentences
#         ...
#         distance_matrix = np.inf * ...
#
#         for i in range(num_vectors):
#             for j in range(i + 1, num_vectors):
#                 distance_matrix[i, j] = ...
#
#         return distance_matrix
#
#     for metric in ['euclidean', 'cosine', 'manhattan']:
#         dist = calculate_distance_matrix(embeddings, distance_type=metric)
#         assert dist[0, 1] <= dist[0, 2]
#         assert dist[0, 1] <= dist[0, 3]
#
#
# # Implementing the KNN algorithm
# def k_nearest_neighbors(train_set, k=3, distance_type='euclidean'):
#     """
#     Defines a k-nearest neighbors classifier for classification based on distance metrics.
#
#     Args:
#     train_set (pandas.DataFrame): The training dataset containing 'vector' and 'label' columns.
#     k (int): Number of nearest neighbors to consider (default is 3).
#     distance_type (str): Type of distance metric to use ('euclidean', 'manhattan', or 'cosine').
#
#     Returns:
#     function: A function to classify test samples based on the k-nearest neighbors approach.
#
#     Example:
#     >>> # Suppose train_set contains vectors and labels for training data
#     >>> knn_classifier = k_nearest_neighbors(train_set, k=3, distance_type='euclidean')
#     >>> test_sample = np.array([1.2, 3.4, 5.6])  # Sample test vector
#     >>> predicted_label = knn_classifier(test_sample)  # Predict the label for the test sample
#     """
#     def classify(test_sample):
#         train_vectors = np.array(train_set['vector'].tolist())
#         distances = ...
#         sorted_indices = np.argsort(...)[:...]
#         k_nearest_labels = train_set['label'].iloc[...].tolist()
#
#         majority_vote = ...
#
#         return majority_vote
#
#     return classify
#
#
# # Calculate accuracy
# def calculate_accuracy(true_labels, predicted_labels):
#     """
#     Calculates the accuracy of predicted labels compared to true labels.
#
#     Args:
#     true_labels (numpy.ndarray or list): Array or list of true labels.
#     predicted_labels (numpy.ndarray or list): Array or list of predicted labels.
#
#     Returns:
#     float: Accuracy score representing the proportion of correct predictions.
#     """
#     ...
#     return accuracy
#

if __name__ == '__main__':
    file_path = "emails.csv"
    df = load_dataset(file_path)
    print((df[df['label'] == 'spam']['text'].apply(lambda text : len(text)).sum()))
    print(explore_email_dataset(df))
    # Calculating basic stats. Note the printing format of floats.
    # stats = explore_email_dataset(df)
    # print(f"Number of samples: {stats['num_samples']}")
    # print(f"Number of spam emails: {stats['spam_count']}")
    # print(f"Number of ham emails: {stats['ham_count']}")
    # print(f"Average email length (Overall): {stats['overall_average_length']:.2f}")
    # print(f"Average Spam email length: {stats['average_spam_length']:.2f}")
    # print(f"Average Ham email length: {stats['average_ham_length']:.2f}")
    #
    # df_with_vectors = text2vector(df)
    # train_set, test_set = train_test_split(df_with_vectors, test_size=0.2)
    # test_calculate_distance()
    # knn_classifier = k_nearest_neighbors(train_set, k=3, distance_type='euclidean') # Choose the parameeters that returned the best results.
    # predictions = [knn_classifier(vector) for vector in test_set['vector']]
    # accuracy = calculate_accuracy(test_set['label'], predictions)
    # print(f"Test set accuracy: {accuracy}")
