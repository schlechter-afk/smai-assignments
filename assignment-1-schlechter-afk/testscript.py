import sys
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class KNN:
    def _init_(self):
        self.encoder_type = None
        self.k = None
        self.distance_metric = None
        self.data = None

    def setParameters(self, encoder_type, k, distance_metric):
        self.encoder_type = encoder_type
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, data):
        self.data = data

    def euclidean_distance(self, x1, x2):
        query_point = x1
        embeddings = x2
        # Chatgpt prompt to convert 1200 1-d lists of size 512 each to a 2d array of 1200*512 shape.
        embeddings_array = np.vstack(embeddings)
        squared_diffs = (embeddings_array - query_point) ** 2
        squared_distances = np.sum(squared_diffs, axis=-1)
        euclidean_distances = squared_distances ** 0.5
        return euclidean_distances

    def manhattan_distance(self, x1, x2):
        query_point = x1
        embeddings = x2
        embeddings_array = np.vstack(embeddings)
        manhattan_distances = np.sum(np.abs(embeddings_array - query_point), axis=-1)
        return manhattan_distances
    
    def cosine_distance(self, x1, x2):
        query_point = x1
        embeddings = x2
        embeddings_array = np.vstack(embeddings)
        dot_product = np.dot(embeddings_array, query_point)
        query_magnitude = np.linalg.norm(query_point)
        embedding_magnitudes = np.linalg.norm(embeddings_array, axis=1)
        cosine_similarity = dot_product / (embedding_magnitudes * query_magnitude)
        cosine_distances = 1 - cosine_similarity
        return cosine_distances

    def predict(self, query_point):
        distances = []
        if self.encoder_type == 'resnet':
            embeddings = self.data[:, 1]
        else:
            embeddings = self.data[:, 2]
        labels = self.data[:, 3]
        if self.distance_metric == 'euclidean':
            distances = self.euclidean_distance(query_point, embeddings)
        elif self.distance_metric == 'manhattan':
            distances = self.manhattan_distance(query_point, embeddings)
        else:
            distances = self.cosine_distance(query_point[0], embeddings)

        distances = np.column_stack((distances, labels))
        sorted_indices = np.argsort(distances[:, 0])
        k_nearest_indices = sorted_indices[:self.k] 
        k_distance_labels = distances[k_nearest_indices]

        inverse_distances = 1.0 / (k_distance_labels[:, 0] + 1).astype(float)
        invlabels = k_distance_labels[:, 1]

        unique_labels, label_inverse = np.unique(invlabels, return_inverse=True)
        inverse_sum = np.bincount(label_inverse, weights=inverse_distances)

        max_vote_index = np.argmax(inverse_sum)
        prediction = unique_labels[max_vote_index]
        return prediction
 
    def evaluate(self, validation_data):
        true_labels = validation_data[:, 3]
        predicted_labels = []
        for row in validation_data:
            if self.encoder_type == 'resnet':
                query_point = row[1]
            else:
                query_point = row[2]

            predicted_label = self.predict(query_point)
            predicted_labels.append(predicted_label)

        # Used ChatGPT to get the following scores.
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted',zero_division =1)
        recall = recall_score(true_labels, predicted_labels, average='weighted',zero_division=1)
        f1 = f1_score(true_labels, predicted_labels, average='weighted',zero_division=1)
        return accuracy, precision, recall, f1
    
def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluate_knn.py <test_data_file.npy>")
        sys.exit(1)

    test_data_file = sys.argv[1]

    try:
        test_data = np.load(test_data_file, allow_pickle=True)
    except FileNotFoundError:
        print("Error: Test data file not found.")
        sys.exit(1)

    # Instantiate the KNN model and fit with training data
    knn_model = KNN()
    train_data = np.load('data.npy', allow_pickle=True)  # Replace with your train data file
    knn_model.setParameters('vit',10,'euclidean')
    knn_model.fit(train_data)

    # Evaluate the KNN model using test data
    max_accuracy, prec, rec, f1sc = knn_model.evaluate(test_data)
    hyper_dist = "euclidean"
    hyper_k = 5
    hyper_enc = 'vit'
    # Print the evaluation results in a table
    print("Evaluation Results:")
    print("=================================")
    print(f"Accuracy: {max_accuracy:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1sc:.4f}")
    print(f"Distance Metric: {hyper_dist}")
    print(f"Encoder: {hyper_enc}")
    print(f"k: {hyper_k}")
    print("=================================")

if __name__ == "__main__":
    main()
