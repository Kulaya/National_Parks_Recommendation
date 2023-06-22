The code imports the required libraries, including csv, torch, BertTokenizer and BertModel from the transformers library, and cosine_similarity from sklearn.metrics.pairwise.

It loads a pre-trained BERT tokenizer and model.

The code reads national park data from a CSV file and stores it in a list called National_Parks, extracting relevant information such as the park name, best time to visit, and description.

It encodes the park descriptions using the BERT tokenizer and generates BERT embeddings for each description. The embeddings are stored in a list called embeddings.

The embeddings are reshaped and converted to a tensor.

The code computes the cosine similarity matrix using the embeddings.

The user specifies a national park they like, and the code finds the index of the corresponding park in the National_Parks list.

Based on the cosine similarity, the code identifies the most similar national parks to the one the user likes. It excludes the liked park itself and selects the top 2 recommendations.

The recommended national parks are stored in a list called recommended_national_parks.

Finally, the code prints a message stating the user's liked national park and lists the recommended national parks based on the cosine similarity.
