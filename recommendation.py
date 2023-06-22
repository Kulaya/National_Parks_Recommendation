import csv
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Read national park data from the CSV file
National_Parks = []
with open('/content/National_Parks.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        national_park = {
            'NATIONAL_PARKS': row['NATIONAL_PARKS'],
            'BEST_TIME_TO_VISIT': row['BEST_TIME_TO_VISIT'],
            'DESCRIPTION': row['DESCRIPTION']
        }
        National_Parks.append(national_park)

# Encode the descriptions and generate BERT embeddings
embeddings = []
for row in National_Parks:
    description = row['DESCRIPTION']
    # Tokenize the description
    tokens = tokenizer.encode(description, add_special_tokens=True)
    # Convert tokens to tensors
    input_ids = torch.tensor(tokens).unsqueeze(0)
    # Generate BERT embeddings
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())

# Reshape the embeddings
embeddings = torch.tensor(embeddings)

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(embeddings)

# Now, suppose a user likes "Serengeti National Park". We can recommend another national park based on cosine similarity.
liked_national_parks = "Serengeti National Park"
liked_national_parks_index = next(index for index, national_park in enumerate(National_Parks) if national_park['NATIONAL_PARKS'] == liked_national_parks)

# Find the most similar national parks
similar_national_park_indices = similarity_matrix[liked_national_parks_index].argsort()[::-1][1:3]  # Exclude the liked national park itself and get top 2 recommendations

recommended_national_parks = [National_Parks[index] for index in similar_national_park_indices]

print("Because you liked " + liked_national_parks + ", we recommend the following national parks:")

for recommended_park in recommended_national_parks:
    print(recommended_park['NATIONAL_PARKS'])
