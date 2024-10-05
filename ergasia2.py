import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from node2vec import Node2Vec
from sklearn.manifold import TSNE
# from google.colab import drive


# Βήμα 1ο: Φόρτωση και προετοιμασία των datasets BRCA & LEUK
# Φόρτωση των dataset
brca_path = "brca.tsv"
leuk_path = "leuk.tsv"


# Καθορισμός dataset
file_choice = input("Επιλέξτε το αρχείο που θέλετε να τρέξετε ('BRCA' ή 'LEUK'): ").strip().lower()

if file_choice == 'brca':
    file_path = brca_path
elif file_choice == 'leuk':
    file_path = leuk_path
else:
    raise ValueError("Λάθος επιλογή αρχείου. Παρακαλώ επιλέξτε 'brca' ή 'leuk'.")


# Μετατροπή αρχείου tsv σε γράφο NetworkX
def load_graph_from_tsv(file_path):
    G_df = pd.read_csv(file_path, sep='\t')
    G = nx.from_pandas_edgelist(G_df, source=G_df.columns[0], target=G_df.columns[1])
    return G

# Aποθήκεθση φορτωμένου γράφηματος
protein_graph = load_graph_from_tsv(file_path)



# Βήμα 2ο: Διανυσματοποίηση των Κόμβων με το Node2Vec
# Αρχικοποίηση αντικειμένου Node2Vec
node2vec = Node2Vec(protein_graph, dimensions=64, walk_length=30, num_walks=200, workers=4)


# Εκπαίδευση μοντέλου
model = node2vec.fit(window=10, min_count=1, batch_words=4)


# Αποθήκευση Αναπαραστάσεων
embeddings = {str(node): model.wv[str(node)] for node in protein_graph.nodes()}


# Εκτύπωση Δείγματος Αναπαραστάσεων
print("Μερικά διανύσματα που δημιουργούνται από τον node2vec:\n")
for i, (node, vector) in enumerate(embeddings.items()):
    print(f"Κόμβος {node}: {vector}\n")
    if i == 4:  # Εκτύπωση μόνο των πρώτων 5 διανυσμάτων
        break

# Μετατροπή Αναπαραστάσεων σε Πίνακα
embedding_values = np.array(list(embeddings.values()))



# Βήμα 3ο: Εφαρμογή της Μεθόδου t-SNE για Μείωση Διαστάσεων
# Αρχικοποίηση αντικειμένου t-SNE
tsne = TSNE(n_components=2, random_state=42)

# Μετατροπή αναπαραστάσεων σε 2D 
embeddings_2d = tsne.fit_transform(embedding_values)



# Βήμα 4ο: Οπτικοποίηση των Δεδομένων
# Δημιουργία Οπτικοποίησης
plt.figure(figsize=(10, 10))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=5)

# Προσθήκη ετικετών
for i, node in enumerate(protein_graph.nodes()):
    plt.annotate(str(node), (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)

# Ορισμός Τίτλου
plt.title(f"Οπτικοποίηση t-SNE των Node2Vec Embeddings ({file_choice.upper()})", fontsize=14)

# Εμφάνιση Οπτικοποίησης
plt.show()