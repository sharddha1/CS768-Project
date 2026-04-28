import os
import time
import tracemalloc
import ctypes
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import random
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.datasets import Planetoid
import torch_geometric.utils as pyg_utils
from nodevectors import DeepWalk

# --- 1. Fast C++ LSH Wrapper ---
lib_path = os.path.abspath('./fast_lsh.so')
c_lsh_lib = ctypes.CDLL(lib_path)
c_lsh_lib.compute_lsh_c.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
]

def precompute_lsh_fast(content_features, m, k):
    num_nodes, feature_dim = content_features.shape
    total_hashes = m * k
    random_vectors = np.random.randn(feature_dim, total_hashes)
    random_vectors /= np.linalg.norm(random_vectors, axis=0)
    
    features_c = np.ascontiguousarray(content_features, dtype=np.float64).flatten()
    random_vectors_c = np.ascontiguousarray(random_vectors, dtype=np.float64).flatten()
    output_c = np.zeros((num_nodes * m), dtype=np.int32)
    
    c_lsh_lib.compute_lsh_c(features_c, random_vectors_c, num_nodes, feature_dim, m, k, output_c)
    return output_c.reshape((num_nodes, m))

# --- 2. PyTorch CP-LSH Model ---
class CPLSH_Model(nn.Module):
    def __init__(self, num_hash_funcs, bits_per_hash, embedding_dim):
        super().__init__()
        self.m = num_hash_funcs
        total_buckets = num_hash_funcs * (2 ** bits_per_hash)
        self.source_embeddings = nn.Embedding(total_buckets, embedding_dim)
        self.target_embeddings = nn.Embedding(total_buckets, embedding_dim)
        nn.init.xavier_uniform_(self.source_embeddings.weight)
        nn.init.xavier_uniform_(self.target_embeddings.weight)

    def get_node_embedding(self, node_hash_indices, is_source=True):
        embs = self.source_embeddings(node_hash_indices) if is_source else self.target_embeddings(node_hash_indices)
        return embs.mean(dim=1)

    def forward(self, src_hashes, pos_dst_hashes, neg_dst_hashes):
        s_u = self.get_node_embedding(src_hashes, is_source=True)
        t_v_pos = self.get_node_embedding(pos_dst_hashes, is_source=False)
        pos_loss = -torch.nn.functional.logsigmoid(torch.sum(s_u * t_v_pos, dim=1))
        
        B, num_neg, m = neg_dst_hashes.shape
        t_n_neg = self.get_node_embedding(neg_dst_hashes.view(B * num_neg, m), is_source=False).view(B, num_neg, -1)
        neg_score = torch.bmm(t_n_neg, s_u.unsqueeze(1).transpose(1, 2)).squeeze(2)
        neg_loss = -torch.sum(torch.nn.functional.logsigmoid(-neg_score), dim=1)
        
        return (pos_loss + neg_loss).mean()

# --- 3. Utilities & Evaluation ---
def measure_memory_and_time(func, *args, **kwargs):
    tracemalloc.start()
    start_time = time.time()
    result = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, time.time() - start_time, peak / 10**6

def prepare_link_prediction_data(G, test_ratio=0.2):
    G_train = G.copy()
    edges = list(G.edges())
    num_test = int(len(edges) * test_ratio)
    
    random.shuffle(edges)
    test_pos_edges = edges[:num_test]
    train_edges = edges[num_test:]
    G_train.remove_edges_from(test_pos_edges)
    
    test_neg_edges = []
    nodes = list(G.nodes())
    while len(test_neg_edges) < num_test:
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v) and u != v:
            test_neg_edges.append((u, v))
            
    return G_train, train_edges, test_pos_edges, test_neg_edges

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    print("Loading Cora Dataset...")
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    G = pyg_utils.to_networkx(data, to_undirected=True)
    features = data.x.numpy()
    labels = data.y.numpy()
    num_nodes = G.number_of_nodes()

    # Model Hyperparameters
    m = 6; k = 10; emb_dim = 80; epochs = 5
    
    print("\nPreparing Link Prediction splits...")
    G_train, train_edges, test_pos, test_neg = prepare_link_prediction_data(G)

    # --- Train CP-LSH ---
    def train_cplsh():
        node_hashes = precompute_lsh_fast(features, m, k)
        node_hashes_tensor = torch.tensor(node_hashes, dtype=torch.long)
        
        model = CPLSH_Model(m, k, emb_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        
        batch_size = 512
        for epoch in range(epochs):
            np.random.shuffle(train_edges)
            for i in range(0, len(train_edges), batch_size):
                batch = train_edges[i:i+batch_size]
                src = [e[0] for e in batch]
                dst = [e[1] for e in batch]
                neg = np.random.randint(0, num_nodes, size=(len(batch), 5))
                
                optimizer.zero_grad()
                loss = model(node_hashes_tensor[src], node_hashes_tensor[dst], node_hashes_tensor[neg])
                loss.backward()
                optimizer.step()
                
        model.eval()
        with torch.no_grad():
            embs = model.get_node_embedding(node_hashes_tensor, is_source=True).numpy()
        return embs

    print("\nTraining CP-LSH...")
    cplsh_embs, cplsh_time, cplsh_mem = measure_memory_and_time(train_cplsh)
    print(f"CP-LSH Time: {cplsh_time:.2f}s | Peak Memory: {cplsh_mem:.2f} MB")

    # --- Train DeepWalk Baseline ---
    def train_deepwalk():
        dw = DeepWalk(walklen=10, epochs=epochs, n_components=emb_dim)
        dw.fit(G_train)
        return np.array([dw.predict(str(i)) for i in range(num_nodes)])

    print("\nTraining DeepWalk Baseline...")
    dw_embs, dw_time, dw_mem = measure_memory_and_time(train_deepwalk)
    print(f"DeepWalk Time: {dw_time:.2f}s | Peak Memory: {dw_mem:.2f} MB")

    # --- Evaluation ---
    print("\n--- RESULTS ---")
    
    # Node Classification
    X_train, X_test, y_train, y_test = train_test_split(cplsh_embs, labels, test_size=0.5, random_state=42)
    clf = LinearSVC(max_iter=3000).fit(X_train, y_train)
    print(f"CP-LSH Node Classification F1: {f1_score(y_test, clf.predict(X_test), average='micro'):.4f}")

    # Link Prediction
    y_true = [1]*len(test_pos) + [0]*len(test_neg)
    y_scores = [np.dot(cplsh_embs[u], cplsh_embs[v]) for u, v in test_pos] + \
               [np.dot(cplsh_embs[u], cplsh_embs[v]) for u, v in test_neg]
    print(f"CP-LSH Link Prediction AUC: {roc_auc_score(y_true, y_scores):.4f}")