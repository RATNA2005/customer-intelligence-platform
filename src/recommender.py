import numpy as np
import pandas as pd

class ItemBasedRecommender:
    """
    Item-Item Collaborative Filtering Recommender using Cosine Similarity.
    Zero external dependencies.
    """
    def __init__(self, top_n_similar=10):
        self.top_n_similar = top_n_similar
        self.item_similarity_df = None
        self.user_item_matrix = None

    def fit(self, interactions_df):
        # Construct User-Item interaction matrix (purchase counts)
        self.user_item_matrix = interactions_df.pivot(
            index="customer_id", 
            columns="product_id", 
            values="purchase_count"
        ).fillna(0)
        
        # Calculate item-item similarity matrix
        print("Computing item similarity matrix...")
        matrix = self.user_item_matrix.values
        norms = np.linalg.norm(matrix, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        
        normalized_matrix = matrix / norms
        similarity_matrix = np.dot(normalized_matrix.T, normalized_matrix)
        
        item_ids = self.user_item_matrix.columns
        self.item_similarity_df = pd.DataFrame(
            similarity_matrix, 
            index=item_ids, 
            columns=item_ids
        )
        
    def recommend(self, customer_id, purchase_history, k=5):
        """
        Recommend top K products for a customer.
        """
        if not purchase_history:
            # Cold start: return top popular products
            popular = self.user_item_matrix.sum(axis=0).sort_values(ascending=False).index.tolist()
            return popular[:k]
            
        scores = {}
        for p_id in purchase_history:
            if p_id not in self.item_similarity_df.columns:
                continue
            similar_items = self.item_similarity_df[p_id].sort_values(ascending=False)
            
            for sim_p_id, score in similar_items.items():
                if sim_p_id in purchase_history:
                    continue
                scores[sim_p_id] = scores.get(sim_p_id, 0.0) + score
                
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        recommended = [item for item, score in sorted_scores[:k]]
        
        if len(recommended) < k:
            popular = self.user_item_matrix.sum(axis=0).sort_values(ascending=False).index.tolist()
            for p_id in popular:
                if p_id not in purchase_history and p_id not in recommended:
                    recommended.append(p_id)
                if len(recommended) >= k:
                    break
                    
        return recommended
