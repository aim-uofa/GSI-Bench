import os
import json
import torch
import numpy as np
import compress_pickle
from sentence_transformers import SentenceTransformer


def get_bbox_dims(obj_data):
    """Extract object bounding box dimensions"""
    if "bbox" in obj_data:
        bbox_info = obj_data["bbox"]
        if "x" in bbox_info:
            return bbox_info
        if "size" in bbox_info:
            return bbox_info["size"]
        mins = bbox_info["min"]
        maxs = bbox_info["max"]
    else:
        bbox_info = obj_data
        mins = bbox_info["min_point"]
        maxs = bbox_info["max_point"]
        ret = {}
        ret['x'] = maxs[0] - mins[0]
        ret['y'] = maxs[1] - mins[1]
        ret['z'] = maxs[2] - mins[2]
        return ret
    
    return {k: maxs[k] - mins[k] for k in ["x", "y", "z"]}


class ObjathorRetriever:
    def __init__(self, sbert_model, config: dict):
        # Use paths from config
        annotations_path = config['annotations_path']
        features_path = config['sbert_features_path']
        
        # Load annotation database
        with open(annotations_path, 'r') as f:
            objathor_annotations = json.load(f)
        self.database = objathor_annotations
        
        # Load SBERT features
        objathor_sbert_features_dict = compress_pickle.load(features_path)
        objathor_uids = objathor_sbert_features_dict["uids"]
        objathor_sbert_features = objathor_sbert_features_dict["text_features"].astype(np.float32)
        
        self.sbert_features = objathor_sbert_features
        self.asset_ids = objathor_uids
        self.config = config
        self.sbert_model = sbert_model

    def retrieve_text_size(self, queries: list[str], target_sizes: list[dict], device="cuda"):
        """Batch retrieve objects using text descriptions and target sizes"""
        top_k_text = self.config.get('top_k_text', 20)
        top_k_final = self.config.get('top_k_final', 5)
            
        num_queries = len(queries)
        if num_queries == 0:
            return []
        if len(target_sizes) != num_queries:
            raise ValueError("Number of queries must match number of target sizes")

        results_list = [([], []) for _ in range(num_queries)]

        with torch.no_grad():
            query_feature_sbert = self.sbert_model.encode(
                queries, convert_to_tensor=True, show_progress_bar=False, device=device
            )
            
            sbert_features_gpu = torch.from_numpy(self.sbert_features).to(device)
            sbert_similarities = query_feature_sbert @ sbert_features_gpu.T
            top_text_scores, top_text_indices = torch.topk(sbert_similarities, k=top_k_text, dim=1)

            for i in range(num_queries):
                current_query_indices = top_text_indices[i]
                current_target_size_dict = target_sizes[i]

                candidate_uids_scores = []
                valid_candidate_sizes = []
                valid_candidate_uids = []

                for j, index in enumerate(current_query_indices):
                    uid = self.asset_ids[index.item()]
                    text_score = top_text_scores[i, j].item()
                    try:
                        size = get_bbox_dims(self.database[uid])
                        valid_candidate_sizes.append(size)
                        valid_candidate_uids.append(uid)
                        candidate_uids_scores.append((uid, text_score))
                    except (KeyError, Exception):
                        continue

                if not valid_candidate_uids:
                    continue

                try:
                    target_size_list = [current_target_size_dict['x'], current_target_size_dict['y'], current_target_size_dict['z']]
                    target_size_tensor = torch.tensor([target_size_list], dtype=torch.float32).to(device)
                    points_list = [[item['x'], item['z'], item['y']] for item in valid_candidate_sizes]
                    candidate_sizes_tensor = torch.tensor(points_list, dtype=torch.float32).to(device)
                except Exception:
                    continue
                
                size_similarities_list = self.get_bboxes_similarity(target_size_tensor, candidate_sizes_tensor)
                try:
                    flat_size_scores = [item[0] for item in size_similarities_list if isinstance(item, list) and len(item) > 0]
                    if len(flat_size_scores) != len(valid_candidate_uids):
                        continue
                    size_similarities_tensor = torch.tensor(flat_size_scores, device=device)
                except (TypeError, IndexError):
                    continue

                text_scores_map = {uid: score for uid, score in candidate_uids_scores}
                try:
                    flat_text_scores = [float(text_scores_map[uid]) for uid in valid_candidate_uids if uid in text_scores_map]
                    if len(flat_text_scores) != len(valid_candidate_uids):
                        continue
                    text_scores_tensor = torch.tensor(flat_text_scores, device=device)
                except (ValueError, KeyError, TypeError):
                    continue

                # Use weights from config
                text_weight = self.config.get('text_weight', 0.9)
                size_weight = self.config.get('size_weight', 0.1)
                combined_scores = (text_weight * text_scores_tensor) + (size_weight * size_similarities_tensor)

                actual_k_final = min(top_k_final, len(valid_candidate_uids))
                if actual_k_final <= 0:
                    continue

                top_combined_scores, top_indices_in_valid = torch.topk(combined_scores, k=actual_k_final)
                final_uids = [valid_candidate_uids[idx.item()] for idx in top_indices_in_valid]
                final_scores = top_combined_scores.cpu().tolist()

                results_list[i] = (final_uids, final_scores)

        return results_list

    def get_bboxes_similarity(self, target_bbox: torch.Tensor, candidate_bboxes: torch.Tensor) -> list[list[float]]:
        """Calculate bounding box similarity"""
        target_bbox = target_bbox.reshape(1, -1)
        norm_target = torch.norm(target_bbox, p=2, dim=1, keepdim=True)
        norm_target = torch.clamp(norm_target, min=1e-9)
        target_bbox_norm = target_bbox / norm_target

        norm_candidate = torch.norm(candidate_bboxes, p=2, dim=1, keepdim=True)
        norm_candidate = torch.clamp(norm_candidate, min=1e-9)
        candidate_bboxes_norm = candidate_bboxes / norm_candidate

        cosine_similarity = torch.mm(candidate_bboxes_norm, target_bbox_norm.T)
        return cosine_similarity.tolist()

