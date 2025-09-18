# stephanie/agents/thought/sot_v01_dataset_builder.py
from __future__ import annotations

from typing import List, Dict, Any
from tqdm import tqdm
import json

class SoTV01DatasetBuilder:
    """
    Builds a dataset of thought trajectories for SoT v0.1.
    Each example is: (query, retrieved_turns, your_response, predicted_move)
    """

    def __init__(self, chat_store, embedding_service, logger=None):
        self.chat_store = chat_store
        self.embedding_service = embedding_service
        self.logger = logger

    def build_dataset(self, output_path: str, max_conversations: int = 1000):
        """
        Build the dataset and save it as a JSONL file.
        """
        # Get top conversations (by turns or messages)
        top_convs = self.chat_store.get_top_conversations(limit=max_conversations, by="turns")
        
        dataset = []
        for conv, _ in tqdm(top_convs, desc="Building SoT v0.1 Dataset"):
            turns = self.chat_store.get_turns_for_conversation(conv.id)
            for turn in turns:
                if not turn.user_message or not turn.assistant_message:
                    continue

                user_query = turn.user_message.text.strip()
                your_response = turn.assistant_message.text.strip()

                if not user_query or not your_response:
                    continue

                # Retrieve 1-2 most similar past turns (excluding this one)
                retrieved_turns = self._retrieve_similar_turns(user_query, current_turn_id=turn.id, top_k=2)

                # Get the predicted move (from meta)
                predicted_move = self._get_predicted_move(turn)

                # Create training example
                example = {
                    "query": user_query,
                    "retrieved_turns": [
                        {
                            "user": rt.user_message.text.strip() if rt.user_message else "",
                            "assistant": rt.assistant_message.text.strip() if rt.assistant_message else "",
                            "conversation_title": rt.conversation.title if rt.conversation else "Unknown",
                            "similarity_score": getattr(rt, 'similarity_score', 0.0)
                        }
                        for rt in retrieved_turns
                    ],
                    "response": your_response,
                    "predicted_move": predicted_move,
                    "conversation_id": conv.id,
                    "turn_id": turn.id
                }

                dataset.append(example)

        # Save dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in dataset:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')

        if self.logger:
            self.logger.info(f"[SoTV01DatasetBuilder] Dataset saved to {output_path}. Total examples: {len(dataset)}")

    def _retrieve_similar_turns(self, query: str, current_turn_id: int, top_k: int = 2) -> List[Any]:
        """
        Simple semantic search: find the top_k most similar turns to the query.
        Excludes the current turn.
        """
        query_embedding = self.embedding_service.get_or_create(query)
        all_turns = self._get_all_turns_excluding(current_turn_id)

        similarities = []
        for turn in all_turns:
            if not turn.user_message or not turn.user_message.text:
                continue
            turn_embedding = self.embedding_service.get_or_create(turn.user_message.text)
            similarity = self._cosine_similarity(query_embedding, turn_embedding)
            similarities.append((similarity, turn))

        similarities.sort(key=lambda x: x[0], reverse=True)
        top_turns = [turn for _, turn in similarities[:top_k]]
        for i, turn in enumerate(top_turns):
            turn.similarity_score = similarities[i][0]
        return top_turns

    def _get_all_turns_excluding(self, turn_id_to_exclude: int) -> List[Any]:
        """Fetch all ChatTurnORMs except the one with the given ID."""
        all_conversations = self.chat_store.get_all(limit=1000)
        all_turns = []
        for conv in all_conversations:
            turns = self.chat_store.get_turns_for_conversation(conv.id)
            for turn in turns:
                if turn.id != turn_id_to_exclude:
                    all_turns.append(turn)
        return all_turns

    @staticmethod
    def _cosine_similarity(vec_a, vec_b):
        import numpy as np
        vec_a = np.array(vec_a)
        vec_b = np.array(vec_b)
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        return dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

    @staticmethod
    def _get_predicted_move(turn) -> str:
        """Extract the predicted move from the turn's meta."""
        meta = turn.assistant_message.meta if turn.assistant_message else {}
        return meta.get("reasoning_move", "VOICE")  # Default to "VOICE"