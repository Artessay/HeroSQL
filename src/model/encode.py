import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

class Encode(nn.Module):
    def __init__(
            self,
            embedding_model_name: str = "thenlper/gte-large",
            hidden_dim: int = 128,
            projection_dim: int = 64,
            dropout: float = 0.2,
    ):
        super().__init__()

        # text encoder
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name)
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        text_embedding_dim = self.embedding_model.config.hidden_size
        
        self.text_projector = nn.Sequential(
            nn.Linear(text_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim)
        )

        self.sql_projector = nn.Sequential(
            nn.Linear(text_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(3 * projection_dim, 3 * projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(3 * projection_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, 1),
        )

        # freeze embedding model
        for param in self.embedding_model.parameters():
            param.requires_grad = False

    def forward(
            self, 
            query_input_ids: Tensor, 
            query_attention_mask: Tensor,
            sql_input_ids: Tensor, 
            sql_attention_mask: Tensor,
    ):
        question_embedding = self.encode_text(
            query_input_ids, 
            query_attention_mask,
        )

        sql_embedding = self.encode_text(
            sql_input_ids, 
            sql_attention_mask,
        )
        
        question_embedding = self.text_projector(question_embedding)
        sql_embedding = self.sql_projector(sql_embedding)
        
        hadamard_embedding = question_embedding * sql_embedding
        fusion_embedding = torch.cat([question_embedding, sql_embedding, hadamard_embedding], dim=1)
        logits = self.classifier(fusion_embedding)

        return logits.squeeze(1)

    def encode_text(
            self, 
            query_input_ids, 
            query_attention_mask, 
    ):
        outputs = self.embedding_model(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask
        )
        embeddings = self.last_token_pool(outputs.last_hidden_state, query_attention_mask)
        return embeddings
    
    @staticmethod
    def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


if __name__ == "__main__":
    from src.dataset.triple_stream_dataset import create_dataloader

    # embedding_model_name = "BAAI/bge-large-en"
    # embedding_model_name = "thenlper/gte-large"
    embedding_model_name = "facebook/contriever"
    model = Encode(
        embedding_model_name = embedding_model_name,
    ).to('cuda')
    model.eval()

    print("Loading data...")
    _, _, test_loader = create_dataloader("bird", batch_size=3, mode="test", method_name='encode', embedding_model_name=embedding_model_name)
    for batch in test_loader:
        with torch.no_grad():
            logits = model(
                query_input_ids=batch["query_input_ids"].to('cuda'),
                query_attention_mask=batch["query_attention_mask"].to('cuda'),
                sql_input_ids=batch["sql_input_ids"].to('cuda'),
                sql_attention_mask=batch["sql_attention_mask"].to('cuda'),
            )
            probs = torch.sigmoid(logits).to('cpu')

        print(probs)
        print(batch["labels"])
        break