import torch
import torch.nn as nn
from torch_geometric.data import Batch
from transformers import AutoTokenizer, AutoModel
from src.model.gin_network import GINNet
from src.model.op_id_mapper import OperatorIdMapper

class Satie(nn.Module):
    def __init__(
            self,
            embedding_model_name: str = "microsoft/codebert-base",
            hidden_dim: int = 128,
            projection_dim: int = 64,
            dropout: float = 0.2,
            operator_embedding_dim: int = 16,
            gnn_num_layers: int = 3,
    ):
        super().__init__()

        # text encoder
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name)
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        text_embedding_dim = self.embedding_model.config.hidden_size
        self.need_padding = "longcoder" in embedding_model_name
        
        self.text_projector = nn.Sequential(
            nn.Linear(text_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim)
        )

        num_operators = OperatorIdMapper.NUM_OPERATORS
        self.operator_embedding_layer = nn.Embedding(num_operators, operator_embedding_dim)
        
        graph_input_dim = operator_embedding_dim + text_embedding_dim
        self.graph_encoder = GINNet(
            input_dim=graph_input_dim,
            hidden_dim=hidden_dim,
            num_layers=gnn_num_layers,
            dropout=dropout,
        )
        self.graph_projector = nn.Linear(hidden_dim, projection_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(3 * projection_dim, projection_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, 1),
        )

        # Precompute [CLS], [SEP] token ids for efficiency
        self.cls_token_id = torch.tensor(
            self.embedding_tokenizer.convert_tokens_to_ids([self.embedding_tokenizer.cls_token]),
        )
        self.sep_token_id = torch.tensor(
            self.embedding_tokenizer.convert_tokens_to_ids([self.embedding_tokenizer.sep_token]),
        )

        # freeze embedding model
        for param in self.embedding_model.parameters():
            param.requires_grad = False
        # num_layers = len(self.embedding_model.encoder.layer)
        # for name, param in self.embedding_model.named_parameters():
        #     if not f"layer.{num_layers - 1}" in name: 
        #         param.requires_grad = False

    def forward(
            self, 
            schema_input_ids, 
            schema_attention_mask, 
            question_input_ids, 
            question_attention_mask, 
            graph_batch: Batch
    ):
        question_embedding = self.encode_text_with_schema(
            schema_input_ids, 
            schema_attention_mask, 
            question_input_ids, 
            question_attention_mask
        )

        operator_feature = self.operator_embedding_layer(
            graph_batch.operator_ids
        )

        property_embedding = self.encode_text_with_schema(
            schema_input_ids[graph_batch.batch], 
            schema_attention_mask[graph_batch.batch], 
            graph_batch.property_input_ids, 
            graph_batch.property_attention_mask
        )
        # property_feature = self.property_projector(property_embedding)
        
        # graph_feature = torch.cat([operator_feature, property_feature], dim=1)
        graph_feature = torch.cat([operator_feature, property_embedding], dim=1)
        graph_embedding = self.graph_encoder(graph_feature, graph_batch.edge_index, graph_batch.batch)

        question_embedding = self.text_projector(question_embedding)
        graph_embedding = self.graph_projector(graph_embedding)
        hadamard_embedding = question_embedding * graph_embedding
        fusion_embedding = torch.cat([question_embedding, graph_embedding, hadamard_embedding], dim=1)
        logits = self.classifier(fusion_embedding)

        return logits.squeeze(1)

    def encode_text_with_schema(
            self, 
            schema_input_ids, 
            schema_attention_mask, 
            question_input_ids, 
            question_attention_mask, 
    ):
        batch_size, schema_length = schema_input_ids.shape
        batch_size, question_length = question_input_ids.shape
        
        # Prepare [CLS] and [SEP] tokens (ensure same device)
        cls_token_ids = self.cls_token_id.to(schema_input_ids.device).repeat(batch_size, 1)
        sep_token_ids = self.sep_token_id.to(schema_input_ids.device).repeat(batch_size, 1)
        token_attention_mask = torch.ones(size=(batch_size, 1), device=schema_input_ids.device)

        # Format sequence: [CLS] [SCHEMA] [SEP] [QUESTION] [SEP]
        embedding_input_ids = torch.cat(
            [cls_token_ids, schema_input_ids, sep_token_ids, question_input_ids, sep_token_ids], dim=1
        )
        embedding_attention_mask = torch.cat(
            [token_attention_mask, schema_attention_mask, token_attention_mask, question_attention_mask, token_attention_mask], dim=1
        )

        sequence_length = embedding_input_ids.shape[1]
        attention_window: int = self.embedding_model.config.attention_window[0]
        if self.need_padding and sequence_length % attention_window != 0:
            pad_token_id: int = self.embedding_tokenizer.pad_token_id
            padding_length = ((sequence_length + attention_window - 1) // attention_window) * attention_window - sequence_length
            assert padding_length > 0

            padding_input_ids = torch.full((batch_size, padding_length), pad_token_id, dtype=embedding_input_ids.dtype, device=embedding_input_ids.device)
            padding_attention_mask = torch.zeros((batch_size, padding_length), dtype=embedding_attention_mask.dtype, device=embedding_attention_mask.device)
            embedding_input_ids = torch.cat([embedding_input_ids, padding_input_ids], dim=1)
            embedding_attention_mask = torch.cat([embedding_attention_mask, padding_attention_mask], dim=1)
        
        embedding_output = self.embedding_model(
            input_ids=embedding_input_ids,
            attention_mask=embedding_attention_mask
        )
        question_start_index = 1 + schema_length + 1
        question_end_index = question_start_index + question_length
        question_embeddings = embedding_output.last_hidden_state[:, question_start_index:question_end_index, :]
        question_embedding = question_embeddings.mean(dim=1)
        return question_embedding

if __name__ == "__main__":
    from src.dataset.triple_stream_dataset import create_dataloader

    model = Satie(
        embedding_model_name = "microsoft/longcoder-base",
        # debug=True  # Enable timing information
    ).to('cuda')
    model.eval()

    _, _, test_loader = create_dataloader("bird", batch_size=3, mode="test")
    for batch in test_loader:
        with torch.no_grad():
            logits = model(
                schema_input_ids=batch["schema_input_ids"].to('cuda'),
                schema_attention_mask=batch["schema_attention_mask"].to('cuda'),
                question_input_ids=batch["question_input_ids"].to('cuda'),
                question_attention_mask=batch["question_attention_mask"].to('cuda'),
                graph_batch=batch["graph_batch"].to('cuda')
            )
            probs = torch.sigmoid(logits).to('cpu')

        print(probs)
        print(batch["labels"])
        break