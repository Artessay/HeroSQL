import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch
from transformers import AutoTokenizer, AutoModel
from src.model.gat_network import GATNet
from torch_geometric.nn import global_add_pool

class Syntax(nn.Module):
    def __init__(
            self,
            embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
            hidden_dim: int = 128,
            projection_dim: int = 64,
            dropout: float = 0.2,
            gnn_num_layers: int = 2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

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
        
        self.ast_graph_encoder = GATNet(
            input_dim=text_embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=gnn_num_layers,
            dropout=dropout,
        )
        self.ast_graph_projector = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.lp_graph_projector = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )
        
        self.classifier = nn.Sequential(
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
            graph_batch: Batch
    ):
        question_embedding = self.encode_text(
            query_input_ids, 
            query_attention_mask,
        )

        graph_embedding = self.encode_lp_graph(graph_batch)

        question_embedding = self.text_projector(question_embedding)
        graph_embedding = self.lp_graph_projector(graph_embedding)
        hadamard_embedding = question_embedding * graph_embedding
        fusion_embedding = torch.cat([question_embedding, graph_embedding, hadamard_embedding], dim=1)
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

    def encode_lp_graph(self, graph_batch: Batch):
        node_asts_embedding = []
        for node_ast in graph_batch.graph_node_asts:
            node_ast_embedding = self.encode_ast_graph(node_ast)
            node_asts_embedding.append(node_ast_embedding)
        node_asts_embedding = torch.cat(node_asts_embedding, dim=0)
        node_asts_embedding = self.ast_graph_projector(node_asts_embedding)

        graph_embedding = global_add_pool(node_asts_embedding, graph_batch.batch)
        
        return graph_embedding
    
    def encode_ast_graph(self, graph_batch: Batch):
        
        graph_ptr = graph_batch.ptr
        num_graphs = graph_ptr.shape[0] - 1

        if len(graph_batch.batch) == 0:
            return torch.zeros(num_graphs, self.hidden_dim).to(graph_ptr.device)

        num_agg_graphs = graph_batch.batch.max().item() + 1
        num_padding_graphs = num_graphs - num_agg_graphs

        property_embedding = self.encode_text(
            graph_batch.graph_attributes_input_ids, 
            graph_batch.graph_attributes_attention_mask
        )
        
        agg_graph_embedding = self.ast_graph_encoder(property_embedding, graph_batch.edge_index, graph_batch.batch)
        assert num_agg_graphs == agg_graph_embedding.shape[0]

        padding_graph_embedding = torch.zeros(num_padding_graphs, self.hidden_dim).to(agg_graph_embedding.device)
        # print(agg_graph_embedding.shape, padding_graph_embedding.shape)
        
        graph_embedding = torch.cat([agg_graph_embedding, padding_graph_embedding], dim=0)
        return graph_embedding

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

    embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"
    model = Syntax(
        embedding_model_name = embedding_model_name,
    ).to('cuda')
    model.eval()

    _, _, test_loader = create_dataloader("bird", batch_size=3, mode="test", method_name="syntax", embedding_model_name=embedding_model_name)
    for batch in test_loader:
        with torch.no_grad():
            logits = model(
                query_input_ids=batch["query_input_ids"].to('cuda'),
                query_attention_mask=batch["query_attention_mask"].to('cuda'),
                graph_batch=batch["graph_batch"].to('cuda')
            )
            probs = torch.sigmoid(logits).to('cpu')

        print(probs)
        print(batch["labels"])
        break