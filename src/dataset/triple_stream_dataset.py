import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data, Batch
from transformers import AutoTokenizer

from src.model.op_id_mapper import OperatorIdMapper
from src.dataset.dataset_utils import load_triple_stream_dataset

class TripleStreamDataset(Dataset):
    def __init__(self, dataset_name: str, mode: str, method_name: str, embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        super().__init__()
        
        self.schema_list, self.question_list, self.graph_list, self.label_list = \
            load_triple_stream_dataset(dataset_name, mode)
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

        self.question_template = "Schema: {}\nQuestion: {}"
        self.graph_node_template = "SQL Query: {}\nNode Operator: {}\nNode Property: {}"
        self.max_property_length = 704
        self.max_node_attribute = 768
        self.method_name = method_name

    def __len__(self):
        return len(self.question_list)
    
    def __getitem__(self, idx):
        schema: str = self.schema_list[idx]
        question: str = self.question_list[idx]
        sql: str = self.graph_list[idx].sql
        graph: Data = self.graph_list[idx]
        label: int = self.label_list[idx]
        
        # Process PyG graph
        graph = self._process_graph(graph)

        return {
            'schema_text': schema,
            'question_text': question,
            'sql_text': sql,
            'graph': graph,
            'label': label,
        }
    
    def _process_graph(self, graph: Data):
        num_nodes = graph.num_nodes
        if num_nodes == 0:
            return Data(
                num_nodes=0,
                edge_index=torch.empty((2, 0), dtype=torch.long),
                graph_attributes_input_ids= torch.empty((0, self.max_node_attribute), dtype=torch.long), 
                graph_attributes_attention_mask= torch.empty((0, self.max_node_attribute), dtype=torch.long), 
            )

        edge_index = graph.edge_index

        sql = graph.sql
        types = graph.types
        properties = graph.properties

        if self.method_name in ['satie']:
            # encode operator
            operator_ids = torch.tensor(
                [
                    OperatorIdMapper.operator_to_id(operator)
                    for operator in types
                ],
                dtype=torch.long
            )

            # encode property
            property_enc = self.tokenizer(properties, truncation=False, padding='max_length', max_length=self.max_property_length, return_tensors="pt")
            property_input_ids = property_enc['input_ids']
            property_attention_mask = property_enc['attention_mask']
            
            # check property length
            assert property_input_ids.shape[1] == self.max_property_length

            return Data(
                num_nodes=num_nodes,
                edge_index=edge_index,
                operator_ids=operator_ids,
                property_input_ids=property_input_ids,
                property_attention_mask=property_attention_mask,
            )

        elif self.method_name in ['ast', 'lope', 'syntax', 'hero']:
            # encode attribute
            graph_attributes = [
                self.graph_node_template.format(operator, attribute)
                if self.method_name == 'ast' else self.graph_node_template.format(sql, operator, attribute)
                for operator, attribute in zip(types, properties)
            ]
            graph_attributes_enc = self.tokenizer(graph_attributes, truncation=True, padding='max_length', max_length=self.max_node_attribute, return_tensors="pt")
            graph_attributes_input_ids = graph_attributes_enc['input_ids']
            graph_attributes_attention_mask = graph_attributes_enc['attention_mask']

            # check attribute length
            assert graph_attributes_input_ids.shape[1] == self.max_node_attribute

            if self.method_name in ['ast', 'lope']:
                return Data(
                    num_nodes=num_nodes,
                    edge_index=edge_index,
                    graph_attributes_input_ids=graph_attributes_input_ids,
                    graph_attributes_attention_mask=graph_attributes_attention_mask,
                )
            elif self.method_name in ['syntax', 'hero']:
                conditions = graph.conditions
                for condition in conditions:
                    condition.sql = sql
                
                original_method_name = self.method_name
                self.method_name = 'ast'   # change method name to lope
                self.max_node_attribute = 64
                self.graph_node_template = "Node Operator: {}\nNode Property: {}"
                graph_node_asts = Batch.from_data_list([self._process_graph(condition) for condition in conditions])
                self.method_name = original_method_name
                self.max_node_attribute = 768
                self.graph_node_template = "SQL Query: {}\nNode Operator: {}\nNode Property: {}"

                return Data(
                    num_nodes=num_nodes,
                    edge_index=edge_index,
                    graph_attributes_input_ids=graph_attributes_input_ids,
                    graph_attributes_attention_mask=graph_attributes_attention_mask,
                    graph_node_asts=graph_node_asts,
                )
            else:
                raise ValueError(f"Unknown method name: {self.method_name}")
        else:
            # do nothing
            return Data(
                num_nodes=num_nodes,
                edge_index=edge_index,
            )
        
    def multimodal_collate_fn(self, batch):
        # batch: list of dicts
        schema_texts = [item['schema_text'] for item in batch]
        question_texts = [item['question_text'] for item in batch]
        sql_texts = [item['sql_text'] for item in batch]

        graph_list = [item['graph'] for item in batch]
        graph_batch = Batch.from_data_list(graph_list)

        label_list = [item['label'] for item in batch]
        label_batch = torch.tensor(label_list)

        if self.method_name in ['syntax', 'lope']:
            query_texts = [
                self.question_template.format(schema, question)
                for schema, question in zip(schema_texts, question_texts)
            ]
            query_batch = self.tokenizer(query_texts, padding=True, truncation=True, return_tensors='pt')

            return {
                'query_input_ids': query_batch['input_ids'],
                'query_attention_mask': query_batch['attention_mask'],
                'graph_batch': graph_batch,
                'labels': label_batch
            }
        elif self.method_name == 'hero':
            query_texts = [
                self.question_template.format(schema, question)
                for schema, question in zip(schema_texts, question_texts)
            ]
            query_batch = self.tokenizer(query_texts, padding=True, truncation=True, return_tensors='pt')

            sql_texts = self.tokenizer(sql_texts, padding=True, truncation=True, return_tensors='pt')

            return {
                'query_input_ids': query_batch['input_ids'],
                'query_attention_mask': query_batch['attention_mask'],
                'sql_input_ids': sql_texts['input_ids'],
                'sql_attention_mask': sql_texts['attention_mask'],
                'graph_batch': graph_batch,
                'labels': label_batch
            }
        elif self.method_name == 'plain':
            query_texts = [
                self.question_template.format(schema, question)
                for schema, question in zip(schema_texts, question_texts)
            ]
            query_batch = self.tokenizer(query_texts, padding=True, truncation=True, return_tensors='pt')

            sql_texts = self.tokenizer(sql_texts, padding=True, truncation=True, return_tensors='pt')

            return {
                'query_input_ids': query_batch['input_ids'],
                'query_attention_mask': query_batch['attention_mask'],
                'sql_input_ids': sql_texts['input_ids'],
                'sql_attention_mask': sql_texts['attention_mask'],
                'labels': label_batch
            }
        elif self.method_name == 'encode':
            query_texts = [
                self.question_template.format("None", question)
                for _, question in zip(schema_texts, question_texts)
            ]
            query_batch = self.tokenizer(query_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

            sql_texts = self.tokenizer(sql_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

            return {
                'query_input_ids': query_batch['input_ids'],
                'query_attention_mask': query_batch['attention_mask'],
                'sql_input_ids': sql_texts['input_ids'],
                'sql_attention_mask': sql_texts['attention_mask'],
                'labels': label_batch
            }
        elif self.method_name == 'satie':
            schema = self.tokenizer(schema_texts, padding=True, truncation=True, return_tensors='pt')
            question = self.tokenizer(question_texts, padding=True, truncation=True, return_tensors='pt')
            
            return {
                'schema_input_ids': schema['input_ids'],
                'schema_attention_mask': schema['attention_mask'],
                'question_input_ids': question['input_ids'],
                'question_attention_mask': question['attention_mask'],
                'graph_batch': graph_batch,
                'labels': label_batch
            }
        else:
            raise ValueError(f"Invalid method name: {self.method_name}")

def create_dataloader(dataset_name: str, batch_size: int, num_workers: int = 64, mode: str = 'train', method_name: str = "hero", embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
    assert mode in ['train', 'test']
    # print("Creating dataloader for dataset:", dataset_name, "mode:", mode, "method:", method_name)

    if mode == 'train':
        # load dataset
        base_dataset = TripleStreamDataset(dataset_name, "train", method_name, embedding_model_name)

        # split train dataset into train and val
        train_dataset, val_dataset = random_split(base_dataset, [0.8, 0.2])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=base_dataset.multimodal_collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=base_dataset.multimodal_collate_fn
        )
    else:
        train_loader, val_loader = None, None

    test_dataset = TripleStreamDataset(dataset_name, "dev", method_name, embedding_model_name)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=test_dataset.multimodal_collate_fn
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import torch

    dataset = TripleStreamDataset("bird", "dev", "hero")
    print(len(dataset))

    item = dataset[0]
    print(item['question_text'])
    print(item['graph'])
    print(item['label'])
    print("-----")

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=dataset.multimodal_collate_fn)
    for batch in dataloader:
        print(batch['query_input_ids'].shape)
        print(batch['graph_batch'])
        print(batch['labels'])
        break