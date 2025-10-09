from tqdm import tqdm
from src.dataset.triple_stream_dataset import create_dataloader


def run(args):
    # prepare dataset and model
    train_loader, val_loader, test_loader = create_dataloader(
        dataset_name = "all", 
        batch_size = 32,
        method_name = "hero",
    )
    
    # unit test block for dataloader
    for loader in [train_loader, val_loader, test_loader]:
        for batch in tqdm(loader):
            pass
    print('done')
    
if __name__ == '__main__':
    import dotenv
    dotenv.load_dotenv()

    from src.utils.lightning_params import get_lightning_args
    args = get_lightning_args()
    run(args)