import os
import logging
import argparse
from datetime import datetime
from rivagan import RivaGAN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

def train_model(data_dim, dataset_path, output_dir, batch_size=10, seq_len=8, lr=1e-3, epochs=10, num_workers=0):
    logger.info(f"Начало обучения модели с data_dim={data_dim}")
    
    try:
        model = RivaGAN(data_dim=data_dim)
        logger.info(f"Модель создана: data_dim={data_dim}")
        
        model.fit(
            dataset=dataset_path,
            batch_size=batch_size,
            seq_len=seq_len,
            lr=lr,
            use_critic=True,
            use_adversary=False,
            epochs=epochs,
            use_bit_inverse=True,
            use_noise=True,
            num_workers=num_workers
        )
        logger.info(f"Обучение завершено: data_dim={data_dim}")
        
        checkpoint_name = f"rivagan_data_dim_{data_dim}_epochs_{epochs}.pt"
        checkpoint_path = os.path.join(output_dir, checkpoint_name)
        model.save(checkpoint_path)
        logger.info(f"Модель сохранена: {checkpoint_path}")
        
        return True, checkpoint_path
        
    except Exception as e:
        logger.error(f"Ошибка при обучении модели data_dim={data_dim}: {str(e)}", exc_info=True)
        return False, None

def main():
    parser = argparse.ArgumentParser(description='Обучение нескольких моделей RivaGAN с разными data_dim')
    parser.add_argument('--dataset', type=str, required=True, help='Путь к датасету')
    parser.add_argument('--output', type=str, default='./checkpoints', help='Директория для сохранения чекпоинтов')
    parser.add_argument('--data_dims', type=int, nargs='+', 
                       default=[32, 48, 64, 96, 128, 256, 512],
                       help='Список значений data_dim для обучения')
    parser.add_argument('--batch_size', type=int, default=10, help='Размер батча')
    parser.add_argument('--seq_len', type=int, default=8, help='Длина последовательности')
    parser.add_argument('--lr', type=float, default=1e-3, help='Скорость обучения')
    parser.add_argument('--epochs', type=int, default=10, help='Количество эпох')
    parser.add_argument('--num_workers', type=int, default=0, help='Количество воркеров DataLoader')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    logger.info("НАЧАЛО ОБУЧЕНИЯ МОДЕЛЕЙ RIVAGAN")
    logger.info(f"Датасет: {args.dataset}")
    logger.info(f"Выходная директория: {args.output}")
    logger.info(f"Параметры: batch_size={args.batch_size}, seq_len={args.seq_len}, lr={args.lr}, epochs={args.epochs}, num_workers={args.num_workers}")
    logger.info(f"Значения data_dim: {args.data_dims}")
    
    results = {}
    for i, data_dim in enumerate(args.data_dims, 1):
        logger.info(f"[{i}/{len(args.data_dims)}] Обработка data_dim={data_dim}")
        
        success, checkpoint_path = train_model(
            data_dim=data_dim,
            dataset_path=args.dataset,
            output_dir=args.output,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.lr,
            epochs=args.epochs,
            num_workers=args.num_workers
        )
        
        results[data_dim] = {'success': success, 'checkpoint': checkpoint_path}
    
    successful = [d for d, r in results.items() if r['success']]
    failed = [d for d, r in results.items() if not r['success']]
    
    logger.info("ФИНАЛЬНЫЙ ОТЧЕТ")
    logger.info(f"Успешно обучено: {len(successful)}/{len(args.data_dims)}")
    logger.info(f"Неудачно: {len(failed)}/{len(args.data_dims)}")
    
    if successful:
        logger.info("Успешные модели:")
        for data_dim in successful:
            logger.info(f"  data_dim={data_dim:3d} -> {os.path.basename(results[data_dim]['checkpoint'])}")
    
    if failed:
        logger.info("Неудачные модели:")
        for data_dim in failed:
            logger.info(f"  data_dim={data_dim}")
    
    logger.info("ОБУЧЕНИЕ ЗАВЕРШЕНО")

if __name__ == "__main__":
    main()
