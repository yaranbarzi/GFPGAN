import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite
from tqdm import tqdm
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import time

from gfpgan import GFPGANer


def process_batch(batch_images, restorer, args, batch_paths):
    """پردازش یک batch از تصاویر به صورت همزمان"""
    results = []
    
    with torch.no_grad():  # کاهش مصرف حافظه
        for i, (img, img_path) in enumerate(zip(batch_images, batch_paths)):
            if img is None:
                results.append(None)
                continue
                
            try:
                # استفاده از half precision برای سرعت بیشتر
                if torch.cuda.is_available() and hasattr(restorer.gfpgan, 'half'):
                    restorer.gfpgan.half()
                
                cropped_faces, restored_faces, restored_img = restorer.enhance(
                    img,
                    has_aligned=args.aligned,
                    only_center_face=args.only_center_face,
                    paste_back=True,
                    weight=args.weight)
                
                results.append(restored_img)
                
            except Exception as e:
                print(f"خطا در پردازش {os.path.basename(img_path)}: {e}")
                results.append(None)
    
    return results


def preload_images(img_paths, batch_size=8):
    """پیش‌بارگذاری تصاویر برای سرعت بیشتر"""
    batches = []
    batch_paths = []
    
    for i in range(0, len(img_paths), batch_size):
        batch = []
        paths = img_paths[i:i + batch_size]
        
        for img_path in paths:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            batch.append(img)
        
        batches.append(batch)
        batch_paths.append(paths)
    
    return batches, batch_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs/whole_imgs')
    parser.add_argument('-o', '--output', type=str, default='results')
    parser.add_argument('-v', '--version', type=str, default='1.3')
    parser.add_argument('-s', '--upscale', type=int, default=2)
    parser.add_argument('--bg_upsampler', type=str, default='realesrgan')
    parser.add_argument('--bg_tile', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=4, help='تعداد تصاویر در هر batch')
    parser.add_argument('--suffix', type=str, default=None)
    parser.add_argument('--only_center_face', action='store_true')
    parser.add_argument('--aligned', action='store_true')
    parser.add_argument('--ext', type=str, default='auto')
    parser.add_argument('-w', '--weight', type=float, default=0.5)
    parser.add_argument('--fast_mode', action='store_true', help='حالت سریع با کیفیت کمتر')
    args = parser.parse_args()

    # ------------------------ input & output ------------------------
    if args.input.endswith('/'):
        args.input = args.input[:-1]
    if os.path.isfile(args.input):
        img_list = [args.input]
    else:
        img_list = sorted(glob.glob(os.path.join(args.input, '*')))

    os.makedirs(args.output, exist_ok=True)
    print(f"تعداد کل فریم‌ها: {len(img_list)}")

    # بهینه‌سازی CUDA
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name()}")

    # ------------------------ set up background upsampler ------------------------
    bg_upsampler = None
    if args.bg_upsampler == 'realesrgan' and not args.fast_mode:
        if torch.cuda.is_available():
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=args.bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True)  # استفاده از half precision برای سرعت
    
    if args.fast_mode:
        print("حالت سریع فعال: background upsampler غیرفعال")

    # ------------------------ set up GFPGAN restorer ------------------------
    if args.version == '1.3':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.3'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    else:
        raise ValueError(f'نسخه مدل {args.version} پشتیبانی نمی‌شود.')

    # determine model paths
    model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('gfpgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = url

    restorer = GFPGANer(
        model_path=model_path,
        upscale=args.upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)

    # ------------------------ optimize for speed ------------------------
    if torch.cuda.is_available() and hasattr(restorer.gfpgan, 'half'):
        restorer.gfpgan.half()  # استفاده از half precision
    
    restorer.gfpgan.eval()  # حالت evaluation برای سرعت بیشتر

    # ------------------------ process with batching ------------------------
    processed_count = 0
    start_time = time.time()
    
    # پیش‌بارگذاری تصاویر در batch ها
    print(f"پیش‌بارگذاری تصاویر در batch های {args.batch_size} تایی...")
    image_batches, path_batches = preload_images(img_list, args.batch_size)
    
    with tqdm(total=len(img_list), desc="پردازش فریم‌ها", unit="فریم") as pbar:
        for batch_idx, (img_batch, path_batch) in enumerate(zip(image_batches, path_batches)):
            
            # پردازش batch
            results = process_batch(img_batch, restorer, args, path_batch)
            
            # ذخیره نتایج
            for img_path, restored_img in zip(path_batch, results):
                if restored_img is not None:
                    img_name = os.path.basename(img_path)
                    basename, ext = os.path.splitext(img_name)
                    
                    if args.ext == 'auto':
                        extension = ext[1:]
                    else:
                        extension = args.ext

                    if args.suffix is not None:
                        save_restore_path = os.path.join(args.output, 'restored_imgs', f'{basename}_{args.suffix}.{extension}')
                    else:
                        save_restore_path = os.path.join(args.output, 'restored_imgs', f'{basename}.{extension}')
                    
                    os.makedirs(os.path.dirname(save_restore_path), exist_ok=True)
                    imwrite(restored_img, save_restore_path)
                    processed_count += 1
                
                pbar.update(1)
            
            # گزارش پیشرفت
            if batch_idx % 5 == 0:  # هر 5 batch
                elapsed = time.time() - start_time
                fps = processed_count / elapsed if elapsed > 0 else 0
                pbar.set_postfix({
                    'موفق': processed_count, 
                    'FPS': f'{fps:.1f}',
                    'GPU': f'{torch.cuda.memory_allocated() / 1024**3:.1f}GB' if torch.cuda.is_available() else 'CPU'
                })
    
    total_time = time.time() - start_time
    avg_fps = processed_count / total_time if total_time > 0 else 0
    
    print(f'\nنتایج در پوشه [{args.output}] ذخیره شدند.')
    print(f'تعداد فریم‌های پردازش شده: {processed_count} از {len(img_list)}')
    print(f'زمان کل: {total_time:.1f} ثانیه')
    print(f'سرعت متوسط: {avg_fps:.1f} فریم در ثانیه')
    print(f'حداکثر مصرف GPU: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB' if torch.cuda.is_available() else '')


if __name__ == '__main__':
    main()
