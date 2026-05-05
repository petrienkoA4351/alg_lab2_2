import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import pickle
import traceback

DC_HUFFMAN_TABLE = {
    0: '00', 1: '010', 2: '011', 3: '100', 4: '101',
    5: '110', 6: '1110', 7: '11110', 8: '111110',
    9: '1111110', 10: '11111110', 11: '111111110'
}

DC_HUFFMAN_DECODE = {v: k for k, v in DC_HUFFMAN_TABLE.items()}

AC_HUFFMAN_TABLE = {
    (0, 0): '1010', (0, 1): '00', (0, 2): '01', (0, 3): '100',
    (0, 4): '1011', (0, 5): '11010', (0, 6): '1111000',
    (0, 7): '11111000', (0, 8): '1111110110', (0, 9): '1111111110000010',
    (0, 10): '1111111110000011', (1, 1): '1100', (1, 2): '11011',
    (1, 3): '1111001', (1, 4): '111110110', (1, 5): '11111110110',
    (1, 6): '1111111110000100', (2, 1): '11100', (2, 2): '11111001',
    (2, 3): '1111110111', (2, 4): '111111110100', (3, 1): '111010',
    (3, 2): '111110111', (3, 3): '111111110101', (4, 1): '111011',
    (4, 2): '1111111000', (5, 1): '1111010', (5, 2): '11111110111',
    (6, 1): '1111011', (6, 2): '111111110110', (7, 1): '11111010',
    (7, 2): '111111110111', (8, 1): '111111000', (8, 2): '111111111000011',
    (9, 1): '111111001', (9, 2): '111111111000100', (10, 1): '111111010',
    (10, 2): '111111111000101', (11, 1): '1111111001', (11, 2): '111111111000110',
    (12, 1): '1111111010', (12, 2): '111111111000111', (13, 1): '11111111000',
    (14, 1): '1111111110010', (15, 0): '11111111001', (15, 1): '1111111110011',
}

AC_HUFFMAN_DECODE = {v: k for k, v in AC_HUFFMAN_TABLE.items()}


def bilinear_downsample(channel, factor=2):
    """
    Даунсэмплинг канала с помощью билинейной интерполяции
    factor: коэффициент уменьшения (по умолчанию 2)
    """
    H, W = channel.shape
    new_H, new_W = H // factor, W // factor
    
    y = np.linspace(0, H - 1, new_H)
    x = np.linspace(0, W - 1, new_W)
    xv, yv = np.meshgrid(x, y)
    
    x0 = np.floor(xv).astype(int)
    x1 = np.minimum(x0 + 1, W - 1)
    y0 = np.floor(yv).astype(int)
    y1 = np.minimum(y0 + 1, H - 1)
    
    wa = (x1 - xv) * (y1 - yv)
    wb = (xv - x0) * (y1 - yv)
    wc = (x1 - xv) * (yv - y0)
    wd = (xv - x0) * (yv - y0)
    
    result = (wa * channel[y0, x0] + wb * channel[y0, x1] +
              wc * channel[y1, x0] + wd * channel[y1, x1])
    
    return result


def bilinear_upsample(channel, target_shape):
    """
    Апсэмплинг канала с помощью билинейной интерполяции
    target_shape: целевой размер (height, width)
    """
    h, w = channel.shape
    target_h, target_w = target_shape
    
    y = np.linspace(0, h - 1, target_h)
    x = np.linspace(0, w - 1, target_w)
    xv, yv = np.meshgrid(x, y)
    
    x0 = np.floor(xv).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y0 = np.floor(yv).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    
    wa = (x1 - xv) * (y1 - yv)
    wb = (xv - x0) * (y1 - yv)
    wc = (x1 - xv) * (yv - y0)
    wd = (xv - x0) * (yv - y0)
    
    result = (wa * channel[y0, x0] + wb * channel[y0, x1] +
              wc * channel[y1, x0] + wd * channel[y1, x1])
    
    return result


def rgb_to_ycbcr(img):
    img = img.astype(np.float32)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128
    return np.stack((Y, Cb, Cr), axis=-1)


def ycbcr_to_rgb(img):
    Y, Cb, Cr = img[..., 0], img[..., 1], img[..., 2]
    Cb -= 128
    Cr -= 128
    R = Y + 1.402 * Cr
    G = Y - 0.344136 * Cb - 0.714136 * Cr
    B = Y + 1.772 * Cb
    return np.clip(np.stack((R, G, B), axis=-1), 0, 255).astype(np.uint8)


def create_dct_matrix(N):
    D = np.zeros((N, N))
    for k in range(N):
        for n in range(N):
            alpha = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
            D[k, n] = alpha * np.cos((np.pi * (2 * n + 1) * k) / (2 * N))
    return D


def dct2(block, D):
    return D @ block @ D.T


def idct2(block, D):
    return D.T @ block @ D


def get_quant_matrix(N, quality, chrominance=False):
    Q_base_luminance = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    
    Q_chrom_base = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ])
    
    Q_base = Q_chrom_base if chrominance else Q_base_luminance
    
    if quality <= 0:
        quality = 1
    if quality >= 100:
        quality = 99
    
    if N != 8:
        return np.ones((N, N), dtype=np.float32)
    
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    
    Q_scaled = np.floor((Q_base * scale + 50) / 100)
    Q_scaled = np.clip(Q_scaled, 1, 255)
    return Q_scaled.astype(np.float32)


def quantize(block, Q):
    return np.round(block / Q).astype(np.int32)


def dequantize(block, Q):
    return (block * Q).astype(np.float32)


def zigzag(block):
    h, w = block.shape
    result = []
    for s in range(h + w - 1):
        for y in range(s + 1):
            x = s - y
            if y < h and x < w:
                if s % 2 == 0:
                    result.append(int(block[y, x]))
                else:
                    result.append(int(block[x, y]))
    return np.array(result)


def inverse_zigzag(arr, N):
    block = np.zeros((N, N), dtype=np.float32)
    i = 0
    for s in range(2 * N - 1):
        for y in range(s + 1):
            x = s - y
            if y < N and x < N:
                if s % 2 == 0:
                    block[y, x] = float(arr[i])
                else:
                    block[x, y] = float(arr[i])
                i += 1
    return block


def differential_encode_dc(dc_values):
    if len(dc_values) == 0:
        return np.array([])
    diffs = [int(dc_values[0])]
    for i in range(1, len(dc_values)):
        diffs.append(int(dc_values[i] - dc_values[i - 1]))
    return np.array(diffs)


def differential_decode_dc(diffs):
    if len(diffs) == 0:
        return np.array([])
    values = [int(diffs[0])]
    for i in range(1, len(diffs)):
        values.append(values[-1] + int(diffs[i]))
    return np.array(values)


def pad_image(img, block_size):
    h, w = img.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    return np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant')


def get_category(value):
    value = int(abs(value))
    if value == 0:
        return 0
    return int(np.floor(np.log2(value))) + 1


def encode_value_with_inversion(value, category):
    if category == 0:
        return ''
    abs_val = abs(int(value))
    bin_str = format(abs_val, f'0{category}b')
    if value >= 0:
        return bin_str
    inverted = ''.join('1' if b == '0' else '0' for b in bin_str)
    return inverted


def decode_inverted_bits(bit_str, category):
    if not bit_str or category == 0:
        return 0
    if bit_str[0] == '1':
        return int(bit_str, 2)
    inverted = ''.join('1' if b == '0' else '0' for b in bit_str)
    return -int(inverted, 2)


def variable_length_encode(zz_blocks):
    encoded_blocks = []
    
    for zz in zz_blocks:
        dc = int(zz[0])
        dc_cat = get_category(dc)
        if dc_cat not in DC_HUFFMAN_TABLE:
            dc_cat = min(dc_cat, 11)
        dc_huff = DC_HUFFMAN_TABLE[dc_cat]
        dc_bin = encode_value_with_inversion(dc, dc_cat)
        dc_code = (dc_huff, dc_bin)
        
        ac_codes = []
        zero_count = 0
        
        for val in zz[1:]:
            val = int(val)
            if val == 0:
                zero_count += 1
            else:
                while zero_count > 15:
                    ac_huff = AC_HUFFMAN_TABLE[(15, 0)]
                    ac_codes.append((ac_huff, ''))
                    zero_count -= 16
                
                cat = get_category(val)
                val_bin = encode_value_with_inversion(val, cat)
                key = (zero_count, cat)
                if key not in AC_HUFFMAN_TABLE:
                    key = (0, cat)
                ac_huff = AC_HUFFMAN_TABLE.get(key, '')
                ac_codes.append((ac_huff, val_bin))
                zero_count = 0
        
        if zero_count > 0:
            ac_codes.append((AC_HUFFMAN_TABLE[(0, 0)], ''))
        
        encoded_blocks.append((dc_code, ac_codes))
    
    return encoded_blocks


def variable_length_decode(encoded_blocks, block_size):
    zz_blocks = []
    
    for dc_code, ac_codes in encoded_blocks:
        dc_huff, dc_bin = dc_code
        if dc_huff not in DC_HUFFMAN_DECODE:
            dc_cat = 0
        else:
            dc_cat = DC_HUFFMAN_DECODE[dc_huff]
        dc_val = decode_inverted_bits(dc_bin, dc_cat)
        zz = [dc_val]
        
        ac = []
        for ac_huff, val_bin in ac_codes:
            if ac_huff not in AC_HUFFMAN_DECODE:
                continue
            run_size = AC_HUFFMAN_DECODE[ac_huff]
            run, size = run_size
            
            if (run, size) == (0, 0):
                ac.extend([0] * (block_size * block_size - 1 - len(ac)))
                break
            elif (run, size) == (15, 0):
                ac.extend([0] * 16)
            else:
                val = decode_inverted_bits(val_bin, size)
                ac.extend([0] * run)
                ac.append(val)
        
        while len(ac) < block_size * block_size - 1:
            ac.append(0)
        
        zz.extend(ac[:block_size * block_size - 1])
        zz_blocks.append(np.array(zz[:block_size * block_size]))
    
    return zz_blocks


def encode_channel(channel, Q, block_size, D):
    padded = pad_image(channel, block_size)
    h, w = padded.shape
    
    dc_values = []
    ac_blocks = []
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = padded[i:i + block_size, j:j + block_size]
            dct = dct2(block, D)
            q = quantize(dct, Q)
            zz = zigzag(q)
            dc_values.append(int(zz[0]))
            ac_blocks.append(zz[1:])
    
    dc_diffs = differential_encode_dc(dc_values)
    encoded_ac = variable_length_encode(ac_blocks)
    
    return dc_diffs, encoded_ac, padded.shape


def decode_channel(dc_diffs, ac_blocks, shape, Q, block_size, D):
    dc_values = differential_decode_dc(dc_diffs)
    ac_zz = variable_length_decode(ac_blocks, block_size)
    
    zz_blocks = []
    for dc, ac in zip(dc_values, ac_zz):
        zz = np.concatenate(([float(dc)], ac.astype(np.float32)))
        zz_blocks.append(zz[:block_size * block_size])
    
    blocks = []
    idx = 0
    for zz in zz_blocks:
        q = inverse_zigzag(zz, block_size)
        deq = dequantize(q, Q)
        idct = idct2(deq, D)
        blocks.append(idct)
    
    h, w = shape
    rec = np.zeros((h, w), dtype=np.float32)
    idx = 0
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            rec[i:i + block_size, j:j + block_size] = blocks[idx]
            idx += 1
    
    return rec[:shape[0], :shape[1]]


def compress(img, quality=50, block_size=8):
    D = create_dct_matrix(block_size)
    Q_Y = get_quant_matrix(block_size, quality, chrominance=False)
    Q_C = get_quant_matrix(block_size, quality, chrominance=True)
    
    img_np = np.array(img).astype(np.float32)
    ycbcr = rgb_to_ycbcr(img_np)
    Y, Cb, Cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]
    
    Cb_d = bilinear_downsample(Cb, factor=2)
    Cr_d = bilinear_downsample(Cr, factor=2)
    
    y_dc, y_ac, shape_Y = encode_channel(Y, Q_Y, block_size, D)
    cb_dc, cb_ac, shape_Cb = encode_channel(Cb_d, Q_C, block_size, D)
    cr_dc, cr_ac, shape_Cr = encode_channel(Cr_d, Q_C, block_size, D)
    
    y_dc_list = y_dc.tolist() if isinstance(y_dc, np.ndarray) else y_dc
    cb_dc_list = cb_dc.tolist() if isinstance(cb_dc, np.ndarray) else cb_dc
    cr_dc_list = cr_dc.tolist() if isinstance(cr_dc, np.ndarray) else cr_dc
    
    return {
        'dc': {'y': y_dc_list, 'cb': cb_dc_list, 'cr': cr_dc_list},
        'ac_y': y_ac,
        'ac_cb': cb_ac,
        'ac_cr': cr_ac,
        'size': img_np.shape[:2],
        'quality': quality,
        'shapes': {'y': shape_Y, 'cb': shape_Cb, 'cr': shape_Cr}
    }


def decompress(data, block_size=8):
    quality = data['quality']
    D = create_dct_matrix(block_size)
    Q_Y = get_quant_matrix(block_size, quality, chrominance=False)
    Q_C = get_quant_matrix(block_size, quality, chrominance=True)
    
    y_dc = np.array(data['dc']['y'])
    cb_dc = np.array(data['dc']['cb'])
    cr_dc = np.array(data['dc']['cr'])
    
    Y_rec = decode_channel(
        y_dc, data['ac_y'], data['shapes']['y'], 
        Q_Y, block_size, D
    ).clip(0, 255)
    
    Cb_rec = decode_channel(
        cb_dc, data['ac_cb'], data['shapes']['cb'],
        Q_C, block_size, D
    )
    Cr_rec = decode_channel(
        cr_dc, data['ac_cr'], data['shapes']['cr'],
        Q_C, block_size, D
    )
    
    Cb_rec = bilinear_upsample(Cb_rec, Y_rec.shape)
    Cr_rec = bilinear_upsample(Cr_rec, Y_rec.shape)
    
    final_ycbcr = np.stack((Y_rec, Cb_rec, Cr_rec), axis=-1)
    return ycbcr_to_rgb(final_ycbcr).astype(np.uint8)


def compress_rgb(img, quality=50):
    return compress(img, quality)


def compress_grayscale(img_gray, quality=50):
    img_rgb = Image.merge('RGB', (img_gray, img_gray, img_gray))
    return compress(img_rgb, quality)


def compress_binary(img_binary, quality=50):
    img_array = np.array(img_binary, dtype=np.uint8) * 255
    img_rgb = Image.fromarray(np.stack([img_array, img_array, img_array], axis=-1))
    return compress(img_rgb, quality)


def to_grayscale(img):
    return img.convert('L')


def to_bw_dither(img):
    try:
        return img.convert('1', dither=Image.Dither.FLOYSTEINBERG)
    except AttributeError:
        try:
            return img.convert('1', dither=Image.FLOYSTEINBERG)
        except AttributeError:
            gray = img.convert('L')
            return gray.point(lambda x: 0 if x < 128 else 255, '1')


def to_bw_no_dither(img):
    try:
        return img.convert('1', dither=Image.Dither.NONE)
    except AttributeError:
        try:
            return img.convert('1', dither=Image.NONE)
        except AttributeError:
            gray = img.convert('L')
            return gray.point(lambda x: 0 if x < 128 else 255, '1')


def save_compressed(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def get_file_size(filepath):
    return os.path.getsize(filepath)


def plot_individual_graphs(img_name, var_results, output_dir='results'):
    """Построение 4 отдельных графиков для каждого типа изображения"""
    
    qualities = sorted([q for q in var_results['RGB'].keys() if var_results['RGB'].get(q) is not None])
    
    plot_configs = [
        {'key': 'RGB', 'title': 'Цветное изображение (билинейная интерполяция)', 'color': 'blue', 'filename': f'{img_name}_RGB_size_vs_quality.png'},
        {'key': 'Grayscale', 'title': 'Оттенки серого (билинейная интерполяция)', 'color': 'green', 'filename': f'{img_name}_Grayscale_size_vs_quality.png'},
        {'key': 'BW_Dither', 'title': 'Черно-белое с дизерингом (билинейная интерполяция)', 'color': 'orange', 'filename': f'{img_name}_BW_Dither_size_vs_quality.png'},
        {'key': 'BW_No_Dither', 'title': 'Черно-белое без дизеринга (билинейная интерполяция)', 'color': 'red', 'filename': f'{img_name}_BW_No_Dither_size_vs_quality.png'}
    ]
    
    for config in plot_configs:
        key = config['key']
        if key not in var_results:
            continue
            
        plt.figure(figsize=(10, 6))
        
        sizes = [var_results[key].get(q) for q in qualities]
        valid_qualities = [q for q, s in zip(qualities, sizes) if s is not None]
        valid_sizes = [s for s in sizes if s is not None]
        
        if valid_qualities:
            plt.plot(valid_qualities, valid_sizes, 
                    color=config['color'], marker='o', linewidth=2, markersize=8)
            
            for q, s in zip(valid_qualities, valid_sizes):
                plt.annotate(f'{s}', (q, s), textcoords="offset points", 
                           xytext=(0, 10), ha='center', fontsize=9)
        
        plt.xlabel('Качество сжатия (Quality)', fontsize=12)
        plt.ylabel('Размер файла (байт)', fontsize=12)
        plt.title(f'Зависимость размера сжатого файла от качества\n{img_name} - {config["title"]}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{config["filename"]}', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  График сохранён: {output_dir}/{config['filename']}")


def display_all_restored_images(output_dir='results'):
    """Отображение всех восстановленных изображений"""
    if not os.path.exists(output_dir):
        print(f"Директория {output_dir} не существует")
        return
    
    qualities = [0, 20, 40, 60, 80, 100]
    var_types = ['RGB', 'Grayscale', 'BW_Dither', 'BW_No_Dither']
    var_names_display = {
        'RGB': 'Цветное',
        'Grayscale': 'Оттенки серого',
        'BW_Dither': 'Ч/Б с дизерингом',
        'BW_No_Dither': 'Ч/Б без дизеринга'
    }
    
    images_data = {}
    for f in os.listdir(output_dir):
        for var in var_types:
            if f.endswith('.png') and f'_{var}_restored_q' in f:
                for q in qualities:
                    if f'_restored_q{q}.png' in f:
                        img_name = f.split('_')[0]
                        if img_name not in images_data:
                            images_data[img_name] = {}
                        if var not in images_data[img_name]:
                            images_data[img_name][var] = {}
                        images_data[img_name][var][q] = f
    
    for img_name, var_data in images_data.items():
        originals = {}
        for var in var_types:
            if var == 'RGB':
                orig_path = f'{output_dir}/{img_name}_original.png'
            elif var == 'Grayscale':
                orig_path = f'{output_dir}/{img_name}_gray.png'
            elif var == 'BW_Dither':
                orig_path = f'{output_dir}/{img_name}_dither.png'
            else:
                orig_path = f'{output_dir}/{img_name}_no_dither.png'
            
            if os.path.exists(orig_path):
                originals[var] = Image.open(orig_path)
        
        fig, axes = plt.subplots(4, 7, figsize=(21, 12))
        
        for row, var in enumerate(var_types):
            if var in originals:
                axes[row, 0].imshow(originals[var], cmap='gray' if var != 'RGB' else None)
                axes[row, 0].set_title(f'{var_names_display[var]}\nОригинал', fontsize=10)
            axes[row, 0].axis('off')
            
            for col, q in enumerate(qualities, 1):
                if var in var_data and q in var_data[var]:
                    img_path = f'{output_dir}/{var_data[var][q]}'
                    restored = Image.open(img_path)
                    axes[row, col].imshow(restored, cmap='gray' if var != 'RGB' else None)
                    axes[row, col].set_title(f'Качество {q}', fontsize=10)
                axes[row, col].axis('off')
        
        plt.suptitle(f'Результаты сжатия с билинейной интерполяцией: {img_name}', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{img_name}_all_restored.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Сравнительная таблица сохранена: {output_dir}/{img_name}_all_restored.png")


def test_all_variations(original_img, base_name, output_dir, qualities):
    """Тестирование сжатия для всех вариаций изображения"""
    
    variations = {
        'RGB': original_img,
        'Grayscale': to_grayscale(original_img),
        'BW_Dither': to_bw_dither(original_img),
        'BW_No_Dither': to_bw_no_dither(original_img)
    }
    
    for var_name, img in variations.items():
        if var_name == 'RGB':
            img.save(f'{output_dir}/{base_name}_original.png')
        elif var_name == 'Grayscale':
            img.save(f'{output_dir}/{base_name}_gray.png')
        elif var_name == 'BW_Dither':
            img.save(f'{output_dir}/{base_name}_dither.png')
        elif var_name == 'BW_No_Dither':
            img.save(f'{output_dir}/{base_name}_no_dither.png')
    
    results = {}
    
    for var_name, img in variations.items():
        print(f"\n  Тестирование {var_name}...")
        var_results = {}
        
        for quality in qualities:
            try:
                if var_name == 'RGB':
                    compressed_data = compress_rgb(img, quality=quality)
                elif var_name == 'Grayscale':
                    compressed_data = compress_grayscale(img, quality=quality)
                else:
                    compressed_data = compress_binary(img, quality=quality)
                
                compressed_path = f'{output_dir}/{base_name}_{var_name}_q{quality}_compressed.bin'
                save_compressed(compressed_data, compressed_path)
                
                restored = decompress(compressed_data)
                restored_img = Image.fromarray(restored)
                restored_path = f'{output_dir}/{base_name}_{var_name}_restored_q{quality}.png'
                restored_img.save(restored_path)
                
                file_size = get_file_size(compressed_path)
                var_results[quality] = file_size
                print(f"    Quality {quality:3d}: {file_size:8d} байт")
                
            except Exception as e:
                print(f"    Quality {quality:3d}: Ошибка - {e}")
                var_results[quality] = None
        
        results[var_name] = var_results
    
    return results


def create_test_image(path, size=(512, 512)):
    """Создание тестового изображения"""
    print(f"Создание тестового изображения: {path}")
    x = np.linspace(0, 1, size[0])
    y = np.linspace(0, 1, size[1])
    X, Y_grid = np.meshgrid(x, y)
    
    R = (np.sin(X * np.pi * 4) * 0.5 + 0.5) * 255
    G = (np.cos(Y_grid * np.pi * 4) * 0.5 + 0.5) * 255
    B = (np.sin((X + Y_grid) * np.pi * 4) * 0.5 + 0.5) * 255
    
    test_array = np.stack([R, G, B], axis=-1).astype(np.uint8)
    test_img = Image.fromarray(test_array)
    test_img.save(path)
    print(f"Тестовое изображение создано: {path}")
    return path


def run_full_test(qualities=[0, 20, 40, 60, 80, 100]):
    """Полное тестирование для всех изображений"""
    
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    test_images = []
    
    if os.path.exists('Lena.png'):
      test_images.append(('Lena.png', os.path.splitext('Lena.png')[0]))
    
    if not test_images:
        test_images.append((create_test_image('test_lenna.png', (512, 512)), 'test_lenna'))
    
    if os.path.exists('Image.png'):
      test_images.append(('Image.png', os.path.splitext('Image.png')[0]))
    
    if len(test_images) == 1:
        test_images.append((create_test_image('test_image.png', (512, 512)), 'test_image'))
    
    all_results = {}
    
    for img_path, img_name in test_images:
        print(f"\n{'='*70}")
        print(f"Обработка изображения: {img_name}")
        print(f"{'='*70}")
        
        try:
            original = Image.open(img_path).convert('RGB')
            print(f"Размер изображения: {original.size}")
            
            results = test_all_variations(original, img_name, output_dir, qualities)
            all_results[img_name] = results
            
        except Exception as e:
            print(f"Ошибка при обработке {img_path}: {e}")
            traceback.print_exc()
    
    if all_results:
        print("\n" + "="*70)
        print("ПОСТРОЕНИЕ ГРАФИКОВ (билинейная интерполяция)")
        print("="*70)
        
        for img_name, var_results in all_results.items():
            print(f"\nПостроение графиков для {img_name}:")
            plot_individual_graphs(img_name, var_results, output_dir)
        
        display_all_restored_images(output_dir)
        
        print("\n" + "="*70)
        print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ (размеры в байтах)")
        print("="*70)
        
        for img_name, var_results in all_results.items():
            print(f"\n{img_name}:")
            print(f"{'Тип':<18}", end='')
            for q in qualities:
                print(f"Q{q:<6}", end='')
            print()
            print("-" * (18 + 7 * len(qualities)))
            
            for var_name, quality_results in var_results.items():
                display_name = {
                    'RGB': 'Цветное',
                    'Grayscale': 'Оттенки серого',
                    'BW_Dither': 'Ч/Б с дизерингом',
                    'BW_No_Dither': 'Ч/Б без дизеринга'
                }.get(var_name, var_name)
                
                print(f"{display_name:<18}", end='')
                for q in qualities:
                    size = quality_results.get(q)
                    if size is not None:
                        print(f"{size:<7}", end='')
                    else:
                        print(f"{'Ошибка':<7}", end='')
                print()
    
    return all_results


if __name__ == "__main__":
    print("="*70)
    print("JPEG-подобный компрессор изображений")
    print("Сжатие для: RGB, Grayscale, Ч/Б с дизерингом, Ч/Б без дизеринга")
    print("="*70)
    
    qualities = [0, 20, 40, 60, 80, 100]
    
    results = run_full_test(qualities)
    
    print("\n" + "="*70)
    print("Все результаты сохранены в директории 'results/'")
    print("="*70)
    print("\nСохранённые файлы:")
    print("  - Оригинальные изображения (*_original.png, *_gray.png, *_dither.png, *_no_dither.png)")
    print("  - Восстановленные изображения (*_restored_q*.png)")
    print("  - Графики для каждого типа изображения (с билинейной интерполяцией):")
    print("    * *_RGB_size_vs_quality.png")
    print("    * *_Grayscale_size_vs_quality.png")
    print("    * *_BW_Dither_size_vs_quality.png")
    print("    * *_BW_No_Dither_size_vs_quality.png")
    print("  - Сравнительные таблицы (*_all_restored.png)")
