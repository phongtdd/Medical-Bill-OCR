import math
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np


#Vietnamese full alphabets
#-------------------------------------------------------------------------------------------
lowercase = "aăâbcdđeêghijklmnoôơpqrstuưvwxyz" \
            "áàảãạằắẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ0123456789"
uppercase = lowercase.upper()
special_chars = "/!@#$%^&*()_+:,.-;?{}[]|~` "
full_alphabet = lowercase + uppercase + special_chars

char_to_idx = {char: idx + 1 for idx, char in enumerate(full_alphabet)} 
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
idx_to_char[0] = ''


#Define transform of images
#-------------------------------------------------------------------------------------------
transform = A.Compose([
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])


#Beam search decode
#-------------------------------------------------------------------------------------------
def beam_search_decode(probs, beam_width=5, blank=0):
    seq_len, batch_size, nclass = probs.size()
    decoded_batch = []

    for batch_idx in range(batch_size):
        beam = [(tuple(), 0.0)]

        for t in range(seq_len):
            new_beam = defaultdict(lambda: -math.inf)
            time_step_log_prob = probs[t, batch_idx].cpu().numpy()

            for seq, score in beam:
                for c in range(nclass):
                    p = time_step_log_prob[c]
                    if len(seq) > 0 and c == seq[-1]:
                        new_seq = seq
                    else:
                        new_seq = seq + (c,) if c != blank else seq
                    new_score = score + p
                    if new_score > new_beam[new_seq]:
                        new_beam[new_seq] = new_score

            beam = sorted(new_beam.items(), key=lambda x: x[1], reverse=True)[:beam_width]

        best_seq, best_score = beam[0]

        # Filter blanks and repeated characters here
        decoded = []
        prev = None
        for idx in best_seq:
            if idx != blank and idx != prev:
                # Defensive check in case idx_to_char missing key
                char = idx_to_char.get(idx, '')
                if char != '':
                    decoded.append(char)
            prev = idx

        decoded_str = "".join(decoded)
        decoded_batch.append(decoded_str)

    return decoded_batch


#Clean decoded text
#-------------------------------------------------------------------------------------------
def clean_decoded_text(text, blank_char=''):
    cleaned = []
    prev_char = None
    for ch in text:
        if ch != blank_char and ch != prev_char:
            cleaned.append(ch)
        prev_char = ch
    return ''.join(cleaned)


#Process images
#-------------------------------------------------------------------------------------------
def process_image(image_path):
    image = Image.open(image_path).convert('L')
    image = custom_resize(image, min_width=256, target_height=32)
    transform_image = transform(image=np.array(image))
    tensor_image = transform_image['image']  
    return tensor_image
