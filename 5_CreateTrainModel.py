# Script for training

print('Imports')

import numpy as np
import pandas as pd

from collections import Counter

import matplotlib.pyplot as plt

import random
import tqdm
import os
import math

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import editdistance

import cv2
from PIL import Image
import Augmentor


print('Random')

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    
    os.environ['PYTHONHASHSEED'] = str(seed)


seed = 1234
set_seed(seed)

print('Data')

print('Labels')

work_dir_path = './'

# Data paths
# train_images_path = f'{work_dir_path}data/train_val_split_csv/train/'
# train_labels_path = f'{work_dir_path}data/train_val_split_csv/train.csv'
train_images_path = f'{work_dir_path}data/train_val_split_csv/train10/'
train_labels_path = f'{work_dir_path}data/train_val_split_csv/train10.csv'

val_images_path = f'{work_dir_path}data/train_val_split_csv/val/'
val_labels_path = f'{work_dir_path}data/train_val_split_csv/val.csv'

test_images_path = f'{work_dir_path}data/train_val_split_csv/test/'
test_labels_path = f'{work_dir_path}data/train_val_split_csv/test.csv'

# List of characters that are used for recognition
alphabet = [' ', '!', '"', '%', '(', ')', ',', '-', '+', '=', '.', '/', "'", "№", 
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', '[', 
            ']', '«', '»', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 
            'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 
            'Щ', 'Э', 'Ю', 'Я', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 
            'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 
            'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'ё']


def process_data(image_dir, labels_file, allowed_chars=None):
    
    # Load labels CSV as dataframe
    df = pd.read_csv(labels_file)
    
    # Filter rows by allowed chars if provided
    if not allowed_chars is None:      
        initial_rows = len(df)
        df = df[df['label'].apply(lambda x: set(x).issubset(allowed_chars))]
        removed_rows = initial_rows - len(df)
        print(f'Removed {removed_rows} rows')
    
    # Map image filenames to labels
    img2label = dict(zip(image_dir + df['image'], df['label']))

    # Get unique sorted character set
    chars = sorted(set(''.join(df['label'])))
    
    return img2label, chars


img2label_train, chars_train = process_data(train_images_path, train_labels_path, allowed_chars=alphabet)
print(f'Total train set size is {len(img2label_train)}')
img_names_train, labels_train = list(img2label_train.keys()), list(img2label_train.values())

img2label_val, chars_val = process_data(val_images_path, val_labels_path, allowed_chars=alphabet)
print(f'Total val set size is {len(img2label_train)}')
img_names_val, labels_val = list(img2label_val.keys()), list(img2label_val.values())

img2label_test, chars_test = process_data(test_images_path, test_labels_path)
print(f'Total test set size is {len(img2label_train)}')
img_names_test, labels_test = list(img2label_test.keys()), list(img2label_test.values())


print(set(chars_train).difference(set(alphabet)))
print(set(alphabet).difference(set(chars_train)))
print()

print(set(chars_val).difference(set(alphabet)))
print(set(alphabet).difference(set(chars_val)))
print()

print(set(chars_test).difference(set(alphabet)))
print(set(alphabet).difference(set(chars_test)))


# Add special tokens
alphabet = ['_PAD', '_SOS'] + alphabet + ['_EOS']

# Expected {'_EOS', '_PAD', '_SOS'}
set(alphabet).difference(set(chars_train))

# Create char and indexes mapping dicts
char2idx = {char: idx for idx, char in enumerate(alphabet)}
idx2char = {idx: char for idx, char in enumerate(alphabet)}


print('Images')

def process_image(img, width=256, height=64):
    """
    Resize image to target dimensions while preserving aspect ratio

    Params:
        img : np.array
            Input image
    
    Returns:
        img : np.array
            Output image with target dimensions
    """
    
    # Get input dimensions
    ih, iw, _ = img.shape
    
    # Define target dimensions
    target_w = width
    target_h = height
    
    # Calculate new width to preserve aspect ratio
    new_w = target_w
    new_h = int(ih * (new_w / iw))
    
    # Resize by width to preserve aspect ratio
    img = cv2.resize(img, (new_w, new_h))
    
    # Ensure type is float32 for future operations
    # img = img.astype('float32')

    # Pad height with white pixels to reach target height 
    if new_h < target_h:
        pad = np.full((target_h - new_h, new_w, 3), 255)
        img = np.concatenate((img, pad), axis=0)
        
    # Or crop height if exceeds target height
    elif new_h > target_h:  
        img = cv2.resize(img, (target_w, target_h))
    
    return img


def load_and_preprocess_images(image_paths):
    """
    Load images from disk and preprocess
    
    Args:
        image_paths (list[str]): List of filepaths to images
        
    Returns:
        images (list[numpy.array]): List of preprocessed images
    """
    
    images = []
    
    # Iterate through image paths
    for img_path in tqdm.tqdm(image_paths):
            
        try:
            # Load image as numpy array
            img = np.asarray(Image.open(img_path).convert('RGB'))
            
            # Preprocess image
            img = process_image(img)
            
            # Append and convert to save memory
            images.append(img.astype('uint8'))
            
        except Exception as e:
            # Log error images 
            print(f"Error processing {img_path}")
            
            # Retry preprocessing
            img = process_image(img)
    
    return images


# Load train images
X_train = load_and_preprocess_images(img_names_train)

# Load val images
X_val = load_and_preprocess_images(img_names_val)

# Load test images
X_test = load_and_preprocess_images(img_names_test)

# Create targets for each set
y_train = labels_train
y_val = labels_val
y_test = labels_test


print(('Torch Dataset')

# Create an Augmentor pipeline object to manage image augmentations
p = Augmentor.Pipeline() 

# Add a shear transformation with up to 2 pixels left/right shear
# Apply this shear 70% of the time
p.shear(max_shear_left=2, max_shear_right=2, probability=0.7) 

# Add a random distortion transformation
# Divide image into 3x3 grid and distort each section randomly
# Apply magnitude of up to 11 pixels displacement 
# Apply this distortion 100% of the time
p.random_distortion(probability=1.0, grid_width=3, grid_height=3, magnitude=11)


channels = 1

train_transforms = transforms.Compose([
            # Convert tensor to PIL image 
            transforms.ToPILImage(),
    
            # Convert images to 1 channel grayscale
            transforms.Grayscale(channels),
    
            # Apply Augmentor pipeline distortions
            p.torch_transform(),  # random distortion and shear
    
            # Randomly change contrast and saturation
            transforms.ColorJitter(contrast=(0.5,1),saturation=(0.5,1)),
    
            # Randomly rotate between -9 and +9 degrees
            transforms.RandomRotation(degrees=(-9, 9)),
    
            # Random affine transformation with scaling and shear
            transforms.RandomAffine(10, None, [0.6, 1], 3, fill=255), 
    
            # Random Gaussian blurring
            #transforms.transforms.GaussianBlur(3, sigma=(0.1, 1.9)),
    
            # Convert PIL image to tensor
            transforms.ToTensor()
        ])

# Test time augmentation only converts PIL->tensor
test_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(channels),
            transforms.ToTensor()
        ])


def text_to_labels(text, char2idx):
    """
    Convert text to label vector based on char2idx mapping
    
    Args:
        text (str): Input text 
        char2idx (dict): Mapping of characters to indexes
        
    Returns: 
        labels (List[int]): List of character indexes  
    """
    
    # Initialize labels vector
    labels = [] 
    
    # Add start-of-sentence token at start
    labels.append(char2idx['_SOS'])
    
    # Map characters to indexes
    for char in text:
        # Skip unknown characters
        if char in char2idx:
            labels.append(char2idx[char])
            
    # Add end-of-sentence token at end
    labels.append(char2idx['_EOS'])
    
    return labels


def labels_to_text(labels, idx2char):
    """Convert label vector back to text using idx2char mapping
    
    Args:
        labels (List[int]): Vector of character indexes 
        idx2char (dict): Mapping indexes to characters
        
    Returns: 
        text (str): Reconstructed text
    """
    
    # Initialize empty string
    text = ''
    
    for i in range(len(labels)):
            
        # Get character for each index 
        char = idx2char[labels[i]]
        
        
        # Append character to text
        text += char

    # Replace non alphabet tokens
    text = text.replace('_EOS', '').replace('_PAD', '').replace('_SOS', '')
    
    return text


class TextDataset():
    """Dataset for images and text labels"""
    
    def __init__(self, images, text_labels, transforms, char2idx, idx2char):
        """
        Args:
            images : list[numpy.ndarray]
                List of images (array format)
            text_labels : list[str]
                List of text labels  
            transforms : torchvision.transforms
                Augmentations
            char2idx : dict
                Mapping chars to indexes
            idx2char : dict
                Mapping indexes to chars
        """
        self.images = images
        self.text_labels = text_labels
        self.transforms = transforms
        
        # Text encoding mappings        
        self.char2idx = char2idx  
        self.idx2char = idx2char
        
    def __getitem__(self, index):
        """
        Get image-label pair by index

        Args:
            index : int
                Index of data
        Returns:
            Dict
                Sample of data:
                  'image': Image (Tensor format)  
                  'label': Encoded label (LongTensor)
        """
        
        def transform_image(img, transforms):
            """
            Apply transforms to single image

            Args:
                img : numpy.ndarray
                    Image to transform
                transforms : torchvision.transforms
                    Augmentations
            Returns:
                img : numpy.ndarray
                    Transformed imgae
            """
            img = transforms(img)
            img = img / img.max()
            img = img ** (random.random() * 0.7 + 0.6)

            return img
        
        # Load image and apply transforms
        image = transform_image(self.images[index], self.transforms)
        
        # Convert text to label vector
        label = text_to_labels(self.text_labels[index], self.char2idx)
                
        return {
            'image': image, 
            'label': torch.LongTensor(label)
        }
    
    def __len__(self):
        """Return dataset length"""
        return len(self.text_labels)
    
    def get_statistics(self): 
        """Print dataset statistics"""
        num_samples = len(self)  
        longest_text = max(len(text) for text in self.text_labels)
        char_counts = self.get_char_counts()

        print(f'Dataset size = {num_samples}')
        print(f'Longest text = {longest_text} chars') 
        print(f'Most common char = {list(char_counts.keys())[0]} ({list(char_counts.values())[0]})') 
        print(f'Least common char = {list(char_counts.keys())[-1]} ({list(char_counts.values())[-1]})')
        
    def get_char_counts(self):
        """Get descending sorted char counts"""
        full_text = ''.join(self.text_labels)  
        return dict(sorted(Counter(full_text).items(), key=lambda x: x[1], reverse=True))


# Creare 3 Datasets
train_dataset = TextDataset(X_train, y_train, train_transforms, char2idx, idx2char)
val_dataset = TextDataset(X_val, y_val, test_transforms, char2idx, idx2char)
test_dataset = TextDataset(X_test, y_test, test_transforms, char2idx, idx2char)


# Print Datasets statistics
print('Train DS statistics:')
train_dataset.get_statistics()
print()

print('Val DS statistics:')
val_dataset.get_statistics()
print()

print('Test DS statistics:')
test_dataset.get_statistics()


def TextCollate(batch):
    """
    Collates batches of images and text labels, padding the labels to a fixed length.
    This allows batches with variable length labels to be used for model training.
    
    Args:
        batch : List[Dict] 
            A batch of examples, each containing an 'image' and 'label' field.
    
    Returns:  
        Dict
            A batch compatible for model training:
              'images': Tensor of stacked images  
              'labels': LongTensor of padded labels, 
                        padded to the longest label length in the batch
    """
    # Extract images from the batch
    images = [b['image'] for b in batch]
    
    # Stack them into a tensor
    images = torch.stack(images)
    
    # Extract text labels from the batch  
    labels = [b['label'] for b in batch]
    
    # Find the maximum label length in the batch
    max_batch_lenght = max(len(l) for l in labels)
    
    # Create an empty padded tensor for the labels
    padded_labels = torch.LongTensor(max_batch_lenght, len(batch))
    
    # Initialize it to all zeros
    padded_labels.zero_()
    
    # Copy the labels to the corresponding rows of the padded tensor
    for i in range(len(batch)):
        txt = labels[i]
        padded_labels[:txt.size(0), i] = txt

    return {
        'images': images,
        'labels': padded_labels
    }


# Batch size for all DataLoaders
batch_size = 64

# Create 3 DataLoaders using 3 Datasets
train_loader = DataLoader(train_dataset, shuffle=True, 
                          batch_size=batch_size, pin_memory=True, 
                          drop_last=True, collate_fn=TextCollate)

val_loader = DataLoader(val_dataset, shuffle=False, 
                          batch_size=batch_size, pin_memory=True, 
                          drop_last=True, collate_fn=TextCollate)

test_loader = DataLoader(test_dataset, shuffle=False, 
                          batch_size=batch_size, pin_memory=True, 
                          drop_last=True, collate_fn=TextCollate)
             

print('Model')

print('Create Model')

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model : int
                Dimensionality of the input features.
            dropout : float
                Dropout rate.
            max_len : int
                Maximum length of the input sequence.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        # Create a positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        # Create a vector of positions for each position in the sequence
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Calculate the div_term for sine and cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply sine to even indexes in the array
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indexes in the array
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add a batch dimension and transpose the dimensions
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Register pe as a buffer so it's not considered a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Applies positional encoding to the input tensor.

        Args:
            x: Tensor [L, B, DIM]
                Input tensor of shape 
                L - seq length, B - batch size, DIM - d_model

        Returns:
            Tensor [L, B, DIM]
                Output tensor with positional encoding added and dropout applied
                L - seq length, B - batch size, DIM - d_model
        """
        # Add positional encoding to the input tensor
        x = x + self.scale * self.pe[:x.size(0), :]
        # Apply dropout
        return self.dropout(x)


def count_parameters(model):
    """
    Counts the number of trainable parameters in a model.
    
    Args:
        model (nn.Module): A PyTorch neural network model.
        
    Returns:
        param_count : int
            The total number of trainable parameters in the model.
    """
    
    # Initialize parameter count
    param_count = 0
    
    # Iterate through all model parameters
    for param in model.parameters():
        
        # Add up only trainable parameters
        if param.requires_grad:
            param_count += param.numel()
            
    return param_count


def log_model_config(model):
    """
    Logs configuration details of a model.
    
    Args:
        model: The model to log details of.
    """
    
    # Print number of transformer layers
    print('Number of transformer encoder layers:', model.enc_layers)
    
    # Print number of attention heads
    print('Number of attention heads:', model.transformer.nhead)
    
    # Print decoder embedding dimensionality
    print('Decoder embedding dimensionality:', model.embedding.embedding_dim)
    
    # Print number of classes in output layer
    print('Number of classes in output layer:', model.embedding.num_embeddings)
    
    # Print dropout probability
    print('Dropout probability:', model.encoder_pe.dropout.p)
    
    # Print number of trainable parameters    
    num_params = count_parameters(model)
    print('Number of trainable parameters:', f'{num_params:,}')

    # Print full model info
    print(model)


class TransformerModel(nn.Module):
    def __init__(self, out_token_size, hidden_size, enc_layers=1, dec_layers=1, nhead=1, dropout=0.1):
        super(TransformerModel, self).__init__()

        # Num of transformer layers
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
                
        # Initialize convolutional layers, batch normalization, and max pooling layers
        self.conv_layers, self.batch_norm_layers, self.pool_layers = self._initialize_layers(hidden_size)

        # Initialize activation function
        self.activation = nn.LeakyReLU()

        # Initialize positional encodings for src and tgt
        self.encoder_pe = PositionalEncoding(hidden_size, dropout)
        self.decoder_pe = PositionalEncoding(hidden_size, dropout)

        # Initialize character embedding for labels
        self.embedding = nn.Embedding(out_token_size, hidden_size)
        
        # Initialize transformer
        self.transformer = nn.Transformer(
            d_model=hidden_size, 
            nhead=nhead, 
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers, 
            dim_feedforward=hidden_size * 4, 
            dropout=dropout
        )

        # Initialize fully connected output layer
        self.fc_out = nn.Linear(hidden_size, out_token_size)
        
        # Initialize masks
        self.tgt_mask = None

        # Log configuration
        log_model_config(self)

    def _initialize_layers(self, hidden_size): 
        '''
        Initialize CNN layers

        Args:
            hidden_size : int
        
        Returns:
            conv_layers : nn.ModuleList, batch_norm_layers : nn.ModuleList, pool_layers : nn.ModuleList
        '''
        # Define convolutional layers
        conv_layers = nn.ModuleList([
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(512, hidden_size, kernel_size=(2, 1), stride=(1, 1))
        ])

        # Define batch normalization layers
        batch_norm_layers = nn.ModuleList([
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(256),
            nn.BatchNorm2d(256),
            nn.BatchNorm2d(512),
            nn.BatchNorm2d(512),
            nn.BatchNorm2d(hidden_size)
        ])

        # Define max pooling layers
        pool_layers = nn.ModuleList([
            None,  # No max pooling after the zero convolutional layer
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            None,  # No max pooling after the second convolutional layer
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            None,  # No max pooling after the fourth convolutional layer
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),
            None   # No max pooling after the sixth convolutional layer
        ])

        return conv_layers, batch_norm_layers, pool_layers

    def make_len_mask(self, x):
        '''
        Creates a pad (length) mask for the sequence

        Args:
            x : Tensor [L, B]
                L - max label length (tgt_len), B - batch size
        
        Returns:
            output : Tensor [B, L]
                B - batch size, L - max label length (tgt_len)
        '''
        return (x == 0).transpose(0, 1)
    
    def _get_features(self, x):       
        '''
        Extracts features from the input using convolutional layers
        
        Args:
            src : Tensor [B, C, H, W]
                B - batch size, C - num of channels, H - height, W - width
        
        Returns:
            x : Tensor : [W, B, C*H]
                B - batch size, C - num of channels, H - height, W - width
        '''
        for conv, bn, pool in zip(self.conv_layers, self.batch_norm_layers, self.pool_layers):
            x = self.activation(bn(conv(x)))
            if pool is not None:
                x = pool(x)
        '''
        # [B, C, H, W] -> [B, W, C, H] -> [B, W, C*H] -> [W, B, C*H]
        # [16, 512, 1, 65] -> [16, 65, 1, 512] -> [16, 65, 512] -> [65, 16, 512] (seq_len, batch_size, hidden_size)
        x = x.permute(0, 3, 1, 2).flatten(2).permute(1, 0, 2)
        '''
        
        # [B, C, H, W] -> [B, W, C*H] -> [W, B, C*H]
        # [16, 512, 1, 65] -> [16, 65, 512] -> [65, 16, 512] (seq_len, batch_size, hidden_size)
        x = x.flatten(2).permute(2, 0, 1)
        return x

    def forward(self, src, tgt):
        '''
        Model's forward pass
        
        Args:
            src : Tensor [B, C, H, W]
                B - batch size, C - num of channels, H - height, W - width
            tgt : Tensor [L, B]
                L - max label length (tgt_len), B - batch size
        
        Returns:
            output : Tensor [L, B, O]
                L - max label length (tgt_len), B - batch, O - output token size
        '''
        # 1 Images (src)

        # Extract features from images using Conv Layers
        # [batch_size, channels, height, width] -> [seq_len, batch_size, hidden_size]
        features = self._get_features(src)

        # Apply positinal encoding to extracted features (before encoder's input)
        pos_encoded_features = self.encoder_pe(features)

        # 2 Labels (tgt)

        # Generates a squeare matrix where the each row allows one character more to be seen
        # The masked positions are filled with float('-inf')
        # Unmasked positions are filled with float(0.0).
        if self.tgt_mask is None or self.tgt_mask.size(0) != len(tgt):
            self.tgt_mask = nn.Transformer.generate_square_subsequent_mask(len(tgt)).to(tgt.device) 

        # Create tgt_key_padding_mask (shape [batch_size, tgt_len])
        tgt_pad_mask = self.make_len_mask(tgt)

        # Apply word embedding to labels (characters indexes)
        # [seq_len, batch_size] -> [seq_len, batch_size, hidden_size]
        embedded_tgt = self.embedding(tgt)

        # Apply positinal encoding to characters embeddings 
        pos_encoded_embedded_tgt = self.decoder_pe(embedded_tgt)

        # Transformer pass
        output = self.transformer(
            pos_encoded_features, 
            pos_encoded_embedded_tgt, 
            tgt_mask=self.tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )

        # Final fully connected layer that converts transformer output to characters indexes
        output = self.fc_out(output)

        return output

    
    def predict(self, batch):
        '''
        Method to predict sequences of token indexes based on input data (images) batch.

        Args:
            batch : Tensor [B, C, H, W]
                B - batch, C - channel, H - height, W - width

        Returns:
            result : List [B, L]
                Predicted sequences of token indexes
                B - batch size, L - max label length (tgt_len)
        '''
        result = []
        for i, item in enumerate(batch):
            # Get features from the image
            features = self._get_features(item.unsqueeze(0))
            # Apply position encoding to the features
            pos_encoded_features = self.encoder_pe(features)
            # Apply transformer encoder to the encoded features
            memory = self.transformer.encoder(pos_encoded_features)
            
            # Initialize the list of indexes with the index of the start token
            out_indexes = [alphabet.index('_SOS'), ]

            # Generate the sequence of tokens
            for _ in range(100):
                # Convert the list of indexes to a tensor and add a batch dimension
                tgt_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(item.device)
                # Apply transformer decoder to the tensor and memory
                output = self.transformer.decoder(self.decoder_pe(self.embedding(tgt_tensor)), memory)
                # Apply the fully connected layer to the decoder output
                fc_output = self.fc_out(output)
                # Select the index of the token with the highest probability
                out_token = fc_output.argmax(2)[-1].item()
                # Add the token index to the list
                out_indexes.append(out_token)
                # If the end token is reached, break the loop
                if out_token == alphabet.index('_EOS'):
                    # print(i)
                    break

            # Add the predicted sequence of tokens to the result
            result.append(out_indexes)

        return result


# Model parameters
hidden_size = 512
enc_layers = 2
dec_layers = 2
n_heads = 4
dropout = 0.2

set_seed(1234)

# Initialize model
model = TransformerModel(
    len(alphabet), 
    hidden_size=hidden_size, 
    enc_layers=enc_layers, 
    dec_layers=dec_layers,                
    nhead=n_heads, 
    dropout=dropout
)


print('Train Model')

def character_error_rate(sequence1, sequence2):
    """
    Calculate the Character Error Rate (CER) between two sequences of characters.

    Args:
        sequence1 : str
            The first sequence of characters.
        sequence2 : str
            The second sequence of characters.

    Returns:
        cer : float
            The Character Error Rate (CER) between the two sequences.
    """
    # Create a set of unique characters from both sequences
    vocabulary = set(sequence1 + sequence2)
    
    # Create a dictionary that maps each unique character to a unique integer
    char_to_index = dict(zip(vocabulary, range(len(vocabulary))))
    
    # Convert the sequences to lists of characters using the mapping
    char_sequence1 = [chr(char_to_index[char]) for char in sequence1]
    char_sequence2 = [chr(char_to_index[char]) for char in sequence2]
    
    # Calculate the distance between the two sequences
    distance = editdistance.eval(''.join(char_sequence1), ''.join(char_sequence2))
    
    # Calculate the Character Error Rate as the ratio of the distance to the maximum length of the two sequences
    cer = distance / max(len(sequence1), len(sequence2))
    
    return cer


# Total number of epochs
n_epochs = 200

# Early stopping patience
patience = 15

# Base LR
learning_rate = 1e-4

# CP frequency
checkpoint_n_epochs = 5

# Models path
models_path = f'{work_dir_path}models/'

# CP path
checkpoint_path = f'{models_path}checkpoints/'

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=char2idx['_PAD'])

# LR scheduler patience
lr_reduce_patience = 7

# Create LR scheduler (ReduceLROnPlateau)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=lr_reduce_patience)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

# ES
best_val_loss = 1e+10
best_model_state = None

# Metrics log
train_losses = []
val_losses = []
val_cer_scores = []
val_wer_scores = []

# TensorBoard writer
writer = SummaryWriter()

for epoch in tqdm.tqdm(range(n_epochs)):
    # Learning
    model.train()

    # Epoch metrics
    train_loss = []
    val_loss = []
    val_cer_score = []
    val_wer_score = []

    for i, batch in enumerate(test_loader):
        inputs = batch['images'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs, labels[:-1, :])
        loss = criterion(outputs.view(-1, outputs.shape[-1]), torch.reshape(labels[1:, :], (-1,)))
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.item())

    # Validation
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inputs = batch['images'].to(device)
            labels = batch['labels'].to(device)

            # Use forward method to calculate loss
            outputs = model(inputs, labels[:-1, :])
            loss = criterion(outputs.view(-1, outputs.shape[-1]), torch.reshape(labels[1:, :], (-1,)))
            
            val_loss.append(loss.item())


    # Get all metrics
    mean_train_loss = np.mean(train_loss)
    mean_val_loss = np.mean(val_loss)
    # mean_val_cer_score = np.mean(val_cer_score)
    # mean_val_wer_score = np.mean(val_wer_score)

    train_losses.append(mean_train_loss)
    val_losses.append(mean_val_loss) 
    # val_cer_scores.append(mean_val_cer_score)
    # val_wer_scores.append(mean_val_wer_score)
    
    writer.add_scalar('Loss/train', mean_train_loss, epoch)
    writer.add_scalar('Loss/val', mean_val_loss, epoch)
    # writer.add_scalar('CER/val', mean_val_cer_score, epoch)
    # writer.add_scalar('WER/val', mean_val_wer_score, epoch)
    writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

    # Print epoch result
    print()
    print(f'Epoch {epoch+1}/{n_epochs}: \
    Train loss: {mean_train_loss}; \
    Val loss: {mean_val_loss}; \
    LR: {optimizer.param_groups[0]['lr']}')

    # Save CP
    if (epoch + 1) % checkpoint_n_epochs == 0:
        torch.save(model.state_dict(), f'{checkpoint_path}model_cp(epoch={epoch + 1}_TL={mean_train_loss}_VL={mean_val_loss})_BS={batch_size}_LR={optimizer.param_groups[0]['lr']}.pth')

    # LR scheduler step
    scheduler.step(mean_val_loss)
    
    # Check ES
    if mean_val_loss <= best_val_loss:
        best_val_loss = mean_val_loss
        patience_counter = 0
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping after {patience} epochs without improvement.')
            model.load_state_dict(best_model_state)
            break


# Save result model
torch.save(model.state_dict(), f'{models_path}model_bs={batch_size}_epochs={epoch + 1}_ES={patience}_lr={learning_rate}.pth')

