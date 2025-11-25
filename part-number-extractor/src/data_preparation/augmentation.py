"""
Data augmentation techniques for BOM data
"""

import random
import copy
from typing import List, Dict


class BOMDataAugmenter:
    """BOM 데이터 증강 클래스"""

    def __init__(self, random_seed: int = 42):
        random.seed(random_seed)

    def shuffle_columns(self, item: Dict) -> Dict:
        """
        컬럼 순서를 무작위로 섞기 (Part Number 위치 변경)
        
        Args:
            item: Original data item
        
        Returns:
            Augmented data item with shuffled columns
        """
        augmented = copy.deepcopy(item)
        
        # Create list of indices
        indices = list(range(len(augmented['cells'])))
        random.shuffle(indices)
        
        # Reorder cells and labels
        augmented['cells'] = [augmented['cells'][i] for i in indices]
        if 'labels' in augmented:
            augmented['labels'] = [augmented['labels'][i] for i in indices]
        
        return augmented

    def add_noise(self, item: Dict, noise_prob: float = 0.1) -> Dict:
        """
        셀에 노이즈 추가 (오타, 공백 변경)
        Part Number는 변경하지 않음
        
        Args:
            item: Original data item
            noise_prob: Probability of adding noise to each character
        
        Returns:
            Augmented data item with noise
        """
        augmented = copy.deepcopy(item)
        
        for idx, (cell, label) in enumerate(zip(augmented['cells'], augmented.get('labels', []))):
            # Don't add noise to Part Number
            if label == 'PART_NUMBER':
                continue
            
            # Add noise to other cells
            if random.random() < 0.3:  # 30% chance per cell
                cell_str = str(cell)
                noisy_cell = self._add_character_noise(cell_str, noise_prob)
                augmented['cells'][idx] = noisy_cell
        
        return augmented

    def _add_character_noise(self, text: str, noise_prob: float) -> str:
        """Add character-level noise to text"""
        if not text:
            return text
        
        chars = list(text)
        for i in range(len(chars)):
            if random.random() < noise_prob:
                noise_type = random.choice(['swap', 'duplicate', 'delete', 'space'])
                
                if noise_type == 'swap' and i < len(chars) - 1:
                    chars[i], chars[i+1] = chars[i+1], chars[i]
                elif noise_type == 'duplicate':
                    chars.insert(i, chars[i])
                elif noise_type == 'delete':
                    chars[i] = ''
                elif noise_type == 'space':
                    chars[i] = ' '
        
        return ''.join(chars)

    def vary_part_number_format(self, item: Dict) -> Dict:
        """
        Part Number 형식 변형 (하이픈 추가/제거, 대소문자)
        
        Args:
            item: Original data item
        
        Returns:
            Augmented data item with varied Part Number format
        """
        augmented = copy.deepcopy(item)
        
        for idx, label in enumerate(augmented.get('labels', [])):
            if label == 'PART_NUMBER':
                part_number = str(augmented['cells'][idx])
                
                # Random transformation
                transform = random.choice(['add_hyphen', 'remove_hyphen', 'uppercase', 'lowercase'])
                
                if transform == 'add_hyphen' and '-' not in part_number and len(part_number) > 4:
                    # Add hyphen at random position
                    pos = random.randint(2, len(part_number) - 2)
                    part_number = part_number[:pos] + '-' + part_number[pos:]
                elif transform == 'remove_hyphen':
                    part_number = part_number.replace('-', '')
                elif transform == 'uppercase':
                    part_number = part_number.upper()
                elif transform == 'lowercase':
                    part_number = part_number.lower()
                
                augmented['cells'][idx] = part_number
        
        return augmented

    def augment_dataset(
        self,
        data: List[Dict],
        target_size: int,
        methods: List[str] = None
    ) -> List[Dict]:
        """
        데이터셋 전체 증강
        
        Args:
            data: Original dataset
            target_size: Target number of samples after augmentation
            methods: List of augmentation methods to use
        
        Returns:
            Augmented dataset
        """
        if methods is None:
            methods = ['shuffle', 'noise', 'format']
        
        augmented_data = data.copy()
        
        while len(augmented_data) < target_size:
            # Select random item to augment
            item = random.choice(data)
            
            # Apply random augmentation method
            method = random.choice(methods)
            
            if method == 'shuffle':
                aug_item = self.shuffle_columns(item)
            elif method == 'noise':
                aug_item = self.add_noise(item)
            elif method == 'format':
                aug_item = self.vary_part_number_format(item)
            else:
                aug_item = item
            
            # Update row_id
            aug_item['row_id'] = f"aug_{len(augmented_data):05d}"
            augmented_data.append(aug_item)
        
        return augmented_data[:target_size]
