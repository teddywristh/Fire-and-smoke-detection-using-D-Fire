# -*- coding: utf-8 -*-
"""
Lưu ý:
- Augmentation chỉ được áp dụng cho tập train.
- Không được ghi đè hay làm thay đổi dữ liệu validation/test.
- Validation và test phải luôn deterministic (không ngẫu nhiên).
"""


def describe_recommended_augmentation():

    policy = {
        "horizontal_flip": True,        
        "rotation_range_degrees": 10,     
        "width_shift_range": 0.1,        
        "height_shift_range": 0.1,      
        "zoom_range": 0.1,                
        "brightness_range": [0.8, 1.2],  
        "apply_to": "train_only",         
    }
    return policy