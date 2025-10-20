import json
import copy
import os.path as osp
from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS, HOOKS
from mmengine.hooks import Hook

@DATASETS.register_module()
class AdaptiveCocoDataset(CocoDataset):
    """
    Adaptive COCO Dataset that can dynamically add samples during training
    """
    
    def __init__(self, 
                max_samples=30,
                samples_per_stage=1,
                **kwargs):
        
        self.max_samples = max_samples
        self.samples_per_stage = samples_per_stage
        self.current_sample_count = 0
        
        # Build full path manually
        data_root = kwargs.get('data_root', '')
        ann_file = kwargs['ann_file']
        
        # Combine data_root and ann_file for full path
        if data_root and not ann_file.startswith('/'):
            self.full_ann_file = data_root + ann_file
        else:
            self.full_ann_file = ann_file
        
        print(f"DEBUG: Looking for annotation file at: {self.full_ann_file}")
        
        # Create temporary ann_file - RELATIVE path only (without data_root)
        temp_ann_relative = ann_file.replace('.json', '_adaptive_temp.json')
        self.temp_ann_file = data_root + temp_ann_relative
        
        print(f"DEBUG: Will create temp file at: {self.temp_ann_file}")
        
        # Load full dataset first
        with open(self.full_ann_file, 'r') as f:
            self.full_coco_data = json.load(f)
        
        # Add first sample BEFORE calling super().__init__()
        self.add_next_samples()
        
        # Update kwargs to point to RELATIVE temp file path (MMDetection will add data_root)
        kwargs['ann_file'] = temp_ann_relative
        
        super().__init__(**kwargs)
    
    def _create_temp_annotation_file(self):
        """Create temporary COCO annotation file with current samples"""
        temp_data = {
            'categories': self.full_coco_data['categories'],
            'images': [],
            'annotations': []
        }
        
        if self.current_sample_count > 0:
            # Get current sample images
            current_images = self.full_coco_data['images'][:self.current_sample_count]
            current_image_ids = {img['id'] for img in current_images}
            
            # Get corresponding annotations
            current_annotations = [
                ann for ann in self.full_coco_data['annotations'] 
                if ann['image_id'] in current_image_ids
            ]
            
            temp_data['images'] = current_images
            temp_data['annotations'] = current_annotations
        
        # Save temporary file
        with open(self.temp_ann_file, 'w') as f:
            json.dump(temp_data, f)
    
    def add_next_samples(self):
        """Add next batch of samples and reload dataset"""
        if self.current_sample_count >= self.max_samples:
            print(f"Already at maximum samples: {self.max_samples}")
            return False
        
        # Add samples (but not exceed max_samples)
        new_count = min(
            self.current_sample_count + self.samples_per_stage,
            self.max_samples
        )
        
        if new_count == self.current_sample_count:
            return False
            
        self.current_sample_count = new_count
        
        # Update temporary annotation file
        self._create_temp_annotation_file()
        
        # Reload dataset if it's already initialized
        if hasattr(self, 'coco'):
            self.coco = self.coco_api.COCO(self.temp_ann_file)
            self.cat_ids = self.coco.get_cat_ids(cat_names=self.metainfo['classes'])
            self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
            self.img_ids = self.coco.get_img_ids()
            self.data_list = self.load_data_list()
        
        print(f"Added samples. Current count: {self.current_sample_count}/{self.max_samples}")
        return True
    
    def get_current_sample_count(self):
        """Return current number of samples in dataset"""
        return self.current_sample_count
    def load_data_list(self):
        """Load annotations and add text prompts for GroundingDINO"""
        data_list = super().load_data_list()
        
        class_names = self.metainfo['classes']
        text_prompt = '. '.join(class_names) + '.' 
        
        for data_info in data_list:
            data_info['text'] = text_prompt
            data_info['custom_entities'] = True
            
        return data_list


@HOOKS.register_module()
class AdaptiveLearningHook(Hook):
    """
    Hook to manage adaptive learning process:
    - Add new samples every N epochs
    - Save checkpoints with meaningful names
    - Log adaptive learning metrics
    """
    
    def __init__(self, 
                 epochs_per_stage=10,
                 save_checkpoint=True):
        self.epochs_per_stage = epochs_per_stage
        self.save_checkpoint = save_checkpoint
        self.stage_count = 0
    
    def before_train_epoch(self, runner):
        """Check if we need to add new samples at the start of epoch"""
        current_epoch = runner.epoch
        
        # Check if it's time to add new samples (every epochs_per_stage epochs, but not at epoch 0)
        if current_epoch > 0 and current_epoch % self.epochs_per_stage == 0:
            dataset = runner.train_dataloader.dataset
            
            if hasattr(dataset, 'add_next_samples'):
                old_count = dataset.get_current_sample_count()
                success = dataset.add_next_samples()
                
                if success:
                    self.stage_count += 1
                    new_count = dataset.get_current_sample_count()
                    
                    # Log stage change
                    runner.logger.info(
                        f"Adaptive Learning Stage {self.stage_count}: "
                        f"Added samples {old_count} -> {new_count}"
                    )
                    
                    # Log to tensorboard
                    runner.visualizer.add_scalar(
                        'adaptive/sample_count', 
                        new_count, 
                        current_epoch
                    )
                    runner.visualizer.add_scalar(
                        'adaptive/stage', 
                        self.stage_count, 
                        current_epoch
                    )
                    
                    # Save checkpoint with meaningful name
                    if self.save_checkpoint:
                        self._save_adaptive_checkpoint(runner, new_count)
    
    def _save_adaptive_checkpoint(self, runner, sample_count):
        """Save checkpoint with adaptive learning info"""
        checkpoint_name = f"epoch_{runner.epoch}_samples_{sample_count}.pth"
        
        # Correct API call for save_checkpoint
        runner.save_checkpoint(
            runner.work_dir,
            checkpoint_name,
            save_optimizer=True,
            save_param_scheduler=True
        )
        
        runner.logger.info(f"Saved adaptive checkpoint: {checkpoint_name}")
    
    def after_train_epoch(self, runner):
        """Log additional metrics after each epoch"""
        if hasattr(runner.train_dataloader.dataset, 'get_current_sample_count'):
            current_samples = runner.train_dataloader.dataset.get_current_sample_count()
            
            # Log current sample count
            runner.visualizer.add_scalar(
                'adaptive/current_samples', 
                current_samples, 
                runner.epoch
            )
        
    