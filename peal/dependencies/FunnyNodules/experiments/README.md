# Run instructions
## HierViT, Proto-Caps, ResNet-50, DenseNet-121
All four models use the same entry point:

python main.py
### Training vs. Testing

For these models, the workflow is unified:

- Training

        --train=True

- Testing

        --test=True
  and additionally provide:

        --model_path="[path to the trained checkpoint]"
        --epoch=[epoch number of the saved model]

Choosing the Architecture

Select the model via the --base_model argument:

| Model  | --base_model value |
| ------------- | ------------- |
| HierViT  | "ViT"  |
| Proto-Caps  | "ConvNet"  |
| ResNet-50  | "ResNetMT"  |
| DenseNet-121  | "DenseNetMT"  |

### Example Commands
- Training

        python main.py
        --train=True
        --base_model="ViT"
        --batch_size=2
        --resize_shape
        224
        224
        --lr=0.0001
        --warmup=150
        --push_step=1  
- Testing

        python main.py
        --test=True
        --base_model="ViT"
        --batch_size=2
        --resize_shape
        224
        224
        --lr=0.0001
        --warmup=150
        --push_step=1
        --model_path="2025-10-20 15:52:59.172496_0.885_145.pth"
        --epoch=145

        
## Concept Bottleneck
The Concept Bottleneck Model uses a separate script and different arguments:


        python ConceptBottleneck-master_FunnyNodules/experiments.py
        FunnyNodules
        Joint
        --seed=42
        --fc_layers
        256
        128
        6
        5
        --C_fc_name=fc3
        --y_fc_name=fc4
        --y_loss_type=cls
        --num_epochs=150


- FunnyNodules specifies the dataset.
- Joint selects the training strategy.
- Additional flags configure the classifier heads and training settings.

The code is modified by original work by Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, Percy Liang 2021  [ConceptBottleneck](https://github.com/yewsiang/ConceptBottleneck).
