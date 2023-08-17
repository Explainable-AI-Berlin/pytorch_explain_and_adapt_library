apptainer run --nv ../python_container.sif python main.py  --lpips                  \
                --config configs/celeba.yml         \
                --exp ./runs/tmp         \
                --edit_attr test         \
                --n_train_img 100        \
                --n_inv_step 1000
