import numpy as np
from vip_runner import vip_runner
from vlms import GPT4V
import json
from PIL import Image
import matplotlib.pyplot as plt

# Adjust radius of annotations based on size of the image
radius_per_pixel = 0.05


def run_vip(
    im,
    query,
    n_samples_init,
    n_samples_opt,
    n_iters,
    n_parallel_trials,
    openai_api_key,
    im_name,
):
    if not openai_api_key:
        print('Must provide OpenAI API Key')
        return []
    if im is None:
        print('Must specify image')
        return []
    if not query:
        print('Must specify description')
        return []

    img_size = np.min(im.shape[:2])
    print(int(img_size * radius_per_pixel))
    # add some action spec
    style = {
        'num_samples': 12,
        'circle_alpha': 0.6,
        'alpha': 0.8,
        'arrow_alpha': 0.0,
        'radius': int(img_size * radius_per_pixel),
        'thickness': 2,
        'fontsize': int(img_size * radius_per_pixel),
        'rgb_scale': 255,
        'focal_offset': 1,  # camera distance / std of action in z
    }

    action_spec = {
        'loc': [0, 0, 0],
        'scale': [0.0, 100, 100],
        'min_scale': [0.0, 30, 30],
        'min': [0, -300.0, -300],
        'max': [0, 300, 300],
        'action_to_coord': 250,
        'robot': None,
    }

    vlm = GPT4V(openai_api_key=openai_api_key)
    vip_gen = vip_runner(
        vlm,
        im,
        query,
        style,
        action_spec,
        n_samples_init=n_samples_init,
        n_samples_opt=n_samples_opt,
        n_iters=n_iters,
        n_parallel_trials=n_parallel_trials,
    )
    result = None
    for rst in vip_gen:
        result = rst
    for itr in range(len(result[0])):
        img = Image.fromarray(np.squeeze(result[0][itr]).astype('uint8'))
        img.save(f'assets/{im_name}_{itr+1}.png')


if __name__ == '__main__':
    with open('assets/api.json', 'r') as file:
        api = json.load(file)
    name = 'tools'
    run_vip(
        im=np.array(Image.open(f"assets/{name}.png")),
        query="what should I use pull a nail",
        n_samples_init=25,
        n_samples_opt=10,
        n_iters=3,
        n_parallel_trials=1,
        openai_api_key=api['key'],
        im_name=name
    )
