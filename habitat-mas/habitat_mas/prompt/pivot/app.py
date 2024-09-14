"""PIVOT Demo."""

import gradio as gr
import numpy as np
from vip_runner import vip_runner
from vlms import GPT4V

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
    progress=gr.Progress(track_tqdm=False),
):

  if not openai_api_key:
    return [], 'Must provide OpenAI API Key'
  if im is None:
    return [], 'Must specify image'
  if not query:
    return [], 'Must specify description'

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
  for rst in vip_gen:
    yield rst


examples = [
    {
        'im_path': 'ims/aloha.png',
        'desc': 'a point between the fork and the cup',
    },
    {
        'im_path': 'ims/robot.png',
        'desc': 'the toy in the middle of the table',
    },
    {
        'im_path': 'ims/parking.jpg',
        'desc': 'a place to park if I am handicapped',
    },
    {
        'im_path': 'ims/tools.png',
        'desc': 'what should I use pull a nail'
    },
]


with gr.Blocks() as demo:
  gr.Markdown("""
# PIVOT: Prompting with Iterative Visual Optimization

[website](https://pivot-prompt.github.io/)
[view on huggingface](https://huggingface.co/spaces/pivot-prompt/pivot-prompt-demo/)

The demo below showcases a version of the PIVOT algorithm, which uses iterative visual prompts to optimize and guide the reasoning of Vision-Langauge-Models (VLMs).
Given an image and a description of an object or region,
PIVOT iteratively searches for the point in the image that best corresponds to the description.
This is done through visual prompting, where instead of reasoning with text, the VLM reasons over images annotated with sampled points,
in order to pick the best points.
In each iteration, we take the points previously selected by the VLM, resample new points around the their mean, and repeat the process.

To get started, you can use the provided example image and query pairs, or
upload your own images.
This demo uses GPT-4V, so it requires an OpenAI API key.

Hyperparameters to set:
* N Samples for Initialization - how many initial points are sampled for the first PIVOT iteration.
* N Samples for Optimiazation - how many points are sampled for subsequent iterations.
* N Iterations - how many optimization iterations to perform.
* N Ensemble Recursions - how many ensembles for recursive PIVOT.

Note that each iteration takes about ~10s, and each additional ensemble adds a multiple number of N Iterations.

After PIVOT finishes, the image gallery below will visualize PIVOT results throughout all the iterations.
There are two images for each iteration - the first one shows all the sampled points, and the second one shows which one PIVOT picked.
The Info textbox will show the final selected pixel coordinate that PIVOT converged to.

**To use the example images, right click on the image -> copy image, then click the clipboard icon in the Input Image box.**
""".strip())

  gr.Markdown(
      '## Example Images and Queries\n Drag images into the image box below (Try safari on Mac if dragging does not work)'
  )
  with gr.Row(equal_height=True):
    for example in examples:
      gr.Image(value=example['im_path'], type='numpy', label=example['desc'])

  gr.Markdown('## New Query')
  with gr.Row():
    with gr.Column():
      inp_im = gr.Image(
          label='Input Image',
          type='numpy',
          show_label=True,
          value=examples[0]['im_path'],
      )
      inp_query = gr.Textbox(
          label='Description',
          lines=1,
          placeholder=examples[0]['desc'],
      )

    with gr.Column():
      inp_openai_api_key = gr.Textbox(
          label='OpenAI API Key (not saved)', lines=1
      )
      with gr.Group():
        inp_n_samples_init = gr.Slider(
            label='N Samples for Initialization',
            minimum=10,
            maximum=40,
            value=25,
            step=1,
        )
        inp_n_samples_opt = gr.Slider(
            label='N Samples for Optimization',
            minimum=3,
            maximum=20,
            value=10,
            step=1,
        )
        inp_n_iters = gr.Slider(
            label='N Iterations', minimum=1, maximum=5, value=3, step=1
        )
        inp_n_parallel_trials = gr.Slider(
            label='N Parallel Trials', minimum=1, maximum=3, value=1, step=1
        )
      btn_run = gr.Button('Run')

  with gr.Group():
    out_ims = gr.Gallery(
        label='Images with Sampled and Chosen Points',
        columns=4,
        rows=1,
        interactive=False,
        object_fit="contain", height="auto"
    )
    out_info = gr.Textbox(label='Info', lines=1)

  btn_run.click(
      run_vip,
      inputs=[
          inp_im,
          inp_query,
          inp_n_samples_init,
          inp_n_samples_opt,
          inp_n_iters,
          inp_n_parallel_trials,
          inp_openai_api_key,
      ],
      outputs=[out_ims, out_info],
  )

demo.launch()
