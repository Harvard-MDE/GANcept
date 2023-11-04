#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('python --version')


# 

# In[2]:


# Install condacolab
# !pip install -q condacolab
# import condacolab
# condacolab.install()


# In[3]:

get_ipython().system('env')


# In[4]:


# Check conda installation
import condacolab

condacolab.check()


# In[5]:


#!conda create --name gan python=3.10.12


# In[6]:


get_ipython().system('source activate gan')


# In[7]:


#!conda env update -n gan -f environment.yml


# In[8]:


get_ipython().system('source activate gan')


# In[9]:


# # Only run this once if you are in google colab

#%%bash
# !git clone https://github.com/fengqingthu/CLIP_Steering.git
# !git clone https://github.com/openai/CLIP.git


# In[10]:


# # Only run this once if you are in google colab
# %%bash

# !pip install ninja 2>> install.log
# !git clone https://github.com/SIDN-IAP/global-model-repr.git tutorial_code 2>> install.log


# In[11]:


# # Only run this if you are in google colab
# %%bash
# pip install torch==1.9.1+cu111 torchvision==0.10.1+cu118 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# pip install   pytorch-pretrained-biggan   ftfy   regex   tqdm   git+https://github.com/openai/CLIP.git   click   requests   pyspng   ninja   imageio-ffmpeg==0.4.3   scipy


# In[12]:


# import google.colab
import sys, torch

sys.path.append('tutorial_code')
if not torch.cuda.is_available():
    print("Change runtime type to include a GPU.")


# ## Import all dependencies
# 
# Make sure to add CLIP_steering into the path so that we can use the GANAlyze tools.

# In[13]:


import logging
import os
import pathlib

import clip
import IPython.display
import numpy as np
import torch.nn.functional as F
import torchvision
import torch.hub
from netdissect import proggan
from PIL import Image
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, "./CLIP_Steering")
try:
    import ganalyze_common_utils as common
    import ganalyze_transformations as transformations
except ImportError:
    print("Could not import ganalyze_common_utils or ganalyze_transformations")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running pytorch', torch.__version__, 'using', device.type)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# # Import and test the Pro GAN model
# 
# The GAN generator is just a function z->x that transforms random z to realistic images x.
# 
# To generate images, all we need is a source of random z.  Let's make a micro dataset with a few random z.

# In[14]:


import torchvision
import torch.hub
from netdissect import nethook, proggan

#n = 'proggan_bedroom-d8a89ff1.pth'
# n = 'proggan_churchoutdoor-7e701dd5.pth'
# n = 'proggan_conferenceroom-21e85882.pth'
# n = 'proggan_diningroom-3aa0ab80.pth'
# n = 'proggan_kitchen-67f1e16c.pth'
n = 'proggan_livingroom-5ef336dd.pth'
# n = 'proggan_restaurant-b8578299.pth'

url = 'http://gandissect.csail.mit.edu/models/' + n
try:
    sd = torch.hub.load_state_dict_from_url(url) # pytorch 1.1
except:
    sd = torch.hub.model_zoo.load_url(url) # pytorch 1.0
proggan_model = proggan.from_state_dict(sd).to(device)
proggan_model


# In[15]:


from netdissect import zdataset,renormalize

SAMPLE_SIZE = 6 # Increase this for better results (but slower to run)
zds = zdataset.z_dataset_for_model(proggan_model, size=SAMPLE_SIZE, seed=5555)
len(zds), zds[0][0].shape


# # Import and test CLIP model
# 
# Check that the CLIP model works fine. We import CLIP by installing it through PIP. We cloned CLIP's repo to get the CLIP/CLIP.png test image.

# In[16]:


clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()
clip_model.to(device)

image = preprocess(Image.open("CLIP/CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a livingroom", "a bedroom", "a church"]).to(device)

with torch.no_grad():
    image_features = clip_model.encode_image(image)
    text_features = clip_model.encode_text(text)

    logits_per_image, logits_per_text = clip_model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)


# In[17]:


torch.cuda.empty_cache()
latent_space_dim = zds[0][0][:,0,0].shape[0]
context_length = clip_model.context_length
vocab_size = clip_model.vocab_size

print(
    "Model parameters:",
    f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}",
)
print("Context length:", context_length)
print("Vocab size:", vocab_size)
print("Latent space dimension:", latent_space_dim)


# In[18]:


#@title Helper functions
from typing import List, Optional, Tuple
from PIL import Image


def show_images(
        images: list[Image.Image],
        resize: Optional[Tuple[int, int]] = None
    ):
    """Show a list of images in a row."""
    images = [np.array(img) for img in images]
    images = np.concatenate(images, axis=1)
    images = Image.fromarray(images)

    if resize:
        images.thumbnail(resize)

    IPython.display.display(images)


def show_and_save_images(
    images: list[Image.Image], batch: int, path: str, variant: str = "original"
):
    show_images(images)

    if not os.path.exists(path):
        os.makedirs(path)

    for i, img in enumerate(images):
        img.save(f"{path}/image_{batch}_{i}_{variant}.png")


def show_gan_results(gan_results: List[List[Tuple[Image.Image, np.ndarray]]]):
    for batch_results in gan_results:
        batch_size = len(batch_results[0][0])

        for i in range(batch_size):
            steering_images = [res[0][i] for res in batch_results]
            steering_scores = np.stack(
                [res[1][i].detach().cpu().numpy() for res in batch_results]
            ).tolist()
            print(steering_scores)
            show_images(steering_images, resize=(1024, 256))

def get_clip_probs(image_inputs, text_features, model, attribute_index=0):
    image_inputs = torch.stack([preprocess(img.resize((512, 512))) for img in image_inputs]).to(device)
    image_features = model.encode_image(image_inputs).float()

    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # cosine similarity as logits
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()

    clip_probs = logits_per_image.softmax(dim=-1)

    return clip_probs.narrow(dim=-1, start=attribute_index, length=1).squeeze(dim=-1)

def show_gan_results(gan_results: list):
    for batch_results in gan_results:
        batch_size = len(batch_results[0][0])

        for i in range(batch_size):
            steering_images = [res[0][i] for res in batch_results]
            steering_scores = np.stack(
                [res[1][i].detach().cpu().numpy() for res in batch_results]
            ).tolist()
            print(steering_scores)
            show_images(steering_images, resize=(1024, 256))


def make_images_and_probs(
    model, zdataset, clip_model, encoded_text, attribute_index=0
):
    gan_images = []
    for z in zdataset:
      gan_output = model(z[None,...])[0]
      gan_images.append(renormalize.as_image(gan_output))

    clip_probs = get_clip_probs(gan_images, encoded_text, clip_model, attribute_index)

    return gan_images, clip_probs


# ## Use CLIP to extract target text attributes
# 
# Use the CLIP model to extract target text attributes for steering the output of a GAN. We use the CLIP tokenizer and encoder to extract text features and normalize them.
# 
# The resulting text features are used later to steer the GAN output towards the desired attribute.

# In[19]:


# Extract text features for clip

# Here is how we specify the desired attributes
attributes = ["a santorini room", "a playroom"]
attribute_index = 0  # which attribute do we want to maximize
text_descriptions = [f"{label}" for label in attributes]



# #================================================================
# # Read the list of words from a text file
# with open('word_list.txt', 'r') as file:
#     word_list = [line.strip() for line in file]

# # Loop through each word in the list and execute the code
# for attribute in word_list:
#     print(f"Processing attribute: {attribute}")
#     run Automatic_GAN_Dissection.py
# #================================================================




with torch.no_grad():
    text_tokens = clip.tokenize(text_descriptions).to(device)
    text_features = clip_model.encode_text(text_tokens).float()
    # text_features = F.normalize(text_features, p=2, dim=-1)

text_features.shape


# # Declare the GAN streering model
# 
# This is the model in charge of changing the vectors `z` so that it aligns with the objective declared in `text_features`.

# In[20]:


transformation = transformations.OneDirection(latent_space_dim, vocab_size)
transformation = transformation.to(device)


# In[21]:


checkpoint_dir = f"checkpoints/results_maximize_classifier_probability"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_name = f"pytorch_model_progran_steering_{attributes[attribute_index]}_final.pth"
print (checkpoint_dir)
print (checkpoint_name)


# # Traning steps
# 
# Now we're ready to train the GAN steering model called `transformation`. The overall algorithm looks like this:
# 
# 1. Generate the noise and class vectors for a given number of training samples.
# 2. For each batch, generate the GAN images and calculate the CLIP scores comparing the images features correlation with the target text features.
# 3. Use the `transformation` model to adjust the original noise. Repeat step 2 for the transformed noise `z_transformed`.
# 4. Compare the scoring output, and make the model optimize to minimize the difference between the target scores and the resulting one after transforming the original noise.
# 
# 

# In[22]:


optimizer = torch.optim.Adam(
    transformation.parameters(), lr=0.0002
)  # as specified in GANalyze
losses = common.AverageMeter(name="Loss")

#  training settings
optim_iter = 0
batch_size = 64  # Do not change
train_alpha_a = -0.5  # Lower limit for step sizes
train_alpha_b = 0.5  # Upper limit for step sizes
#
# Number of samples to train for # Ganalyze uses 400,000 samples.
# Use smaller number for testing.
#
#num_samples = 90_000
#num_samples = 450
num_samples = 64



attribute_index = 0
checkpoint_dir = f"checkpoints/results_maximize_{attributes[attribute_index]}_probability"
pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

zds = zdataset.z_dataset_for_model(proggan_model, size=num_samples, seed=5555)
zds_dataloader = DataLoader(zds, batch_size=batch_size, shuffle=True)

progress = tqdm("Training", total=len(range(0, num_samples, batch_size)))

# loop over data batches
for batch_start, z_batch in enumerate(zds_dataloader):
    z_batch = z_batch[0].squeeze().to(device)

    step_sizes = (train_alpha_b - train_alpha_a) * np.random.random(
        size=(batch_size)
    ) + train_alpha_a  # sample step_sizes

    step_sizes_broadcast = np.repeat(step_sizes, latent_space_dim).reshape(
        [batch_size, latent_space_dim]
    )
    step_sizes_broadcast = (
        torch.from_numpy(step_sizes_broadcast).type(torch.FloatTensor).to(device)
    )

    #
    # Generate the original images and get their clip scores
    #
    gan_images, out_scores = make_images_and_probs(
        model=proggan_model,
        zdataset = z_batch,
        clip_model=clip_model,
        encoded_text=text_features,
        attribute_index=attribute_index,
    )

    # TODO: ignore z vectors with less confident clip scores
    target_scores = torch.clip(
        out_scores + torch.from_numpy(step_sizes).to(device).float(),
        0.0,
        1.0
    )

    #
    # Transform the z vector and get the clip scores for the transformed images
    #
    zb_transformed = transformation.transform(z_batch, None, step_sizes = step_sizes_broadcast)
    gan_images_transformed, out_scores_transformed = make_images_and_probs(
        model=proggan_model,
        zdataset = zb_transformed,
        clip_model=clip_model,
        encoded_text=text_features,
        attribute_index=attribute_index,
    )

    #
    # Compute loss and backpropagate
    #
    loss = transformation.criterion(out_scores_transformed, target_scores)

    loss.backward()
    optimizer.step()

    #
    # Print and save intermediate results
    #
    losses.update(loss.item(), batch_size)
    if optim_iter % 50 == 0:
        print(
            f"[Maximizing score for {attributes[attribute_index]}] "
            f"Progress: [{batch_start}/{num_samples}] {losses}"
        )

        print(
            f"[Scores] "
            f"Target: {target_scores} Out: {out_scores_transformed}"
        )

    if optim_iter % 200 == 0:
        batch_checkpoint_name = f"pytorch_model_progran_{batch_start}.pth"
        torch.save(
            transformation.state_dict(),
            os.path.join(checkpoint_dir, batch_checkpoint_name)
        )

        # plot and save sample images
        # show_and_save_images(gan_images, batch_start, checkpoint_dir)
        # show_and_save_images(
        #     gan_images_transformed, batch_start, checkpoint_dir, "transformed"
        # )

    optim_iter = optim_iter + 1
    progress.update(1)


# In[23]:


checkpoint_path = os.path.join(checkpoint_dir, batch_checkpoint_name)
torch.save(
    transformation.state_dict(),
    checkpoint_path
)
print (checkpoint_path)
print (checkpoint_name)


# In[24]:


# Only run this if you are in google colab
get_ipython().system('rm -rf Weights')
get_ipython().system('mkdir -p Weights')

# mount files from google drive
# and follow the steps here
# from google.colab import drive
# drive.mount('/content/gdrive')

import shutil

shutil.copy(checkpoint_path, f"/content/Weights/{checkpoint_name}")
shutil.copy(checkpoint_path, f"Weights/{checkpoint_name}")


# In[25]:


checkpoint_path


# # Testing steps
# 
# Now that the model is trained, we can test it and see how the output changes when incrementing and decrementing the step_sizes on our `z` vectors.
# 
# We take the latest saved checkpoint located at `{checkpoint_dir}/pytorch_model_final.pth`.

# In[26]:



# Now that the model is trained, we can test it.
#
# Testing the model involves using the transformation model to transform a z vector and then
# using the GAN model to generate an image from the transformed z vector.
# We will change the step size and see how the image changes.
#
batch_size = 6  # Do not change
alpha = 0.2
num_samples = 6

iters = 10

transformation = transformations.OneDirection(latent_space_dim)

# transformation.load_state_dict(
#     torch.load(
#         os.path.join("Weights", checkpoint_name),
#     ),
#     strict=True,
# )

transformation.load_state_dict(
   torch.load(checkpoint_path),
   strict=True,
)


transformation.to(device)
transformation.eval()



gan_results = []

with torch.no_grad():
    zds = zdataset.z_dataset_for_model(proggan_model, size=num_samples, seed=5555)
    zds_dataloader = DataLoader(zds, batch_size=batch_size, shuffle=True)

    progress = tqdm("Training", total=len(range(0, num_samples, batch_size)))

    # loop over data batches
    for batch_start, z_batch in enumerate(zds_dataloader):
        #
        # Setup the batch z and y vectors. Also sample step sizes.
        #
        z_batch = z_batch[0].squeeze().to(device)

        step_sizes = (
            (torch.ones((batch_size, latent_space_dim)) * alpha).float().to(device)
        )

        gan_images, out_scores = make_images_and_probs(
            model=proggan_model,
            zdataset = z_batch,
            clip_model=clip_model,
            encoded_text=text_features,
            attribute_index=attribute_index,
        )

        batch_results = [(gan_images, out_scores)]

        # Generate images by transforming the z vector in the negative direction
        z_negative = z_batch.clone()

        for iter in range(iters):
            z_negative = transformation.transform(z_negative, None, -step_sizes)
            batch_results.insert(
                0,
                make_images_and_probs(
                    model=proggan_model,
                    zdataset = z_negative,
                    clip_model=clip_model,
                    encoded_text=text_features,
                    attribute_index=attribute_index,
                )
            )

        # Generate images by transforming the z vector in the positive direction
        z_positive = z_batch.clone()

        for iter in range(iters):
            z_positive = transformation.transform(z_positive, None, step_sizes)
            batch_results.append(
                make_images_and_probs(
                    model=proggan_model,
                    zdataset = z_positive,
                    clip_model=clip_model,
                    encoded_text=text_features,
                    attribute_index=attribute_index,
                )
            )

        gan_results.append(batch_results)

        progress.update(1)


# In[27]:


show_gan_results(gan_results)


# ## Hooking a model with InstrumentedModel
# 
# To analyze what a model is doing inside, we can wrap it with an InstrumentedModel, which makes it easy to hook or modify a particular layer.
# 
# InstrumentedModel adds a few useful functions for inspecting a model, including:
#    * `model.retain_layer('layername')` - hooks a layer to hold on to its output after computation
#    * `model.retained_layer('layername')` - returns the retained data from the last computation
#    * `model.edit_layer('layername', rule=...)` - runs the `rule` function after the given layer
#    * `model.remove_edits()` - removes editing rules
# 
# Let's setup `retain_layer` now.  We'll pick a layer sort of in the early-middle of the generator.  You can pick whatever you like.

# In[28]:


transformation = transformations.OneDirection(latent_space_dim)
transformation.load_state_dict(
    torch.load(
#         "/home/ubuntu/GANSteering/proggan/pytorch_model_progran_steering_a floral bedroom_final.pth",
        # "/home/ubuntu/GANSteering/proggan/pytorch_model_progran_steering_a floral bedroom_final.pth",
#        "/./checkpoints/results_maximize_a santorini room_probability/pytorch_model_progran_0.pth",
      "/content/checkpoints/results_maximize_a santorini room_probability/pytorch_model_progran_0.pth",
    ),
    strict=True,
)
transformation.to(device)
transformation.eval()


# In[29]:


from netdissect import nethook
from netdissect import imgviz
from netdissect import show
from netdissect import tally
from netdissect import upsample
from netdissect import segviz

# Don't re-wrap it, if it's already wrapped (e.g., if you press enter twice)
if not isinstance(proggan_model, nethook.InstrumentedModel):
    proggan_model = nethook.InstrumentedModel(proggan_model)
proggan_model.retain_layer('layer4')


# In[30]:


# Run the model
img = proggan_model(zds[0][0][None,...].to(device))

# As a side-effect, the proggan_model has retained the output of layer4.
acts = proggan_model.retained_layer('layer4')

# We can look at it.  How much data is it?
acts.shape


# In[31]:


# Let's just look at the 0th convolutional channel.
print(acts[0,0])


# ## Visualizing activation data
# 
# It can be informative to visualize activation data instead of just looking at the numbers.
# 
# Net dissection comes with an ImageVisualizer object for visualizing grid data as an image in a few different ways.  Here is a heatmap of the array above:

# In[32]:


iv = imgviz.ImageVisualizer(100)
iv.heatmap(acts[0,1], mode='nearest')


# In[33]:



show(
    [['unit %d' % u,
      [iv.image(img[0])],
      [iv.masked_image(img[0], acts, (0,u))],
      [iv.heatmap(acts, (0,u), mode='nearest')],
     ] for u in range(1, 6)]
)


# ## Collecting quantile statistics for every unit
# 
# We want to know per-channel minimum or maximum values, means, medians, quantiles, etc.
# 
# We want to treat each pixel as its own sample for all the channels.  For example, here are the activations for one image as an 8x8 tensor over with 512 channels.  We can disregard the geometry and just look at it as a 64x512 sample matrix, that is 64 samples of 512-dimensional vectors.

# In[34]:


print(acts.shape)
print(acts.permute(0, 2, 3, 1).contiguous().view(-1, acts.shape[1]).shape)


# Net dissection has a tally package that tracks quantiles over large samples.
# 
# To use it, just define a function that returns sample matrices like the 64x512 above, and then it will call your function on every batch and tally up the statistics.

# In[35]:


# To collect stats, define a function that returns 2d [samples, units]
def compute_samples(zbatch):
    _ = proggan_model(zbatch.to(device))          # run the proggan_model
    acts = proggan_model.retained_layer('layer4') # get the activations, and flatten
    return acts.permute(0, 2, 3, 1).contiguous().view(-1, acts.shape[1])

# Then tally_quantile will run your function over the whole dataset to collect quantile stats
rq = tally.tally_quantile(compute_samples, zds)

# Print out the median value for the first 20 channels
rq.quantiles(0.5)[:20]


# ## Exploring quantiles
# 
# The rq object tracks a sketch of all the quantiles of the sampled data.  For example, what is the mean, median, and percentile value for each unit?

# In[36]:


# This tells me now, for example, what the means are for channel,
# rq.mean()
# what median is,
# rq.quantiles(0.5)
# Or what the 99th percentile quantile is.
# rq.quantiles(0.99)

(rq.quantiles(0.8) > 0).sum()


# The quantiles can be plugged directly into the ImageVisualizer to put heatmaps on an informative per-unit scale.  When you do this:
# 
#    * Heatmaps are shown on a scale from black to white from 1% lowest to the 99% highest value.
#    * Masked image lassos are shown at a 95% percentile level (by default, can be changed).

# In[37]:


iv = imgviz.ImageVisualizer(100, quantiles=rq)
show([
    [  # for every unit, make a block containing
       'unit %d' % u,         # the unit number
       [iv.image(img[0])],    # the unmodified image
       [iv.masked_image(img[0], acts, (0,u))], # the masked image
       [iv.heatmap(acts, (0,u), mode='nearest')], # the heatmap
    ]
    for u in range(414, 420)
])


# In[38]:


def compute_image_max(zbatch):
    image_batch = proggan_model(zbatch.to(device))
    return proggan_model.retained_layer('layer4').max(3)[0].max(2)[0]

topk = tally.tally_topk(compute_image_max, zds)
topk.result()[1].shape


# In[39]:


# For each unit, this function prints out unit masks from the top-activating images
def unit_viz_row(unitnum, percent_level=0.95):
    out = []
    for imgnum in topk.result()[1][unitnum][:8]:
        img = proggan_model(zds[imgnum][0][None,...].to(device))
        acts = proggan_model.retained_layer('layer4')
        out.append([imgnum.item(),
                    [iv.masked_image(img[0], acts, (0, unitnum), percent_level=percent_level)],
                   ])
    return out

show(unit_viz_row(30))


# # Evaluate matches with semantic concepts
# 
# Do the filters match any semantic concepts?  To systematically examine this question,
# we have pretrained (using lots of labeled data) a semantic segmentation network to recognize
# a few hundred classes of objects, parts, and textures.
# 
# Run the code in this section to look for matches between filters in our GAN and semantic
# segmentation clases.
# 
# ## Labeling semantics within the generated images
# 
# Let's quantify what's inside these images by segmenting them.
# 
# First, we create a segmenter network.  (We use the Unified Perceptual Parsing segmenter by Xiao, et al. (https://arxiv.org/abs/1807.10221).
# 
# Note that the segmenter we use here requires a GPU.

# Then we create segmentation images for the dataset.  Here tally_cat just concatenates batches of image (or segmentation) data.
# 
#   * `segmodel.segment_batch` segments an image
#   * `iv.segmentation(seg)` creates a solid-color visualization of a segmentation
#   * `iv.segment_key(seg, segmodel)` makes a small legend for the segmentation

# In[40]:


# !git clone https://github.com/vacancy/PreciseRoIPooling.git


# In[41]:


# !rm -r ./PreciseRoIPooling/

#!rm -r /content/tutorial_code/netdissect/upsegmodel/prroi_pool


# In[42]:


import torch
import json
from netdissect import segmenter, setting,upsegmodel

segmodel_dir = '/content/datasets/segmodel/upp-resnet50-upernet/'
# Load json of class names and part/object structure
with open(os.path.join(segmodel_dir, 'labels.json')) as f:
    labeldata = json.load(f)
nr_classes={k: len(labeldata[k])
            for k in ['object', 'scene', 'material']}
nr_classes['part'] = sum(len(p) for p in labeldata['object_part'].values())
# Create a segmentation model
segbuilder = upsegmodel.ModelBuilder()
# example segmodel_arch = ('resnet101', 'upernet')
epoch = 40
segmodel_arch = ('resnet50', 'upernet')
seg_encoder = segbuilder.build_encoder(
        arch=segmodel_arch[0],
        fc_dim=2048,
        weights=os.path.join(segmodel_dir, 'encoder_epoch_%d.pth' % epoch))
seg_decoder = segbuilder.build_decoder(
        arch=segmodel_arch[1],
        fc_dim=2048, use_softmax=True,
        nr_classes=nr_classes,
        weights=os.path.join(segmodel_dir, 'decoder_epoch_%d.pth' % epoch))
segmodel = upsegmodel.SegmentationModule(
        seg_encoder, seg_decoder, labeldata)
segmodel.categories = ['object', 'part', 'material']
segmodel.eval()


# In[43]:


from netdissect import segmenter, setting

# egmenter.UnifiedParsingSegmenter(segsizes=[256])
segmodel, seglabels, _ = setting.load_segmenter('netpq')
# seglabels = [l for l, c in segmodel.get_label_and_category_names()[0]]
print('segmenter has', len(seglabels), 'labels')


# In[44]:


import os
if os.getenv("COLAB_RELEASE_TAG"):
  print("Running in Colab")
else:
  print("NOT in Colab")


# In[45]:


imgs = tally.tally_cat(lambda zbatch: proggan_model(zbatch.to(device)), zds)
seg = tally.tally_cat(lambda img: segmodel.segment_batch(img.cuda(), downsample=1), imgs)

from netdissect.segviz import seg_as_image, segment_key
show([
    (iv.image(imgs[i]),
     iv.segmentation(seg[i,0]),
     iv.segment_key(seg[i,0], segmodel)
    )
    for i in range(min(len(seg), 5))
])


# In[46]:


'/content/tutorial_code/netdissect/upsegmodel/prroi_pool/src/prroi_pooling_gpu_impl.cu'


# In[47]:


torch.cuda.empty_cache()
get_ipython().system('nvidia-smi')


# In[48]:


# We upsample activations to measure them at each segmentation location.
upfn8 = upsample.upsampler((64, 64), (8, 8)) # layer4 is resolution 8x8

def compute_conditional_samples(zbatch):
    image_batch = proggan_model(zbatch.to(device))
    seg = segmodel.segment_batch(image_batch, downsample=4)
    upsampled_acts = upfn8(proggan_model.retained_layer('layer4'))
    return tally.conditional_samples(upsampled_acts, seg)

# Run this function once to sample one image
sample = compute_conditional_samples(zds[0][0].cuda()[None,...])

# The result is a list of all the conditional subsamples
[(seglabels[c], d.shape) for c, d in sample]


# In[49]:


cq = tally.tally_conditional_quantile(compute_conditional_samples, zds)


# Conditional quantile statistics let us compute lots of relationships between units and visual concepts.
# 
# For example, IoU is the "intersection over union" ratio, measuring how much overlap there is between the top few percent activations of a unit and the presence of a visual concept.  We can estimate the IoU ratio for all pairs between units and concepts with these stats:

# In[50]:


iou_table = tally.iou_from_conditional_quantile(cq, cutoff=0.99)
iou_table.shape


# Now let's view a few of the units, labeled with an associated concept, sorted from highest to lowest IoU.

# In[51]:


unit_list = sorted(enumerate(zip(*iou_table.max(1))), key=lambda k: -k[1][0])

for unit, (iou, segc) in unit_list[:5]:
    print('unit %d: %s (iou %.2f)' % (unit, seglabels[segc], iou))
    show(unit_viz_row(unit))


# We can quantify the overall match between units and segmentation concepts by counting the number of units that match a segmentation concept (omitting low-scoring matches).

# In[52]:


print('Number of units total:', len(unit_list))
print('Number of units that match a segmentation concept with IoU > 0.04:',
   len([i for i in range(len(unit_list)) if unit_list[i][1][0] > 0.04]))


# ## Examining units that select for lamp
# 
# Now let's filter just units that were labeled as 'lamp' units.

# In[53]:


lamp_index = seglabels.index('lamp')
lamp_units = [(unit, iou) for iou, unit in
              list(zip(*(iou_table[:,lamp_index].sort(descending=True))))[:10]]


# In[54]:


lamp_units


# In[55]:


# tree_units = [(unit, iou, segc) for unit, (iou, segc) in unit_list if seglabels[segc] == 'tree'][:10]
# If you can't run the segmenter, uncomment the line below and comment the one above.
# tree_units = [365, 157, 119, 374, 336, 195, 278, 76, 408, 125]

for unit, iou in lamp_units:
    print('unit %d, iou %.2f' % (unit, iou))
    show(unit_viz_row(unit))


# ## Editing a model by altering units
# 
# Now let's try changing some units directly to see what they do.
# 
# We we will use `model.edit_layer` to do that.
# 
# This works by just allowing you to define a function that edits the output of a layer.
# 
# We will edit the output of `layer4` by zeroing ten of the tree units.

# In[56]:


# Leo: Changed here how the tree units are used as indexes
lamp_units_ix = [ix for ix, _ in lamp_units]
lamp_units_ix


# In[57]:


def zero_out_lamp_units(data, model):
    data[:, lamp_units_ix, :, :] = 0.0
    return data

proggan_model.edit_layer('layer4', rule=zero_out_lamp_units)
edited_imgs = tally.tally_cat(lambda zbatch: proggan_model(zbatch.to(device)), zds)
show([
    (['Before', [renormalize.as_image(imgs[i])]],
     ['After', [renormalize.as_image(edited_imgs[i])]])
      for i in range(min(10, len(zds)))])
proggan_model.remove_edits()


# # Testing causal effects of representation units
# 
# Now it's your turn.
# 
# Now try the following experiments:
#    * Instead of zeroing the lamp units, try setting them negative, e.g., to -5.
#    * Instead of turning the lamp units off, try turning them on, e.g., set them to 10.
# 

# In[58]:


def zero_out_lamp_units(data, proggan_model):
    data[:, lamp_units_ix, :, :] = -5
    return data

proggan_model.edit_layer('layer4', rule=zero_out_lamp_units)
edited_imgs = tally.tally_cat(lambda zbatch: proggan_model(zbatch.to(device)), zds)

show([
    (['Before', [renormalize.as_image(imgs[i])]],
     ['After', [renormalize.as_image(edited_imgs[i])]])
      for i in range(min(10, len(zds)))])

proggan_model.remove_edits()


# # Examining units for other concepts
# 
# 
# Find a set of `blanket`-selective units `blanket_units` (instead of `lamp_units`), or choose another concept.
# 
# Then create a set of examples show the effect of setting the these off, to `-5.0` or on, at `10.0`.
# 
# What is the effect on blankets?  What is the effect on other types of objects in the generated scenes?

# In[59]:


# Your solution to exercise 4.

blanket_index = seglabels.index('chair')
blanket_units = [(unit, iou) for iou, unit in
              list(zip(*(iou_table[:,blanket_index].sort(descending=True))))[:10]]
blanket_units_ix = [ix for ix, _ in blanket_units]

def turn_off_blanket_units(data, model):
    data[:, blanket_units_ix, :, :] = -5.0
    return data

# Then visualize the effect
proggan_model.edit_layer('layer4', rule=turn_off_blanket_units)
edited_imgs = tally.tally_cat(lambda zbatch: proggan_model(zbatch.to(device)), zds)

show([
    (['Before', [renormalize.as_image(imgs[i])]],
     ['After', [renormalize.as_image(edited_imgs[i])]])
      for i in range(min(10, len(zds)))])
proggan_model.remove_edits()


# # Steered ProGAN
# 
# We will take the steering model, and compare what happens with the proggan with the original and the steered noise.
# 
# First, we take 5 original noises and steer them

# In[60]:


alpha = 0.2
iters = 5
batch_size = 5

# Generate images by transforming the z vector in the positive direction
zds_dataloader = DataLoader(zds, batch_size=batch_size, shuffle=False)


with torch.no_grad():
    step_sizes = (
        (torch.ones((batch_size, latent_space_dim)) * alpha).float().to(device)
    )

    for zbatch in zds_dataloader:
        original_noise = zbatch[0].squeeze().to(device)
        transformed_noise = original_noise.clone()

        for iter in range(iters):
            transformed_noise = transformation.transform(transformed_noise, None, step_sizes)

        break


# In[61]:


transformed_noise = transformed_noise[..., None, None]
original_noise = original_noise[..., None, None]

transformed_noise.shape, original_noise.shape


# ## Compare the GAN dissect with ProGAN and steered ProGAN
# 
# Now we've got the transformed noise, which we can use it to create images with the ProGAN.
# 
# First, let's compare the **highest activations** in the fourth layer using the transformed noise to the ones using
# the original noise vector.

# In[62]:


def find_all_activated_units(acts, threshold = 0,image_num = None):
    n_images = acts.shape[0]
    n_units = acts.shape[1]
    unit_activation_shape = acts.shape[2]*acts.shape[3]
    if image_num > -1:
        activation = acts[image_num]
    else:
        print("Averaging all images")
        reshaped_tensor = acts.view(n_images*n_units, acts.shape[2], acts.shape[3])
        activation = torch.mean(reshaped_tensor, dim=0)
    flattened_activation = activation.view(n_units, unit_activation_shape)

    # Find the indices of units with activation greater than the threshold
    unit_indices = torch.nonzero(torch.any(flattened_activation > threshold, dim=1), as_tuple=False)[:, 0]

    average_activations_filtered = torch.mean(flattened_activation[unit_indices], dim=1)

#     row_numbers = torch.nonzero(torch.any(flattened_activation > threshold, dim=1), as_tuple=False)[:, 0]
    sorted_unit_indices = torch.argsort(average_activations_filtered, descending=True)

    return sorted_unit_indices


# In[63]:


transformed_img = proggan_model(transformed_noise)
transformed_acts = proggan_model.retained_layer('layer4')


# In[64]:


transformed_activated_units = find_all_activated_units(transformed_acts, threshold = 0,image_num = 0)
transformed_activated_units


# In[65]:


rq_transformed = tally.tally_quantile(compute_samples, transformed_noise)
iv = imgviz.ImageVisualizer(100, quantiles=rq_transformed)

show([
    [  # for every unit, make a block containing
       'unit %d' % u,         # the unit number
       [iv.image(transformed_img[0])],    # the unmodified image
       [iv.masked_image(transformed_img[0], transformed_acts, (0,u))], # the masked image
       [iv.heatmap(transformed_acts, (0,u), mode='nearest')], # the heatmap
    ]
    for u in transformed_activated_units[:5]
])


# In[66]:


def find_semantic_concepts(activated_units, iou_table):
    semantic_concept = []
    semantic_conecpt_iou = []
    for u in activated_units:
        semantic_concept_index = torch.argmax(iou_table[u])
        iou = iou_table[u,semantic_concept_index ]
        semantic_concept.append(seglabels[semantic_concept_index])
        semantic_conecpt_iou.append(iou)
    return semantic_concept,semantic_conecpt_iou


# Create a dataframe with the information of each activated unit, for image 0. In case you want to do the average of images, instead of having only one, use image_num = None

# In[67]:


import pandas as pd

def create_image_dataframe(noise, threshold = 0, image_num = None):
    img = proggan_model(noise)
    acts = proggan_model.retained_layer('layer4')
    activated_units = find_all_activated_units(acts, threshold = threshold,image_num = image_num)
    cq = tally.tally_conditional_quantile(compute_conditional_samples, noise)
    iou_table = tally.iou_from_conditional_quantile(cq, cutoff=0.99)
    associated_concepts, iou = find_semantic_concepts(activated_units, iou_table)
    data = {
        'Semantic Concept': associated_concepts,
        'IoU':  [float(tensor.item()) for tensor in iou]
    }

    df = pd.DataFrame(data, index=activated_units.tolist())
    df.index.name = 'Activated Units'
    return df


# In[68]:


df_transformed = create_image_dataframe(transformed_noise, 0,0)
df_transformed


# Let's do the same with the original noise

# In[69]:


original_img = proggan_model(original_noise)
original_acts = proggan_model.retained_layer('layer4')


# In[70]:


original_activated_units = find_all_activated_units(original_acts, threshold = 0,image_num = 0)
original_activated_units


# In[71]:


show([
    [  # for every unit, make a block containing
       'unit %d' % u,         # the unit number
       [iv.image(original_img[0])],    # the unmodified image
       [iv.masked_image(original_img[0], original_acts, (0,u))], # the masked image
       [iv.heatmap(original_acts, (0,u), mode='nearest')], # the heatmap
    ]
    for u in original_activated_units[:5]
])


# In[72]:


df_original = create_image_dataframe(original_noise, 0,0)
df_original


# Now, let's do some statistics to the transformed images using the steering model and then the original images

# In[73]:


# Steered images
transformed_unit_counts = df_transformed['Semantic Concept'].value_counts()

transformed_average_iou = df_transformed.groupby('Semantic Concept')['IoU'].mean()

transformed_result = transformed_unit_counts.to_frame().join(transformed_average_iou)

transformed_result.columns = ['Unit Count', 'Average IoU']

transformed_result.head(10)


# In[74]:


# Original images
original_unit_counts = df_original['Semantic Concept'].value_counts()

original_average_iou = df_original.groupby('Semantic Concept')['IoU'].mean()

original_result = original_unit_counts.to_frame().join(original_average_iou)

original_result.columns = ['Unit Count', 'Average IoU']

original_result.head(10)


# In[75]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[76]:


plt.figure(figsize=(25, 12))
sns.barplot(x=transformed_result.index, y='Unit Count', data=transformed_result, color='blue')
plt.xticks(rotation=90)
plt.xlabel('Semantic Concept')
plt.ylabel('Unit Count')
plt.title('Unit Count by Semantic Concept')
plt.show()


# In[77]:


plt.figure(figsize=(25, 12))
sns.barplot(x=original_result.index, y='Unit Count', data=original_result, color='orange')
plt.xticks(rotation=90)
plt.xlabel('Semantic Concept')
plt.ylabel('Unit Count')
plt.title('Unit Count by Semantic Concept')
plt.show()


# In[78]:


common_concepts = set(transformed_result.index).intersection(original_result.index)

# Reindex the DataFrames with the common concepts
transformed_result = transformed_result.reindex(common_concepts)
original_result = original_result.reindex(common_concepts)

# Set the width of each bar
bar_width = 0.35

# Set the position of each bar on the x-axis
r1 = np.arange(len(transformed_result.index))
r2 = [x + bar_width for x in r1]

# Set the figure size
plt.figure(figsize=(30, 12))

# Plot the transformed_result
plt.bar(r1, transformed_result['Unit Count'], color='blue', width=bar_width, label='Transformed Result')
plt.bar(r1, transformed_result['Average IoU'], color='lightblue', width=bar_width, alpha=0.7)

# Plot the original_result
plt.bar(r2, original_result['Unit Count'], color='orange', width=bar_width, label='Original Result')
plt.bar(r2, original_result['Average IoU'], color='lightsalmon', width=bar_width, alpha=0.7)

# Customize the x-axis ticks and labels
plt.xticks([r + bar_width/2 for r in range(len(transformed_result.index))], transformed_result.index, rotation=90)

# Set labels and title
plt.xlabel('Semantic Concept')
plt.ylabel('Count / Average IoU')
plt.title('Comparison of Unit Count and Average IoU by Semantic Concept')

# Add a legend
plt.legend()

# Show the plot
plt.show()


# In[78]:





# In[78]:




