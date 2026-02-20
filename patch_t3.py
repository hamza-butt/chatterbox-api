import torch
import chatterbox.models.t3.t3 as t3_module
from tqdm import tqdm

# We store the original T3.inference method just in case, but we rewrite it to disable output_attentions=True
original_inference = getattr(t3_module.T3, "inference")

def patched_inference(self, *args, **kwargs):
    # This is a hacky way to override the default arguments in the generate loop if we need to.
    # However, since the generation loop inside T3.inference explicitly hardcodes `output_attentions=True`,
    # the easiest way to override is to patch out the patched_model object __call__.
    pass

