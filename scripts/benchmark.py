import time
from modules import shared
from modules.processing import StableDiffusionProcessingTxt2Img, Processed, process_images

args = {
    'sd_model': shared.sd_model,
    'prompt': 'postapocalyptic steampunk city, exploration, cinematic, realistic, hyper detailed, photorealistic maximum detail, volumetric light, (((focus))), wide-angle, (((brightly lit))), (((vegetation))), lightning, vines, destruction, devastation, wartorn, ruins',
    'sampler_name': 'Euler a',
    'batch_size': 1,
    'n_iter': 1,
    'steps': 10,
    'cfg_scale': 15.0,
    'width': 512,
    'height': 512,
    'restore_faces': False,
    'tiling': False,
    'do_not_save_samples': True,
    'do_not_save_grid': True,
    'negative_prompt': '(((blurry))), ((foggy)), (((dark))), ((monochrome)), sun, (((depth of field)))',
    'do_not_reload_embeddings': True
}

def run_benchmark(batch: int, extra: bool):
    shared.state.begin()
    args['batch_size'] = batch
    args['steps'] = 20 if not extra else 50
    p = StableDiffusionProcessingTxt2Img(**args)
    t0 = time.time()
    try:
        processed: Processed = process_images(p)
    except Exception as e:
        print(f'benchmark error: {batch} {e}')
        return 'error'
    t1 = time.time()
    shared.state.end()
    its = args['steps'] * batch / (t1 - t0)
    return round(its, 2)
