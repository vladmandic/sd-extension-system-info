import time
import logging
from modules import shared
from modules.processing import StableDiffusionProcessingTxt2Img, Processed, process_images


log = logging.getLogger('sd')

args = {
    'sd_model': None,
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
    shared.state.begin('benchmark')
    if args['sd_model'] is None:
        args['sd_model'] = shared.sd_model
    args['batch_size'] = batch
    args['steps'] = 20 if not extra else 50
    mp = 0
    p = StableDiffusionProcessingTxt2Img(**args)
    t0 = time.time()
    try:
        processed: Processed = process_images(p)
        if processed is None or processed.images is None:
            log.error(f'SD-System-Info benchmark error: {batch} no results')
            return 'error'
        if len(processed.images) != batch:
            log.error(f'SD-System-Info benchmark error: {batch} results mismatch')
            return 'error'
        for image in processed.images:
            if image.width * image.height < 65536:
                log.error(f'SD-System-Info benchmark error: {batch} image too small')
                return 'error'
            mp += image.width * image.height
    except Exception as e:
        log.error(f'SD-System-Info benchmark error: {batch} {e}')
        return 'error'
    t1 = time.time()
    shared.state.end()
    its = args['steps'] * batch / (t1 - t0)
    mp = mp / 1024 / 1024
    mps = mp / (t1 - t0)
    log.debug(f'SD-System-Info benchmark: batch={batch} time={t1-t0:.2f} steps={args["steps"]} its={its:.2f} mps={mps:.2f}')
    if its > 300:
        log.error(f'SD-System-Info benchmark: its={its:.2f} too high')
        return 'error'
    return round(its, 2)


class LogFilter(logging.Filter):
    import socket
    hostname = socket.gethostname()
    def filter(self, record):
        record.hostname = LogFilter.hostname
        return True


def submit_benchmark(data, username):
    from logging.handlers import SysLogHandler
    from hashlib import sha256

    syslog = SysLogHandler(address=('logs3.papertrailapp.com', 32554))
    syslog.addFilter(LogFilter())
    formatter = logging.Formatter(f'%(asctime)s %(hostname)s SDBENCHMARK: {username} %(message)s', datefmt='%b %d %H:%M:%S')
    syslog.setFormatter(formatter)
    remote = logging.getLogger('SDBENCHMARK')
    remote.setLevel(logging.INFO)
    for h in remote.handlers: # remove local handlers
        remote.removeHandler(h)
    remote.addHandler(syslog)
    for line in data:
        message = '|'.join(line).replace('  ', ' ').replace('"', '').strip()
        hash256 = sha256(message.encode('utf-8')).hexdigest()[:6]
        message = message + '|' + hash256
        remote.info(message)
