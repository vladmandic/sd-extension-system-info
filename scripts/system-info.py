import datetime
import os
import sys
import platform
import subprocess
import time
from html.parser import HTMLParser

import accelerate
import gradio as gr
import psutil
import pytorch_lightning
import safetensors
import torch
import transformers
from modules import paths, script_callbacks, sd_hijack, sd_models, sd_samplers, shared, extensions

data = {}

def get_cuda():
    if not torch.cuda.is_available():
        return {}
    else:
        try:
            return {
                'version': torch.version.cuda,
                'devices': torch.cuda.device_count(),
                'current': torch.cuda.get_device_name(torch.cuda.current_device()),
                'arch': torch.cuda.get_arch_list()[-1],
                'capability': torch.cuda.get_device_capability(shared.device),
                'cudnn': torch.backends.cudnn.version(),
            }
        except Exception as e:
            return { 'error': e }

def get_uptime():
    s = vars(shared.state)
    return time.strftime('%c', time.localtime(s.get('server_start', time.time())))

class HTMLFilter(HTMLParser):
    text = ""
    def handle_data(self, data):
        self.text += data

def get_state():
    s = vars(shared.state)
    flags = 'skipped ' if s.get('skipped', False) else ''
    flags += 'interrupted ' if s.get('interrupted', False) else ''
    flags += 'needs restart' if s.get('need_restart', False) else ''
    text = s.get('textinfo', '')
    if text is not None and len(text) > 0:
        f = HTMLFilter()
        f.feed(text)
        text = os.linesep.join([s for s in f.text.splitlines() if s])
    return {
        'started': time.strftime('%c', time.localtime(s.get('time_start', time.time()))),
        'step': f'{s.get("sampling_step", 0)} / {s.get("sampling_steps", 0)}',
        'jobs': f'{s.get("job_no", 0)} / {s.get("job_count", 0)}', # pylint: disable=consider-using-f-string
        'flags': flags,
        'job': s.get('job', ''),
        'text-info': text,
    }

def get_memory():
    def gb(val: float):
        return round(val / 1024 / 1024 / 1024, 2)
    mem = {}
    try:
        process = psutil.Process(os.getpid())
        res = process.memory_info()
        ram_total = 100 * res.rss / process.memory_percent()
        ram = { 'free': gb(ram_total - res.rss), 'used': gb(res.rss), 'total': gb(ram_total) }
        mem.update({ 'ram': ram })
    except Exception as e:
        mem.update({ 'ram': e })
    try:
        if torch.cuda.is_available():
            s = torch.cuda.mem_get_info()
            gpu = { 'free': gb(s[0]), 'used': gb(s[1] - s[0]), 'total': gb(s[1]) }
            s = dict(torch.cuda.memory_stats(shared.device))
            allocated = { 'current': gb(s['allocated_bytes.all.current']), 'peak': gb(s['allocated_bytes.all.peak']) }
            reserved = { 'current': gb(s['reserved_bytes.all.current']), 'peak': gb(s['reserved_bytes.all.peak']) }
            active = { 'current': gb(s['active_bytes.all.current']), 'peak': gb(s['active_bytes.all.peak']) }
            inactive = { 'current': gb(s['inactive_split_bytes.all.current']), 'peak': gb(s['inactive_split_bytes.all.peak']) }
            warnings = { 'retries': s['num_alloc_retries'], 'oom': s['num_ooms'] }
            mem.update({
                'gpu': gpu,
                'gpu-active': active,
                'gpu-allocated': allocated,
                'gpu-reserved': reserved,
                'gpu-inactive': inactive,
                'events': warnings,
            })
    except:
        pass
    return mem

def get_optimizations():
    ram = []
    if shared.cmd_opts.medvram:
        ram.append('medvram')
    if shared.cmd_opts.lowvram:
        ram.append('lowvram')
    if shared.cmd_opts.lowram:
        ram.append('lowram')
    if len(ram) == 0:
        ram.append('none')
    return ram

def get_libs():
    try:
        import xformers # pylint: disable=import-outside-toplevel
        xversion = xformers.__version__
    except:
        xversion = 'unavailable'
    return {
        'xformers': xversion,
        'accelerate': accelerate.__version__,
        'transformers': transformers.__version__,
        'safetensors': safetensors.__version__,
        'lightning': pytorch_lightning.__version__,
    }

def get_repos():
    repos = {}
    for key, val in paths.paths.items():
        try:
            cmd = f'git -C {val} log --pretty=format:"%h %ad" -1 --date=short'
            res = subprocess.run(f'{cmd} {val}', stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True, check=True)
            stdout = res.stdout.decode(encoding = 'utf8', errors='ignore') if len(res.stdout) > 0 else ''
            words = stdout.split(' ')
            repos[key] = f'[{words[0]}] {words[1]}'
        except:
            repos[key] = '(unknown)'
    return repos

def get_platform():
    try:
        return {
            'host': platform.node(),
            'arch': platform.machine(),
            'cpu': platform.processor(),
            'system': platform.system(),
            'platform': platform.platform(aliased = True, terse = False),
            'release': platform.release(),
            'version': platform.version(),
            'python': platform.python_version(),
        }
    except Exception as e:
        return { 'error': e }

def get_torch():
    return {
        'version': torch.__version__,
        'precision': shared.cmd_opts.precision + (' fp32' if shared.cmd_opts.no_half else ' fp16'),
    }

def get_version():
    try:
        res = subprocess.run('git log --pretty=format:"%h %ad" -1 --date=short', stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True, check=True)
        ver = res.stdout.decode(encoding = 'utf8', errors='ignore') if len(res.stdout) > 0 else ''
        githash, updated = ver.split(' ')
        res = subprocess.run('git remote get-url origin', stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True, check=True)
        origin = res.stdout.decode(encoding = 'utf8', errors='ignore') if len(res.stdout) > 0 else ''
        res = subprocess.run('git branch --show-current', stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True, check=True)
        branch = res.stdout.decode(encoding = 'utf8', errors='ignore') if len(res.stdout) > 0 else ''
        return {
            'updated': updated,
            'hash': githash,
            'origin': origin.replace('\n', ''),
            'branch': branch.replace('\n', ''),
        }
    except:
        return {}

def get_embeddings():
    return sorted([f'{v} ({sd_hijack.model_hijack.embedding_db.word_embeddings[v].vectors})' for i, v in enumerate(sd_hijack.model_hijack.embedding_db.word_embeddings)])

def get_skipped():
    return sorted([k for k in sd_hijack.model_hijack.embedding_db.skipped_embeddings.keys()])

def get_crossattention():
    try:
        return sd_hijack.model_hijack.optimization_method
    except:
        return 'unknown'

def get_models():
    return sorted([x.title for x in sd_models.checkpoints_list.values()])

def get_samplers():
    return sorted([sampler[0] for sampler in sd_samplers.all_samplers])

def get_extensions():
    return sorted([f"{e.name} ({'enabled' if e.enabled else 'disabled'}{' builtin' if e.is_builtin else ''})" for e in extensions.extensions])

def get_loras():
    loras = []
    try:
        sys.path.append(extensions.extensions_builtin_dir)
        from Lora import lora
        loras = sorted([l for l in lora.available_loras.keys()])
    except:
        pass
    return loras

def get_full_data():
    global data # pylint: disable=global-statement
    data = {
        'date': datetime.datetime.now().strftime('%c'),
        'timestamp': datetime.datetime.now().strftime('%X'),
        'uptime': get_uptime(),
        'version': get_version(),
        'torch': get_torch(),
        'cuda': get_cuda(),
        'state': get_state(),
        'memory': get_memory(),
        'optimizations': get_optimizations(),
        'libs': get_libs(),
        'repos': get_repos(),
        'models': get_models(),
        'hypernetworks': [name for name in shared.hypernetworks],
        'embeddings': get_embeddings(),
        'skipped': get_skipped(),
        'loras': get_loras(),
        'schedulers': get_samplers(),
        'extensions': get_extensions(),
        'platform': get_platform(),
        'crossattention': get_crossattention(),
        'api': shared.cmd_opts.api,
        'webui': not shared.cmd_opts.nowebui,
    }
    return data

def get_quick_data():
    data['timestamp'] = datetime.datetime.now().strftime('%X')
    data['state'] = get_state()
    data['memory'] = get_memory()

def list2text(lst: list):
    return '\n'.join(lst)

def dict2str(d: dict):
    arr = [f'{name}: {d[name]}' for i, name in enumerate(d)]
    return ' '.join(arr)

def dict2text(d: dict):
    arr = ['{name}: {val}'.format(name = name, val = d[name] if not type(d[name]) is dict else dict2str(d[name])) for i, name in enumerate(d)] # pylint: disable=consider-using-f-string
    return list2text(arr)

def refresh_info_quick():
    get_quick_data()
    return dict2text(data['state']), dict2text(data['memory']), data['timestamp'], data

def refresh_info_full():
    get_full_data()
    return dict2text(data['state']), dict2text(data['memory']), data['models'], data['hypernetworks'], data['loras'], data['embeddings'], data['skipped'], dict2text(data['model']), dict2text(data['vae']), data['timestamp'], data

def on_ui_tabs():
    get_full_data()
    with gr.Blocks(analytics_enabled = False) as system_info_tab:
        with gr.Row(elem_id = 'system_info_tab'):
            with gr.Column(scale = 9):
                with gr.Box():
                    with gr.Row():
                        with gr.Column():
                            gr.Textbox(data['uptime'], label = 'Server start time', lines = 1)
                            gr.Textbox(dict2text(data['version']), label = 'Version', lines = len(data['version']))
                        with gr.Column():
                            state = gr.Textbox(dict2text(data['state']), label = 'State', lines = len(data['state']))
                        with gr.Column():
                            memory = gr.Textbox(dict2text(data['memory']), label = 'Memory', lines = len(data['memory']))
                with gr.Box():
                    with gr.Accordion('System data', open = True, visible = True):
                        with gr.Row():
                            with gr.Column():
                                gr.Textbox(dict2text(data['platform']), label = 'Platform', lines = len(data['platform']))
                            with gr.Column():
                                gr.Textbox(dict2text(data['torch']), label = 'Torch', lines = len(data['torch']))
                                gr.Textbox(dict2text(data['cuda']), label = 'CUDA', lines = len(data['cuda']))
                                with gr.Row():
                                    gr.Textbox(list2text(data['optimizations']), label = 'Memory optimization')
                                    gr.Textbox(data['crossattention'], label = 'Cross-attention')
                                    gr.Textbox((data['api']), label = 'API')
                            with gr.Column():
                                gr.Textbox(dict2text(data['libs']), label = 'Libs', lines = len(data['libs']))
                                gr.Textbox(dict2text(data['repos']), label = 'Repos', lines = len(data['repos']))
                with gr.Box():
                    with gr.Accordion('Models...', open = False, visible = True):
                        with gr.Row():
                            with gr.Column():
                                models = gr.JSON(data['models'], label = 'Models', lines = len(data['models']))
                                hypernetworks = gr.JSON(data['hypernetworks'], label = 'Hypernetworks', lines = len(data['hypernetworks']))
                            with gr.Column():
                                embeddings = gr.JSON(data['embeddings'], label = 'Embeddings: loaded', lines = len(data['embeddings']))
                                skipped = gr.JSON(data['skipped'], label = 'Embeddings: skipped', lines = len(data['embeddings']))
                                loras = gr.JSON(data['loras'], label = 'Available LORAs', lines = len(data['loras']))
                with gr.Box():
                    with gr.Accordion('Info object', open = False, visible = True):
                        # reduce json data to avoid private info
                        data.pop('models', None)
                        data.pop('embeddings', None)
                        data.pop('skipped', None)
                        data.pop('hypernetworks', None)
                        data.pop('schedulers', None)
                        data.pop('loras', None)
                        json = gr.JSON(data)
            with gr.Column(scale = 1, min_width = 120):
                timestamp = gr.Text(data['timestamp'], label = '', elem_id = 'system_info_tab_last_update')
                refresh_quick = gr.Button('Refresh state', elem_id = 'system_info_tab_refresh_btn', visible = False).style(full_width = False) # quick refresh is used from js interval
                refresh_quick.click(refresh_info_quick, inputs = [], outputs = [state, memory, timestamp, json])
                refresh_full = gr.Button('Refresh data', elem_id = 'system_info_tab_refresh_full_btn').style(full_width = False)
                refresh_full.click(refresh_info_full, inputs = [], outputs = [state, memory, models, hypernetworks, loras, embeddings, skipped, timestamp, json])
                interrupt = gr.Button('Send interrupt', elem_id = 'system_info_tab_interrupt_btn')
                interrupt.click(shared.state.interrupt, inputs = [], outputs = [])
    return (system_info_tab, 'System Info', 'system_info_tab'),

script_callbacks.on_ui_tabs(on_ui_tabs)
