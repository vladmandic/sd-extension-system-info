import os
import sys
import time
import json
import platform
import subprocess
import datetime
from hashlib import sha256
from html.parser import HTMLParser

import torch
import accelerate
import gradio as gr
import psutil
import transformers

from modules import paths, script_callbacks, sd_hijack, sd_models, sd_samplers, shared, extensions
from modules.ui_components import FormRow

from scripts.benchmark import run_benchmark, submit_benchmark # pylint: disable=E0401,E0611

### system info globals

data = {}

### benchmark globals

bench_text = ''
bench_file = os.path.join(os.path.dirname(__file__), 'benchmark-data-local.json')
bench_headers = ['timestamp', 'performance', 'version', 'system', 'libraries', 'gpu', 'optimizations', 'model', 'username', 'note', 'hash']
bench_data = []
console_logging = None


### system info module

def get_user():
    user = ''
    if user == '':
        try:
            user = os.getlogin()
        except:
            pass
    if user == '':
        try:
            import pwd
            user = pwd.getpwuid(os.getuid())[0]
        except:
            pass
    return user


def get_gpu():
    if not torch.cuda.is_available():
        return {}
    else:
        try:
            if torch.version.cuda:
                return {
                    'device': f'{torch.cuda.get_device_name(torch.cuda.current_device())} ({str(torch.cuda.device_count())}) ({torch.cuda.get_arch_list()[-1]}) {str(torch.cuda.get_device_capability(shared.device))}',
                    'cuda': torch.version.cuda,
                    'cudnn': torch.backends.cudnn.version(),
                }
            elif torch.version.hip:
                return {
                    'device': f'{torch.cuda.get_device_name(torch.cuda.current_device())} ({str(torch.cuda.device_count())})',
                    'hip': torch.version.hip,
                }
            else:
                return {
                    'device': 'unknown'
                }
        except Exception as e:
            return { 'error': e }

def get_uptime():
    s = vars(shared.state)
    return time.strftime('%c', time.localtime(s.get('server_start', time.time())))

class HTMLFilter(HTMLParser):
    text = ""
    def handle_data(self, data): # pylint: disable=redefined-outer-name
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
        import xformers # pylint: disable=import-outside-toplevel, import-error
        xversion = xformers.__version__
    except:
        xversion = 'unavailable'
    return {
        'xformers': xversion,
        'accelerate': accelerate.__version__,
        'transformers': transformers.__version__,
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
        if platform.system() == 'Windows':
            release = platform.platform(aliased = True, terse = False)
        else:
            release = platform.release()
        return {
            # 'host': platform.node(),
            'arch': platform.machine(),
            'cpu': platform.processor(),
            'system': platform.system(),
            'release': release,
            # 'platform': platform.platform(aliased = True, terse = False),
            # 'version': platform.version(),
            'python': platform.python_version(),
        }
    except Exception as e:
        return { 'error': e }


def get_torch():
    try:
        ver = torch.__long_version__
    except:
        ver = torch.__version__
    return f"{ver} {shared.cmd_opts.precision} {' nohalf' if shared.cmd_opts.no_half else ' half'}"


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
            'url': origin.replace('\n', '') + '/tree/' + branch.replace('\n', '')
        }
    except:
        return {}


def get_embeddings():
    return sorted([f'{v} ({sd_hijack.model_hijack.embedding_db.word_embeddings[v].vectors})' for i, v in enumerate(sd_hijack.model_hijack.embedding_db.word_embeddings)])


def get_skipped():
    return sorted([k for k in sd_hijack.model_hijack.embedding_db.skipped_embeddings.keys()])


def get_crossattention():
    try:
        ca = sd_hijack.model_hijack.optimization_method
        if ca is None:
            return 'none'
        else: return ca
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
        from Lora import lora # pylint: disable=E0401
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
        'gpu': get_gpu(),
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
    }
    return data


def get_quick_data():
    data['timestamp'] = datetime.datetime.now().strftime('%X')
    data['state'] = get_state()
    data['memory'] = get_memory()


def list2text(lst: list):
    return '\n'.join(lst)


def dict2str(d: dict):
    arr = [f'{name}:{d[name]}' for i, name in enumerate(d)]
    return ' '.join(arr)


def dict2text(d: dict):
    arr = ['{name}: {val}'.format(name = name, val = d[name] if not type(d[name]) is dict else dict2str(d[name])) for i, name in enumerate(d)] # pylint: disable=consider-using-f-string
    return list2text(arr)


def refresh_info_quick():
    get_quick_data()
    return dict2text(data['state']), dict2text(data['memory']), data['crossattention'], data['timestamp'], data


def refresh_info_full():
    get_full_data()
    return dict2text(data['state']), dict2text(data['memory']), data['crossattention'], data['models'], data['hypernetworks'], data['loras'], data['embeddings'], data['skipped'], data['timestamp'], data


### ui definition

def on_ui_tabs():
    get_full_data()
    with gr.Blocks(analytics_enabled = False) as system_info:
        with gr.Row(elem_id = 'system_info'):
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
                                gr.Textbox(data['torch'], label = 'Torch', lines = 1)
                                gr.Textbox(dict2text(data['gpu']), label = 'GPU', lines = len(data['gpu']))
                                with gr.Row():
                                    gr.Textbox(list2text(data['optimizations']), label = 'Memory optimization')
                                    crossattention = gr.Textbox(data['crossattention'], label = 'Cross-attention')
                            with gr.Column():
                                gr.Textbox(dict2text(data['libs']), label = 'Libs', lines = len(data['libs']))
                                gr.Textbox(dict2text(data['repos']), label = 'Repos', lines = len(data['repos']))
                with gr.Box():
                    with gr.Accordion('Benchmarks...', open = True, visible = True):
                        bench_load()
                        with gr.Row():
                            benchmark_data = gr.DataFrame(bench_data, label = 'Benchmark Data', elem_id = 'system_info_benchmark_data', show_label = True, interactive = False, wrap = True, overflow_row_behaviour = 'paginate', max_rows = 10, headers = bench_headers)
                        with gr.Row():
                            with gr.Column(scale=3):
                                username = gr.Textbox(get_user, label = 'Username', placeholder='enter username for submission', elem_id='system_info_tab_username')
                                note = gr.Textbox('', label = 'Note', placeholder='enter any additional notes', elem_id='system_info_tab_note')
                            with gr.Column(scale=1):
                                with FormRow():
                                    global console_logging # pylint: disable=global-statement
                                    console_logging = gr.Checkbox(label = 'Console logging', value = False, elem_id = 'system_info_tab_console', interactive = True)
                                    warmup = gr.Checkbox(label = 'Perform warmup', value = True, elem_id = 'system_info_tab_warmup')
                                    extra = gr.Checkbox(label = 'Extra steps', value = False, elem_id = 'system_info_tab_extra')
                                level = gr.Radio(['quick', 'normal', 'extensive'], value = 'normal', label = 'Benchmark level', elem_id = 'system_info_tab_level')
                                # batches = gr.Textbox('1, 2, 4, 8', label = 'Batch sizes', elem_id = 'system_info_tab_batch_size', interactive = False)
                            with gr.Column(scale=1):
                                bench_run_btn = gr.Button('Run benchmark', elem_id = 'system_info_tab_benchmark_btn').style(full_width = False)
                                bench_run_btn.click(bench_init, inputs = [username, note, warmup, level, extra], outputs = [benchmark_data])
                                bench_submit_btn = gr.Button('Submit results', elem_id = 'system_info_tab_submit_btn').style(full_width = False)
                                bench_submit_btn.click(bench_submit, inputs = [username], outputs = [])
                                _bench_link = gr.HTML('<a href="https://vladmandic.github.io/sd-extension-system-info/pages/benchmark.html" target="_blank">Link to online results</a>')
                        with gr.Row():
                            _bench_note = gr.HTML(elem_id = 'system_info_tab_bench_note', value = """
                                <span>performance is measured in iterations per second (it/s) and reported for different batch sizes (e.g. 1, 2, 4, 8, 16...)</span><br>
                                <span>running benchmark may take a while. extensive tests may result in gpu out-of-memory conditions.</span>""")
                        with gr.Row():
                            bench_label = gr.HTML('', elem_id = 'system_info_tab_bench_label')
                            refresh_bench_btn = gr.Button('Refresh bench', elem_id = 'system_info_tab_refresh_bench_btn', visible = False).style(full_width = False) # quick refresh is used from js interval
                            refresh_bench_btn.click(bench_refresh, inputs = [], outputs = [bench_label])
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
                        js = gr.JSON(data)
            with gr.Column(scale = 1, min_width = 120):
                timestamp = gr.Text(default=data['timestamp'], label = '', elem_id = 'system_info_tab_last_update')
                refresh_quick_btn = gr.Button('Refresh state', elem_id = 'system_info_tab_refresh_btn', visible = False).style(full_width = False) # quick refresh is used from js interval
                refresh_quick_btn.click(refresh_info_quick, inputs = [], outputs = [state, memory, crossattention, timestamp, js], show_progress = False)
                refresh_full_btn = gr.Button('Refresh data', elem_id = 'system_info_tab_refresh_full_btn', variant='primary').style(full_width = False)
                refresh_full_btn.click(refresh_info_full, inputs = [], outputs = [state, memory, crossattention, models, hypernetworks, loras, embeddings, skipped, timestamp, js])
                interrupt_btn = gr.Button('Send interrupt', elem_id = 'system_info_tab_interrupt_btn', variant='primary')
                interrupt_btn.click(shared.state.interrupt, inputs = [], outputs = [])
    return (system_info, 'System Info', 'system_info'),


### benchmarking module

def bench_log(msg: str):
    global bench_text # pylint: disable=global-statement
    bench_text = msg
    if console_logging is not None and console_logging.value:
        print('benchmark', msg)


def bench_submit(username: str):
    if username is None or username == '':
        bench_log('username is required to submit results')
        return
    submit_benchmark(bench_data, username, console_logging.value)
    bench_log(f'data submitted: {len(bench_data)} records')


def bench_run(batches: list = [1], extra: bool = False):
    results = []
    for batch in batches:
        bench_log(f'running for batch size {batch}')
        res = run_benchmark(batch, extra)
        bench_log(f'results batch size {batch}: {res} it/s')
        results.append(str(res))
    its = ' / '.join(results)
    return its


def bench_init(username: str, note: str, warmup: bool, level: str, extra: bool):
    bench_log('starting')
    hash256 = sha256((dict2str(data['platform']) + data['torch'] + dict2str(data['libs']) + dict2str(data['gpu']) + ','.join(data['optimizations']) + data['crossattention']).encode('utf-8')).hexdigest()[:6]
    existing = [x for x in bench_data if (x[-1] is not None and x[-1][:6] == hash256)]
    if len(existing) > 0:
        bench_log('replacing existing entry')
        d = existing[0]
    elif bench_data[-1][0] is not None:
        bench_log('new entry')
        bench_data.append([None] * len(bench_headers))
        d = bench_data[-1]
    else:
        d = bench_data[-1]

    if level == 'quick':
        batches = [1]
    elif level == 'normal':
        batches = [1, 2, 4]
    elif level == 'extensive':
        batches = [1, 2, 4, 8, 16]
    else:
        batches = []

    model_hash = shared.opts.data['sd_model_checkpoint'].split('[')[-1].split(']')[0]
    if model_hash != 'cc6cb27103':
        bench_log('using non standard model')

    if warmup:
        bench_run([1], False)

    try:
        mem = data['memory']['gpu']['total']
    except:
        mem = 0

    # bench_headers = ['timestamp', 'performance', 'version', 'system', 'libraries', 'gpu', 'optimizations', 'model', 'username', 'note', 'hash']
    d[0] = str(datetime.datetime.now())
    d[1] = bench_run(batches, extra)
    d[2] = dict2str(data['version'])
    d[3] = dict2str(data['platform'])
    d[4] = f"torch:{data['torch']} {dict2str(data['libs'])}"
    d[5] = dict2str(data['gpu']) + f' {str(round(mem))}GB'
    d[6] = data['crossattention'] + ' ' + ','.join(data['optimizations'])
    d[7] = shared.opts.data['sd_model_checkpoint']
    d[8] = username
    d[9] = note
    d[10] = hash256

    md = '| ' + ' | '.join(d) + ' |'
    bench_log(md)

    bench_save()
    return bench_data


def bench_load():
    global bench_data # pylint: disable=global-statement
    tmp = []
    if os.path.isfile(bench_file) and os.path.getsize(bench_file) > 0:
        try:
            with open(bench_file, 'r', encoding='utf-8') as f:
                tmp = json.load(f)
                bench_data = tmp
                bench_log('data loaded: ' + bench_file)
        except Exception as err:
            bench_log('error loading: ' + bench_file + ' ' + str(err))
    if len(bench_data) == 0:
        bench_data.append([None] * len(bench_headers))
    return bench_data


def bench_save():
    if bench_data[-1][0] is None:
        del bench_data[-1]
    try:
        with open(bench_file, 'w', encoding='utf-8') as f:
            json.dump(bench_data, f, indent=2, default=str, skipkeys=True)
            bench_log('data saved: ' + bench_file)
    except Exception as err:
        bench_log('error saving: ' + bench_file + ' ' + str(err))


def bench_refresh():
    return gr.HTML.update(value = bench_text)


script_callbacks.on_ui_tabs(on_ui_tabs)
