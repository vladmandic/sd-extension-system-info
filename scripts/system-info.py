import os
import sys
import time
import json
import platform
import subprocess
import datetime
import logging
from html.parser import HTMLParser
import torch
import gradio as gr
from modules import paths, script_callbacks, sd_models, sd_samplers, shared, extensions, devices
import benchmark # pylint: disable=wrong-import-order


### system info globals

log = logging.getLogger('sd')
data = {
    'date': '',
    'timestamp': '',
    'uptime': '',
    'version': {},
    'torch': '',
    'gpu': {},
    'state': {},
    'memory': {},
    'flags': [],
    'libs': {},
    'repos': {},
    'device': {},
    'schedulers': [],
    'extensions': [],
    'platform': '',
    'crossattention': '',
    'backend': getattr(devices, 'backend', ''),
    'pipeline': ('native' if shared.native else 'original') if hasattr(shared, 'native') else 'a1111',
    'model': {},
}
networks = {
    'models': [],
    'hypernetworks': [],
    'embeddings': [],
    'skipped': [],
    'loras': [],
    'lycos': [],
}

### benchmark globals

bench_text = ''
bench_file = os.path.join(os.path.dirname(__file__), 'benchmark-data-local.json')
bench_headers = ['timestamp', 'it/s', 'version', 'system', 'libraries', 'gpu', 'flags', 'settings', 'username', 'note', 'hash']
bench_data = []

### system info module

def get_user():
    user = ''
    if user == '':
        try:
            user = os.getlogin()
        except Exception:
            pass
    if user == '':
        try:
            import pwd
            user = pwd.getpwuid(os.getuid())[0]
        except Exception:
            pass
    return user


def get_gpu():
    if not torch.cuda.is_available():
        try:
            if shared.cmd_opts.use_openvino:
                from modules.intel.openvino import get_openvino_device
                return {
                    'device': get_openvino_device(),
                    'openvino': get_package_version("openvino")
                }
            else:
                return {}
        except Exception:
            return {}
    else:
        try:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return {
                    'device': f'{torch.xpu.get_device_name(torch.xpu.current_device())} ({str(torch.xpu.device_count())})',
                    'ipex': get_package_version('intel-extension-for-pytorch'),
                }
            elif torch.version.cuda:
                return {
                    'device': f'{torch.cuda.get_device_name(torch.cuda.current_device())} ({str(torch.cuda.device_count())}) ({torch.cuda.get_arch_list()[-1]}) {str(torch.cuda.get_device_capability(shared.device))}',
                    'cuda': torch.version.cuda,
                    'cudnn': torch.backends.cudnn.version(),
                    'driver': get_driver(),
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


def get_driver():
    if torch.cuda.is_available() and torch.version.cuda:
        try:
            result = subprocess.run('nvidia-smi --query-gpu=driver_version --format=csv,noheader', shell=True, check=False, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            version = result.stdout.decode(encoding="utf8", errors="ignore").strip()
            return version
        except Exception:
            return ''
    else:
        return ''


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


def get_docker_limit():
    try:
        with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r', encoding='utf8') as f:
            docker_limit = float(f.read())
    except Exception:
        docker_limit = sys.float_info.max
    return docker_limit


def get_memory():
    def gb(val: float):
        return round(val / 1024 / 1024 / 1024, 2)
    mem = {}
    try:
        import psutil
        process = psutil.Process(os.getpid())
        res = process.memory_info()
        ram_total = 100 * res.rss / process.memory_percent()
        ram_total = min(ram_total, get_docker_limit())
        ram = { 'free': gb(ram_total - res.rss), 'used': gb(res.rss), 'total': gb(ram_total) }
        mem.update({ 'ram': ram })
    except Exception as e:
        mem.update({ 'ram': e })
    if torch.cuda.is_available():
        try:
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
                'utilization': 0,
            })
            mem.update({ 'utilization': torch.cuda.utilization() }) # do this one separately as it may fail
        except Exception:
            pass
    else:
        try:
            from openvino.runtime import Core as OpenVINO_Core
            from modules.intel.openvino import get_device as get_raw_openvino_device
            openvino_core = OpenVINO_Core()
            mem.update({
                'gpu': { 'total': gb(openvino_core.get_property(get_raw_openvino_device(), 'GPU_DEVICE_TOTAL_MEM_SIZE')) },
            })
        except Exception:
            pass
    return mem


def get_flags():
    ram = []
    if getattr(shared.cmd_opts, 'medvram', False):
        ram.append('medvram')
    if getattr(shared.cmd_opts, 'medvram_sdxl', False):
        ram.append('medvram-sdxl')
    if getattr(shared.cmd_opts, 'lowvram', False):
        ram.append('lowvram')
    if getattr(shared.cmd_opts, 'lowvam', False):
        ram.append('lowram')
    if len(ram) == 0:
        ram.append('none')
    return ram


def get_package_version(pkg: str):
    import pkg_resources
    spec = pkg_resources.working_set.by_key.get(pkg, None) # more reliable than importlib
    version = pkg_resources.get_distribution(pkg).version if spec is not None else ''
    return version


def get_libs():
    return {
        'xformers': get_package_version('xformers'),
        'diffusers': get_package_version('diffusers'),
        'transformers': get_package_version('transformers'),
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
        except Exception:
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
    except Exception:
        ver = torch.__version__
    return f"{ver}"


def get_version():
    version = {}
    try:
        res = subprocess.run('git log --pretty=format:"%h %ad" -1 --date=short', stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True, check=True)
        ver = res.stdout.decode(encoding = 'utf8', errors='ignore') if len(res.stdout) > 0 else ''
        githash, updated = ver.split(' ')
        res = subprocess.run('git remote get-url origin', stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True, check=True)
        origin = res.stdout.decode(encoding = 'utf8', errors='ignore') if len(res.stdout) > 0 else ''
        res = subprocess.run('git rev-parse --abbrev-ref HEAD', stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True, check=True)
        branch = res.stdout.decode(encoding = 'utf8', errors='ignore') if len(res.stdout) > 0 else ''
        url = origin.replace('\n', '') + '/tree/' + branch.replace('\n', '')
        app = origin.replace('\n', '').split('/')[-1]
        if app == 'automatic':
            app = 'SD.next'
        version = {
            'app': app,
            'updated': updated,
            'hash': githash,
            'url': url
        }
    except Exception:
        pass
    return version


def get_crossattention():
    try:
        ca = getattr(shared.opts, 'cross_attention_optimization', None)
        if ca is None:
            from modules import sd_hijack
            ca = sd_hijack.model_hijack.optimization_method
        return ca
    except Exception:
        return 'unknown'


def get_model():
    obj = {
        'base': shared.opts.data.get('sd_model_checkpoint', 'none'),
        'refiner': shared.opts.data.get('sd_model_refiner', 'none'),
        'vae': shared.opts.data.get('sd_vae', 'none'),
        'te': shared.opts.data.get('sd_text_encoder', 'none'),
        'unet': shared.opts.data.get('sd_unet', 'none'),
    }
    return obj


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
        loras = sorted(lora.available_loras.keys())
    except Exception:
        pass
    return loras


def get_device():
    dev = {
        'active': str(devices.device),
        'dtype': str(devices.dtype),
        'vae': str(devices.dtype_vae),
        'unet': str(devices.dtype_unet),
    }
    return dev


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
        'flags': get_flags(),
        'libs': get_libs(),
        'repos': get_repos(),
        'device': get_device(),
        'model': get_model(),
        'schedulers': get_samplers(),
        'extensions': get_extensions(),
        'platform': get_platform(),
        'crossattention': get_crossattention(),
        'backend': getattr(devices, 'backend', ''),
        'pipeline': ('native' if shared.native else 'original') if hasattr(shared, 'native') else 'a1111',
    }
    global networks # pylint: disable=global-statement
    networks = {
        'models': get_models(),
        'loras': get_loras(),
    }
    return data


def get_quick_data():
    data['timestamp'] = datetime.datetime.now().strftime('%X')
    data['state'] = get_state()
    data['memory'] = get_memory()
    data['model'] = get_model()


def list2text(lst: list):
    return '\n'.join(lst)


def dict2str(d: dict):
    arr = [f'{name}:{d[name]}' for i, name in enumerate(d)]
    return ' '.join(arr)


def dict2text(d: dict):
    arr = ['{name}: {val}'.format(name = name, val = d[name] if type(d[name]) is not dict else dict2str(d[name])) for i, name in enumerate(d)] # pylint: disable=consider-using-f-string
    return list2text(arr)


def refresh_info_quick(_old_data = None):
    get_quick_data()
    return dict2text(data['state']), dict2text(data['memory']), data['crossattention'], data['timestamp'], data


def refresh_info_full():
    get_full_data()
    return data['uptime'], dict2text(data['version']), dict2text(data['state']), dict2text(data['memory']), dict2text(data['platform']), data['torch'], dict2text(data['gpu']), list2text(data['flags']), data['crossattention'], data['backend'], data['pipeline'], dict2text(data['libs']), dict2text(data['repos']), dict2text(data['device']), dict2text(data['model']), networks['models'], networks['loras'], data['timestamp'], data


### ui definition

def create_ui(blocks: gr.Blocks = None):
    try:
        if shared.cmd_opts.api_only:
            return []
    except Exception:
        pass
    if not standalone:
        from modules.ui import ui_system_tabs # pylint: disable=redefined-outer-name
    else:
        ui_system_tabs = None

    with gr.Blocks(analytics_enabled = False) if standalone else blocks as system_info:
        with gr.Row(elem_id = 'system_info'):
            with ui_system_tabs or gr.Tabs(elem_id = 'system_info_tabs'):
                with gr.TabItem('System Info'):
                    with gr.Row():
                        timestamp = gr.Textbox(value=data['timestamp'], label = '', elem_id = 'system_info_tab_last_update', container=False)
                        refresh_quick_btn = gr.Button('Refresh state', elem_id = 'system_info_tab_refresh_btn', visible = False) # quick refresh is used from js interval
                        refresh_full_btn = gr.Button('Refresh data', elem_id = 'system_info_tab_refresh_full_btn', variant='primary')
                        interrupt_btn = gr.Button('Send interrupt', elem_id = 'system_info_tab_interrupt_btn', variant='primary')
                    with gr.Row():
                        with gr.Column():
                            uptimetxt = gr.Textbox(data['uptime'], label = 'Server start time', lines = 1)
                            versiontxt = gr.Textbox(dict2text(data['version']), label = 'Version', lines = len(data['version']))
                        with gr.Column():
                            statetxt = gr.Textbox(dict2text(data['state']), label = 'State', lines = len(data['state']))
                        with gr.Column():
                            memorytxt = gr.Textbox(dict2text(data['memory']), label = 'Memory', lines = len(data['memory']))
                    with gr.Row():
                        with gr.Column():
                            platformtxt = gr.Textbox(dict2text(data['platform']), label = 'Platform', lines = len(data['platform']))
                            with gr.Row():
                                backendtxt = gr.Textbox(data['backend'], label = 'Backend')
                                pipelinetxt = gr.Textbox(data['pipeline'], label = 'Pipeline')
                        with gr.Column():
                            torchtxt = gr.Textbox(data['torch'], label = 'Torch', lines = 1)
                            gputxt = gr.Textbox(dict2text(data['gpu']), label = 'GPU', lines = len(data['gpu']))
                            with gr.Row():
                                opttxt = gr.Textbox(list2text(data['flags']), label = 'Memory optimization')
                                attentiontxt = gr.Textbox(data['crossattention'], label = 'Cross-attention')
                        with gr.Column():
                            libstxt = gr.Textbox(dict2text(data['libs']), label = 'Libs', lines = len(data['libs']))
                            repostxt = gr.Textbox(dict2text(data['repos']), label = 'Repos', lines = len(data['repos']), visible = False)
                            devtxt = gr.Textbox(dict2text(data['device']), label = 'Device Info', lines = len(data['device']))
                            modeltxt = gr.Textbox(dict2text(data['model']), label = 'Model Info', lines = len(data['model']))
                    with gr.Row():
                        gr.HTML('Load<br><div id="si-sparkline-load"></div>')
                        gr.HTML('Memory<br><div id="si-sparkline-memo"></div>')
                    with gr.Accordion('Info object', open = False, visible = True):
                        # reduce json data to avoid private info
                        refresh_info_quick()
                        js = gr.JSON(data)

                with gr.TabItem('Benchmark'):
                    bench_load()
                    with gr.Row():
                        benchmark_data = gr.DataFrame(bench_data, label = 'Benchmark Data', elem_id = 'system_info_benchmark_data', show_label = True, interactive = False, wrap = True, overflow_row_behaviour = 'paginate', max_rows = 10, headers = bench_headers)
                    with gr.Row():
                        with gr.Column(scale=1):
                            username = gr.Textbox(get_user, label = 'Username', placeholder='enter username for submission', elem_id='system_info_tab_username')
                        with gr.Column(scale=6):
                            note = gr.Textbox('', label = 'Note', placeholder='enter any additional notes', elem_id='system_info_tab_note')
                    with gr.Row():
                        with gr.Accordion('Settings'):
                            warmup = gr.Checkbox(label = 'Perform warmup', value = True, elem_id = 'system_info_tab_warmup')
                            with gr.Row():
                                steps = gr.Dropdown(['turbo', 'default', 'long'], value = 'normal', label = 'Benchmark steps', elem_id = 'system_info_tab_steps')
                                level = gr.Dropdown(['quick', 'normal', 'extensive', 'ludicrous'], value = 'normal', label = 'Benchmark level', elem_id = 'system_info_tab_level')
                            with gr.Row():
                                width = gr.Number(512, label = 'Image width', elem_id = 'system_info_tab_width')
                                height = gr.Number(512, label = 'Image height', elem_id = 'system_info_tab_height')
                        with gr.Column():
                            bench_run_btn = gr.Button('Run benchmark', elem_id = 'system_info_tab_benchmark_btn', variant='primary')
                            bench_run_btn.click(bench_init, inputs = [username, note, warmup, level, steps, width, height], outputs = [benchmark_data])
                            bench_submit_btn = gr.Button('Submit results', elem_id = 'system_info_tab_submit_btn', variant='primary')
                            bench_submit_btn.click(bench_submit, inputs = [username], outputs = [])
                            _bench_link = gr.HTML('<a href="https://vladmandic.github.io/sd-extension-system-info/pages/benchmark.html" target="_blank">Link to online results</a>')
                            _bench_note = gr.HTML(elem_id = 'system_info_tab_bench_note', value = """
                                <span>performance is measured in iterations per second (it/s) and reported for different batch sizes (e.g. 1, 2, 4, 8, 16...)</span><br>
                                <span>running benchmark may take a while. extensive tests may result in gpu out-of-memory conditions.</span>""")
                    with gr.Row():
                        bench_label = gr.HTML('', elem_id = 'system_info_tab_bench_label')
                        refresh_bench_btn = gr.Button('Refresh bench', elem_id = 'system_info_tab_refresh_bench_btn', visible = False) # quick refresh is used from js interval
                        refresh_bench_btn.click(bench_refresh, inputs = [], outputs = [bench_label])

                with gr.TabItem('Models & Networks'):
                    with gr.Row():
                        with gr.Column():
                            models = gr.JSON(networks['models'], label = 'Models')
                        with gr.Column():
                            loras = gr.JSON(networks['loras'], label = 'Available LORA')

                refresh_quick_btn.click(refresh_info_quick, _js='receive_system_info', show_progress = False,
                    inputs = [js],
                    outputs = [statetxt, memorytxt, attentiontxt, timestamp, js]
                )
                refresh_full_btn.click(refresh_info_full, show_progress = False,
                    inputs = [],
                    outputs = [uptimetxt, versiontxt, statetxt, memorytxt, platformtxt, torchtxt, gputxt, opttxt, attentiontxt, backendtxt, pipelinetxt, libstxt, repostxt, devtxt, modeltxt, models, loras, timestamp, js]
                )
                interrupt_btn.click(shared.state.interrupt, inputs = [], outputs = [])

    return [(system_info, 'System Info', 'system_info')]


### benchmarking module

def bench_submit(username: str):
    if username is None or username == '':
        log.error('SD-System-Info: username is required to submit results')
        return
    benchmark.submit_benchmark(bench_data, username)
    log.info(f'SD-System-Info: benchmark data submitted: {len(bench_data)} records')


def bench_run(batches: list = [1], steps: str = 'normal', width: int = 512, height: int = 512):
    results = []
    for batch in batches:
        log.debug(f'SD-System-Info: benchmark starting: batch={batch}')
        res = benchmark.run_benchmark(batch, steps, width, height)
        log.info(f'SD-System-Info: benchmark batch={batch} its={res}')
        results.append(str(res))
    its = ' / '.join(results)
    return its


def get_settings():
    settings = {
        'pipeline': shared.sd_model.__class__.__name__,
        'model': shared.opts.data.get('sd_model_checkpoint', 'none').lower(),
        'vae': shared.opts.data.get('sd_vae', 'none').lower(),
        'width': benchmark.args.get('width', 512),
        'height': benchmark.args.get('height', 512),
    }
    return settings


def get_optimizations():
    optimizations = {
        'pipeline': data['pipeline'],
        'crossattention': data['crossattention'],
        'flags': ','.join(data['flags']).strip(),
        'compute': str(devices.backend) if hasattr(devices, 'backend') else 'unknown',
        'compile': shared.opts.data.get('cuda_compile_backend', 'none'),
        'dtype': str(devices.dtype) if hasattr(devices, 'dtype') else 'unknown',
    }
    return optimizations


def bench_init(username: str, note: str, warmup: bool, level: str, steps: str, width: int, height: int):
    from hashlib import sha256

    log.debug('SD-System-Info: benchmark starting')
    get_full_data()
    hash256 = sha256((dict2str(data['platform']) + data['torch'] + dict2str(data['libs']) + dict2str(data['gpu']) + ','.join(data['flags']) + data['crossattention']).encode('utf-8')).hexdigest()[:6]
    existing = [x for x in bench_data if (x[-1] is not None and x[-1][:6] == hash256)]
    if len(existing) > 0:
        log.debug('SD-System-Info: benchmark replacing existing entry')
        d = existing[0]
    elif bench_data[-1][0] is not None:
        log.debug('SD-System-Info: benchmark new entry')
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
    elif level == 'ludicrous':
        batches = [1, 2, 4, 8, 16, 32, 64]
    else:
        batches = []

    if warmup:
        bench_run([1], False, width, height)

    try:
        mem = data['memory']['gpu']['total']
    except Exception:
        mem = 0

    # bench_headers = ['timestamp', 'performance', 'version', 'system', 'libraries', 'gpu', 'flags', 'settings', 'username', 'note', 'hash']
    d[0] = str(datetime.datetime.now())
    d[1] = bench_run(batches, steps, width, height)
    d[2] = dict2str(data['version'])
    d[3] = dict2str(data['platform'])
    d[4] = f"torch:{data['torch']} {dict2str(data['libs'])}"
    d[5] = dict2str(data['gpu']) + f' {str(round(mem))}GB'
    d[6] = dict2str(get_optimizations())
    d[7] = dict2str(get_settings())
    d[8] = username
    d[9] = note
    d[10] = hash256

    md = '| ' + ' | '.join(d) + ' |'
    log.info(f'SD-System-Info: benchmark result: {md}')

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
                log.debug(f'SD-System-Info: benchmark data loaded: {bench_file}')
        except Exception as err:
            log.debug(f'SD-System-Info: benchmark error loading: {bench_file} {str(err)}')
    if len(bench_data) == 0:
        bench_data.append([None] * len(bench_headers))
    return bench_data


def bench_save():
    if bench_data[-1][0] is None:
        del bench_data[-1]
    try:
        with open(bench_file, 'w', encoding='utf-8') as f:
            json.dump(bench_data, f, indent=2, default=str, skipkeys=True)
            log.debug(f'SD-System-Info: benchmark data saved: {bench_file}')
    except Exception as err:
        log.error(f'SD-System-Info: benchmark error saving:  {bench_file} {str(err)}')


def bench_refresh():
    return gr.HTML.update(value = bench_text)


### API

from typing import Optional # pylint: disable=wrong-import-order
from fastapi import FastAPI, Depends # pylint: disable=wrong-import-order
from pydantic import BaseModel, Field # pylint: disable=wrong-import-order,no-name-in-module


class StatusReq(BaseModel): # definition of http request
    state: bool = Field(title="State", description="Get server state", default=False)
    memory: bool = Field(title="Memory", description="Get server memory status", default=False)
    full: bool = Field(title="FullInfo", description="Get full server info", default=False)
    refresh: bool = Field(title="FullInfo", description="Force refresh server info", default=False)


class StatusRes(BaseModel): # definition of http response
    version: dict = Field(title="Version", description="Server version")
    uptime: str = Field(title="Uptime", description="Server uptime")
    timestamp: str = Field(title="Timestamp", description="Data timestamp")
    state: Optional[dict] = Field(title="State", description="Server state", default=None)
    memory: Optional[dict] = Field(title="Memory", description="Server memory status", default=None)
    platform: Optional[dict] = Field(title="Platform", description="Server platform", default=None)
    torch: Optional[str] = Field(title="Torch", description="Torch version", default=None)
    gpu: Optional[dict] = Field(title="GPU", description="GPU info", default=None)
    flags: Optional[list] = Field(title="Optimizations", description="Memory optimizations", default=None)
    crossatention: Optional[str] = Field(title="CrossAttention", description="Cross-attention optimization", default=None)
    device: Optional[dict] = Field(title="Device", description="Device info", default=None)
    backend: Optional[str] = Field(title="Backend", description="Backend", default=None)
    pipeline: Optional[str] = Field(title="Pipeline", description="Pipeline", default=None)


def get_status_api(req: StatusReq = Depends()):
    if req.refresh:
        get_full_data()
    else:
        get_quick_data()
    res = StatusRes(
        version = data['version'],
        timestamp = data['timestamp'],
        uptime = data['uptime']
    )
    if req.state or req.full:
        res.state = data['state']
    if req.memory or req.full:
        res.memory = data['memory']
    if req.full:
        res.platform = data['platform']
        res.torch = data['torch']
        res.gpu = data['gpu']
        res.flags = data['flags']
        res.crossatention = data['crossattention']
        res.device = data['device']
        res.backend = data['backend']
        res.pipeline = data['pipeline']
    return res


def register_api(app: FastAPI):
    app.add_api_route("/sdapi/v1/system-info/status", get_status_api, methods=["GET"], response_model=StatusRes)


### Entry point

def on_app_started(blocks, app): # register api
    register_api(app)
    if not standalone:
        create_ui(blocks)
    """
    @app.get("/sdapi/v1/system-info/status")
    async def sysinfo_api():
        get_quick_data()
        res = { 'state': data['state'], 'memory': data['memory'], 'timestamp': data['timestamp'] }
        return res
    """

try:
    from modules.ui import ui_system_tabs # pylint: disable=unused-import,ungrouped-imports
    standalone = False
except Exception:
    standalone = True

if standalone:
    script_callbacks.on_ui_tabs(create_ui)
script_callbacks.on_app_started(on_app_started)
