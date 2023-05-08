// this would not be needed if automatic run gradio with loop enabled

let loaded = false;
let interval_sys;
let interval_bench;

function refresh_info() {
  const btn = gradioApp().getElementById('system_info_tab_refresh_btn') // we could cache this dom element
  if (btn) btn.click() // but ui may get destroyed, actual refresh is done from python code we just trigger it but simulating button click
}

function refresh_info_full() {
  const btn = gradioApp().getElementById('system_info_tab_refresh_full_btn') // we could cache this dom element
  if (btn) btn.click() // but ui may get destroyed, actual refresh is done from python code we just trigger it but simulating button click
}

function refresh_bench() {
  const btn = gradioApp().getElementById('system_info_tab_refresh_bench_btn') // we could cache this dom element
  if (btn) btn.click() // but ui may get destroyed, actual refresh is done from python code we just trigger it but simulating button click
}

function onHidden() { // stop refresh interval when tab is not visible
  if (interval_sys) {
    clearInterval(interval_sys);
    interval_sys = undefined;
  }
  if (interval_bench) {
    clearInterval(interval_bench);
    interval_bench = undefined;
  }
}

function onVisible() { // start refresh interval tab is when visible
  if (!interval_sys) {
    setTimeout(refresh_info_full, 50); // do full refresh on first show
    refresh_info_full(); // do full refresh on first show
    interval_sys = setInterval(refresh_info, 1500); // check interval already started so dont start it again
  }
  if (!interval_bench) interval_bench = setInterval(refresh_bench, 1000); // check interval already started so dont start it again
}

function initLoading() { // triggered on gradio change to monitor when ui gets sufficiently constructed
  if (loaded) return
  const block = gradioApp().getElementById('system_info');
  if (!block) return
  intersectionObserver = new IntersectionObserver((entries) => {
    if (entries[0].intersectionRatio <= 0) onHidden();
    if (entries[0].intersectionRatio > 0) onVisible();
  });
  intersectionObserver.observe(block); // monitor visibility of tab
}

function initInitial() { // just setup monitor for gradio events
  const mutationObserver = new MutationObserver(initLoading)
  mutationObserver.observe(gradioApp(), { childList: true, subtree: true }); // monitor changes to gradio
}

document.addEventListener('DOMContentLoaded', initInitial);
