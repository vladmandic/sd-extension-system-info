# SD WebUI Benchmark Data

## Run Benchmark & Submit

- Benchmark data is created using **SD WebUI Extension** [**System Info**](https://github.com/vladmandic/sd-extension-system-info)  
  <https://github.com/vladmandic/sd-extension-system-info>  
- Record is appended if any of the system properties change  
  Else benchmark data replaces existing matching record
- Data is submitted to cloud logger and stored online for 48 hours
  <https://papertrailapp.com>  

## Update Results

- GitHub Action runs periodically  
  <https://vladmandic.github.io/sd-data/pages/benchmark-download.js>
  - Downloads latest benchmark data  
  - Merges downloaded data with local data  
    <https://vladmandic.github.io/sd-data/pages/benchmark-raw.json>
  - Creates JSON with sorted unique records  
    <https://vladmandic.github.io/sd-data/pages/benchmark-data.json>
  - Renders as Markdown table  
    <https://github.com/vladmandic/sd-data/blob/main/pages/benchmark.md>

- GitPages Serves HTML page which downloads and renders JSON data  
  <https://vladmandic.github.io/sd-extension-system-info/pages/benchmark.html>
