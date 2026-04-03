import os, sys, json, datetime, platform, subprocess, re, math
from pathlib import Path

# 脚本依赖（在HTML末尾单独展示）
SCRIPT_DEPENDENCIES = ["psutil", "requests", "plotly"]

#--------------------
# 1. 环境检测
#--------------------
def get_system_info():
    info = {}
    info["python"] = sys.version.split()[0]
    info["platform"] = platform.platform()
    info["machine"] = platform.machine()
    info["cpu_count"] = os.cpu_count()
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["total_mem_gb"] = round(mem.total / (1024**3), 2)
        info["psutil_missing"] = False
    except ModuleNotFoundError:
        info["total_mem_gb"] = None
        info["psutil_missing"] = True
        info["psutil_tip"] = "pip install psutil"
    except Exception as e:
        info["total_mem_gb"] = None
        info["psutil_missing"] = False
        info["psutil_error"] = str(e)
    # GPU 探测
    gpus = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpus.append({
                    "name": torch.cuda.get_device_name(i),
                    "capability": torch.cuda.get_device_capability(i),
                    "memory_gb": round(torch.cuda.get_device_properties(i).total_memory/(1024**3),2)
                })
    except Exception:
        pass
    # nvidia-smi 备用
    if not gpus:
        try:
            out = subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader"], stderr=subprocess.DEVNULL, text=True)
            for line in out.strip().splitlines():
                n,m,d = [x.strip() for x in line.split(",")]
                gpus.append({"name": n, "memory_gb": float(re.findall(r"(\d+)", m)[0])/1024, "driver": d})
        except Exception:
            pass
    info["gpus"] = gpus
    for tool in ["docker", "kubectl"]:
        try:
            v = subprocess.check_output([tool, "--version"], stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            v = None
        info[f"{tool}_version"] = v
    return info

#--------------------
# 2. 已安装支持
#--------------------
def check_installed_packages():
    targets = ["deepseek","qwen","ollama","transformers","torch","llama_cpp","accelerate","sentence-transformers","plotly","requests","psutil"]
    installed = {}
    for pkg in targets:
        try:
            mod = __import__(pkg)
            installed[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            installed[pkg] = None
    return installed

#--------------------
# 3. 在线抓取最新资讯
#--------------------
def fetch_latest_model_info():
    base = "llm_info_cache.json"
    data = {"updated": None, "source": "local", "models": {}, "note": "未获取网络数据"}

    local_cache = {}
    if Path(base).exists():
        try:
            local_cache = json.loads(Path(base).read_text(encoding="utf-8")).get("models", {})
        except Exception:
            local_cache = {}

    try:
        import requests
    except ModuleNotFoundError:
        data["note"] = "requests 未安装，跳过在线更新"
        if Path(base).exists():
            return json.loads(Path(base).read_text(encoding="utf-8"))
        return data

    def get_json(url, params=None):
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    try:
        model_list = {}
        for name in ["qwen", "llama", "gpt2", "mistral", "orca", "mpt", "deepseek"]:
            try:
                info = get_json(f"https://huggingface.co/api/models/{name}")
                model_list[name] = {
                    "id": name,
                    "lastModified": info.get("lastModified"),
                    "pipeline_tag": info.get("pipeline_tag"),
                    "downloads": info.get("downloads", 0),
                    "tags": info.get("tags", []),
                }
            except Exception:
                if local_cache.get(name):
                    model_list[name] = {**local_cache.get(name), "note": "使用本地缓存"}
                else:
                    model_list[name] = {"error": "获取失败"}

        try:
            gh = get_json("https://api.github.com/repos/ollama/ollama/releases/latest")
            model_list["ollama"] = {"version": gh.get("tag_name"), "published_at": gh.get("published_at")}
        except Exception:
            if local_cache.get("ollama"):
                model_list["ollama"] = {**local_cache.get("ollama"), "note": "使用本地缓存"}
            else:
                model_list.setdefault("ollama", {})
                model_list["ollama"]["error"] = "获取失败"

        data = {"updated": datetime.datetime.now().isoformat(), "source": "internet", "models": model_list, "note": ""}
        Path(base).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data
    except Exception as e:
        if Path(base).exists():
            local = json.loads(Path(base).read_text(encoding="utf-8"))
            local["note"] = f"网络请求失败，使用本地缓存: {e}"
            return local
        data["note"] = f"网络请求失败，且无本地缓存: {e}"
        return data

#--------------------
# 4. 模型配置推荐逻辑
#--------------------
DEFAULT_MODELS = [
    {"name":"deepseek", "vram_min_gb":2, "ram_min_gb":8, "use":"小型本地推理 & API融合", "package":"deepseek"},
    {"name":"qwen",     "vram_min_gb":6, "ram_min_gb":16, "use":"中大型对话/推理", "package":"qwen"},
    {"name":"ollama",   "vram_min_gb":3, "ram_min_gb":12, "use":"跨平台定制模型", "package":"ollama"},
    {"name":"llama.cpp","vram_min_gb":0, "ram_min_gb":8 , "use":"CPU环境下小模型", "package":"llama_cpp"},
    {"name":"gpt-neox","vram_min_gb":8, "ram_min_gb":32, "use":"高性能GPU推理", "package":"transformers"},
]

MODEL_SPECIFIC = [
    {"name":"qwen-7b", "vram_min_gb":14, "ram_min_gb":24, "use":"中大型对话/推理", "package":"qwen"},
    {"name":"qwen-14b", "vram_min_gb":28, "ram_min_gb":48, "use":"大模型高精度", "package":"qwen"},
    {"name":"llama-2-13b", "vram_min_gb":25, "ram_min_gb":40, "use":"LLama 2 通用推理", "package":"llama_cpp"},
    {"name":"mistral-7b", "vram_min_gb":14, "ram_min_gb":20, "use":"Mistral 富表达能力", "package":"transformers"},
]

def decide_compatibility_sys(sysinfo, installed):
    available = []
    for m in DEFAULT_MODELS + MODEL_SPECIFIC:
        ok = True
        reason = []
        pkg = m.get("package", m["name"])
        if installed.get(pkg) is None:
            ok = False
            reason.append(f"依赖未安装({pkg})")
        if sysinfo.get("total_mem_gb") is None or sysinfo.get("total_mem_gb",0) < m["ram_min_gb"]:
            ok = False
            reason.append(f"RAM不足({sysinfo.get('total_mem_gb')}<{m['ram_min_gb']})")
        gpus = sysinfo.get("gpus", [])
        if m.get("vram_min_gb",0) > 0:
            if not gpus or max([g.get("memory_gb",0) for g in gpus]) < m["vram_min_gb"]:
                ok = False
                reason.append(f"显存不足({m['vram_min_gb']}GB)")
        available.append({"name": m["name"], "can_run": ok, "reason": ",".join(reason) if reason else "OK", "type": m["use"], "ram": m["ram_min_gb"], "vram": m["vram_min_gb"]})
    return available

#--------------------
# 5. 基准测试 (PONG/bench)
#--------------------
def run_inference_bench():
    bench = {}
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        model_name = "sshleifer/tiny-gpt2"
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
        bench["transformers_load_s"] = round(time.time()-t0, 3)

        t1 = time.time()
        _ = gen("Hello", max_new_tokens=10, do_sample=False)
        bench["transformers_infer_s"] = round(time.time()-t1, 3)
    except Exception as e:
        bench["transformers_error"] = str(e)

    try:
        from llama_cpp import Llama
        model_path = os.getenv("LLAMA_CPP_MODEL", "./models/llama-mini-7b.gguf")
        if Path(model_path).exists():
            t0 = time.time()
            llm = Llama(model_path=model_path)
            bench["llama_cpp_load_s"] = round(time.time()-t0, 3)
            t1 = time.time()
            _ = llm("Hello", max_tokens=8)
            bench["llama_cpp_infer_s"] = round(time.time()-t1, 3)
        else:
            bench["llama_cpp_error"] = f"模型文件不存在: {model_path}"
    except Exception as e:
        bench["llama_cpp_error"] = str(e)

    return bench

#--------------------
# 6. 生成 HTML
#--------------------
def create_html_report(sysinfo, installed, internet, available, bench_results):
    def row(key, value):
        return f"<tr><th>{key}</th><td>{value}</td></tr>"

    report = """
    <html><head><meta charset='utf-8'><title>LLM 运行能力报告</title>
    <style>body{font-family:Arial,Helvetica,sans-serif;}table{border-collapse:collapse;width:100%;}th,td{border:1px solid #ccc;padding:8px;text-align:left;}.ok{background:#e6ffe6;}.no{background:#ffe6e6;}.warn{background:#fff0b5;}button{margin-right:8px;}</style>
    <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
    <script>
        var i18n = {
            zh: {
                title: 'LLM 运行能力报告',
                main_title: 'LLM 运行能力报告',
                system_info: '系统信息',
                installed: '已安装组件',
                network: '网络最新情报来源',
                model_compat: '推荐模型与兼容性',
                chart: '模型 RAM/VRAM 需求对比图',
                benchmark: '预测基准结果',
                conclusion: '结论',
                deps: 'Python脚本依赖（可通过 pip 安装）',
                model: '模型',
                can_run: '可运行',
                ram: 'RAM需求',
                vram: 'VRAM需求',
                reason: '说明',
                usage: '建议用途',
                yes: 'Yes',
                no: 'No',
                models: 'models'
            },
            en: {
                title: 'LLM Capability Report',
                main_title: 'LLM Capability Report',
                system_info: 'System Info',
                installed: 'Installed Packages',
                network: 'Latest LLM Info',
                model_compat: 'Model Compatibility',
                chart: 'Model RAM/VRAM Comparison',
                benchmark: 'Benchmark Results',
                conclusion: 'Conclusion',
                deps: 'Python script dependencies (pip install)',
                model: 'Model',
                can_run: 'Can Run',
                ram: 'RAM',
                vram: 'VRAM',
                reason: 'Reason',
                usage: 'Usage',
                yes: 'Yes',
                no: 'No',
                models: 'models'
            },
            ja: {
                title: 'LLM 実行能力レポート',
                main_title: 'LLM 実行能力レポート',
                system_info: 'システム情報',
                installed: 'インストール済みパッケージ',
                network: '最新LLM情報',
                model_compat: 'モデル互換性',
                chart: 'モデル RAM/VRAM 比較',
                benchmark: 'ベンチマーク結果',
                conclusion: '結論',
                deps: 'Pythonスクリプト依存関係 (pip install)',
                model: 'モデル',
                can_run: '可実行',
                ram: 'RAM',
                vram: 'VRAM',
                reason: '説明',
                usage: '用途',
                yes: 'Yes',
                no: 'No',
                models: 'models'
            },
            ko: {
                title: 'LLM 실행 능력 보고서',
                main_title: 'LLM 실행 능력 보고서',
                system_info: '시스템 정보',
                installed: '설치된 패키지',
                network: '최신 LLM 정보',
                model_compat: '모델 호환성',
                chart: '모델 RAM/VRAM 비교',
                benchmark: '벤치마크 결과',
                conclusion: '결론',
                deps: 'Python 스크립트 의존성 (pip install)',
                model: '모델',
                can_run: '실행 가능',
                ram: 'RAM',
                vram: 'VRAM',
                reason: '설명',
                usage: '사용 용도',
                yes: 'Yes',
                no: 'No',
                models: 'models'
            }
        };
        function switchLang(lang) {
            var t = i18n[lang] || i18n.zh;
            document.title = t.title;
            document.getElementById('main-title').innerText = t.main_title;
            document.getElementById('section-system').innerText = t.system_info;
            document.getElementById('section-installed').innerText = t.installed;
            document.getElementById('section-network').innerText = t.network;
            document.getElementById('section-model').innerText = t.model_compat;
            document.getElementById('section-chart').innerText = t.chart;
            document.getElementById('section-benchmark').innerText = t.benchmark;
            document.getElementById('section-conclusion').innerText = t.conclusion;
            document.getElementById('section-deps').innerText = t.deps;
            document.getElementById('th-model').innerText = t.model;
            document.getElementById('th-canrun').innerText = t.can_run;
            document.getElementById('th-ram').innerText = t.ram;
            document.getElementById('th-vram').innerText = t.vram;
            document.getElementById('th-reason').innerText = t.reason;
            document.getElementById('th-usage').innerText = t.usage;
            for (let el of document.querySelectorAll('.yes-no')) {
                if (el.dataset.value === 'yes') el.innerText = t.yes;
                if (el.dataset.value === 'no') el.innerText = t.no;
            }
            var mEl = document.getElementById('th-models-entry');
            if (mEl) mEl.innerText = t.models;
        }
        document.addEventListener('DOMContentLoaded', function() { switchLang('zh'); });
    </script>
    </head><body>
    <div style='margin-bottom:12px;'><button onclick="switchLang('zh')">中文</button><button onclick="switchLang('en')">English</button><button onclick="switchLang('ja')">日本語</button><button onclick="switchLang('ko')">한국어</button></div>"""

    report += "<h1 id='main-title'>LLM 运行能力报告（" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "）</h1>"
    report += "<h2 id='section-system'>系统信息</h2><table><tbody>"

    for k,v in sysinfo.items():
        report += row(k, v)

    report += """</tbody></table><h2 id='section-installed'>已安装组件</h2><table><thead><tr><th>软件</th><th>版本</th></tr></thead><tbody>"""
    for k,v in installed.items():
        if k in SCRIPT_DEPENDENCIES:
            continue
        report += f"<tr><td>{k}</td><td>{v}</td></tr>"
    report += "</tbody></table>"

    report += "<h2 id='section-network'>网络最新情报来源</h2><table><tbody>"
    report += row("source", internet.get("source"))
    report += row("updated", internet.get("updated"))
    note = internet.get("note") or ""
    if note:
        report += row("note", note)
    if internet.get("models"):
        report += f"<tr><th id='th-models-entry'>models</th><td><pre>{json.dumps(internet.get('models'), ensure_ascii=False, indent=2)}</pre></td></tr>"
    report += "</tbody></table>"

    report += "<h2 id='section-model'>推荐模型与兼容性</h2><table><thead><tr><th id='th-model'>模型</th><th id='th-canrun'>可运行</th><th id='th-ram'>RAM需求</th><th id='th-vram'>VRAM需求</th><th id='th-reason'>说明</th><th id='th-usage'>建议用途</th></tr></thead><tbody>"
    for r in available:
        cls = "ok" if r["can_run"] else "no"
        status = 'yes' if r['can_run'] else 'no'
        report += f"<tr class='{cls}'><td>{r['name']}</td><td><span class='yes-no' data-value='{status}'>{'Yes' if r['can_run'] else 'No'}</span></td><td>{r['ram']}</td><td>{r['vram']}</td><td>{r['reason']}</td><td>{r['type']}</td></tr>"
    report += "</tbody></table>"

    chart_data = {
        "x": [m["name"] for m in DEFAULT_MODELS + MODEL_SPECIFIC],
        "ram": [m["ram_min_gb"] for m in DEFAULT_MODELS + MODEL_SPECIFIC],
        "vram": [m["vram_min_gb"] for m in DEFAULT_MODELS + MODEL_SPECIFIC],
    }

    report += "<h2 id='section-chart'>模型 RAM/VRAM 需求对比图</h2><div id='chart' style='height:450px;'></div>"
    report += "<script>var data=[{" + f"x:{json.dumps(chart_data['x'])},y:{json.dumps(chart_data['ram'])},name:'RAM(GB)',type:'bar'" + "},{" + f"x:{json.dumps(chart_data['x'])},y:{json.dumps(chart_data['vram'])},name:'VRAM(GB)',type:'bar'" + "}];var layout={barmode:'group',title:'模型内存需求'};Plotly.newPlot('chart',data,layout);</script>"

    report += "<h2 id='section-benchmark'>预测基准结果</h2><table><thead><tr><th>项</th><th>耗时（秒）或状态</th></tr></thead><tbody>"
    for k,v in bench_results.items():
        report += f"<tr><td>{k}</td><td>{v}</td></tr>"
    report += "</tbody></table>"

    report += "<h2 id='section-conclusion'>结论</h2><p>若模型不可运行，请根据“说明”列补装组件、升级 RAM / GPU 或使用轻量级本地模型。</p>"

    report += "<h2 id='section-deps'>Python脚本依赖（可通过 pip 安装）</h2><table><thead><tr><th>包</th><th>版本</th><th>安装命令</th></tr></thead><tbody>"
    for pkg in SCRIPT_DEPENDENCIES:
        ver = installed.get(pkg, None)
        status = ver if ver else "未安装"
        report += f"<tr><td>{pkg}</td><td>{status}</td><td>pip install {pkg}</td></tr>"
    report += "</tbody></table>"

    f = "llm_compat_report.html"
    Path(f).write_text(report, encoding="utf-8")
    return f

if __name__ == "__main__":
    sysinfo = get_system_info()
    installed = check_installed_packages()
    internet = fetch_latest_model_info()
    available = decide_compatibility_sys(sysinfo, installed)
    bench_results = run_inference_bench()
    out = create_html_report(sysinfo, installed, internet, available, bench_results)
    print(f"生成完成：{out}，请用浏览器打开查看。")