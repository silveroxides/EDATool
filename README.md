# 📊 NiceGUI Dataset Explorer & Trimming Tool

A high-performance, memory-efficient, and lightweight Exploratory Data Analysis (EDA) and dataset pruning tool built specifically for massive caption JSONL datasets.

---

## ⚙️ Installation & Quick Start

### 1. Install Dependencies
Install all required packages declared in [`requirements.txt`](requirements.txt):
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
Run the NiceGUI application entrypoint:
```bash
python main.py
```

### 3. Dynamic CLI Configuration
Customize the dataset directory and server port dynamically:
```bash
python main.py -d /path/to/dataset_folder -p 8080
# Or using full parameter names
python main.py --dir /path/to/dataset_folder --port 8080
```
This boots up the application on the specified port (defaulting to `8080`) and opens the web GUI at `http://localhost:8080` (or your custom port).

---

## ⌨️ Tactile Keyboard Controls

Navigate and curate thousands of image metadata entries directly from your keyboard:

| Key Bind | Action Description |
| :--- | :--- |
| **`Q`** | Previous Page |
| **`E`** | Next Page |
| **`ArrowLeft`** | Highlight Left Card (wraps to previous page's final card if crossing boundary) |
| **`ArrowRight`** | Highlight Right Card (wraps to next page's first card if crossing boundary) |
| **`Space`** or **`T`** | Toggle Trim State of the active highlighted card |

---

## 🖥️ Interface Overview

The interface is structured into three main viewports inside a persistent zero-scrolling lightbox dashboard:
- **Left Sidebar**: Manages dataset file selection, multi-key dynamic filter building, standard filters (score, favorites, rating, tags, Gemini description text), sorting, and live interactive Plotly analytics. Also houses the filtered JSONL and trimmed JSONL exporters.
- **Center Gallery**: Displays a highly responsive grid of image cards with glowing cyan selection indicators (`border-cyan-400 shadow-[0_0_15px_rgba(34,211,238,0.6)]`) and inline trim checkboxes.
- **Right Inspector**: Shows persistent high-resolution CDN live previews, detailed metadata properties grid, and alphabetized badge-chip tags.

---

## 🚀 High-Performance Architecture

Built from the ground up for extreme speed, rich responsiveness, and minimal memory consumption:
- **Binary Offset Indexing (`.idx`)**: Scans the JSONL once at first load to save starting byte positions as 64-bit integers. Unpacks instantly to allow $O(1)$ random-access seeking of any record.
- **Columnar Parquet Caching (`.parquet`)**: Generates an optimized, highly compressed columnar cache of searchable fields in the [`.eda_cache/`](.eda_cache/) directory, reducing active RAM footprint to just ~14MB.
- **Asynchronous IO-Bound Pool**: Utilizes NiceGUI's concurrent worker threads (`run.io_bound`) to run blocking disk seeks and JSON parsing, ensuring zero UI stutter.
- **WebSocket Keyboard Listener (`ui.keyboard`)**: Listens to global window keydown events via a persistent WebSocket connection, bypassing focus-stealing sandbox issues.

---

## 📁 Repository Files

- [`main.py`](main.py): The main presentation layer managing application states, global keybinds, layout rendering, and server configuration.
- [`eda_backend.py`](eda_backend.py): The data operations layer managing binary `.idx` offsets, columnar Parquet structures, lazy seeking, and streaming exports.
- [`requirements.txt`](requirements.txt): Declares Python dependencies for the NiceGUI and data analysis stack.
- [`example_line.jsonl`](example_line.jsonl): A small sample dataset file for testing and verification.
