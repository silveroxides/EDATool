import os
import struct
import json
import collections
import asyncio
import argparse
import pandas as pd
import plotly.graph_objects as go
from nicegui import ui, run, app
from nicegui.events import KeyEventArguments
from nicegui.element import Element
import eda_backend

# Monkeypatch NiceGUI Element.parent_slot to return None instead of raising RuntimeError
# when the parent slot has been deleted. This gracefully handles asynchronous event/value updates
# during rapid re-renders or page clears without triggering startup tracebacks.
_original_parent_slot_getter = Element.parent_slot.fget
def _patched_parent_slot_getter(self):
    try:
        return _original_parent_slot_getter(self)
    except RuntimeError as e:
        if 'The parent slot of the element has been deleted.' in str(e):
            return None
        raise
Element.parent_slot = property(_patched_parent_slot_getter)

# Custom background styling matching instructions:
# Outer page styled with h-screen w-screen overflow-hidden flex flex-row bg-zinc-950 text-white font-sans.
# NiceGUI loads global CSS or inline styles natively.
ui.add_head_html("""
<style>
/* Custom scrollbar styling for slick dark theme */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #09090b; /* zinc-950 */
}
::-webkit-scrollbar-thumb {
    background: #27272a; /* zinc-800 */
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #3f3f46; /* zinc-700 */
}
</style>
""", shared=True)

def parse_args():
    parser = argparse.ArgumentParser(description="NiceGUI EDA & Trimming Tool")
    parser.add_argument(
        "--dir", "-d",
        type=str,
        default=r"F:\datasets\deepghs\deepghs--danbooru2024-captions-gemini-flash-1.5",
        help="Dataset directory path"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="Server port"
    )
    args, _ = parser.parse_known_args()
    return args

args = parse_args()

class AppState:
    def __init__(self, initial_dir=None):
        self.current_dir = initial_dir or r"F:\datasets\deepghs\deepghs--danbooru2024-captions-gemini-flash-1.5"

        # Fallback to workspace root if directory doesn't exist
        if not os.path.exists(self.current_dir) or not os.path.isdir(self.current_dir):
            self.current_dir = os.path.abspath(".")

        self.jsonl_files = []
        self.current_file = None

        # Byte Offsets & Metadata
        self.offsets = []
        self.metadata_df = pd.DataFrame()
        self.filtered_df = pd.DataFrame()
        self.page_records = []  # Full records loaded for the current page

        # User Selection and Pagination State
        self.page = 1
        self.page_size = 4
        self.num_columns_per_row = 4
        self.active_index = 0  # 0-indexed offset within the active page grid

        # Sorting
        self.sort_col = "score"
        self.sort_order = "Descending"  # Descending or Ascending

        # Standard Filters
        self.rating_selection = []
        self.ext_selection = []
        self.score_range = [0, 100]
        self.fav_range = [0, 100]
        self.w_range = [0, 2000]
        self.h_range = [0, 2000]
        self.text_query = ""
        self.tag_query = ""
        self.filter_conditions = []  # For Dynamic Multi-key Filter Builder

        # Selection tracking for trimmed data
        self.trimmed_offsets = set()

        # Bounds populated dynamically
        self.all_ratings = []
        self.all_exts = []
        self.min_score = 0
        self.max_score = 100
        self.min_fav = 0
        self.max_fav = 100
        self.min_w = 0
        self.max_w = 2000
        self.min_h = 0
        self.max_h = 2000
        self.loading = False

        self.jsonl_files = self.list_jsonl_files()
        if self.jsonl_files:
            self.current_file = self.jsonl_files[0]

    def get_full_path(self):
        if not self.current_file:
            return ""
        return os.path.join(self.current_dir, self.current_file)

    def list_jsonl_files(self):
        if not os.path.exists(self.current_dir) or not os.path.isdir(self.current_dir):
            return []
        try:
            files = [f for f in os.listdir(self.current_dir) if f.endswith('.jsonl')]
            return sorted(files)
        except Exception:
            return []

    def update_bounds(self):
        df = self.metadata_df
        if df.empty:
            return
        self.all_ratings = sorted([r for r in df['rating'].unique() if r])
        self.all_exts = sorted([e for e in df['file_ext'].unique() if e])

        self.min_score = int(df['score'].min())
        self.max_score = int(df['score'].max())
        if self.min_score == self.max_score:
            self.max_score += 1

        self.min_fav = int(df['fav_count'].min())
        self.max_fav = int(df['fav_count'].max())
        if self.min_fav == self.max_fav:
            self.max_fav += 1

        self.min_w = int(df['image_width'].min())
        self.max_w = int(df['image_width'].max())
        if self.min_w == self.max_w:
            self.max_w += 1

        self.min_h = int(df['image_height'].min())
        self.max_h = int(df['image_height'].max())
        if self.min_h == self.max_h:
            self.max_h += 1

        self.rating_selection = list(self.all_ratings)
        self.ext_selection = list(self.all_exts)
        self.score_range = [self.min_score, self.max_score]
        self.fav_range = [self.min_fav, self.max_fav]
        self.w_range = [self.min_w, self.max_w]
        self.h_range = [self.min_h, self.max_h]

    def apply_all_filters(self):
        df = self.metadata_df.copy()
        if df.empty:
            self.filtered_df = df
            return

        # Standard filters
        if self.rating_selection:
            df = df[df['rating'].isin(self.rating_selection)]
        else:
            df = df[df['rating'].isin([])]

        if self.ext_selection:
            df = df[df['file_ext'].isin(self.ext_selection)]
        else:
            df = df[df['file_ext'].isin([])]

        df = df[(df['score'] >= self.score_range[0]) & (df['score'] <= self.score_range[1])]
        df = df[(df['fav_count'] >= self.fav_range[0]) & (df['fav_count'] <= self.fav_range[1])]
        df = df[(df['image_width'] >= self.w_range[0]) & (df['image_width'] <= self.w_range[1])]
        df = df[(df['image_height'] >= self.h_range[0]) & (df['image_height'] <= self.h_range[1])]

        if self.text_query:
            df = df[df['regular_summary'].str.contains(self.text_query, case=False, na=False)]

        if self.tag_query:
            terms = self.tag_query.strip().split()
            for term in terms:
                if term.startswith('-'):
                    neg_term = term[1:].strip()
                    if neg_term:
                        df = df[~df['tags'].str.contains(neg_term, case=False, na=False)]
                else:
                    pos_term = term.strip()
                    if pos_term:
                        df = df[df['tags'].str.contains(pos_term, case=False, na=False)]

        # Dynamic filter builder
        for cond in self.filter_conditions:
            col = cond['column']
            op = cond['operator']
            val = cond['value']

            is_numeric = col in ['id', 'score', 'fav_count', 'image_width', 'image_height']

            if op == 'is empty/null':
                if is_numeric:
                    df = df[df[col].isna() | (df[col] == 0)]
                else:
                    df = df[df[col].isna() | (df[col].astype(str).str.strip() == '')]
                continue

            if is_numeric:
                try:
                    val_num = float(val) if '.' in str(val) else int(val)
                except ValueError:
                    continue

                if op == '=':
                    df = df[df[col] == val_num]
                elif op == '!=':
                    df = df[df[col] != val_num]
                elif op == '>':
                    df = df[df[col] > val_num]
                elif op == '<':
                    df = df[df[col] < val_num]
                elif op == '>=':
                    df = df[df[col] >= val_num]
                elif op == '<=':
                    df = df[df[col] <= val_num]
            else:
                val_str = str(val)
                if op == '=':
                    df = df[df[col].astype(str) == val_str]
                elif op == '!=':
                    df = df[df[col].astype(str) != val_str]
                elif op == 'contains':
                    df = df[df[col].astype(str).str.contains(val_str, case=False, na=False)]
                elif op == 'starts with':
                    df = df[df[col].astype(str).str.startswith(val_str, na=False)]
                elif op == 'ends with':
                    df = df[df[col].astype(str).str.endswith(val_str, na=False)]

        # Apply sorting
        if not df.empty:
            ascending = (self.sort_order == "Ascending")
            df = df.sort_values(by=self.sort_col, ascending=ascending)

        self.filtered_df = df

        # Clamp page and active index
        total_records = len(self.filtered_df)
        total_pages = max(1, (total_records + self.page_size - 1) // self.page_size)
        if self.page > total_pages:
            self.page = total_pages
        if self.page < 1:
            self.page = 1

        start_idx = (self.page - 1) * self.page_size
        end_idx = min(start_idx + self.page_size, total_records)
        num_items = end_idx - start_idx
        if num_items > 0:
            self.active_index = max(0, min(self.active_index, num_items - 1))
        else:
            self.active_index = 0

state = AppState(initial_dir=args.dir)

# Helpers
def format_bytes(n):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"

def get_ratings_chart(filtered_df):
    if filtered_df.empty:
        return go.Figure()
    rating_counts = filtered_df['rating'].value_counts()
    fig = go.Figure(data=[go.Bar(
        x=rating_counts.index.tolist(),
        y=rating_counts.values.tolist(),
        marker_color='#1f77b4'
    )])
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=150,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#27272a')
    )
    return fig

def get_tags_chart(filtered_df):
    if filtered_df.empty:
        return go.Figure()
    tags_series = filtered_df['tags'].dropna().astype(str)
    all_tags = tags_series.str.cat(sep=' ').split()
    tag_counts = collections.Counter(all_tags)
    if '' in tag_counts:
        del tag_counts['']
    top_20 = tag_counts.most_common(20)
    if not top_20:
        return go.Figure()
    tags, freqs = zip(*top_20)
    fig = go.Figure(data=[go.Bar(
        x=list(freqs),
        y=list(tags),
        orientation='h',
        marker_color='#00f3ff'
    )])
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(showgrid=True, gridcolor='#27272a'),
        yaxis=dict(showgrid=False, autocrange="reversed")
    )
    return fig

# UI Container references
header_container = None
grid_container = None
inspector_container = None
sidebar_conditions_container = None
batch_stats_container = None
analytics_container = None
file_status_container = None

# Global dialogs and their controls
loading_dialog = None
progress_bar = None
progress_label = None

export_dialog = None
export_progress_bar = None
export_progress_label = None

trim_dialog = None
trim_progress_bar = None
trim_progress_label = None

# Async-safe dynamic page re-renders
async def refresh_grid():
    if not grid_container:
        return
    grid_container.clear()
    with grid_container:
        if state.filtered_df.empty:
            with ui.column().classes('w-full h-full flex items-center justify-center text-zinc-500 gap-2'):
                ui.icon('image_not_supported', size='4rem')
                ui.label('No matching records found. Try relaxing your filters.').classes('text-lg')
            return

        start_idx = (state.page - 1) * state.page_size
        end_idx = min(start_idx + state.page_size, len(state.filtered_df))
        page_df = state.filtered_df.iloc[start_idx:end_idx]

        offsets = page_df['byte_offset'].tolist()

        # Loading page records asynchronously (Thread Pool)
        records = await run.io_bound(eda_backend.read_records_lazy, state.get_full_path(), offsets)
        state.page_records = records

        cols = state.num_columns_per_row
        grid_classes = f'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-{cols} gap-4 w-full'
        with ui.row().classes(grid_classes):
            for idx, ((row_idx, row), record) in enumerate(zip(page_df.iterrows(), records)):
                offset = row['byte_offset']
                is_active = (idx == state.active_index)
                is_trimmed = (offset in state.trimmed_offsets)

                # Selection highlight layout style
                if is_active:
                    card_classes = 'border-2 border-cyan-400 shadow-[0_0_15px_rgba(34,211,238,0.6)] bg-cyan-950/10 cursor-pointer rounded-xl overflow-hidden p-3 flex flex-col gap-2 transition-all duration-200'
                else:
                    card_classes = 'border border-zinc-800 bg-zinc-900/50 hover:border-zinc-700 cursor-pointer rounded-xl overflow-hidden p-3 flex flex-col gap-2 transition-all duration-200'

                card = ui.card().classes(card_classes)
                # Keep local index variable binding using helper maker
                def make_click_handler(i):
                    return lambda: select_active_card(i)
                card.on('click', make_click_handler(idx))

                with card:
                    preview_url = record.get('large_file_url') or record.get('file_url') or record.get('preview_file_url') if record else None
                    if preview_url:
                        ui.image(preview_url).classes('h-48 object-contain bg-black rounded-lg w-full')
                    else:
                        with ui.column().classes('h-48 w-full bg-zinc-950 rounded-lg flex items-center justify-center text-zinc-600'):
                            ui.icon('broken_image', size='2rem')
                            ui.label('No Image URL').classes('text-xs')

                    seq_num = start_idx + idx + 1
                    rec_id = record.get('id', 'N/A') if record else 'N/A'
                    ui.label(f"[{seq_num}] ID: {rec_id}").classes('text-sm font-bold text-zinc-200 truncate')

                    # Trim checkbox
                    def make_change_handler(offs):
                        return lambda e: toggle_trim_offset(offs, e.value)
                    ui.checkbox('Mark for Trim', value=is_trimmed, on_change=make_change_handler(offset)).classes('text-xs text-zinc-400').props('keep-color color=red')

                    # Manual Select Button
                    ui.button('🔍 Select', on_click=make_click_handler(idx)).classes('w-full mt-auto').props('size=sm outline color=cyan')

async def refresh_inspector():
    if not inspector_container:
        return
    inspector_container.clear()
    with inspector_container:
        if state.filtered_df.empty or not state.page_records or state.active_index >= len(state.page_records):
            with ui.column().classes('w-full h-full flex items-center justify-center text-zinc-500 gap-2'):
                ui.icon('find_in_page', size='4rem')
                ui.label('No active selection. Select a record to view details.').classes('text-lg')
            return

        record = state.page_records[state.active_index]
        row_idx = (state.page - 1) * state.page_size + state.active_index
        row = state.filtered_df.iloc[row_idx]
        offset = row['byte_offset']
        is_trimmed = (offset in state.trimmed_offsets)

        rec_id = record.get('id', 'N/A')
        ui.label(f"🔑 Record ID: {rec_id}").classes('text-xl font-extrabold text-cyan-400 border-b border-zinc-850 pb-2 w-full')

        # High fidelity preview
        preview_url = record.get('large_file_url') or record.get('file_url') or record.get('preview_file_url')
        if preview_url:
            ui.image(preview_url).classes('w-full max-h-[400px] object-contain bg-black rounded-lg shadow-md border border-zinc-800')
        else:
            with ui.column().classes('w-full h-48 bg-zinc-950 rounded-lg flex items-center justify-center text-zinc-600'):
                ui.icon('broken_image', size='3rem')
                ui.label('No preview available.').classes('text-sm')

        # Toggle buttons
        if is_trimmed:
            ui.button('🟢 Remove from Trim List', on_click=lambda: toggle_trim_offset(offset, False)).classes('w-full py-2 text-md font-bold').props('color=green')
        else:
            ui.button('❌ Add to Trim List', on_click=lambda: toggle_trim_offset(offset, True)).classes('w-full py-2 text-md font-bold').props('color=red')

        # Metadata Properties Table
        with ui.column().classes('w-full bg-zinc-900/50 p-4 rounded-xl border border-zinc-800 gap-2'):
            ui.label('Properties').classes('text-xs font-bold uppercase tracking-wider text-zinc-500 border-b border-zinc-850 pb-1 mb-1')

            props = [
                ("ID", str(record.get('id', 'N/A'))),
                ("Created At", str(record.get('created_at', 'N/A'))),
                ("Rating", str(record.get('rating', 'N/A'))),
                ("File Ext", str(record.get('file_ext', 'N/A'))),
                ("Resolution", f"{record.get('image_width', '0')} x {record.get('image_height', '0')}"),
                ("Score", str(record.get('score', 0))),
                ("Favorites", str(record.get('fav_count', 0)))
            ]

            for label, val in props:
                with ui.row().classes('w-full justify-between py-1 border-b border-zinc-850/40 text-sm'):
                    ui.label(label).classes('text-zinc-400 font-medium')
                    ui.label(val).classes('text-zinc-100 font-semibold')

        # AI Summary (Gemini Description)
        summary_text = record.get('regular_summary') or '*No summary available.*'
        with ui.expansion('🤖 AI Summary', value=True).classes('w-full border border-zinc-850 rounded-xl overflow-hidden bg-zinc-900/30 text-sm font-bold'):
            ui.label(summary_text).classes('text-sm text-zinc-300 p-3 leading-relaxed font-normal')

        # AI Layout Analysis
        if 'individual_parts' in record and record['individual_parts']:
            with ui.expansion('📊 AI Layout Analysis', value=False).classes('w-full border border-zinc-850 rounded-xl overflow-hidden bg-zinc-900/30 text-sm font-bold'):
                ui.markdown(record['individual_parts']).classes('text-xs text-zinc-300 p-3 leading-relaxed font-normal')

        # Space-separated tags list formatted into neat badge chips
        tags_str = record.get('tags') or record.get('tag_string') or ''
        tags_list = sorted([t.strip() for t in tags_str.split() if t.strip()])
        with ui.column().classes('w-full gap-2 mt-2'):
            ui.label(f'🏷️ Associated Tags ({len(tags_list)})').classes('text-xs font-bold uppercase tracking-wider text-zinc-500')
            with ui.row().classes('flex flex-wrap gap-1.5 w-full max-h-48 overflow-y-auto p-1'):
                for t in tags_list:
                    ui.badge(t).classes('px-2 py-1 text-xs font-medium bg-zinc-800 text-zinc-300 border border-zinc-700/50 hover:bg-cyan-950 hover:text-cyan-400 transition-colors cursor-pointer rounded-md')

async def refresh_header():
    if not header_container:
        return
    header_container.clear()
    with header_container:
        total_records = len(state.filtered_df)
        total_pages = max(1, (total_records + state.page_size - 1) // state.page_size)

        with ui.row().classes('items-center gap-2'):
            ui.button('⏪ First', on_click=lambda: jump_to_page(1)).props('size=sm outline color=cyan')
            ui.button('◀ Prev', on_click=lambda: page_turn(-1)).props('size=sm outline color=cyan').classes('w-16')

            ui.label(f"Page {state.page} of {total_pages}").classes('text-sm font-extrabold text-zinc-100 bg-zinc-800 border border-zinc-700 px-3 py-1.5 rounded-lg text-center')

            ui.button('Next ▶', on_click=lambda: page_turn(1)).props('size=sm outline color=cyan').classes('w-16')
            ui.button('Last ⏩', on_click=lambda: jump_to_page(total_pages)).props('size=sm outline color=cyan')

        with ui.row().classes('items-center gap-2'):
            ui.label("Jump:").classes('text-xs text-zinc-400')
            def on_jump(e):
                try:
                    val = int(e.value)
                    jump_to_page(val)
                except (ValueError, TypeError):
                    pass
            ui.number(value=state.page, min=1, max=total_pages, step=1, on_change=on_jump).classes('w-16 text-center').props('size=xs dense outlined dark color=cyan')

        ui.label(f"Matching: {total_records:,} / {len(state.offsets):,} records").classes('text-sm font-semibold text-cyan-400 border border-cyan-950 bg-cyan-950/20 px-3 py-1.5 rounded-lg')

async def refresh_batch_stats():
    if not batch_stats_container:
        return
    batch_stats_container.clear()
    with batch_stats_container:
        total_trimmed = len(state.trimmed_offsets)
        filtered_offsets_set = set(state.filtered_df['byte_offset'].tolist()) if not state.filtered_df.empty else set()
        current_filtered_trimmed = len(filtered_offsets_set.intersection(state.trimmed_offsets))

        ui.label(f"Total Trimmed in File: {total_trimmed:,}").classes('text-sm font-semibold text-red-400')
        ui.label(f"Current Filtered Trimmed: {current_filtered_trimmed:,}").classes('text-sm font-semibold text-orange-400')

async def refresh_sidebar_conditions():
    if not sidebar_conditions_container:
        return
    sidebar_conditions_container.clear()
    with sidebar_conditions_container:
        if state.filter_conditions:
            ui.label("Active Conditions:").classes('text-xs font-bold text-zinc-400 uppercase tracking-wider mt-2')
            for idx, cond in enumerate(state.filter_conditions):
                with ui.row().classes('w-full justify-between items-center bg-zinc-900 border border-zinc-800 p-1.5 rounded-lg'):
                    if cond['operator'] == 'is empty/null':
                        cond_str = f"{cond['column']} is empty/null"
                    else:
                        cond_str = f"{cond['column']} {cond['operator']} '{cond['value']}'"
                    ui.label(cond_str).classes('text-xs text-zinc-300 truncate max-w-[200px]')

                    def make_remove_handler(i):
                        return lambda: remove_condition(i)
                    ui.button('❌', on_click=make_remove_handler(idx)).props('flat dense size=xs').classes('text-red-400')

async def refresh_analytics():
    if not analytics_container:
        return
    analytics_container.clear()
    with analytics_container:
        if state.filtered_df.empty:
            ui.label("No data to compute charts.").classes('text-sm text-zinc-500')
            return

        avg_score = state.filtered_df['score'].mean()
        avg_favs = state.filtered_df['fav_count'].mean()
        avg_w = state.filtered_df['image_width'].mean()
        avg_h = state.filtered_df['image_height'].mean()

        num_filtered = len(state.filtered_df)
        num_trimmed_filtered = len(set(state.filtered_df['byte_offset']).intersection(state.trimmed_offsets))
        pct_trimmed = (num_trimmed_filtered / num_filtered * 100) if num_filtered > 0 else 0.0

        with ui.grid(columns=2).classes('w-full gap-2 text-center text-xs mt-2 border-b border-zinc-800 pb-3'):
            with ui.column().classes('bg-zinc-900 p-2 rounded-lg border border-zinc-800'):
                ui.label("Avg Score").classes('text-zinc-500 font-bold')
                ui.label(f"{avg_score:.1f}").classes('text-lg font-extrabold text-cyan-400')
            with ui.column().classes('bg-zinc-900 p-2 rounded-lg border border-zinc-800'):
                ui.label("Avg Favs").classes('text-zinc-500 font-bold')
                ui.label(f"{avg_favs:.1f}").classes('text-lg font-extrabold text-pink-400')
            with ui.column().classes('bg-zinc-900 p-2 rounded-lg border border-zinc-800'):
                ui.label("Avg Resolution").classes('text-zinc-500 font-bold')
                ui.label(f"{int(avg_w)}x{int(avg_h)}").classes('text-xs font-extrabold text-zinc-300 mt-1')
            with ui.column().classes('bg-zinc-900 p-2 rounded-lg border border-zinc-800'):
                ui.label("% Trimmed").classes('text-zinc-500 font-bold')
                ui.label(f"{pct_trimmed:.1f}%").classes('text-lg font-extrabold text-red-400')

        ui.label("Rating Distribution").classes('text-xs font-bold text-zinc-400 uppercase mt-2')
        try:
            rating_fig = get_ratings_chart(state.filtered_df)
            ui.plotly(rating_fig).classes('w-full bg-transparent')
        except Exception as e:
            ui.label(f"Rating chart err: {str(e)}").classes('text-xs text-zinc-600')

        ui.label("Top 20 Tags").classes('text-xs font-bold text-zinc-400 uppercase mt-2')
        try:
            tag_fig = get_tags_chart(state.filtered_df)
            ui.plotly(tag_fig).classes('w-full bg-transparent')
        except Exception as e:
            ui.label(f"Tag chart err: {str(e)}").classes('text-xs text-zinc-600')

async def refresh_file_status():
    if not file_status_container:
        return
    file_status_container.clear()
    with file_status_container:
        if not state.current_file:
            ui.label("No file loaded.").classes('text-xs text-zinc-500')
            return

        full_path = state.get_full_path()
        idx_path, parquet_path = eda_backend.get_cache_paths(full_path)
        file_size_val = os.path.getsize(full_path) if os.path.exists(full_path) else 0
        idx_size_val = os.path.getsize(idx_path) if os.path.exists(idx_path) else 0
        parquet_size_val = os.path.getsize(parquet_path) if os.path.exists(parquet_path) else 0

        with ui.column().classes('w-full text-xs text-zinc-400 gap-1 bg-zinc-950 p-3 rounded-lg border border-zinc-850'):
            ui.label(f"📁 File: {state.current_file}").classes('truncate font-bold text-zinc-300 w-full')
            ui.label(f"Total Lines: {len(state.offsets):,}")
            ui.label(f"JSONL Size: {format_bytes(file_size_val)}")
            ui.label(f"Offset Index: {format_bytes(idx_size_val)}")
            ui.label(f"Metadata Cache: {format_bytes(parquet_size_val)}")

async def refresh_full_ui():
    await refresh_header()
    await refresh_grid()
    await refresh_inspector()
    await refresh_batch_stats()
    await refresh_sidebar_conditions()
    await refresh_analytics()
    await refresh_file_status()

# Selection & Trimming Handlers
def select_active_card(idx):
    state.active_index = idx
    asyncio.create_task(refresh_grid())
    asyncio.create_task(refresh_inspector())

def toggle_trim_offset(offset, value):
    if value:
        state.trimmed_offsets.add(offset)
    else:
        state.trimmed_offsets.discard(offset)
    asyncio.create_task(refresh_grid())
    asyncio.create_task(refresh_inspector())
    asyncio.create_task(refresh_batch_stats())

def remove_condition(idx):
    state.filter_conditions.pop(idx)
    state.apply_all_filters()
    asyncio.create_task(refresh_full_ui())

def page_turn(delta):
    total_pages = max(1, (len(state.filtered_df) + state.page_size - 1) // state.page_size)
    new_page = state.page + delta
    if 1 <= new_page <= total_pages:
        state.page = new_page
        state.active_index = 0
        asyncio.create_task(refresh_grid())
        asyncio.create_task(refresh_header())
        asyncio.create_task(refresh_inspector())

def jump_to_page(p):
    total_pages = max(1, (len(state.filtered_df) + state.page_size - 1) // state.page_size)
    p = max(1, min(p, total_pages))
    if p != state.page:
        state.page = p
        state.active_index = 0
        asyncio.create_task(refresh_grid())
        asyncio.create_task(refresh_header())
        asyncio.create_task(refresh_inspector())

async def load_dataset(file_path):
    if state.loading:
        return
    state.loading = True
    loading_dialog.open()
    try:
        loop = asyncio.get_running_loop()

        def update_index_progress(percent, text):
            progress_bar.value = percent
            progress_label.text = text

        def run_index():
            def cb(curr, tot):
                p = curr / tot if tot > 0 else 0.0
                loop.call_soon_threadsafe(update_index_progress, p, f"Building offset index: {curr:,} / {tot:,} bytes")
            return eda_backend.get_or_build_index(file_path, progress_callback=cb)

        offsets = await run.io_bound(run_index)

        def update_meta_progress(percent, text):
            progress_bar.value = percent
            progress_label.text = text

        def run_meta():
            def cb(curr, tot):
                p = curr / tot if tot > 0 else 0.0
                loop.call_soon_threadsafe(update_meta_progress, p, f"Creating metadata cache: {curr:,} / {tot:,} rows")
            return eda_backend.get_or_build_metadata(file_path, offsets, progress_callback=cb)

        df = await run.io_bound(run_meta)

        state.offsets = offsets
        state.metadata_df = df
        state.trimmed_offsets = set()
        state.filter_conditions = []
        state.page = 1
        state.active_index = 0

        state.update_bounds()
        state.apply_all_filters()
        update_filters_ui()

    except Exception as e:
        ui.notify(f"Error loading dataset: {str(e)}", type="negative")
    finally:
        loading_dialog.close()
        state.loading = False
        await refresh_full_ui()

# Multiselect / bounds dynamic updating helper
def update_filters_ui():
    rating_select.options = state.all_ratings
    rating_select.value = state.rating_selection

    ext_select.options = state.all_exts
    ext_select.value = state.ext_selection

    score_slider.props(f'min={state.min_score} max={state.max_score}')
    score_slider.value = {'min': state.min_score, 'max': state.max_score}

    fav_slider.props(f'min={state.min_fav} max={state.max_fav}')
    fav_slider.value = {'min': state.min_fav, 'max': state.max_fav}

    w_slider.props(f'min={state.min_w} max={state.max_w}')
    w_slider.value = {'min': state.min_w, 'max': state.max_w}

    h_slider.props(f'min={state.min_h} max={state.max_h}')
    h_slider.value = {'min': state.min_h, 'max': state.max_h}

# Filter property change events
def on_rating_select(e):
    state.rating_selection = e.value
    state.apply_all_filters()
    asyncio.create_task(refresh_full_ui())

def on_ext_select(e):
    state.ext_selection = e.value
    state.apply_all_filters()
    asyncio.create_task(refresh_full_ui())

def on_slider_change(field, val):
    if not val:
        return
    min_val, max_val = val['min'], val['max']
    if field == 'score':
        state.score_range = [min_val, max_val]
    elif field == 'fav':
        state.fav_range = [min_val, max_val]
    elif field == 'w':
        state.w_range = [min_val, max_val]
    elif field == 'h':
        state.h_range = [min_val, max_val]
    state.apply_all_filters()
    asyncio.create_task(refresh_full_ui())

def on_text_query_change(e):
    state.text_query = e.value
    state.apply_all_filters()
    asyncio.create_task(refresh_full_ui())

def on_tag_query_change(e):
    state.tag_query = e.value
    state.apply_all_filters()
    asyncio.create_task(refresh_full_ui())

def on_sort_col_change(e):
    state.sort_col = e.value
    state.apply_all_filters()
    asyncio.create_task(refresh_full_ui())

def on_sort_order_change(e):
    state.sort_order = e.value
    state.apply_all_filters()
    asyncio.create_task(refresh_full_ui())

def on_page_size_change(e):
    state.page_size = e.value
    state.apply_all_filters()
    asyncio.create_task(refresh_full_ui())

def on_columns_change(e):
    state.num_columns_per_row = e.value
    asyncio.create_task(refresh_grid())

def mark_all_filtered_for_trim():
    if not state.filtered_df.empty:
        state.trimmed_offsets.update(state.filtered_df['byte_offset'].tolist())
        ui.notify(f"Marked {len(state.filtered_df):,} rows for trim.", type="info")
        state.apply_all_filters()
        asyncio.create_task(refresh_full_ui())

def unmark_all_filtered_for_trim():
    if not state.filtered_df.empty:
        state.trimmed_offsets.difference_update(state.filtered_df['byte_offset'].tolist())
        ui.notify(f"Unmarked {len(state.filtered_df):,} rows.", type="info")
        state.apply_all_filters()
        asyncio.create_task(refresh_full_ui())

# Export dialogs and actions
async def trigger_filtered_export(export_path):
    if not export_path:
        ui.notify("Please provide a destination path.", type="warning")
        return
    export_dialog.open()
    loop = asyncio.get_running_loop()

    def update_prog(percent, text):
        export_progress_bar.value = percent
        export_progress_label.text = text

    def do_export():
        def cb(curr, tot):
            p = curr / tot if tot > 0 else 1.0
            loop.call_soon_threadsafe(update_prog, p, f"Exported: {curr:,} / {tot:,} rows ({int(p*100)}%)")
        eda_backend.stream_export_jsonl(
            state.get_full_path(),
            export_path,
            state.filtered_df['byte_offset'].tolist(),
            progress_callback=cb
        )

    try:
        await run.io_bound(do_export)
        ui.notify(f"Successfully exported to {export_path}", type="positive")
    except Exception as e:
        ui.notify(f"Export failed: {str(e)}", type="negative")
    finally:
        await asyncio.sleep(1)
        export_dialog.close()

async def trigger_trim_export(trim_path):
    if len(state.trimmed_offsets) == 0:
        ui.notify("Your Trim List is empty. Select records to exclude first.", type="warning")
        return
    if not trim_path:
        ui.notify("Please provide a destination path.", type="warning")
        return
    trim_dialog.open()
    loop = asyncio.get_running_loop()

    def update_prog(percent, text):
        trim_progress_bar.value = percent
        trim_progress_label.text = text

    def do_trim():
        def cb(curr, tot):
            p = curr / tot if tot > 0 else 1.0
            loop.call_soon_threadsafe(update_prog, p, f"Writing trimmed file: {curr:,} / {tot:,} rows ({int(p*100)}%)")
        return eda_backend.trim_jsonl(
            state.get_full_path(),
            trim_path,
            state.offsets,
            state.trimmed_offsets,
            progress_callback=cb
        )

    try:
        remaining_count = await run.io_bound(do_trim)
        ui.notify(f"Successfully saved {remaining_count:,} remaining rows to {trim_path}", type="positive")
    except Exception as e:
        ui.notify(f"Trim failed: {str(e)}", type="negative")
    finally:
        await asyncio.sleep(1)
        trim_dialog.close()

# Keyboard Navigation Shortcuts via Native global ui.keyboard
def setup_keyboard_shortcuts():
    def handle_keydown(e: KeyEventArguments):
        # Ignore keyboard shortcuts when user is typing inside text or number input fields
        try:
            if e.active_element:
                tag = getattr(e.active_element, 'tagName', '').upper()
                if tag in ['INPUT', 'TEXTAREA'] or getattr(e.active_element, 'name', '') in ['input', 'textarea']:
                    return
        except Exception:
            pass

        if not e.action.keydown:
            return

        key = str(e.key.value).lower() if hasattr(e.key, 'value') else str(e.key).lower()

        # 1. Q / E: Page turns
        if key == 'q':
            page_turn(-1)
        elif key == 'e':
            page_turn(1)

        # 2. ArrowLeft / ArrowRight: selects items, turning pages when crossing bounds
        elif key == 'arrowleft':
            if state.active_index > 0:
                state.active_index -= 1
                asyncio.create_task(refresh_grid())
                asyncio.create_task(refresh_inspector())
            elif state.page > 1:
                state.page -= 1
                total_records = len(state.filtered_df)
                start_idx = (state.page - 1) * state.page_size
                end_idx = min(start_idx + state.page_size, total_records)
                prev_page_items = end_idx - start_idx
                state.active_index = prev_page_items - 1
                asyncio.create_task(refresh_full_ui())

        elif key == 'arrowright':
            total_records = len(state.filtered_df)
            start_idx = (state.page - 1) * state.page_size
            end_idx = min(start_idx + state.page_size, total_records)
            current_page_items = end_idx - start_idx
            total_pages = max(1, (total_records + state.page_size - 1) // state.page_size)

            if state.active_index < current_page_items - 1:
                state.active_index += 1
                asyncio.create_task(refresh_grid())
                asyncio.create_task(refresh_inspector())
            elif state.page < total_pages:
                state.page += 1
                state.active_index = 0
                asyncio.create_task(refresh_full_ui())

        # 3. Space / T: Toggles active card trim status
        elif key in [' ', 't']:
            if state.page_records and state.active_index < len(state.page_records):
                start_idx = (state.page - 1) * state.page_size
                row_idx = start_idx + state.active_index
                row = state.filtered_df.iloc[row_idx]
                offset = row['byte_offset']
                is_trimmed = offset in state.trimmed_offsets
                toggle_trim_offset(offset, not is_trimmed)

    ui.keyboard(on_key=handle_keydown)

# Main Application Frame Layout
def build_ui():
    global header_container, grid_container, inspector_container, sidebar_conditions_container, batch_stats_container, analytics_container, file_status_container
    global rating_select, ext_select, score_slider, fav_slider, w_slider, h_slider, cond_value_input, op_select, col_select
    global loading_dialog, progress_bar, progress_label
    global export_dialog, export_progress_bar, export_progress_label
    global trim_dialog, trim_progress_bar, trim_progress_label

    # Instantiate dialogs within page context
    loading_dialog = ui.dialog().props('persistent')
    with loading_dialog, ui.card().classes('w-96 bg-zinc-900 border border-zinc-800 p-4 flex flex-col gap-3'):
        ui.label("⚙️ Preparing Cache & Index...").classes('text-lg font-bold text-cyan-400')
        progress_bar = ui.linear_progress(value=0.0).props('color=cyan')
        progress_label = ui.label("Initializing preparation process...").classes('text-sm text-zinc-400')

    export_dialog = ui.dialog()
    with export_dialog, ui.card().classes('w-96 bg-zinc-900 border border-zinc-800 p-4 flex flex-col gap-3'):
        ui.label("Exporting Filtered JSONL...").classes('text-lg font-bold text-cyan-400')
        export_progress_bar = ui.linear_progress(value=0.0).props('color=cyan')
        export_progress_label = ui.label("0% (0 / 0 rows)").classes('text-sm text-zinc-400')

    trim_dialog = ui.dialog()
    with trim_dialog, ui.card().classes('w-96 bg-zinc-900 border border-zinc-800 p-4 flex flex-col gap-3'):
        ui.label("Writing Trimmed JSONL...").classes('text-lg font-bold text-red-400')
        trim_progress_bar = ui.linear_progress(value=0.0).props('color=red')
        trim_progress_label = ui.label("0% (0 / 0 rows)").classes('text-sm text-zinc-400')

    # Outer page structure: h-screen w-screen overflow-hidden flex flex-col bg-zinc-950 text-white font-sans
    with ui.column().classes('h-screen w-screen overflow-hidden bg-zinc-950 text-white font-sans gap-0 p-0'):

        # 1. Main Header Panel (h-14 flex-none bg-zinc-900 border-b border-zinc-800 flex items-center justify-between px-4 gap-4)
        header_container = ui.row().classes('h-14 w-full bg-zinc-900 border-b border-zinc-800 flex items-center justify-between px-4 flex-none gap-4')

        # 2. Main content row: flex-grow overflow-hidden flex flex-row flex-nowrap gap-0
        with ui.row().classes('flex-grow w-full overflow-hidden flex flex-row flex-nowrap gap-0'):

            # Left Sidebar: approx. 20% or 300px (w-80 flex-none h-full bg-zinc-900 border-r border-zinc-850 p-4 flex flex-col gap-4 overflow-y-auto)
            with ui.column().classes('w-80 flex-none h-full bg-zinc-900 border-r border-zinc-850 p-4 flex flex-col gap-4 overflow-y-auto'):

                # Section 1: Dataset Selector
                with ui.column().classes('w-full gap-2 border-b border-zinc-800 pb-3'):
                    ui.label("📁 Dataset Selection").classes('text-sm font-bold uppercase tracking-wider text-cyan-400')

                    def on_dir_change(e):
                        state.current_dir = e.value
                        state.jsonl_files = state.list_jsonl_files()
                        file_sel.options = state.jsonl_files
                        if state.jsonl_files:
                            file_sel.value = state.jsonl_files[0]
                    ui.input(label="JSONL Directory", value=state.current_dir, on_change=on_dir_change).classes('w-full').props('outlined dark dense color=cyan')

                    async def on_file_change(e):
                        if e.value:
                            state.current_file = e.value
                            await load_dataset(state.get_full_path())
                    file_sel = ui.select(state.jsonl_files, value=state.current_file, label="Select JSONL File", on_change=on_file_change).classes('w-full').props('outlined dark dense color=cyan')

                    file_status_container = ui.column().classes('w-full')

                # Section 2: Dynamic Query Builder
                with ui.column().classes('w-full gap-2 border-b border-zinc-800 pb-3'):
                    ui.label("🛠️ Multi-Key Filter Builder").classes('text-sm font-bold uppercase tracking-wider text-cyan-400')

                    col_opts = ['id', 'score', 'fav_count', 'rating', 'file_ext', 'image_width', 'image_height', 'tags', 'regular_summary', 'created_at']
                    col_select = ui.select(col_opts, value='score', label="Filter Column").classes('w-full').props('outlined dark dense color=cyan')

                    op_opts = ['=', '!=', '>', '<', '>=', '<=', 'contains', 'starts with', 'ends with', 'is empty/null']

                    def on_op_change(e):
                        if e.value == 'is empty/null':
                            cond_value_input.disable()
                        else:
                            cond_value_input.enable()
                    op_select = ui.select(op_opts, value='=', label="Filter Operator", on_change=on_op_change).classes('w-full').props('outlined dark dense color=cyan')

                    cond_value_input = ui.input(label="Filter Value", value="").classes('w-full').props('outlined dark dense color=cyan')

                    def add_condition():
                        val = cond_value_input.value
                        op = op_select.value
                        col = col_select.value
                        if val or op == 'is empty/null':
                            new_cond = {
                                'column': col,
                                'operator': op,
                                'value': val.strip() if val else ""
                            }
                            if new_cond not in state.filter_conditions:
                                state.filter_conditions.append(new_cond)
                                state.apply_all_filters()
                                asyncio.create_task(refresh_full_ui())
                    ui.button('➕ Add Condition', on_click=add_condition).classes('w-full py-1').props('color=cyan')

                    sidebar_conditions_container = ui.column().classes('w-full gap-1.5')

                # Section 3: Standard Filters
                with ui.column().classes('w-full gap-2 border-b border-zinc-800 pb-3'):
                    ui.label("🔍 Standard Filters").classes('text-sm font-bold uppercase tracking-wider text-cyan-400')

                    rating_select = ui.select(state.all_ratings, value=state.rating_selection, multiple=True, label="Content Rating", on_change=on_rating_select).classes('w-full').props('outlined dark dense color=cyan')
                    ext_select = ui.select(state.all_exts, value=state.ext_selection, multiple=True, label="File Extension", on_change=on_ext_select).classes('w-full').props('outlined dark dense color=cyan')

                    ui.label("Score Range").classes('text-xs font-bold text-zinc-400 mt-1')
                    score_slider = ui.range(min=state.min_score, max=state.max_score, value={'min': state.min_score, 'max': state.max_score}, on_change=lambda e: on_slider_change('score', e.value)).classes('w-full').props('dark color=cyan font-xs label')

                    ui.label("Favorites Range").classes('text-xs font-bold text-zinc-400 mt-1')
                    fav_slider = ui.range(min=state.min_fav, max=state.max_fav, value={'min': state.min_fav, 'max': state.max_fav}, on_change=lambda e: on_slider_change('fav', e.value)).classes('w-full').props('dark color=cyan label')

                    ui.label("Image Width Range").classes('text-xs font-bold text-zinc-400 mt-1')
                    w_slider = ui.range(min=state.min_w, max=state.max_w, value={'min': state.min_w, 'max': state.max_w}, on_change=lambda e: on_slider_change('w', e.value)).classes('w-full').props('dark color=cyan label')

                    ui.label("Image Height Range").classes('text-xs font-bold text-zinc-400 mt-1')
                    h_slider = ui.range(min=state.min_h, max=state.max_h, value={'min': state.min_h, 'max': state.max_h}, on_change=lambda e: on_slider_change('h', e.value)).classes('w-full').props('dark color=cyan label')

                    ui.input(label="Summary Text Search", value=state.text_query, on_change=on_text_query_change).classes('w-full').props('outlined dark dense color=cyan clearable')
                    ui.input(label="Tag Query (AND / -EXCLUDE)", value=state.tag_query, on_change=on_tag_query_change).classes('w-full').props('outlined dark dense color=cyan clearable placeholder="1girl smile -wolf_girl"')

                # Section 4: Collapsible Analytics Charts
                with ui.expansion("📈 Real-Time Analytics", value=False).classes('w-full border border-zinc-800 rounded-lg overflow-hidden bg-zinc-950/20 text-sm font-bold'):
                    analytics_container = ui.column().classes('w-full gap-2 p-2 font-normal')

                # Section 5: Batch Operations & Save/Export
                with ui.column().classes('w-full gap-2'):
                    ui.label("⚡ Batch Operations").classes('text-sm font-bold uppercase tracking-wider text-cyan-400')

                    batch_stats_container = ui.column().classes('w-full')

                    ui.button('🚨 Mark ALL Matching for Trim', on_click=mark_all_filtered_for_trim).classes('w-full text-xs font-bold py-1.5').props('color=red')
                    ui.button('♻️ Unmark ALL Matching', on_click=unmark_all_filtered_for_trim).classes('w-full text-xs font-bold py-1.5').props('color=green')

                    ui.label("🔃 Sorting").classes('text-xs font-bold text-zinc-400 mt-1')
                    ui.select(["score", "fav_count", "image_width", "image_height", "created_at", "id", "byte_offset"], value=state.sort_col, label="Sort By", on_change=on_sort_col_change).classes('w-full').props('outlined dark dense color=cyan')
                    ui.radio(["Descending", "Ascending"], value=state.sort_order, on_change=on_sort_order_change).classes('w-full text-xs').props('dark inline dense')

                    ui.label("🖼️ Gallery Settings").classes('text-sm font-bold uppercase tracking-wider text-cyan-400 mt-2')
                    ui.select([1, 2, 4, 8, 12], value=state.page_size, label="Page Size", on_change=on_page_size_change).classes('w-full').props('outlined dark dense color=cyan')
                    ui.select([1, 2, 3, 4, 6], value=state.num_columns_per_row, label="Columns per Row", on_change=on_columns_change).classes('w-full').props('outlined dark dense color=cyan')

                # Section 6: Exporter Panels
                with ui.expansion("📥 Exporters", value=False).classes('w-full border border-zinc-800 rounded-lg overflow-hidden bg-zinc-950/20 text-sm font-bold'):
                    with ui.column().classes('w-full gap-3 p-2 font-normal'):
                        ui.label("Export Filtered JSONL").classes('text-xs font-bold text-cyan-400 border-b border-zinc-800 pb-1 w-full')
                        default_export_path = os.path.join(state.current_dir, f"{os.path.splitext(state.current_file or 'file')[0]}_filtered.jsonl")
                        filtered_export_input = ui.input(label="Destination Path", value=default_export_path).classes('w-full').props('outlined dark dense color=cyan')
                        ui.button("Start Filtered Export", on_click=lambda: trigger_filtered_export(filtered_export_input.value)).classes('w-full text-xs py-1.5').props('color=cyan')

                        ui.label("Write Trimmed JSONL").classes('text-xs font-bold text-red-400 border-b border-zinc-800 pb-1 w-full mt-2')
                        default_trim_path = os.path.join(state.current_dir, f"{os.path.splitext(state.current_file or 'file')[0]}_trimmed.jsonl")
                        trimmed_export_input = ui.input(label="Destination Path", value=default_trim_path).classes('w-full').props('outlined dark dense color=cyan')
                        ui.button("Start Trim & Save", on_click=lambda: trigger_trim_export(trimmed_export_input.value)).classes('w-full text-xs py-1.5').props('color=red')

            # Left Panel - Gallery Image Grid (60% content column: bg-zinc-950 h-full overflow-y-auto p-4 flex flex-col gap-4)
            grid_container = ui.column().classes('w-3/5 flex-grow h-full bg-zinc-950 p-4 overflow-y-auto flex flex-col gap-4')

            # Right Panel - Persistent Detailed Record Inspector (40% content column: bg-zinc-900 border-l border-zinc-850 h-full overflow-y-auto p-4 flex flex-col gap-4)
            inspector_container = ui.column().classes('w-2/5 flex-none h-full bg-zinc-900 border-l border-zinc-850 p-4 overflow-y-auto flex flex-col gap-4')

    setup_keyboard_shortcuts()

@ui.page('/')
async def index():
    build_ui()
    if state.current_file:
        await load_dataset(state.get_full_path())
    else:
        await refresh_full_ui()

# Start application server
if __name__ in {"__main__", "__mp_main__"}:
    show_browser = os.environ.get("NICEGUI_SHOW", "true").lower() == "true"
    ui.run(
        title="NiceGUI EDA & Trimming Tool",
        dark=True,
        reload=False, # Disable auto-reload for stability in testing
        port=args.port,
        host='127.0.0.1',
        show=show_browser
    )
