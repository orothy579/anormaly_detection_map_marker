#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PNG 맵 좌표·방향 등록기 (YAML 지원) — v1.8 (New Episode 기능)
- New Episode: start, goal 입력 후 path 입력 (waypoint는 점만, 방향 없음)
- Path 연결 선분 및 화살표 표시
- Path 입력 완료 후 start/goal 재입력 여부 확인
- start/goal 고정 후 여러 trajectory 생성 가능
"""

import os
import math
import time
import csv
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

try:
    from PIL import Image, ImageTk
except ImportError as e:
    raise SystemExit("Pillow가 필요합니다. `pip install pillow`") from e

try:
    import yaml
except ImportError as e:
    raise SystemExit("PyYAML이 필요합니다. `pip install pyyaml`") from e

try:
    import numpy as np
    from scipy import ndimage
    from scipy.ndimage import label
except ImportError as e:
    raise SystemExit("numpy와 scipy가 필요합니다. `pip install numpy scipy`") from e


@dataclass
class Marker:
    id: str
    type: str  # 'start' | 'waypoint' | 'goal'
    x: float   # image px
    y: float   # image px
    theta_deg: float  # heading in degrees [0..360)
    source: str = "live"  # 'live' or 'file'
    route_id: int = 0     # 같은 경로 묶음 번호
    seq: int = 0          # 경로 내 순서 (1..N)


@dataclass
class MapMeta:
    resolution: float = 0.05
    origin_x: float = 0.0
    origin_y: float = 0.0
    origin_yaw: float = 0.0  # radians
    image_path: Optional[str] = None
    negate: Optional[int] = None
    occupied_thresh: Optional[float] = None
    free_thresh: Optional[float] = None


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def img_to_map(u: float, v: float, theta_img_deg: float, W: int, H: int, meta: MapMeta, use_center: bool = True):
    x0, y0, yaw = meta.origin_x, meta.origin_y, meta.origin_yaw
    res = meta.resolution
    offs = 0.5 if use_center else 0.0
    x_map = x0 + (u + offs) * res
    y_map = y0 + (H - (v + offs)) * res
    theta_map = wrap_to_pi(math.radians(theta_img_deg))
    return x_map, y_map, theta_map


def map_to_img(x_map: float, y_map: float, theta_map_rad: float, W: int, H: int, meta: MapMeta, use_center: bool = True):
    x0, y0, yaw = meta.origin_x, meta.origin_y, meta.origin_yaw
    res = meta.resolution
    offs = 0.5 if use_center else 0.0
    u = (x_map - x0) / res - offs
    v = H - (y_map - y0) / res - offs
    theta_img = wrap_to_pi(theta_map_rad)
    return u, v, math.degrees(theta_img)


class MapMarkerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("맵 마커 등록기 v1.8")
        self.geometry("1280x880")
        self.minsize(980, 640)

        # Image & canvas
        self.img: Optional[Image.Image] = None
        self.img_tk: Optional[ImageTk.PhotoImage] = None
        self.img_path: Optional[str] = None
        self.img_size: Tuple[int, int] = (0, 0)  # (W, H)

        # YAML meta
        self.yaml_path: Optional[str] = None
        self.meta: MapMeta = MapMeta()

        # Route state
        self.use_center = tk.BooleanVar(value=True)
        self.hide_live = tk.BooleanVar(value=True)

        self.markers: List[Marker] = []  # stack top at index 0
        self.current_route_id: int = 0
        self.next_seq: int = 0  # within current route

        # Episode state: 'idle' | 'start_input' | 'goal_input' | 'path_input' | 'fixed_start_goal'
        self.episode_mode: str = 'idle'
        self.fixed_start: Optional[Marker] = None
        self.fixed_goal: Optional[Marker] = None
        self.current_path_waypoints: List[Marker] = []  # 현재 path의 waypoint들

        # preview
        self.drag_start: Optional[Tuple[float, float]] = None
        self.preview_line = None
        self.preview_head = None

        # highlight state
        self.selected_idx: Optional[int] = None
        self.last_hover_idx: Optional[int] = None
        self.hovered_route_id: Optional[int] = None  # 호버된 마커의 route_id

        # Barrier state
        self.barrier_map: Optional[np.ndarray] = None  # 2D boolean array, True = barrier
        self.show_barrier = tk.BooleanVar(value=True)
        self.robot_width_m: float = 1.0  # meters
        self.robot_depth_m: float = 1.0  # meters

        # Zoom state
        self.zoom_level: float = 1.0  # 현재 줌 레벨
        self.zoom_factor: float = 1.1  # 줌 인/아웃 배율
        
        # Cached images for performance
        self._barrier_pil_original: Optional[Image.Image] = None  # 원본 barrier 이미지 (PIL)
        self._barrier_tk_cached: Optional[ImageTk.PhotoImage] = None  # 캐시된 barrier 이미지
        self._barrier_zoom_cached: float = 0.0  # 캐시된 barrier의 줌 레벨

        # Build UI
        self._build_ui()
        self._bind_scroll_events()

    # -------------- UI --------------
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        root = ttk.Frame(self)
        root.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=0)  # left panel
        root.columnconfigure(1, weight=1)  # canvas
        root.rowconfigure(0, weight=1)

        # Left panel
        left = ttk.Frame(root, padding=10)
        left.grid(row=0, column=0, sticky="ns")
        for i in range(80):
            left.rowconfigure(i, weight=0)
        left.rowconfigure(79, weight=1)

        # --- YAML load ---
        ttk.Label(left, text="1) YAML 불러오기 (map_server)", font=("", 11, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Button(left, text="YAML 열기...", command=self.on_open_yaml).grid(row=1, column=0, sticky="ew")
        self.yaml_info = ttk.Label(left, text="경로: -", foreground="#666")
        self.yaml_info.grid(row=2, column=0, sticky="w", pady=(4, 4))
        self.meta_info = ttk.Label(left, text="resolution: -, origin: -, yaw: -", foreground="#666")
        self.meta_info.grid(row=3, column=0, sticky="w", pady=(0, 6))
        self.img_info = ttk.Label(left, text="이미지: -", foreground="#666")
        self.img_info.grid(row=4, column=0, sticky="w", pady=(0, 12))

        # --- Barrier generation ---
        ttk.Label(left, text="1-1) 로봇 배리어 생성", font=("", 11, "bold")).grid(row=5, column=0, sticky="w", pady=(0, 6))
        barrier_frame = ttk.Frame(left)
        barrier_frame.grid(row=7, column=0, sticky="ew", pady=(0, 12))
        ttk.Label(barrier_frame, text="로봇 가로(m):").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.robot_width_entry = ttk.Entry(barrier_frame, width=8)
        self.robot_width_entry.insert(0, "1.0")
        self.robot_width_entry.grid(row=0, column=1, sticky="w", padx=(0, 8))
        ttk.Label(barrier_frame, text="로봇 세로(m):").grid(row=0, column=2, sticky="w", padx=(0, 4))
        self.robot_depth_entry = ttk.Entry(barrier_frame, width=8)
        self.robot_depth_entry.insert(0, "1.0")
        self.robot_depth_entry.grid(row=0, column=3, sticky="w")
        ttk.Button(left, text="배리어 생성", command=self.on_generate_barrier).grid(row=8, column=0, sticky="ew", pady=(4, 4))
        ttk.Checkbutton(left, text="배리어 표시", variable=self.show_barrier, command=self._redraw_all_markers).grid(row=9, column=0, sticky="w", pady=(0, 0))

        # --- Episode control ---
        ttk.Label(left, text="2) 경로 입력", font=("", 11, "bold")).grid(row=10, column=0, sticky="w", pady=(0, 6))
        self.new_episode_btn = ttk.Button(left, text="새 경로 입력", command=self.on_new_episode)
        self.new_episode_btn.grid(row=11, column=0, sticky="ew", pady=(4, 4))
        self.complete_path_btn = ttk.Button(left, text="경로 입력 완료", command=self.on_complete_path, state="disabled")
        self.complete_path_btn.grid(row=12, column=0, sticky="ew", pady=(4, 4))
        self.episode_status = ttk.Label(left, text="status: idle", foreground="#666", font=("", 9))
        self.episode_status.grid(row=13, column=0, sticky="w", pady=(0, 6))
        
        # --- 좌표 직접 입력 (맵 좌표계) ---
        coord_input_frame = ttk.LabelFrame(left, text="맵 좌표 직접 입력 (미터)", padding=6)
        coord_input_frame.grid(row=14, column=0, sticky="ew", pady=(6, 6))
        
        # Start 좌표 입력
        ttk.Label(coord_input_frame, text="시작점 (x, y, θ):").grid(row=0, column=0, sticky="w", pady=(0, 4))
        start_coord_frame = ttk.Frame(coord_input_frame)
        start_coord_frame.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        self.start_x_entry = ttk.Entry(start_coord_frame, width=8)
        self.start_x_entry.grid(row=0, column=0, padx=(0, 4))
        self.start_y_entry = ttk.Entry(start_coord_frame, width=8)
        self.start_y_entry.grid(row=0, column=1, padx=(0, 4))
        self.start_theta_entry = ttk.Entry(start_coord_frame, width=8)
        self.start_theta_entry.grid(row=0, column=2, padx=(0, 4))
        ttk.Button(start_coord_frame, text="입력", command=self.on_input_start_coord, width=6).grid(row=0, column=3)
        
        # Goal 좌표 입력
        ttk.Label(coord_input_frame, text="목적지 (x, y, θ):").grid(row=2, column=0, sticky="w", pady=(0, 4))
        goal_coord_frame = ttk.Frame(coord_input_frame)
        goal_coord_frame.grid(row=3, column=0, sticky="ew", pady=(0, 0))
        self.goal_x_entry = ttk.Entry(goal_coord_frame, width=8)
        self.goal_x_entry.grid(row=0, column=0, padx=(0, 4))
        self.goal_y_entry = ttk.Entry(goal_coord_frame, width=8)
        self.goal_y_entry.grid(row=0, column=1, padx=(0, 4))
        self.goal_theta_entry = ttk.Entry(goal_coord_frame, width=8)
        self.goal_theta_entry.grid(row=0, column=2, padx=(0, 4))
        ttk.Button(goal_coord_frame, text="입력", command=self.on_input_goal_coord, width=6).grid(row=0, column=3)
        
        ttk.Checkbutton(left, text="이전 라이브 경로 숨김(최신 경로만 표시)", variable=self.hide_live, command=self._redraw_all_markers).grid(row=15, column=0, sticky="w", pady=(6, 12))
        ttk.Label(left, text="캔버스: 좌클릭 → 점 입력 / 가운데버튼 드래그: 팬 / Ctrl+휠: 줌", foreground="#666").grid(row=16, column=0, sticky="w", pady=(0, 12))

        # --- Save/Load/Clear ---
        ttk.Label(left, text="3) 저장 / 불러오기 / 관리", font=("", 11, "bold")).grid(row=17, column=0, sticky="w", pady=(0, 6))
        btns = ttk.Frame(left)
        btns.grid(row=18, column=0, sticky="ew", pady=(0, 2))
        ttk.Button(btns, text="저장(.json)", command=self.on_save).pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="불러오기(.json/.txt)", command=self.on_load_txt).pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="전체 삭제", command=self.on_clear_all).pack(side="left")
        ttk.Label(left, text="각도: 0°→오른쪽(+X), 90°→위(+Y) / map은 위(+Y)", foreground="#666").grid(row=19, column=0, sticky="w", pady=(10, 0))

        ttk.Label(left, text="마커 목록", font=("", 11, "bold")).grid(row=20, column=0, sticky="w", pady=(12, 6))
        self.marker_list = tk.Listbox(left, height=18)
        self.marker_list.grid(row=21, column=0, sticky="nsew")
        ttk.Button(left, text="선택 삭제", command=self.on_delete_selected).grid(row=22, column=0, sticky="ew", pady=(6, 0))

        # Listbox interactions: hover + select
        self.marker_list.bind("<<ListboxSelect>>", self.on_list_select)
        self.marker_list.bind("<Motion>", self.on_list_hover_motion)
        self.marker_list.bind("<Leave>", self.on_list_hover_leave)

        # Canvas area + scrollbars
        canvas_wrap = ttk.Frame(root, padding=(0, 10, 10, 10))
        canvas_wrap.grid(row=0, column=1, sticky="nsew")
        canvas_wrap.rowconfigure(0, weight=1)
        canvas_wrap.columnconfigure(0, weight=1)

        # Scrollbars
        self.v_scroll = ttk.Scrollbar(canvas_wrap, orient="vertical")
        self.h_scroll = ttk.Scrollbar(canvas_wrap, orient="horizontal")

        self.canvas = tk.Canvas(canvas_wrap, bg="#f5f5f5", highlightthickness=0,
                                yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.v_scroll.config(command=self.canvas.yview)
        self.h_scroll.config(command=self.canvas.xview)
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")

        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Middle-button pan
        self.canvas.bind("<Button-2>", self.on_middle_down)
        self.canvas.bind("<B2-Motion>", self.on_middle_drag)

    # -------------- Scroll bindings --------------
    def _bind_scroll_events(self):
        # Windows/macOS vertical/horizontal
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)              # vertical
        self.canvas.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)  # horizontal
        # Linux (X11) vertical
        self.canvas.bind("<Button-4>", self._on_linux_wheel)
        self.canvas.bind("<Button-5>", self._on_linux_wheel)
        # Ctrl + Wheel for zoom
        self.canvas.bind("<Control-MouseWheel>", self._on_zoom)
        self.canvas.bind("<Control-Button-4>", self._on_zoom)
        self.canvas.bind("<Control-Button-5>", self._on_zoom)

    def _on_mousewheel(self, event):
        # Ctrl 누르면 줌, Shift 누르면 수평 스크롤, 아니면 수직 스크롤
        if event.state & 0x0004:  # Control key
            self._on_zoom(event)
        elif event.state & 0x0001:  # Shift key
            self.canvas.xview_scroll(-1 * int(event.delta / 120), "units")
        else:
            self.canvas.yview_scroll(-1 * int(event.delta / 120), "units")

    def _on_shift_mousewheel(self, event):
        self.canvas.xview_scroll(-1 * int(event.delta / 120), "units")

    def _on_linux_wheel(self, event):
        # Linux wheel events
        # Ctrl 키가 눌려있으면 _on_zoom이 처리하도록 함 (바인딩된 이벤트 사용)
        if event.state & 0x0004:  # Control key
            # _on_zoom이 처리하도록 이벤트를 전달하지 않고 스크롤만 처리
            # Ctrl+Button-4/5는 이미 _on_zoom에 바인딩되어 있으므로 여기서는 처리하지 않음
            return
        # Ctrl 키가 없을 때만 스크롤 처리
        if event.num == 4:
            self.canvas.yview_scroll(-3, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(3, "units")

    def _on_zoom(self, event):
        """Ctrl + 마우스 휠로 줌 인/아웃"""
        if not self.img:
            return
        
        # Get mouse position in canvas coordinates
        if hasattr(event, 'x') and hasattr(event, 'y'):
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
        else:
            # Fallback: use canvas center
            x = self.canvas.winfo_width() / 2
            y = self.canvas.winfo_height() / 2
        
        # Determine zoom direction
        if hasattr(event, 'delta') and event.delta != 0:
            # Windows/macOS
            delta = event.delta
            if delta > 0:
                zoom_factor = self.zoom_factor  # 확대
            else:
                zoom_factor = 1.0 / self.zoom_factor  # 축소
        elif hasattr(event, 'num'):
            # Linux
            if event.num == 4:
                zoom_factor = self.zoom_factor  # 확대
            elif event.num == 5:
                zoom_factor = 1.0 / self.zoom_factor  # 축소
            else:
                return  # 알 수 없는 이벤트
        else:
            return  # 알 수 없는 이벤트
        
        self._zoom_at_point(x, y, zoom_factor)

    def _zoom_at_point(self, x: float, y: float, zoom_factor: float):
        """특정 점을 중심으로 줌"""
        if not self.img:
            return
        
        # Limit zoom level
        new_zoom = self.zoom_level * zoom_factor
        if new_zoom < 0.1 or new_zoom > 10.0:
            return
        
        # Get mouse position in canvas coordinates (before zoom)
        canvas_x = self.canvas.canvasx(x)
        canvas_y = self.canvas.canvasy(y)
        
        # Update zoom level
        old_zoom = self.zoom_level
        self.zoom_level = new_zoom
        
        # Calculate new image size
        new_width = int(self.img_size[0] * self.zoom_level)
        new_height = int(self.img_size[1] * self.zoom_level)
        
        # Resize image (NEAREST는 너무 품질이 낮으므로, 크기가 크면 BILINEAR 사용)
        # 큰 이미지의 경우 BILINEAR이 LANCZOS보다 빠르면서도 품질이 좋음
        if new_width > 2000 or new_height > 2000:
            resized_img = self.img.resize((new_width, new_height), Image.Resampling.BILINEAR)
        else:
            resized_img = self.img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.img_tk = ImageTk.PhotoImage(resized_img)
        
        # Delete old map image and redraw
        self.canvas.delete("map")
        self.canvas.create_image(0, 0, anchor="nw", image=self.img_tk, tags=("map",))
        
        # Update scroll region
        self.canvas.config(scrollregion=(0, 0, new_width, new_height))
        
        # Calculate mouse position in scroll region coordinates (before zoom)
        sx0, sx1 = self.canvas.xview()
        sy0, sy1 = self.canvas.yview()
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        
        old_scroll_width = self.img_size[0] * old_zoom
        old_scroll_height = self.img_size[1] * old_zoom
        
        mouse_x_in_scroll = sx0 * old_scroll_width + (canvas_x / canvas_width) * (sx1 - sx0) * old_scroll_width
        mouse_y_in_scroll = sy0 * old_scroll_height + (canvas_y / canvas_height) * (sy1 - sy0) * old_scroll_height
        
        # Calculate new scroll position to keep mouse point fixed
        new_sx0 = (mouse_x_in_scroll - (x / canvas_width) * new_width) / new_width
        new_sy0 = (mouse_y_in_scroll - (y / canvas_height) * new_height) / new_height
        
        # Clamp scroll position
        new_sx0 = max(0.0, min(1.0, new_sx0))
        new_sy0 = max(0.0, min(1.0, new_sy0))
        
        self.canvas.xview_moveto(new_sx0)
        self.canvas.yview_moveto(new_sy0)
        
        # Redraw markers and paths (they will be scaled correctly)
        self._redraw_all_markers()

    def on_middle_down(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def on_middle_drag(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    # -------------- YAML handling --------------
    def on_open_yaml(self):
        path = filedialog.askopenfilename(
            title="YAML 열기",
            filetypes=[("YAML", "*.yaml *.yml *.YAML *.YML"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            messagebox.showerror("에러", f"YAML을 열 수 없습니다.\n{e}")
            return

        try:
            resolution = float(data.get("resolution", 0.05))
            origin = data.get("origin", [0.0, 0.0, 0.0])
            if not isinstance(origin, (list, tuple)) or len(origin) < 3:
                raise ValueError("origin 필드는 [x, y, yaw] 리스트여야 합니다.")
            image_rel = data.get("image", None)
        except Exception as e:
            messagebox.showerror("에러", f"YAML 파싱 오류: {e}")
            return

        self.yaml_path = path
        self.meta = MapMeta(
            resolution=resolution,
            origin_x=float(origin[0]),
            origin_y=float(origin[1]),
            origin_yaw=float(origin[2]),
            image_path=image_rel,
            negate=data.get("negate"),
            occupied_thresh=data.get("occupied_thresh"),
            free_thresh=data.get("free_thresh"),
        )
        self.yaml_info.config(text=f"경로: {path}")
        yaw_deg = math.degrees(self.meta.origin_yaw)
        self.meta_info.config(text=f"resolution: {self.meta.resolution}, origin: ({self.meta.origin_x:.3f}, {self.meta.origin_y:.3f}), yaw: {self.meta.origin_yaw:.3f} rad ({yaw_deg:.2f}°)")
        
        # Update barrier info
        if hasattr(self, 'barrier_info'):
            self.barrier_info.config(
                text=f"YAML 로드됨: resolution={self.meta.resolution:.4f} m/pixel, occupied_thresh={self.meta.occupied_thresh or 0.65}, negate={self.meta.negate or 0}",
                foreground="#0066cc"
            )

        if self.meta.image_path:
            img_path = self.meta.image_path
            if not os.path.isabs(img_path):
                img_path = os.path.join(os.path.dirname(self.yaml_path), img_path)
            if os.path.exists(img_path):
                self._load_image(img_path)
            else:
                messagebox.showwarning("경고", f"YAML의 image 경로를 찾을 수 없습니다:\n{img_path}\n수동으로 이미지를 불러오세요.")

    def _load_image(self, path: str):
        try:
            img = Image.open(path).convert("RGBA")
        except Exception as e:
            messagebox.showerror("에러", f"이미지를 열 수 없습니다.\n{e}")
            return
        self.img = img
        self.img_path = path
        self.img_size = img.size  # (W, H)
        self.img_tk = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        # Reset zoom when loading new image
        self.zoom_level = 1.0
        self.canvas.config(scrollregion=(0, 0, self.img_size[0], self.img_size[1]))
        self.canvas.create_image(0, 0, anchor="nw", image=self.img_tk, tags=("map",))
        # Reset barrier when loading new image
        self.barrier_map = None
        # 캐시 초기화
        self._barrier_pil_original = None
        self._barrier_tk_cached = None
        self._barrier_zoom_cached = 0.0
        self._redraw_all_markers()
        self.img_info.config(text=f"원본 크기: {self.img_size[0]} x {self.img_size[1]} px")
        self.update_idletasks()

    # -------------- Barrier generation --------------
    def on_generate_barrier(self):
        if not self.img:
            messagebox.showwarning("경고", "먼저 이미지를 로드하세요.")
            return
        
        if self.meta.resolution <= 0:
            messagebox.showerror("에러", "맵 resolution이 설정되지 않았습니다. YAML 파일을 먼저 로드하세요.")
            return
        
        if not self.yaml_path:
            messagebox.showwarning("경고", "YAML 파일이 로드되지 않았습니다. YAML 파일을 먼저 로드하여 resolution 정보를 확인하세요.")
            return
        
        try:
            w_m = float(self.robot_width_entry.get())
            d_m = float(self.robot_depth_entry.get())
            if w_m <= 0 or d_m <= 0:
                raise ValueError("로봇 크기는 0보다 커야 합니다.")
        except ValueError as e:
            messagebox.showerror("에러", f"로봇 크기 입력 오류: {e}")
            return
        
        self.robot_width_m = w_m
        self.robot_depth_m = d_m
        
        # Convert to pixels
        resolution = self.meta.resolution
        w_px = w_m / resolution
        d_px = d_m / resolution
        
        # Get occupancy map
        img_array = np.array(self.img.convert("L"))  # Grayscale
        # PIL Image shape: (H, W), numpy array shape: (H, W)
        H, W = img_array.shape
        
        # Determine occupied pixels based on YAML settings
        negate = self.meta.negate if self.meta.negate is not None else 0
        occupied_thresh = self.meta.occupied_thresh if self.meta.occupied_thresh is not None else 0.65
        free_thresh = self.meta.free_thresh if self.meta.free_thresh is not None else 0.196
        
        # Normalize to 0-1 range
        # In grayscale: 0 = black (dark), 255 = white (bright)
        # After normalization: 0.0 = black, 1.0 = white
        img_normalized = img_array.astype(np.float32) / 255.0
        
        # Apply negate if needed
        if negate:
            img_normalized = 1.0 - img_normalized
        
        # Threshold to find occupied cells
        # In ROS occupancy grid:
        #   - White (bright, high value) = free space
        #   - Black/Gray (dark, low value) = occupied
        # So we need to find pixels that are DARK (low normalized value)
        # Use free_thresh: pixels below this are considered occupied
        # Or use (1.0 - occupied_thresh): pixels below this are occupied
        # For safety, use the lower threshold to catch both black and gray
        occupied = img_normalized <= (1.0 - occupied_thresh)
        
        # Create elliptical kernel for dilation
        # Kernel size should match robot dimensions (radius = half of robot size)
        # Use actual robot dimensions without extra margin to avoid over-expansion
        kernel_w = max(3, int(np.ceil(w_px)))  # Minimum 3 pixels, but use actual size
        kernel_h = max(3, int(np.ceil(d_px)))
        # Make kernel size odd for proper centering
        if kernel_w % 2 == 0:
            kernel_w += 1
        if kernel_h % 2 == 0:
            kernel_h += 1
        
        # Create elliptical kernel
        y, x = np.ogrid[:kernel_h, :kernel_w]
        center_x = kernel_w // 2
        center_y = kernel_h // 2
        
        # Elliptical kernel: (x-center_x)^2/(w/2)^2 + (y-center_y)^2/(d/2)^2 <= 1
        # Use actual pixel dimensions (half of robot size as radius)
        half_w = max(1.0, w_px / 2.0)  # Avoid division by zero
        half_d = max(1.0, d_px / 2.0)
        kernel = ((x - center_x) / half_w) ** 2 + ((y - center_y) / half_d) ** 2 <= 1.0
        
        # Dilate occupied cells with the kernel
        # This expands each occupied pixel by the robot's footprint
        barrier = ndimage.binary_dilation(occupied, structure=kernel.astype(np.uint8))
        
        # Optional: Remove small isolated barrier regions (noise reduction)
        # This helps if there are small artifacts that create unwanted barriers
        # Use connected components to filter out very small barrier regions
        labeled_barrier, num_features = label(barrier)
        # Count pixels in each connected component
        component_sizes = np.bincount(labeled_barrier.ravel())
        # Keep only components that are reasonably large (at least 4 pixels)
        # This removes tiny noise but keeps real barriers
        min_component_size = 4
        mask = component_sizes >= min_component_size
        mask[0] = False  # Don't keep background (label 0)
        barrier = mask[labeled_barrier]
        
        # Verify barrier map shape matches image
        if barrier.shape != (H, W):
            messagebox.showerror("에러", f"배리어 맵 크기 불일치: 이미지 {img_array.shape}, 배리어 {barrier.shape}")
            return
        
        self.barrier_map = barrier
        
        # 캐시 초기화 (새 barrier 생성 시)
        self._barrier_pil_original = None
        self._barrier_tk_cached = None
        self._barrier_zoom_cached = 0.0
        
        # Calculate actual map dimensions for info
        map_width_m = W * resolution
        map_height_m = H * resolution
        
        self._redraw_all_markers()
        
        # Show detailed info including YAML source
        yaml_name = os.path.basename(self.yaml_path) if self.yaml_path else "N/A"
        info_msg = (
            f"배리어 생성 완료\n\n"
            f"=== YAML 정보 ===\n"
            f"파일: {yaml_name}\n"
            f"Resolution: {resolution:.4f} m/pixel\n"
            f"Occupied threshold: {occupied_thresh}\n"
            f"Negate: {negate}\n\n"
            f"=== 맵 정보 ===\n"
            f"이미지 크기: {W}px x {H}px\n"
            f"맵 크기: {map_width_m:.2f}m x {map_height_m:.2f}m\n\n"
            f"=== 로봇 정보 ===\n"
            f"로봇 크기: {w_m:.3f}m x {d_m:.3f}m\n"
            f"픽셀 크기: {w_px:.2f}px x {d_px:.2f}px\n"
            f"Kernel 크기: {kernel_w}px x {kernel_h}px"
        )
        messagebox.showinfo("완료", info_msg)
    
    def _is_in_barrier(self, x: float, y: float) -> bool:
        """Check if a point (in image pixel coordinates) is in barrier area."""
        if self.barrier_map is None:
            return False
        W, H = self.img_size
        # Clamp coordinates to image bounds
        px = int(max(0, min(W - 1, x)))
        py = int(max(0, min(H - 1, y)))
        # Note: y coordinate might need flipping depending on image orientation
        # Assuming image coordinates: (0,0) at top-left
        if py < self.barrier_map.shape[0] and px < self.barrier_map.shape[1]:
            return bool(self.barrier_map[py, px])
        return False
    
    def _draw_barrier(self):
        """Draw barrier overlay on canvas."""
        if not self.show_barrier.get() or self.barrier_map is None:
            return
        
        W, H = self.img_size
        if self.barrier_map.shape[0] != H or self.barrier_map.shape[1] != W:
            return
        
        # 원본 barrier 이미지가 없거나 변경되었으면 생성
        if self._barrier_pil_original is None:
            barrier_alpha = 0.3  # Semi-transparent
            # Create RGBA image
            barrier_img = np.zeros((H, W, 4), dtype=np.uint8)
            # Set red color where barrier is True
            barrier_mask = self.barrier_map
            barrier_img[barrier_mask, 0] = 255  # R
            barrier_img[barrier_mask, 1] = 107  # G
            barrier_img[barrier_mask, 2] = 107  # B
            barrier_img[barrier_mask, 3] = int(255 * barrier_alpha)  # A
            # Convert to PIL Image and cache
            self._barrier_pil_original = Image.fromarray(barrier_img, mode='RGBA')
            self._barrier_zoom_cached = 0.0  # 캐시 무효화
        
        # 줌 레벨이 변경되었거나 캐시가 없으면 리사이즈
        if abs(self._barrier_zoom_cached - self.zoom_level) > 0.001 or self._barrier_tk_cached is None:
            if self.zoom_level != 1.0:
                new_width = int(W * self.zoom_level)
                new_height = int(H * self.zoom_level)
                # NEAREST를 사용하여 더 빠른 리사이즈 (barrier는 단순한 오버레이이므로)
                barrier_pil_resized = self._barrier_pil_original.resize((new_width, new_height), Image.Resampling.NEAREST)
            else:
                barrier_pil_resized = self._barrier_pil_original
            
            self._barrier_tk_cached = ImageTk.PhotoImage(barrier_pil_resized)
            self._barrier_zoom_cached = self.zoom_level
        
        # 캐시된 이미지 사용
        self.canvas.create_image(0, 0, anchor="nw", image=self._barrier_tk_cached, tags=("barrier",))

    # -------------- Episode control --------------
    def on_new_episode(self):
        """New Episode 시작: start, goal 입력 모드로 전환"""
        if self.episode_mode in ('start_input', 'goal_input', 'path_input', 'fixed_start_goal'):
            if not messagebox.askyesno("확인", "현재 입력 중인 episode가 있습니다. 새로 시작할까요?"):
                return
            # 현재 입력 중인 마커들 제거
            self._clear_current_episode_markers()
        
        # 이전 episode의 모든 live 마커 제거 (새 episode 시작 시)
        to_remove = [i for i, mk in enumerate(self.markers) if mk.source == "live"]
        for i in reversed(to_remove):
            del self.markers[i]
        
        self.episode_mode = 'start_input'
        self.current_route_id = self._get_next_route_id()
        self.next_seq = 0
        self.current_path_waypoints = []
        self.fixed_start = None
        self.fixed_goal = None
        self._redraw_all_markers()
        self._refresh_marker_list()
        self._update_episode_status("시작점을 클릭 후 드래그하여 방향을 설정하거나 좌표를 직접 입력하세요")
        self.complete_path_btn.config(state="disabled")
        messagebox.showinfo("안내", "새 Episode 시작\n1. 시작점을 클릭 후 드래그하여 방향 설정 또는 좌표 직접 입력\n2. 목적지를 클릭 후 드래그하여 방향 설정 또는 좌표 직접 입력\n3. 경로(waypoint)를 클릭하세요 (방향 없음)\n4. 'Path 입력 완료' 버튼을 누르세요")
    
    def on_input_start_coord(self):
        """맵 좌표계에서 시작점 좌표를 직접 입력받아 마커 생성"""
        if not self.img:
            messagebox.showwarning("경고", "먼저 이미지를 로드하세요.")
            return
        
        if self.episode_mode != 'start_input':
            messagebox.showwarning("경고", "시작점 입력 모드가 아닙니다.")
            return
        
        try:
            x_map = float(self.start_x_entry.get())
            y_map = float(self.start_y_entry.get())
            theta_map_deg = float(self.start_theta_entry.get()) if self.start_theta_entry.get().strip() else 0.0
        except ValueError:
            messagebox.showerror("에러", "좌표값이 올바르지 않습니다. 숫자를 입력하세요.")
            return
        
        # 맵 좌표를 이미지 픽셀 좌표로 변환
        W, H = self.img_size
        u, v, theta_img_deg = map_to_img(x_map, y_map, math.radians(theta_map_deg), W, H, self.meta, self.use_center.get())
        
        # 이미지 범위 체크
        if u < 0 or u >= W or v < 0 or v >= H:
            messagebox.showwarning("경고", f"입력한 좌표가 이미지 범위를 벗어났습니다.\n이미지 크기: {W}x{H}, 변환된 픽셀: ({u:.1f}, {v:.1f})")
            return
        
        # 배리어 영역 체크
        if self._is_in_barrier(u, v):
            messagebox.showwarning("경고", f"배리어 영역에는 시작점을 생성할 수 없습니다.\n맵 좌표: ({x_map:.2f}, {y_map:.2f})")
            return
        
        # 마커 생성
        mk = Marker(
            id=f"mk_{int(time.time()*1000)}",
            type="start",
            x=float(u),
            y=float(v),
            theta_deg=float(theta_img_deg),
            source="live",
            route_id=self.current_route_id,
            seq=0
        )
        self.fixed_start = mk
        self.markers.insert(0, mk)
        self.episode_mode = 'goal_input'
        self._update_episode_status("목적지를 클릭 후 드래그하여 방향을 설정하거나 좌표를 직접 입력하세요")
        self._redraw_all_markers()
        self._refresh_marker_list()
        
        # 입력 필드 초기화
        self.start_x_entry.delete(0, tk.END)
        self.start_y_entry.delete(0, tk.END)
        self.start_theta_entry.delete(0, tk.END)
    
    def on_input_goal_coord(self):
        """맵 좌표계에서 목적지 좌표를 직접 입력받아 마커 생성"""
        if not self.img:
            messagebox.showwarning("경고", "먼저 이미지를 로드하세요.")
            return
        
        if self.episode_mode != 'goal_input':
            messagebox.showwarning("경고", "목적지 입력 모드가 아닙니다.")
            return
        
        try:
            x_map = float(self.goal_x_entry.get())
            y_map = float(self.goal_y_entry.get())
            theta_map_deg = float(self.goal_theta_entry.get()) if self.goal_theta_entry.get().strip() else 0.0
        except ValueError:
            messagebox.showerror("에러", "좌표값이 올바르지 않습니다. 숫자를 입력하세요.")
            return
        
        # 맵 좌표를 이미지 픽셀 좌표로 변환
        W, H = self.img_size
        u, v, theta_img_deg = map_to_img(x_map, y_map, math.radians(theta_map_deg), W, H, self.meta, self.use_center.get())
        
        # 이미지 범위 체크
        if u < 0 or u >= W or v < 0 or v >= H:
            messagebox.showwarning("경고", f"입력한 좌표가 이미지 범위를 벗어났습니다.\n이미지 크기: {W}x{H}, 변환된 픽셀: ({u:.1f}, {v:.1f})")
            return
        
        # 배리어 영역 체크
        if self._is_in_barrier(u, v):
            messagebox.showwarning("경고", f"배리어 영역에는 목적지를 생성할 수 없습니다.\n맵 좌표: ({x_map:.2f}, {y_map:.2f})")
            return
        
        # 마커 생성
        mk = Marker(
            id=f"mk_{int(time.time()*1000)}",
            type="goal",
            x=float(u),
            y=float(v),
            theta_deg=float(theta_img_deg),
            source="live",
            route_id=self.current_route_id,
            seq=999  # 나중에 waypoint 개수에 따라 설정됨
        )
        self.fixed_goal = mk
        self.markers.insert(0, mk)
        self.episode_mode = 'path_input'
        self._update_episode_status("경로(waypoint)를 클릭하여 추가하세요")
        self.complete_path_btn.config(state="normal")
        self._redraw_all_markers()
        self._refresh_marker_list()
        
        # 입력 필드 초기화
        self.goal_x_entry.delete(0, tk.END)
        self.goal_y_entry.delete(0, tk.END)
        self.goal_theta_entry.delete(0, tk.END)

    def on_complete_path(self):
        """Path 입력 완료: start/goal 재입력 여부 확인"""
        if self.episode_mode not in ('path_input', 'fixed_start_goal'):
            return
        
        if self.episode_mode == 'path_input':
            # path_input 모드: 첫 번째 trajectory 완료
            # goal의 seq를 waypoint 개수 + 1로 설정
            if self.fixed_goal:
                self.fixed_goal.seq = len(self.current_path_waypoints) + 1
            
            # 현재 path의 waypoint들을 마커에 추가
            for wp in self.current_path_waypoints:
                self.markers.insert(0, wp)
            
            self._redraw_all_markers()
            self._refresh_marker_list()
            
            # start/goal 재입력 여부 확인
            result = messagebox.askyesno("확인", "시작점과 목적지를 새로 입력하시겠습니까?")
            
            if result:
                # Yes: start/goal 재입력
                # 현재 trajectory의 모든 마커를 saved로 변경 (마커 목록에 계속 표시되도록)
                for mk in self.markers:
                    if mk.route_id == self.current_route_id and mk.source == "live":
                        mk.source = "saved"
                
                # live 마커가 남아있으면 제거 (혹시 모를 경우를 대비)
                to_remove = [i for i, mk in enumerate(self.markers) if mk.source == "live"]
                for i in reversed(to_remove):
                    del self.markers[i]
                
                self.episode_mode = 'start_input'
                self.current_route_id = self._get_next_route_id()
                self.next_seq = 0
                self.current_path_waypoints = []
                self.fixed_start = None
                self.fixed_goal = None
                self._redraw_all_markers()
                self._refresh_marker_list()  # saved 마커는 여전히 목록에 표시됨
                self._update_episode_status("시작점을 클릭 후 드래그하여 방향을 설정하세요")
                self.complete_path_btn.config(state="disabled")
            else:
                # No: start/goal 고정 후 여러 trajectory 생성
                # 첫 번째 trajectory의 모든 마커를 saved로 변경 (마커 목록에 계속 표시되도록)
                for mk in self.markers:
                    if mk.route_id == self.current_route_id and mk.source == "live":
                        mk.source = "saved"
                
                # fixed_start_goal 모드로 전환
                self.episode_mode = 'fixed_start_goal'
                self.current_route_id = self._get_next_route_id()  # 새로운 trajectory를 위한 route_id
                self.next_seq = 0
                self.current_path_waypoints = []
                
                # 새로운 route_id로 start/goal만 복사 (고정된 start/goal 사용)
                if self.fixed_start:
                    new_start = Marker(
                        id=f"mk_{int(time.time()*1000)}",
                        type="start",
                        x=self.fixed_start.x,
                        y=self.fixed_start.y,
                        theta_deg=self.fixed_start.theta_deg,
                        source="live",
                        route_id=self.current_route_id,
                        seq=0
                    )
                    self.markers.insert(0, new_start)
                if self.fixed_goal:
                    new_goal = Marker(
                        id=f"mk_{int(time.time()*1000)+1}",
                        type="goal",
                        x=self.fixed_goal.x,
                        y=self.fixed_goal.y,
                        theta_deg=self.fixed_goal.theta_deg,
                        source="live",
                        route_id=self.current_route_id,
                        seq=999  # 나중에 waypoint 개수에 따라 설정됨
                    )
                    self.markers.insert(0, new_goal)
                
                self._redraw_all_markers()
                self._refresh_marker_list()  # saved 마커는 여전히 목록에 표시됨
                self._update_episode_status("start/goal 고정: 경로를 계속 입력하세요")
                self.complete_path_btn.config(state="normal")
                messagebox.showinfo("안내", "시작점과 목적지가 고정되었습니다.\n여러 경로(trajectory)를 생성할 수 있습니다.\n경로를 입력한 후 'Path 입력 완료' 버튼을 누르세요.")
        else:
            # fixed_start_goal 모드: 새로운 trajectory 생성
            # 현재 trajectory의 goal seq 업데이트
            if self.fixed_goal:
                # 마커 리스트에서 현재 route_id의 goal 찾아서 seq 업데이트
                for mk in self.markers:
                    if mk.route_id == self.current_route_id and mk.type == "goal":
                        mk.seq = len(self.current_path_waypoints) + 1
                        break
            
            # 현재 trajectory의 waypoint들을 마커에 추가 (아직 추가되지 않은 경우)
            for wp in self.current_path_waypoints:
                # 이미 markers에 있는지 확인
                if not any(m.id == wp.id for m in self.markers):
                    self.markers.insert(0, wp)
            
            # 현재 trajectory의 모든 마커를 'saved'로 변경 (GUI에서 숨김, 저장 시에는 포함)
            # 데이터는 유지하되 GUI에서는 사라지게 함
            for mk in self.markers:
                if mk.route_id == self.current_route_id and mk.source == "live":
                    mk.source = "saved"
            
            # start/goal 재입력 여부 확인
            result = messagebox.askyesno("확인", "시작점과 목적지를 새로 입력하시겠습니까?")
            
            if result:
                # Yes: start/goal 재입력
                # live 마커만 제거 (saved 마커는 유지 - 마커 목록에 계속 표시)
                to_remove = [i for i, mk in enumerate(self.markers) if mk.source == "live"]
                for i in reversed(to_remove):
                    del self.markers[i]
                
                self.episode_mode = 'start_input'
                self.current_route_id = self._get_next_route_id()
                self.next_seq = 0
                self.current_path_waypoints = []
                self.fixed_start = None
                self.fixed_goal = None
                self._redraw_all_markers()
                self._refresh_marker_list()  # saved 마커는 여전히 목록에 표시됨
                self._update_episode_status("시작점을 클릭 후 드래그하여 방향을 설정하세요")
                self.complete_path_btn.config(state="disabled")
            else:
                # No: start/goal 고정 유지, 새로운 trajectory 생성
                # 새로운 trajectory를 위한 route_id
                self.current_route_id = self._get_next_route_id()
                self.next_seq = 0
                self.current_path_waypoints = []  # 새로운 trajectory의 waypoint를 위해 초기화
                
                # 새로운 route_id로 start/goal만 복사 (고정된 start/goal 사용)
                if self.fixed_start:
                    new_start = Marker(
                        id=f"mk_{int(time.time()*1000)}",
                        type="start",
                        x=self.fixed_start.x,
                        y=self.fixed_start.y,
                        theta_deg=self.fixed_start.theta_deg,
                        source="live",
                        route_id=self.current_route_id,
                        seq=0
                    )
                    self.markers.insert(0, new_start)
                if self.fixed_goal:
                    new_goal = Marker(
                        id=f"mk_{int(time.time()*1000)+1}",
                        type="goal",
                        x=self.fixed_goal.x,
                        y=self.fixed_goal.y,
                        theta_deg=self.fixed_goal.theta_deg,
                        source="live",
                        route_id=self.current_route_id,
                        seq=999  # 나중에 waypoint 개수에 따라 설정됨
                    )
                    self.markers.insert(0, new_goal)
                
                self._redraw_all_markers()
                self._refresh_marker_list()
                self._update_episode_status("start/goal 고정: 경로를 계속 입력하세요")

    def _clear_current_episode_markers(self):
        """현재 episode의 마커들 제거 (start, goal, waypoints)"""
        # 현재 route_id의 마커들 제거
        to_remove = [i for i, mk in enumerate(self.markers) if mk.route_id == self.current_route_id and mk.source == "live"]
        for i in reversed(to_remove):
            del self.markers[i]
        self.current_path_waypoints = []

    def _get_next_route_id(self) -> int:
        """다음 route_id를 반환. 현재 존재하는 route_id 개수를 세어서 다음 번호를 할당"""
        # 현재 존재하는 모든 route_id 수집
        existing_route_ids = set()
        for mk in self.markers:
            if mk.route_id > 0:  # 0은 무시 (초기값)
                existing_route_ids.add(mk.route_id)
        
        # 1부터 시작해서 빈 번호 찾기
        next_id = 1
        while next_id in existing_route_ids:
            next_id += 1
        
        return next_id
    
    def _renumber_route_ids(self):
        """모든 경로 ID를 1, 2, 3... 순서로 재정렬"""
        # 현재 존재하는 모든 route_id 수집 (0 제외)
        existing_route_ids = set()
        for mk in self.markers:
            if mk.route_id > 0:
                existing_route_ids.add(mk.route_id)
        
        if not existing_route_ids:
            return  # 재정렬할 경로가 없음
        
        # route_id를 정렬하여 매핑 생성 (1, 2, 3... 순서로)
        sorted_ids = sorted(existing_route_ids)
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted_ids, start=1)}
        
        # 모든 마커의 route_id 재할당
        for mk in self.markers:
            if mk.route_id > 0 and mk.route_id in id_mapping:
                mk.route_id = id_mapping[mk.route_id]
        
        # fixed_start, fixed_goal도 업데이트
        if self.fixed_start and self.fixed_start.route_id > 0 and self.fixed_start.route_id in id_mapping:
            self.fixed_start.route_id = id_mapping[self.fixed_start.route_id]
        if self.fixed_goal and self.fixed_goal.route_id > 0 and self.fixed_goal.route_id in id_mapping:
            self.fixed_goal.route_id = id_mapping[self.fixed_goal.route_id]
        
        # current_path_waypoints도 업데이트
        for wp in self.current_path_waypoints:
            if wp.route_id > 0 and wp.route_id in id_mapping:
                wp.route_id = id_mapping[wp.route_id]
        
        # current_route_id도 업데이트 (현재 episode가 진행 중인 경우)
        if self.current_route_id > 0 and self.current_route_id in id_mapping:
            self.current_route_id = id_mapping[self.current_route_id]

    def _update_episode_status(self, text: str):
        """Episode 상태 텍스트 업데이트"""
        mode_text = {
            'idle': 'idle',
            'start_input': 'input start',
            'goal_input': 'input goal',
            'path_input': 'input path',
            'fixed_start_goal': 'start/goal fixed'
        }
        mode = mode_text.get(self.episode_mode, 'Unknown')
        self.episode_status.config(text=f"status: {mode} - {text}")

    # -------------- Canvas interactions --------------
    def on_mouse_down(self, event):
        if not self.img:
            return
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.drag_start = (x, y)
        self._clear_preview()

    def on_mouse_drag(self, event):
        # start/goal 입력 모드일 때만 드래그 미리보기 표시
        if not self.img or not self.drag_start:
            return
        if self.episode_mode not in ('start_input', 'goal_input'):
            return
        
        x0, y0 = self.drag_start
        x1 = self.canvas.canvasx(event.x)
        y1 = self.canvas.canvasy(event.y)
        self._draw_preview(x0, y0, x1, y1)

    def on_mouse_up(self, event):
        if not self.img or not self.drag_start:
            self._clear_preview()
            return
        
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        x0, y0 = self.drag_start
        
        self._clear_preview()
        
        # Episode mode에 따라 처리
        if self.episode_mode == 'start_input':
            # Start 입력 (드래그로 방향 설정)
            # 줌 레벨을 고려하여 원본 좌표로 변환
            orig_x0 = x0 / self.zoom_level
            orig_y0 = y0 / self.zoom_level
            
            if self._is_in_barrier(orig_x0, orig_y0):
                messagebox.showwarning("경고", f"배리어 영역에는 시작점을 생성할 수 없습니다.\n위치: ({orig_x0:.1f}, {orig_y0:.1f})")
                self.drag_start = None
                return
            
            # 드래그 거리에 따라 각도 계산
            dx, dy = (x - x0), (y - y0)
            if dx**2 + dy**2 < 3**2:
                # 클릭만 (드래그 없음) - 기본 방향 0도
                theta_deg = 0.0
            else:
                # 드래그로 방향 설정 (위쪽이 +90도가 되도록 -dy 사용)
                theta_deg = math.degrees(math.atan2(-dy, dx))
                if theta_deg < 0:
                    theta_deg += 360.0
            
            mk = Marker(
                id=f"mk_{int(time.time()*1000)}",
                type="start",
                x=float(orig_x0),
                y=float(orig_y0),
                theta_deg=float(theta_deg),
                source="live",
                route_id=self.current_route_id,
                seq=0
            )
            self.fixed_start = mk
            self.markers.insert(0, mk)
            self.episode_mode = 'goal_input'
            self._update_episode_status("목적지를 클릭 후 드래그하여 방향을 설정하세요")
            self._redraw_all_markers()
            self._refresh_marker_list()
            
        elif self.episode_mode == 'goal_input':
            # Goal 입력 (드래그로 방향 설정)
            # 줌 레벨을 고려하여 원본 좌표로 변환
            orig_x0 = x0 / self.zoom_level
            orig_y0 = y0 / self.zoom_level
            
            if self._is_in_barrier(orig_x0, orig_y0):
                messagebox.showwarning("경고", f"배리어 영역에는 목적지를 생성할 수 없습니다.\n위치: ({orig_x0:.1f}, {orig_y0:.1f})")
                self.drag_start = None
                return
            
            # 드래그 거리에 따라 각도 계산
            dx, dy = (x - x0), (y - y0)
            if dx**2 + dy**2 < 3**2:
                # 클릭만 (드래그 없음) - 기본 방향 0도
                theta_deg = 0.0
            else:
                # 드래그로 방향 설정 (위쪽이 +90도가 되도록 -dy 사용)
                theta_deg = math.degrees(math.atan2(-dy, dx))
                if theta_deg < 0:
                    theta_deg += 360.0
            
            mk = Marker(
                id=f"mk_{int(time.time()*1000)}",
                type="goal",
                x=float(orig_x0),
                y=float(orig_y0),
                theta_deg=float(theta_deg),
                source="live",
                route_id=self.current_route_id,
                seq=999  # goal은 나중에 waypoint 개수에 따라 설정됨
            )
            self.fixed_goal = mk
            self.markers.insert(0, mk)
            self.episode_mode = 'path_input'
            self.next_seq = 0
            self.current_path_waypoints = []
            self._update_episode_status("경로(waypoint)를 클릭하세요 (방향 없음)")
            self.complete_path_btn.config(state="normal")
            self._redraw_all_markers()
            self._refresh_marker_list()
            
        elif self.episode_mode == 'path_input' or self.episode_mode == 'fixed_start_goal':
            # Waypoint 입력 (방향 없이 점만, 클릭만 허용)
            if (x - x0) ** 2 + (y - y0) ** 2 > 3**2:
                # 드래그가 있었으면 무시 (waypoint는 클릭만)
                self.drag_start = None
                return
            
            # 줌 레벨을 고려하여 원본 좌표로 변환
            orig_x0 = x0 / self.zoom_level
            orig_y0 = y0 / self.zoom_level
            
            if self._is_in_barrier(orig_x0, orig_y0):
                messagebox.showwarning("경고", f"배리어 영역에는 waypoint를 생성할 수 없습니다.\n위치: ({orig_x0:.1f}, {orig_y0:.1f})")
                self.drag_start = None
                return
            
            self.next_seq += 1
            mk = Marker(
                id=f"mk_{int(time.time()*1000)}",
                type="waypoint",
                x=float(orig_x0),
                y=float(orig_y0),
                theta_deg=0.0,  # 방향 없음
                source="live",
                route_id=self.current_route_id,
                seq=self.next_seq
            )
            
            if self.episode_mode == 'path_input':
                # path_input 모드: 임시로 current_path_waypoints에 저장
                self.current_path_waypoints.append(mk)
            else:
                # fixed_start_goal 모드: current_path_waypoints에도 추가하고 마커에도 추가
                self.current_path_waypoints.append(mk)
                self.markers.insert(0, mk)
            
            self._redraw_all_markers()
            self._refresh_marker_list()
        
        self.drag_start = None

    def _draw_preview(self, x0, y0, x1, y1):
        # start/goal 입력 모드일 때만 미리보기 표시
        self._clear_preview()
        if self.episode_mode in ('start_input', 'goal_input'):
            line_width = max(1, int(2 * self.zoom_level))
            self.preview_line = self.canvas.create_line(x0, y0, x1, y1, fill="#2563eb", dash=(4, 3), width=line_width, tags=("preview",))
            self.preview_head = self._draw_arrow_head(x0, y0, x1, y1, fill="#2563eb", tag="preview")
            self.canvas.addtag_withtag("preview", self.preview_head)

    def _clear_preview(self):
        for item in self.canvas.find_withtag("preview"):
            self.canvas.delete(item)
        self.preview_line = None
        self.preview_head = None

    # -------------- Redraw / Draw helpers --------------
    def _redraw_all_markers(self):
        # clear drawn markers and barrier
        for item in self.canvas.find_withtag("marker"):
            self.canvas.delete(item)
        for item in self.canvas.find_withtag("path"):
            self.canvas.delete(item)
        for item in self.canvas.find_withtag("barrier"):
            self.canvas.delete(item)
        # clear highlights (will be restored below if selection exists)
        self._clear_highlight()

        if not self.img:
            return

        # Draw barrier first (behind markers)
        self._draw_barrier()

        # hide_live: show only most recent live route_id
        # saved 마커는 GUI에서 숨김 (데이터는 유지)
        allowed_live_route = None
        if self.hide_live.get():
            live_routes = [mk.route_id for mk in self.markers if mk.source == "live"]
            if live_routes:
                allowed_live_route = max(live_routes)
        
        # 호버된 route_id가 있으면 해당 route만 표시
        filter_route_id = self.hovered_route_id

        # Draw markers first (live 마커만 표시, saved는 숨김)
        for mk in self.markers:
            if mk.source == "saved":
                continue  # saved 마커는 GUI에서 숨김
            if mk.source == "live" and allowed_live_route is not None and mk.route_id != allowed_live_route:
                continue
            # 호버된 route_id가 있으면 해당 route만 표시
            if filter_route_id is not None and mk.route_id != filter_route_id:
                continue
            self._draw_marker(mk)
        
        # Draw current path waypoints (if in path_input mode only)
        # fixed_start_goal 모드에서는 waypoint가 이미 markers에 추가되어 있으므로 중복 그리기 방지
        if self.episode_mode == 'path_input':
            for wp in self.current_path_waypoints:
                # 호버된 route_id가 있으면 해당 route만 표시
                if filter_route_id is not None and wp.route_id != filter_route_id:
                    continue
                self._draw_marker(wp)
        
        # Draw path connections and arrows (after markers)
        self._draw_path_connections(allowed_live_route, filter_route_id)

        # restore selection highlight if any
        self._redraw_selection_highlight()

    def _draw_path_connections(self, allowed_live_route: Optional[int] = None, filter_route_id: Optional[int] = None):
        """경로 연결 선분 및 화살표 그리기"""
        # route_id별로 마커 그룹화 (saved 마커는 GUI에서 숨김)
        route_markers = {}
        for mk in self.markers:
            if mk.source == "saved":
                continue  # saved 마커는 GUI에서 숨김
            if mk.source == "live" and allowed_live_route is not None and mk.route_id != allowed_live_route:
                continue
            # 호버된 route_id가 있으면 해당 route만 표시
            if filter_route_id is not None and mk.route_id != filter_route_id:
                continue
            if mk.route_id not in route_markers:
                route_markers[mk.route_id] = []
            route_markers[mk.route_id].append(mk)
        
        # 현재 path_input 또는 fixed_start_goal 모드의 waypoint도 포함
        if self.episode_mode in ('path_input', 'fixed_start_goal') and self.fixed_start and self.fixed_goal:
            route_id = self.current_route_id
            # 호버된 route_id가 있으면 해당 route만 표시
            if filter_route_id is not None and route_id != filter_route_id:
                pass  # 해당 route를 건너뜀
            else:
                # route_markers에 이미 있는 마커들 확인
                existing_marker_ids = set()
                if route_id in route_markers:
                    existing_marker_ids = {mk.id for mk in route_markers[route_id]}
                
                # start/goal이 route_markers에 없으면 추가
                if self.fixed_start.id not in existing_marker_ids:
                    if route_id not in route_markers:
                        route_markers[route_id] = []
                    route_markers[route_id].append(self.fixed_start)
                
                if self.fixed_goal.id not in existing_marker_ids:
                    if route_id not in route_markers:
                        route_markers[route_id] = []
                    route_markers[route_id].append(self.fixed_goal)
                
                # current_path_waypoints의 waypoint 추가 (중복 제거)
                for wp in self.current_path_waypoints:
                    if wp.id not in existing_marker_ids:
                        if route_id not in route_markers:
                            route_markers[route_id] = []
                        route_markers[route_id].append(wp)
        
        # 마커가 없으면 path를 그리지 않음
        if not route_markers:
            return
        
        # 각 route에 대해 경로 그리기
        for route_id, markers in route_markers.items():
            # start, waypoints, goal 순서로 정렬
            start_mk = None
            goal_mk = None
            waypoints = []
            
            for mk in markers:
                if mk.type == "start":
                    start_mk = mk
                elif mk.type == "goal":
                    goal_mk = mk
                elif mk.type == "waypoint":
                    waypoints.append(mk)
            
            # seq 순서로 waypoint 정렬
            waypoints.sort(key=lambda m: m.seq)
            
            # 경로 연결: start -> waypoints -> goal (줌 레벨 적용)
            path_points = []
            if start_mk:
                path_points.append((start_mk.x * self.zoom_level, start_mk.y * self.zoom_level))
            for wp in waypoints:
                path_points.append((wp.x * self.zoom_level, wp.y * self.zoom_level))
            if goal_mk:
                path_points.append((goal_mk.x * self.zoom_level, goal_mk.y * self.zoom_level))
            
            # 선분 그리기
            if len(path_points) >= 2:
                line_width = max(1, int(2 * self.zoom_level))
                for i in range(len(path_points) - 1):
                    x0, y0 = path_points[i]
                    x1, y1 = path_points[i + 1]
                    # 선분
                    self.canvas.create_line(x0, y0, x1, y1, fill="#3b82f6", width=line_width, tags=("path",))
                    # 화살표
                    self._draw_arrow_head(x0, y0, x1, y1, fill="#3b82f6", tag="path")

    def _draw_marker(self, mk: Marker):
        # Apply zoom level to coordinates
        x = mk.x * self.zoom_level
        y = mk.y * self.zoom_level
        
        # colors by type/source
        if mk.type == "start":
            color = "#16a34a"  # green
        elif mk.type == "goal":
            color = "#dc2626"  # red
        else:
            color = "#1f4f8b" if mk.source == "file" else "#3b82f6"  # waypoint: blue-ish

        # v1.8: waypoint는 방향 없이 점만
        if mk.type == "waypoint":
            r = 10 * self.zoom_level
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="#ffffff", outline=color, width=max(1, int(2 * self.zoom_level)), tags=("marker",))
            font_size = max(6, int(9 * self.zoom_level))
            self.canvas.create_text(x, y, text=str(mk.seq), font=("Arial", font_size, "bold"), fill=color, tags=("marker",))
        else:
            # start/goal은 방향 표시 (theta_deg가 0이면 기본 방향)
            length = 40.0 * self.zoom_level
            theta = math.radians(mk.theta_deg) if mk.theta_deg != 0.0 else 0.0
            x1 = x + math.cos(theta) * length
            # 화면 좌표계에서 위쪽은 y 감소이므로 -sin 사용
            y1 = y - math.sin(theta) * length

            # body + head
            self.canvas.create_line(x, y, x1, y1, fill=color, width=max(1, int(3 * self.zoom_level)), tags=("marker",))
            head = self._draw_arrow_head(x, y, x1, y1, fill=color)
            self.canvas.addtag_withtag("marker", head)

            # label: Start/Goal letters
            font_size = max(6, int(10 * self.zoom_level))
            offset = 8 * self.zoom_level
            if mk.type == "start":
                self.canvas.create_text(x + offset, y - offset, text="S", font=("Arial", font_size, "bold"), fill="#111", tags=("marker",))
            elif mk.type == "goal":
                self.canvas.create_text(x + offset, y - offset, text="G", font=("Arial", font_size, "bold"), fill="#111", tags=("marker",))

    def _draw_arrow_head(self, x0, y0, x1, y1, fill="#000", tag="marker"):
        angle = math.atan2(y1 - y0, x1 - x0)
        ah = 10.0 * self.zoom_level
        back = 6.0 * self.zoom_level
        bx = x1 - math.cos(angle) * back
        by = y1 - math.sin(angle) * back
        left = (bx - math.cos(angle - math.pi/2) * ah, by - math.sin(angle - math.pi/2) * ah)
        right = (bx - math.cos(angle + math.pi/2) * ah, by - math.sin(angle + math.pi/2) * ah)
        return self.canvas.create_polygon((x1, y1, left[0], left[1], right[0], right[1]), fill=fill, outline="", tags=(tag,))

    # ---------- Highlight helpers ----------
    def _clear_highlight(self, kind: Optional[str] = None):
        """kind=None → 모두, 'hover' 또는 'selected'만 삭제"""
        tag = "highlight" if kind is None else f"hl_{kind}"
        for item in self.canvas.find_withtag(tag):
            self.canvas.delete(item)

    def _draw_highlight(self, mk: Marker, kind: str = "selected"):
        # Apply zoom level to coordinates
        x = mk.x * self.zoom_level
        y = mk.y * self.zoom_level
        
        color = "#f59e0b" if kind == "hover" else "#22c55e"  # amber for hover, green for selected
        r1, r2 = 18 * self.zoom_level, 26 * self.zoom_level
        # two concentric dashed rings
        o1 = self.canvas.create_oval(x - r1, y - r1, x + r1, y + r1,
                                     outline=color, width=max(1, int(3 * self.zoom_level)), dash=(4, 3), tags=("highlight", f"hl_{kind}"))
        o2 = self.canvas.create_oval(x - r2, y - r2, x + r2, y + r2,
                                     outline=color, width=max(1, int(2 * self.zoom_level)), dash=(2, 3), tags=("highlight", f"hl_{kind}"))
        # crosshair
        offset1 = 6 * self.zoom_level
        offset2 = 4 * self.zoom_level
        ch1 = self.canvas.create_line(x - r2 - offset1, y, x - offset2, y, fill=color, width=max(1, int(2 * self.zoom_level)), tags=("highlight", f"hl_{kind}"))
        ch2 = self.canvas.create_line(x + offset2, y, x + r2 + offset1, y, fill=color, width=max(1, int(2 * self.zoom_level)), tags=("highlight", f"hl_{kind}"))
        ch3 = self.canvas.create_line(x, y - r2 - offset1, x, y - offset2, fill=color, width=max(1, int(2 * self.zoom_level)), tags=("highlight", f"hl_{kind}"))
        ch4 = self.canvas.create_line(x, y + offset2, x, y + r2 + offset1, fill=color, width=max(1, int(2 * self.zoom_level)), tags=("highlight", f"hl_{kind}"))
        # make sure highlight is above markers
        self.canvas.tag_raise("highlight")
        return (o1, o2, ch1, ch2, ch3, ch4)

    def _redraw_selection_highlight(self):
        if self.selected_idx is None:
            return
        if 0 <= self.selected_idx < len(self.markers):
            mk = self.markers[self.selected_idx]
            self._draw_highlight(mk, kind="selected")
        else:
            self.selected_idx = None

    def _scroll_to_center(self, x: float, y: float):
        """Scroll canvas so that (x,y) is centered if possible."""
        W, H = self.img_size
        if W <= 0 or H <= 0:
            return
        self.update_idletasks()
        vw = max(1, self.canvas.winfo_width())
        vh = max(1, self.canvas.winfo_height())
        denom_x = max(1, W - vw)
        denom_y = max(1, H - vh)
        fx = (x - vw / 2) / denom_x
        fy = (y - vh / 2) / denom_y
        fx = 0.0 if fx < 0 else 1.0 if fx > 1 else fx
        fy = 0.0 if fy < 0 else 1.0 if fy > 1 else fy
        self.canvas.xview_moveto(fx)
        self.canvas.yview_moveto(fy)

    # ---------- Listbox callbacks ----------
    def on_list_select(self, event=None):
        sel = self.marker_list.curselection()
        if not sel:
            self.selected_idx = None
            self._clear_highlight(kind="selected")
            return
        listbox_idx = sel[0]
        # listbox 인덱스는 markers 인덱스와 동일 (모든 마커 표시)
        if listbox_idx < 0 or listbox_idx >= len(self.markers):
            return
        self.selected_idx = listbox_idx
        mk = self.markers[self.selected_idx]
        self._clear_highlight(kind="selected")
        # saved 마커도 하이라이트 표시 (캔버스에는 안 보이지만)
        self._draw_highlight(mk, kind="selected")
        self._scroll_to_center(mk.x, mk.y)

    def on_list_hover_motion(self, event):
        listbox_idx = self.marker_list.nearest(event.y)
        # listbox 인덱스는 markers 인덱스와 동일 (모든 마커 표시)
        if listbox_idx < 0 or listbox_idx >= len(self.markers):
            self._clear_highlight(kind="hover")
            self.last_hover_idx = None
            self.hovered_route_id = None
            self._redraw_all_markers()
            return
        if self.last_hover_idx == listbox_idx:
            return
        self.last_hover_idx = listbox_idx
        mk = self.markers[listbox_idx]
        self._clear_highlight(kind="hover")
        # saved 마커도 하이라이트 표시 (캔버스에는 안 보이지만)
        self._draw_highlight(mk, kind="hover")
        # 호버된 마커의 route_id 저장하고 해당 route만 표시
        self.hovered_route_id = mk.route_id
        self._redraw_all_markers()
    
    def on_list_hover_leave(self, event):
        """리스트박스에서 마우스가 벗어나면 호버 상태 초기화"""
        self._clear_highlight(kind="hover")
        self.last_hover_idx = None
        self.hovered_route_id = None
        self._redraw_all_markers()

    # -------------- List / Save / Load / Delete --------------
    def _refresh_marker_list(self):
        self.marker_list.delete(0, tk.END)
        visible_indices = []
        for i, mk in enumerate(self.markers):
            # 모든 마커를 목록에 표시 (saved 포함)
            visible_indices.append(i)
            if mk.source == "live":
                tag = "L"
            elif mk.source == "saved":
                tag = "S"  # Saved 표시
            else:
                tag = "F"  # File
            self.marker_list.insert(
                tk.END,
                f"{mk.id[-6:]} [{tag}] route={mk.route_id} seq={mk.seq} {mk.type.upper()}  u={mk.x:.1f} v={mk.y:.1f} θ={mk.theta_deg:.1f}°"
            )
        # selection index 유효성 보정
        if self.selected_idx is not None:
            if self.selected_idx in visible_indices:
                listbox_idx = visible_indices.index(self.selected_idx)
                self.marker_list.selection_clear(0, tk.END)
                self.marker_list.selection_set(listbox_idx)
            else:
                self.selected_idx = None

    def on_delete_selected(self):
        sel = self.marker_list.curselection()
        if not sel:
            return
        idx = sel[0]
        try:
            marker = self.markers[idx]
        except IndexError:
            return
        
        # start 또는 goal을 삭제하면 해당 route의 모든 마커 삭제
        if marker.type in ("start", "goal"):
            route_id = marker.route_id
            
            # 삭제되는 route가 현재 episode의 route인지 확인
            is_current_episode_route = (self.episode_mode in ('path_input', 'fixed_start_goal') and 
                                       self.current_route_id == route_id)
            
            # 같은 route_id를 가진 모든 마커 삭제
            to_remove = [i for i, mk in enumerate(self.markers) if mk.route_id == route_id]
            # 역순으로 삭제 (인덱스가 변경되지 않도록)
            for i in reversed(to_remove):
                del self.markers[i]
            
            # 현재 episode의 route를 삭제한 경우 episode 상태 정리
            if is_current_episode_route:
                # fixed_start/fixed_goal이 삭제된 route의 것인지 확인
                if self.fixed_start and self.fixed_start.route_id == route_id:
                    self.fixed_start = None
                if self.fixed_goal and self.fixed_goal.route_id == route_id:
                    self.fixed_goal = None
                # current_path_waypoints에서도 삭제된 route의 waypoint 제거
                self.current_path_waypoints = [wp for wp in self.current_path_waypoints 
                                              if wp.route_id != route_id]
                # episode 모드 초기화
                if not self.fixed_start and not self.fixed_goal:
                    self.episode_mode = 'idle'
                    self._update_episode_status("대기 중")
                    self.complete_path_btn.config(state="disabled")
            
            # 선택 인덱스 초기화 (삭제된 마커가 많으므로)
            self.selected_idx = None
            
            # 경로 ID 재정렬 (1, 2, 3... 순서로)
            self._renumber_route_ids()
        else:
            # waypoint는 단일 삭제
            # 현재 episode의 waypoint인지 확인
            if (self.episode_mode in ('path_input', 'fixed_start_goal') and 
                marker.route_id == self.current_route_id):
                # current_path_waypoints에서도 제거
                self.current_path_waypoints = [wp for wp in self.current_path_waypoints 
                                              if wp.id != marker.id]
            
            del self.markers[idx]
            # 선택 인덱스 보정
            if self.selected_idx is not None:
                if idx < self.selected_idx:
                    self.selected_idx -= 1
                elif idx == self.selected_idx:
                    self.selected_idx = None
        
        self._refresh_marker_list()
        self._redraw_all_markers()

    def on_clear_all(self):
        if not self.markers:
            return
        if messagebox.askyesno("확인", "모든 마커를 삭제할까요? (불러온 파일 마커 포함)"):
            # reset episode state first
            self.episode_mode = 'idle'
            self.fixed_start = None
            self.fixed_goal = None
            self.current_path_waypoints = []
            self.next_seq = 0
            
            # Clear all markers
            self.markers.clear()
            self.selected_idx = None
            
            # 명시적으로 캔버스의 모든 마커, path, highlight 삭제
            for item in self.canvas.find_withtag("marker"):
                self.canvas.delete(item)
            for item in self.canvas.find_withtag("path"):
                self.canvas.delete(item)
            for item in self.canvas.find_withtag("highlight"):
                self.canvas.delete(item)
            
            self._refresh_marker_list()
            self._redraw_all_markers()  # barrier만 다시 그리기 위해 호출
            self._update_episode_status("대기 중")
            self.complete_path_btn.config(state="disabled")

    def on_save(self):
        if not self.markers:
            messagebox.showinfo("안내", "저장할 마커가 없습니다.")
            return
        if not self.img:
            messagebox.showwarning("경고", "이미지가 로드되어야 저장할 수 있습니다.")
            return

        W, H = self.img_size
        meta = self.meta
        default_name = f"markers_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path = filedialog.asksaveasfilename(
            title="저장할 파일 선택",
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            # Prepare metadata
            metadata = {
                "version": "1.8",
                "saved_by": "map_marker_app_v_1_8.py",
                "yaml_path": self.yaml_path,
                "image_path": self.img_path,
                "resolution": meta.resolution,
                "origin": [meta.origin_x, meta.origin_y, meta.origin_yaw],
                "use_center": self.use_center.get(),
            }
            
            # Prepare markers data
            markers_data = []
            for mk in reversed(self.markers):  # save bottom->top for readability
                x_map, y_map, theta_map = img_to_map(
                    mk.x, mk.y, mk.theta_deg, W, H, meta, use_center=self.use_center.get()
                )
                marker_dict = {
                    "id": mk.id,
                    "type": mk.type,
                    "source": mk.source,
                    "route_id": mk.route_id,
                    "seq": mk.seq,
                    "u_px": round(mk.x, 2),
                    "v_px": round(mk.y, 2),
                    "theta_img_deg": round(mk.theta_deg, 2),
                    "x_map": round(x_map, 6),
                    "y_map": round(y_map, 6),
                    "theta_map_rad": round(theta_map, 6),
                }
                markers_data.append(marker_dict)
            
            # Create JSON structure
            json_data = {
                "metadata": metadata,
                "markers": markers_data
            }
            
            # Write JSON file with indentation for readability
            with open(path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            messagebox.showinfo("완료", f"저장했습니다:\n{path}")
        except Exception as e:
            messagebox.showerror("에러", f"저장 실패: {e}")

    def on_load_txt(self):
        if not self.img:
            messagebox.showwarning("경고", "먼저 이미지를 로드하세요.")
            return
        path = filedialog.askopenfilename(
            title="마커 파일 열기 (.json/.txt/.csv)",
            filetypes=[("JSON", "*.json"), ("Text/CSV", "*.txt *.csv *.TXT *.CSV"), ("All files", "*.*")],
        )
        if not path:
            return

        # Check if JSON file
        if path.lower().endswith('.json'):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
            except Exception as e:
                messagebox.showerror("에러", f"JSON 파일을 읽을 수 없습니다.\n{e}")
                return
            
            # Parse JSON structure
            if "markers" not in json_data:
                messagebox.showerror("에러", "JSON 파일 형식이 올바르지 않습니다. 'markers' 필드가 없습니다.")
                return
            
            # Load metadata if available
            if "metadata" in json_data:
                meta_info = json_data["metadata"]
                if meta_info.get("yaml_path"):
                    self.yaml_path = meta_info["yaml_path"]
                if meta_info.get("image_path") and os.path.exists(meta_info["image_path"]):
                    self._load_image(meta_info["image_path"])
                if meta_info.get("resolution"):
                    self.meta.resolution = float(meta_info["resolution"])
                if meta_info.get("origin") and len(meta_info["origin"]) >= 3:
                    self.meta.origin_x = float(meta_info["origin"][0])
                    self.meta.origin_y = float(meta_info["origin"][1])
                    self.meta.origin_yaw = float(meta_info["origin"][2])
            
            # Load markers
            tmp: List[Marker] = []
            W, H = self.img_size
            for marker_dict in json_data["markers"]:
                try:
                    _id = marker_dict.get("id", f"mk_{int(time.time()*1000)}")
                    _type = marker_dict.get("type", "waypoint").strip().lower()
                    _source = marker_dict.get("source", "file").strip().lower()
                    
                    # Use pixel coordinates if available, otherwise convert from map coordinates
                    if "u_px" in marker_dict and "v_px" in marker_dict:
                        u = float(marker_dict["u_px"])
                        v = float(marker_dict["v_px"])
                        theta_deg = float(marker_dict.get("theta_img_deg", 0.0))
                    elif "x_map" in marker_dict and "y_map" in marker_dict:
                        x_map = float(marker_dict["x_map"])
                        y_map = float(marker_dict["y_map"])
                        theta_map = float(marker_dict.get("theta_map_rad", 0.0))
                        u, v, theta_deg = map_to_img(x_map, y_map, theta_map, W, H, self.meta, use_center=self.use_center.get())
                    else:
                        continue
                    
                    route_id = int(marker_dict.get("route_id", 0))
                    seq = int(marker_dict.get("seq", 0))
                    
                    tmp.append(Marker(id=_id, type=_type, x=u, y=v, theta_deg=theta_deg, source="file", route_id=route_id, seq=seq))
                except Exception as e:
                    continue
        else:
            # Legacy CSV/TXT format
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = [ln for ln in f if not ln.lstrip().startswith("#") and ln.strip()]
                reader = csv.reader(lines)
                rows = list(reader)
            except Exception as e:
                messagebox.showerror("에러", f"파일을 읽을 수 없습니다.\n{e}")
                return

            if not rows:
                messagebox.showwarning("경고", "파일에 데이터가 없습니다.")
                return

            header = [c.strip().lower() for c in rows[0]]
            has_header = any(name in header for name in ("u_px", "x_map", "source", "route_id", "seq", "type"))
            start_idx = 1 if has_header else 0

            def find(name):
                try:
                    return header.index(name)
                except ValueError:
                    return -1

            if has_header:
                cols = {
                    "id": find("id"),
                    "type": find("type"),
                    "source": find("source"),
                    "route_id": find("route_id"),
                    "seq": find("seq"),
                    "u": find("u_px"),
                    "v": find("v_px"),
                    "theta_deg": find("theta_img_deg"),
                    "x_map": find("x_map"),
                    "y_map": find("y_map"),
                    "theta_map": find("theta_map_rad"),
                }
            else:
                # minimal fallback: id,type,u,v,theta_deg,x_map,y_map,theta_map_rad
                cols = {"id": 0, "type": 1, "u": 2, "v": 3, "theta_deg": 4, "x_map": 5, "y_map": 6, "theta_map": 7}

            # 1) 파일 순서대로 적재
            tmp: List[Marker] = []
            W, H = self.img_size
            for row in rows[start_idx:]:
                try:
                    _id = row[cols["id"]] if cols.get("id", -1) != -1 else f"mk_{int(time.time()*1000)}"
                    _type = (row[cols["type"]].strip().lower() if cols.get("type", -1) != -1 and row[cols["type"]] else "waypoint")
                    _source = (row[cols["source"]].strip().lower() if cols.get("source", -1) != -1 and row[cols["source"]] else "file")
                    if cols.get("u", -1) != -1 and cols.get("v", -1) != -1 and row[cols["u"]] and row[cols["v"]]:
                        u = float(row[cols["u"]])
                        v = float(row[cols["v"]])
                        theta_deg = float(row[cols["theta_deg"]]) if cols.get("theta_deg", -1) != -1 and row[cols["theta_deg"]] else 0.0
                    else:
                        x_map = float(row[cols["x_map"]])
                        y_map = float(row[cols["y_map"]])
                        theta_map = float(row[cols["theta_map"]]) if cols.get("theta_map", -1) != -1 and row[cols["theta_map"]] else 0.0
                        u, v, theta_deg = map_to_img(x_map, y_map, theta_map, W, H, self.meta, use_center=self.use_center.get())

                    route_id = int(row[cols["route_id"]]) if cols.get("route_id", -1) != -1 and row[cols["route_id"]] else 0
                    seq = int(row[cols["seq"]]) if cols.get("seq", -1) != -1 and row[cols["seq"]] else 0

                    tmp.append(Marker(id=_id, type=_type, x=u, y=v, theta_deg=theta_deg, source="file", route_id=route_id, seq=seq))
                except Exception:
                    continue

        # 2) 스택(top-first) 순서로 뒤집기
        tmp = list(reversed(tmp))

        # 3) route_id/seq가 없던 옛 파일이면, 단일 경로로 가정해 route_id=1이고 start/waypoint/goal 추론
        if all(m.route_id == 0 for m in tmp):
            rid = 1
            seq = 0
            start_seen = False
            for i, mk in enumerate(tmp):
                seq += 1
                mk.route_id = rid
                mk.seq = seq
                if mk.type not in ("start", "goal", "waypoint"):
                    mk.type = "start" if not start_seen else "waypoint"
                if mk.type == "start":
                    start_seen = True
            # 마지막이 goal이 아니면 그대로 waypoint로 둠

        # 4) 현재 스택 앞쪽에 추가
        self.markers = tmp + self.markers

        # 5) 라이브 경로 id 갱신
        if tmp:
            self.current_route_id = max(self.current_route_id, max(m.route_id for m in tmp if m.route_id is not None))
            self.episode_mode = 'idle'
            self.next_seq = 0
            self._update_episode_status("대기 중")

        self._refresh_marker_list()
        self._redraw_all_markers()
        messagebox.showinfo("완료", f"불러온 마커: {len(tmp)}개\n{os.path.basename(path)}")

    # -------------- Utils --------------
    def _normalize_angle(self, deg):
        while deg < 0:
            deg += 360.0
        while deg >= 360.0:
            deg -= 360.0
        return deg


if __name__ == "__main__":
    app = MapMarkerApp()
    app.mainloop()

