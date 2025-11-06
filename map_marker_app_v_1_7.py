#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PNG 맵 좌표·방향 등록기 (YAML 지원) — v1.6 (waypoints + list-hover/select highlight)
- 새 경로 시작 시 첫 노드는 자동으로 START
- "마지막 노드(Goal)로 종료" 버튼을 누르기 전까지는 계속 WAYPOINT
- 버튼 누른 뒤 다음 클릭은 GOAL로 확정되고 해당 경로가 닫힘
- hide_live가 켜져 있으면 가장 최근 라이브 경로(start+waypoint+goal)를 한 세트만 표시
- 저장/불러오기 포맷에 route_id, seq 추가 (없어도 호환 로드)
- [NEW] 마커 목록(Listbox)에서 항목에 마우스를 올리면 hover 하이라이트, 선택하면 강조표시 + 캔버스 자동 스크롤/중앙정렬
"""

import os
import math
import time
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass
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
    theta_map = wrap_to_pi(-math.radians(theta_img_deg) + yaw)
    return x_map, y_map, theta_map


def map_to_img(x_map: float, y_map: float, theta_map_rad: float, W: int, H: int, meta: MapMeta, use_center: bool = True):
    x0, y0, yaw = meta.origin_x, meta.origin_y, meta.origin_yaw
    res = meta.resolution
    offs = 0.5 if use_center else 0.0
    u = (x_map - x0) / res - offs
    v = H - (y_map - y0) / res - offs
    theta_img = wrap_to_pi(-(theta_map_rad - yaw))
    return u, v, math.degrees(theta_img)


class MapMarkerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("맵 마커 등록기 v1.6")
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
        self.current_route_open: bool = False
        self.current_route_id: int = 0
        self.next_is_goal: bool = False
        self.next_seq: int = 0  # within current route

        # preview
        self.drag_start: Optional[Tuple[float, float]] = None
        self.preview_line = None
        self.preview_head = None

        # highlight state
        self.selected_idx: Optional[int] = None
        self.last_hover_idx: Optional[int] = None

        # Barrier state
        self.barrier_map: Optional[np.ndarray] = None  # 2D boolean array, True = barrier
        self.show_barrier = tk.BooleanVar(value=True)
        self.robot_width_m: float = 0.5  # meters
        self.robot_depth_m: float = 0.5  # meters

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
        self.barrier_info = ttk.Label(left, text="YAML 로드 필요 (resolution 확인)", foreground="#666", font=("", 8))
        self.barrier_info.grid(row=6, column=0, sticky="w", pady=(0, 4))
        barrier_frame = ttk.Frame(left)
        barrier_frame.grid(row=7, column=0, sticky="ew", pady=(0, 12))
        ttk.Label(barrier_frame, text="로봇 가로(m):").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.robot_width_entry = ttk.Entry(barrier_frame, width=8)
        self.robot_width_entry.insert(0, "0.5")
        self.robot_width_entry.grid(row=0, column=1, sticky="w", padx=(0, 8))
        ttk.Label(barrier_frame, text="로봇 세로(m):").grid(row=0, column=2, sticky="w", padx=(0, 4))
        self.robot_depth_entry = ttk.Entry(barrier_frame, width=8)
        self.robot_depth_entry.insert(0, "0.5")
        self.robot_depth_entry.grid(row=0, column=3, sticky="w")
        ttk.Button(left, text="배리어 생성", command=self.on_generate_barrier).grid(row=8, column=0, sticky="ew", pady=(4, 4))
        ttk.Checkbutton(left, text="배리어 표시", variable=self.show_barrier, command=self._redraw_all_markers).grid(row=9, column=0, sticky="w", pady=(0, 0))

        # --- Waypoint route control ---
        ttk.Label(left, text="2) 경로 입력", font=("", 11, "bold")).grid(row=10, column=0, sticky="w", pady=(0, 6))
        ttk.Button(left, text="Goal 입력", command=self.on_arm_goal_once).grid(row=11, column=0, sticky="ew", pady=(4, 0))
        ttk.Checkbutton(left, text="이전 라이브 경로 숨김(최신 경로만 표시)", variable=self.hide_live, command=self._redraw_all_markers).grid(row=12, column=0, sticky="w", pady=(6, 12))
        ttk.Label(left, text="캔버스: 좌클릭 후 드래그 → 방향 지정 / 가운데버튼 드래그: 팬", foreground="#666").grid(row=13, column=0, sticky="w", pady=(0, 12))

        # --- Save/Load/Clear ---
        ttk.Label(left, text="3) 저장 / 불러오기 / 관리", font=("", 11, "bold")).grid(row=14, column=0, sticky="w", pady=(0, 6))
        btns = ttk.Frame(left)
        btns.grid(row=15, column=0, sticky="ew", pady=(0, 2))
        ttk.Button(btns, text="저장(.txt: 픽셀+월드)", command=self.on_save).pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="불러오기(.txt → 화면 표시)", command=self.on_load_txt).pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="전체 삭제", command=self.on_clear_all).pack(side="left")
        ttk.Label(left, text="각도: 0°→오른쪽(+X), 90°→아래(+Y) / map은 위(+Y)", foreground="#666").grid(row=16, column=0, sticky="w", pady=(10, 0))

        ttk.Label(left, text="마커 목록", font=("", 11, "bold")).grid(row=17, column=0, sticky="w", pady=(12, 6))
        self.marker_list = tk.Listbox(left, height=18)
        self.marker_list.grid(row=18, column=0, sticky="nsew")
        ttk.Button(left, text="선택 삭제", command=self.on_delete_selected).grid(row=19, column=0, sticky="ew", pady=(6, 0))

        # Listbox interactions: hover + select
        self.marker_list.bind("<<ListboxSelect>>", self.on_list_select)
        self.marker_list.bind("<Motion>", self.on_list_hover_motion)
        self.marker_list.bind("<Leave>", lambda e: self._clear_highlight(kind="hover"))

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
        self.canvas.bind("<Button-4>", lambda e: self.canvas.yview_scroll(-3, "units"))
        self.canvas.bind("<Button-5>", lambda e: self.canvas.yview_scroll(+3, "units"))

    def _on_mousewheel(self, event):
        # Shift 누르면 수평 스크롤, 아니면 수직
        if event.state & 0x0001:
            self.canvas.xview_scroll(-1 * int(event.delta / 120), "units")
        else:
            self.canvas.yview_scroll(-1 * int(event.delta / 120), "units")

    def _on_shift_mousewheel(self, event):
        self.canvas.xview_scroll(-1 * int(event.delta / 120), "units")

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
        self.canvas.config(scrollregion=(0, 0, self.img_size[0], self.img_size[1]))
        self.canvas.create_image(0, 0, anchor="nw", image=self.img_tk, tags=("map",))
        # Reset barrier when loading new image
        self.barrier_map = None
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
        
        # Draw barrier as semi-transparent overlay
        # Use numpy for better performance
        barrier_alpha = 0.3  # Semi-transparent
        
        # Create RGBA image
        barrier_img = np.zeros((H, W, 4), dtype=np.uint8)
        # Set red color where barrier is True
        barrier_mask = self.barrier_map
        barrier_img[barrier_mask, 0] = 255  # R
        barrier_img[barrier_mask, 1] = 107  # G
        barrier_img[barrier_mask, 2] = 107  # B
        barrier_img[barrier_mask, 3] = int(255 * barrier_alpha)  # A
        
        # Convert to PIL Image
        barrier_pil = Image.fromarray(barrier_img, mode='RGBA')
        barrier_tk = ImageTk.PhotoImage(barrier_pil)
        self.canvas.create_image(0, 0, anchor="nw", image=barrier_tk, tags=("barrier",))
        # Keep reference to prevent garbage collection
        if not hasattr(self, '_barrier_tk_refs'):
            self._barrier_tk_refs = []
        self._barrier_tk_refs.append(barrier_tk)

    # -------------- Route control --------------
    def on_new_route(self):
        if self.current_route_open:
            if not messagebox.askyesno("확인", "현재 경로가 종료되지 않았습니다. 새 경로를 시작할까요?"):
                return
            self.current_route_open = False
            self.next_is_goal = False
        self.current_route_id += 1
        self.current_route_open = True
        self.next_is_goal = False
        self.next_seq = 0
        messagebox.showinfo("안내", f"새 경로 #{self.current_route_id} 시작: 첫 클릭은 START, 이후는 WAYPOINT입니다.\n'마지막 노드(Goal)로 종료'를 누른 다음 클릭으로 GOAL을 지정하세요.")

    def on_arm_goal_once(self):
        if not self.current_route_open:
            messagebox.showwarning("경고", "열린 경로가 없습니다. 먼저 '새 경로 시작'을 누르세요.")
            return
        self.next_is_goal = True

    # -------------- Canvas interactions --------------
    def on_mouse_down(self, event):
        if not self.img:
            return
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.drag_start = (x, y)
        self._clear_preview()

    def on_mouse_drag(self, event):
        if not self.img or not self.drag_start:
            return
        x0, y0 = self.drag_start
        x1 = self.canvas.canvasx(event.x)
        y1 = self.canvas.canvasy(event.y)
        self._draw_preview(x0, y0, x1, y1)

    def on_mouse_up(self, event):
        if not self.img or not self.drag_start:
            self._clear_preview()
            return
        x0, y0 = self.drag_start
        x1 = self.canvas.canvasx(event.x)
        y1 = self.canvas.canvasy(event.y)
        self._clear_preview()
        if (x1 - x0) ** 2 + (y1 - y0) ** 2 < 3**2:
            self.drag_start = None
            return

        dx, dy = (x1 - x0), (y1 - y0)
        theta_deg = math.degrees(math.atan2(dy, dx))
        if theta_deg < 0:
            theta_deg += 360.0

        if not self.current_route_open:
            # auto-start a new route if user clicks without pressing '새 경로 시작'
            self.current_route_id += 1
            self.current_route_open = True
            self.next_is_goal = False
            self.next_seq = -1

        self.next_seq += 1

        # Check if point is in barrier area (for waypoint only)
        if self.next_seq > 0 and not self.next_is_goal:
            if self._is_in_barrier(x0, y0):
                messagebox.showwarning("경고", f"배리어 영역에는 waypoint를 생성할 수 없습니다.\n위치: ({x0:.1f}, {y0:.1f})")
                self.drag_start = None
                return

        if self.next_seq == 0:
            mtype = "start"
        else:
            if self.next_is_goal:
                mtype = "goal"
                self.current_route_open = False
                self.next_is_goal = False
            else:
                mtype = "waypoint"

        mk = Marker(
            id=f"mk_{int(time.time()*1000)}",
            type=mtype,
            x=float(x0),
            y=float(y0),
            theta_deg=float(theta_deg),
            source="live",
            route_id=self.current_route_id,
            seq=self.next_seq
        )
        self.markers.insert(0, mk)
        self._redraw_all_markers()
        self._refresh_marker_list()

        self.drag_start = None

    def _draw_preview(self, x0, y0, x1, y1):
        self._clear_preview()
        self.preview_line = self.canvas.create_line(x0, y0, x1, y1, fill="#2563eb", dash=(4, 3), width=2, tags=("preview",))
        self.preview_head = self._draw_arrow_head(x0, y0, x1, y1, fill="#2563eb")
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
        for item in self.canvas.find_withtag("barrier"):
            self.canvas.delete(item)
        # clear highlights (will be restored below if selection exists)
        self._clear_highlight()

        if not self.img:
            return

        # Draw barrier first (behind markers)
        self._draw_barrier()

        # hide_live: show only most recent live route_id
        allowed_live_route = None
        if self.hide_live.get():
            live_routes = [mk.route_id for mk in self.markers if mk.source == "live"]
            if live_routes:
                allowed_live_route = max(live_routes)

        for mk in self.markers:
            if mk.source == "live" and allowed_live_route is not None and mk.route_id != allowed_live_route:
                continue
            self._draw_marker(mk)

        # restore selection highlight if any
        self._redraw_selection_highlight()

    def _draw_marker(self, mk: Marker):
        length = 40.0
        theta = math.radians(mk.theta_deg)
        x1 = mk.x + math.cos(theta) * length
        y1 = mk.y + math.sin(theta) * length

        # colors by type/source
        if mk.type == "start":
            color = "#16a34a"  # green
        elif mk.type == "goal":
            color = "#dc2626"  # red
        else:
            color = "#1f4f8b" if mk.source == "file" else "#3b82f6"  # waypoint: blue-ish

        # body + head
        self.canvas.create_line(mk.x, mk.y, x1, y1, fill=color, width=3, tags=("marker",))
        head = self._draw_arrow_head(mk.x, mk.y, x1, y1, fill=color)
        self.canvas.addtag_withtag("marker", head)

        # label: Start/Goal letters; waypoint shows seq index
        if mk.type == "start":
            self.canvas.create_text(mk.x + 8, mk.y - 8, text="S", font=("Arial", 10, "bold"), fill="#111", tags=("marker",))
        elif mk.type == "goal":
            self.canvas.create_text(mk.x + 8, mk.y - 8, text="G", font=("Arial", 10, "bold"), fill="#111", tags=("marker",))
        else:  # waypoint
            r = 10
            self.canvas.create_oval(mk.x - r, mk.y - r, mk.x + r, mk.y + r, fill="#ffffff", outline=color, width=2, tags=("marker",))
            self.canvas.create_text(mk.x, mk.y, text=str(mk.seq), font=("Arial", 9, "bold"), fill=color, tags=("marker",))

    def _draw_arrow_head(self, x0, y0, x1, y1, fill="#000"):
        angle = math.atan2(y1 - y0, x1 - x0)
        ah = 10.0
        back = 6.0
        bx = x1 - math.cos(angle) * back
        by = y1 - math.sin(angle) * back
        left = (bx - math.cos(angle - math.pi/2) * ah, by - math.sin(angle - math.pi/2) * ah)
        right = (bx - math.cos(angle + math.pi/2) * ah, by - math.sin(angle + math.pi/2) * ah)
        return self.canvas.create_polygon((x1, y1, left[0], left[1], right[0], right[1]), fill=fill, outline="", tags=("marker",))

    # ---------- Highlight helpers ----------
    def _clear_highlight(self, kind: Optional[str] = None):
        """kind=None → 모두, 'hover' 또는 'selected'만 삭제"""
        tag = "highlight" if kind is None else f"hl_{kind}"
        for item in self.canvas.find_withtag(tag):
            self.canvas.delete(item)

    def _draw_highlight(self, mk: Marker, kind: str = "selected"):
        color = "#f59e0b" if kind == "hover" else "#22c55e"  # amber for hover, green for selected
        r1, r2 = 18, 26
        # two concentric dashed rings
        o1 = self.canvas.create_oval(mk.x - r1, mk.y - r1, mk.x + r1, mk.y + r1,
                                     outline=color, width=3, dash=(4, 3), tags=("highlight", f"hl_{kind}"))
        o2 = self.canvas.create_oval(mk.x - r2, mk.y - r2, mk.x + r2, mk.y + r2,
                                     outline=color, width=2, dash=(2, 3), tags=("highlight", f"hl_{kind}"))
        # crosshair
        ch1 = self.canvas.create_line(mk.x - r2 - 6, mk.y, mk.x - 4, mk.y, fill=color, width=2, tags=("highlight", f"hl_{kind}"))
        ch2 = self.canvas.create_line(mk.x + 4, mk.y, mk.x + r2 + 6, mk.y, fill=color, width=2, tags=("highlight", f"hl_{kind}"))
        ch3 = self.canvas.create_line(mk.x, mk.y - r2 - 6, mk.x, mk.y - 4, fill=color, width=2, tags=("highlight", f"hl_{kind}"))
        ch4 = self.canvas.create_line(mk.x, mk.y + 4, mk.x, mk.y + r2 + 6, fill=color, width=2, tags=("highlight", f"hl_{kind}"))
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
        idx = sel[0]
        if idx < 0 or idx >= len(self.markers):
            return
        self.selected_idx = idx
        mk = self.markers[idx]
        self._clear_highlight(kind="selected")
        self._draw_highlight(mk, kind="selected")
        self._scroll_to_center(mk.x, mk.y)

    def on_list_hover_motion(self, event):
        idx = self.marker_list.nearest(event.y)
        if idx < 0 or idx >= len(self.markers):
            self._clear_highlight(kind="hover")
            self.last_hover_idx = None
            return
        if self.last_hover_idx == idx:
            return
        self.last_hover_idx = idx
        mk = self.markers[idx]
        self._clear_highlight(kind="hover")
        self._draw_highlight(mk, kind="hover")

    # -------------- List / Save / Load / Delete --------------
    def _refresh_marker_list(self):
        self.marker_list.delete(0, tk.END)
        for mk in self.markers:
            tag = "L" if mk.source == "live" else "F"
            self.marker_list.insert(
                tk.END,
                f"{mk.id[-6:]} [{tag}] route={mk.route_id} seq={mk.seq} {mk.type.upper()}  u={mk.x:.1f} v={mk.y:.1f} θ={mk.theta_deg:.1f}°"
            )
        # selection index 유효성 보정
        if self.selected_idx is not None:
            if 0 <= self.selected_idx < len(self.markers):
                self.marker_list.selection_clear(0, tk.END)
                self.marker_list.selection_set(self.selected_idx)
            else:
                self.selected_idx = None

    def on_delete_selected(self):
        sel = self.marker_list.curselection()
        if not sel:
            return
        idx = sel[0]
        try:
            del self.markers[idx]
        except IndexError:
            return
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
            self.markers.clear()
            self.selected_idx = None
            self._refresh_marker_list()
            self._redraw_all_markers()
            # reset route state
            self.current_route_open = False
            self.next_is_goal = False
            self.next_seq = 0

    def on_save(self):
        if not self.markers:
            messagebox.showinfo("안내", "저장할 마커가 없습니다.")
            return
        if not self.img:
            messagebox.showwarning("경고", "이미지가 로드되어야 저장할 수 있습니다.")
            return

        W, H = self.img_size
        meta = self.meta
        default_name = f"markers_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        path = filedialog.asksaveasfilename(
            title="저장할 파일 선택",
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[("Text", "*.txt"), ("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("# Saved by map_marker_app_yaml_v1_5_waypoints.py\n")
                if self.yaml_path:
                    f.write(f"# yaml: {self.yaml_path}\n")
                f.write(f"# image: {self.img_path}\n")
                f.write(f"# resolution: {meta.resolution}\n")
                f.write(f"# origin: [{meta.origin_x}, {meta.origin_y}, {meta.origin_yaw}]\n")
                f.write(f"# use_center: {self.use_center.get()}\n")
                f.write("id,type,source,route_id,seq,u_px,v_px,theta_img_deg,x_map,y_map,theta_map_rad\n")
                # save bottom->top for readability
                for mk in reversed(self.markers):
                    x_map, y_map, theta_map = img_to_map(
                        mk.x, mk.y, mk.theta_deg, W, H, meta, use_center=self.use_center.get()
                    )
                    f.write(f"{mk.id},{mk.type},{mk.source},{mk.route_id},{mk.seq},{mk.x:.2f},{mk.y:.2f},{mk.theta_deg:.2f},{x_map:.6f},{y_map:.6f},{theta_map:.6f}\n")
            messagebox.showinfo("완료", f"저장했습니다:\n{path}")
        except Exception as e:
            messagebox.showerror("에러", f"저장 실패: {e}")

    def on_load_txt(self):
        if not self.img:
            messagebox.showwarning("경고", "먼저 이미지를 로드하세요.")
            return
        path = filedialog.askopenfilename(
            title="마커 파일 열기 (.txt/.csv)",
            filetypes=[("Text/CSV", "*.txt *.csv *.TXT *.CSV"), ("All files", "*.*")],
        )
        if not path:
            return

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
            self.current_route_open = False
            self.next_is_goal = False
            self.next_seq = 0

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

