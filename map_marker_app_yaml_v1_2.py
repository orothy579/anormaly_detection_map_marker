
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PNG 맵 좌표·방향 등록기 (YAML 지원) — v1.2
변경 사항:
- ✅ "시작 찍으면 자동으로 목적지" 옵션 추가 (Auto Pair)
  - Start를 배치하면 자동으로 Goal 모드로 전환되고, Goal 배치 후 다시 Start로 복귀
- ✅ 캔버스 스크롤/팬 추가
  - 좌/우 스크롤바, 상/하 스크롤바
  - 마우스 휠/Shift+휠 스크롤 (Linux/Windows/macOS 동시 지원)
  - 휠 버튼(가운데 버튼) 드래그로 팬(scan_mark/scan_dragto)

실행:
  pip install pillow pyyaml
  python3 map_marker_app_yaml_v1_2.py
"""
import os
import math
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    from PIL import Image, ImageTk
except ImportError as e:
    raise SystemExit("Pillow가 필요합니다. 먼저 `pip install pillow`로 설치하세요.") from e

try:
    import yaml
except ImportError as e:
    raise SystemExit("PyYAML이 필요합니다. 먼저 `pip install pyyaml`로 설치하세요.") from e


@dataclass
class Marker:
    id: str
    type: str  # 'start' or 'goal'
    x: float   # image px
    y: float   # image px
    theta_deg: float  # heading in degrees [0..360)
    note: str = ""


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
        self.title("PNG 맵 좌표·방향 등록기 (YAML 지원) v1.2")
        self.geometry("1280x860")
        self.minsize(980, 640)

        # Image & canvas
        self.img: Optional[Image.Image] = None
        self.img_tk: Optional[ImageTk.PhotoImage] = None
        self.img_path: Optional[str] = None
        self.img_size: Tuple[int, int] = (0, 0)  # (W, H)

        # YAML meta
        self.yaml_path: Optional[str] = None
        self.meta: MapMeta = MapMeta()

        # Interaction state
        self.mode = tk.StringVar(value="start")
        self.auto_pair = tk.BooleanVar(value=True)  # Start 후 Goal 자동 전환
        self.use_center = tk.BooleanVar(value=True)  # save as cell centers
        self.markers: List[Marker] = []
        self.preview_line = None
        self.preview_head = None
        self.drag_start: Optional[Tuple[float, float]] = None
        self.hover_pos: Optional[Tuple[float, float]] = None

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
        for i in range(50):
            left.rowconfigure(i, weight=0)
        left.rowconfigure(49, weight=1)

        # --- YAML load ---
        ttk.Label(left, text="1) YAML 불러오기 (map_server)", font=("", 11, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Button(left, text="YAML 열기...", command=self.on_open_yaml).grid(row=1, column=0, sticky="ew")
        self.yaml_info = ttk.Label(left, text="경로: -", foreground="#666")
        self.yaml_info.grid(row=2, column=0, sticky="w", pady=(4, 4))
        self.meta_info = ttk.Label(left, text="resolution: -, origin: -, yaw: -", foreground="#666")
        self.meta_info.grid(row=3, column=0, sticky="w", pady=(0, 12))

        # --- Image load (manual) ---
        ttk.Label(left, text="2) Map 이미지 불러오기 (선택)", font=("", 11, "bold")).grid(row=4, column=0, sticky="w")
        ttk.Button(left, text="PNG/JPG 열기...", command=self.on_open_image_manual).grid(row=5, column=0, sticky="ew")
        self.img_info = ttk.Label(left, text="원본 크기: - x - px", foreground="#666")
        self.img_info.grid(row=6, column=0, sticky="w", pady=(4, 12))

        # --- Marker placement ---
        ttk.Label(left, text="3) 마커 배치 (무제한 중첩 가능)", font=("", 11, "bold")).grid(row=7, column=0, sticky="w", pady=(0, 6))
        mode_frame = ttk.Frame(left)
        mode_frame.grid(row=8, column=0, sticky="w")
        ttk.Radiobutton(mode_frame, text="시작(Start)", value="start", variable=self.mode).pack(side="left", padx=(0, 8))
        ttk.Radiobutton(mode_frame, text="목적지(Goal)", value="goal", variable=self.mode).pack(side="left")

        ttk.Checkbutton(left, text="시작 찍으면 다음은 자동으로 목적지(자동 페어)", variable=self.auto_pair).grid(row=9, column=0, sticky="w", pady=(4, 0))
        ttk.Checkbutton(left, text="셀 중심 좌표로 저장(+0.5 오프셋)", variable=self.use_center).grid(row=10, column=0, sticky="w", pady=(4, 12))
        ttk.Label(left, text="캔버스: 좌클릭 후 드래그 → 방향 지정 / 가운데버튼 드래그: 팬", foreground="#666").grid(row=11, column=0, sticky="w", pady=(0, 12))

        # --- Save/Clear ---
        ttk.Label(left, text="4) 저장 / 관리", font=("", 11, "bold")).grid(row=12, column=0, sticky="w", pady=(0, 6))
        btns = ttk.Frame(left)
        btns.grid(row=13, column=0, sticky="ew")
        ttk.Button(btns, text="저장(.txt: 픽셀+월드)", command=self.on_save).pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="전체 삭제", command=self.on_clear_all).pack(side="left")
        ttk.Label(left, text="각도: 0°→오른쪽(+X), 90°→아래(+Y) / map은 위(+Y)", foreground="#666").grid(row=14, column=0, sticky="w", pady=(10, 0))

        ttk.Label(left, text="마커 목록", font=("", 11, "bold")).grid(row=15, column=0, sticky="w", pady=(12, 6))
        self.marker_list = tk.Listbox(left, height=18)
        self.marker_list.grid(row=16, column=0, sticky="nsew")
        ttk.Button(left, text="선택 삭제", command=self.on_delete_selected).grid(row=17, column=0, sticky="ew", pady=(6, 0))

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
        # Windows/Mac: <MouseWheel>, Linux: Button-4/5
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)          # vertical (Win/Mac)
        self.canvas.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)  # horizontal (Win/Mac)

        # Linux
        self.canvas.bind("<Button-4>", lambda e: self.canvas.yview_scroll(-3, "units"))
        self.canvas.bind("<Button-5>", lambda e: self.canvas.yview_scroll(+3, "units"))
        # Horizontal on Linux usually needs custom handling; we skip for simplicity

    def _on_mousewheel(self, event):
        if event.state & 0x0001:  # Shift pressed → horizontal
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

        if self.meta.image_path:
            img_path = self.meta.image_path
            if not os.path.isabs(img_path):
                img_path = os.path.join(os.path.dirname(self.yaml_path), img_path)
            if os.path.exists(img_path):
                self._load_image(img_path)
            else:
                messagebox.showwarning("경고", f"YAML의 image 경로를 찾을 수 없습니다:\n{img_path}\n수동으로 이미지를 불러오세요.")

    # -------------- Image handling --------------
    def on_open_image_manual(self):
        path = filedialog.askopenfilename(
            title="이미지 열기",
            filetypes=[
                ("Image", "*.png *.jpg *.jpeg *.bmp *.PNG *.JPG *.JPEG *.BMP"),
                ("All files", "*.*"),
            ]
        )
        if not path:
            return
        self._load_image(path)

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
        # Scroll region to image bounds
        self.canvas.config(scrollregion=(0, 0, self.img_size[0], self.img_size[1]))
        # Put image inside a window at (0,0)
        self.canvas.create_image(0, 0, anchor="nw", image=self.img_tk, tags=("map",))
        self._redraw_all_markers()
        self.img_info.config(text=f"원본 크기: {self.img_size[0]} x {self.img_size[1]} px")
        self.update_idletasks()

    # -------------- Canvas interactions --------------
    def on_mouse_down(self, event):
        if not self.img:
            return
        # Convert from canvas to image space using canvas coords
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
        self.hover_pos = (x1, y1)
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

        mk = Marker(
            id=f"mk_{int(time.time()*1000)}",
            type=self.mode.get(),
            x=float(x0),
            y=float(y0),
            theta_deg=float(theta_deg),
        )
        self.markers.insert(0, mk)
        self._redraw_all_markers()
        self._refresh_marker_list()

        # Auto pair toggle: Start → Goal → Start
        if self.auto_pair.get():
            if self.mode.get() == "start":
                self.mode.set("goal")
            else:
                self.mode.set("start")

        self.drag_start = None

    def _draw_preview(self, x0, y0, x1, y1):
        self._clear_preview()
        self.preview_line = self.canvas.create_line(x0, y0, x1, y1, fill="#2563eb", dash=(4, 3), width=2)
        self.preview_head = self._draw_arrow_head(x0, y0, x1, y1, fill="#2563eb")

    def _clear_preview(self):
        if self.preview_line:
            self.canvas.delete(self.preview_line)
            self.preview_line = None
        if self.preview_head:
            self.canvas.delete(self.preview_head)
            self.preview_head = None

    def _redraw_all_markers(self):
        # Remove existing markers
        for item in self.canvas.find_withtag("marker"):
            self.canvas.delete(item)
        if not self.img:
            return
        for mk in self.markers:
            self._draw_marker(mk)

    def _draw_marker(self, mk: Marker):
        length = 40.0
        theta = math.radians(mk.theta_deg)
        x1 = mk.x + math.cos(theta) * length
        y1 = mk.y + math.sin(theta) * length
        color = "#16a34a" if mk.type == "start" else "#dc2626"
        self.canvas.create_line(mk.x, mk.y, x1, y1, fill=color, width=3, tags=("marker",))
        head = self._draw_arrow_head(mk.x, mk.y, x1, y1, fill=color)
        self.canvas.addtag_withtag("marker", head)
        self.canvas.create_text(mk.x + 8, mk.y - 8, text=("S" if mk.type == "start" else "G"),
                                font=("Arial", 10, "bold"), fill="#111", tags=("marker",))

    def _draw_arrow_head(self, x0, y0, x1, y1, fill="#000"):
        angle = math.atan2(y1 - y0, x1 - x0)
        ah = 10.0
        back = 6.0
        bx = x1 - math.cos(angle) * back
        by = y1 - math.sin(angle) * back
        left = (bx - math.cos(angle - math.pi/2) * ah, by - math.sin(angle - math.pi/2) * ah)
        right = (bx - math.cos(angle + math.pi/2) * ah, by - math.sin(angle + math.pi/2) * ah)
        return self.canvas.create_polygon((x1, y1, left[0], left[1], right[0], right[1]), fill=fill, outline="", tags=("marker",))

    # -------------- Marker list / Save --------------
    def _refresh_marker_list(self):
        self.marker_list.delete(0, tk.END)
        for mk in self.markers:
            self.marker_list.insert(tk.END, f"{mk.id[-6:]}  {mk.type.upper()}  u={mk.x:.1f}  v={mk.y:.1f}  θ={mk.theta_deg:.1f}°")

    def on_delete_selected(self):
        sel = self.marker_list.curselection()
        if not sel:
            return
        idx = sel[0]
        try:
            del self.markers[idx]
        except IndexError:
            return
        self._refresh_marker_list()
        self._redraw_all_markers()

    def on_clear_all(self):
        if not self.markers:
            return
        if messagebox.askyesno("확인", "모든 마커를 삭제할까요?"):
            self.markers.clear()
            self._refresh_marker_list()
            self._redraw_all_markers()

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
                f.write("# Saved by map_marker_app_yaml_v1_2.py\n")
                if self.yaml_path:
                    f.write(f"# yaml: {self.yaml_path}\n")
                f.write(f"# image: {self.img_path}\n")
                f.write(f"# resolution: {meta.resolution}\n")
                f.write(f"# origin: [{meta.origin_x}, {meta.origin_y}, {meta.origin_yaw}]\n")
                f.write(f"# use_center: {self.use_center.get()}\n")
                f.write("id,type,u_px,v_px,theta_img_deg,x_map,y_map,theta_map_rad\n")
                for mk in self.markers:
                    x_map, y_map, theta_map = img_to_map(
                        mk.x, mk.y, mk.theta_deg, W, H, meta, use_center=self.use_center.get()
                    )
                    f.write(f"{mk.id},{mk.type},{mk.x:.2f},{mk.y:.2f},{mk.theta_deg:.2f},{x_map:.6f},{y_map:.6f},{theta_map:.6f}\n")
            messagebox.showinfo("완료", f"저장했습니다:\n{path}")
        except Exception as e:
            messagebox.showerror("에러", f"저장 실패: {e}")


if __name__ == "__main__":
    app = MapMarkerApp()
    app.mainloop()
