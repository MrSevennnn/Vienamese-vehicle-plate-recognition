
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import os
from datetime import datetime

from alpr_system import StandaloneALPR


class ALPRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống nhận dạng biển số xe - ALPR System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.alpr_system = None
        self.current_image = None
        self.current_image_path = None
        self.results = None
        self.is_processing = False
        
        # Settings with automatic path detection
        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(current_dir, "weights")
        
        self.settings = {
            'device': 'auto',
            'plate_conf': 0.25,
            'ocr_threshold': 0.9,
            'plate_model': os.path.join(weights_dir, "plate_yolov8n_320_2024.pt")
        }
        
        self.setup_ui()
        
        # Tự động khởi tạo hệ thống ALPR khi chạy chương trình
        self.initialize_system()
        
    def setup_ui(self):
        """Thiết lập giao diện người dùng"""
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main menu
        self.create_menu()
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left panel - Controls
        left_frame = ttk.LabelFrame(main_frame, text="Điều khiển", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        self.create_control_panel(left_frame)
        
        # Right panel - Display
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.create_display_panel(right_frame)
        
        # Bottom status bar
        self.create_status_bar()
        
    def create_menu(self):
        """Tạo menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Mở ảnh...", command=self.load_image, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Lưu kết quả...", command=self.save_result, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Thoát", command=self.root.quit, accelerator="Ctrl+Q")
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Cài đặt", menu=settings_menu)
        settings_menu.add_command(label="Cấu hình mô hình...", command=self.open_settings)
        settings_menu.add_command(label="Khởi tạo lại hệ thống", command=self.reinitialize_system)
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self.load_image())
        self.root.bind('<Control-s>', lambda e: self.save_result())
        self.root.bind('<Control-q>', lambda e: self.root.quit())
        
    def create_control_panel(self, parent):
        """Tạo panel điều khiển"""
        # Input controls
        input_frame = ttk.LabelFrame(parent, text="Nguồn đầu vào", padding=5)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(input_frame, text="Chọn ảnh", command=self.load_image, width=15).pack(pady=2)
        ttk.Button(input_frame, text="Video", command=self.load_video, width=15).pack(pady=2)
        
        # Processing controls
        process_frame = ttk.LabelFrame(parent, text="Xử lý", padding=5)
        process_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.process_button = ttk.Button(process_frame, text="Phân tích ảnh", 
                                        command=self.process_image, state=tk.DISABLED, width=15)
        self.process_button.pack(pady=2)
        
        self.clear_button = ttk.Button(process_frame, text="Xóa kết quả", 
                                      command=self.clear_results, width=15)
        self.clear_button.pack(pady=2)
        
        # Progress bar
        self.progress = ttk.Progressbar(process_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)
        
        # Quick settings
        settings_frame = ttk.LabelFrame(parent, text="Cài đặt nhanh", padding=5)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Plate confidence
        ttk.Label(settings_frame, text="Ngưỡng biển số:").pack(anchor=tk.W)
        self.plate_conf_var = tk.DoubleVar(value=self.settings['plate_conf'])
        plate_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, 
                               variable=self.plate_conf_var, orient=tk.HORIZONTAL)
        plate_scale.pack(fill=tk.X)
        plate_label = ttk.Label(settings_frame, textvariable=self.plate_conf_var)
        plate_label.pack()
        
        # OCR threshold
        ttk.Label(settings_frame, text="Ngưỡng OCR:").pack(anchor=tk.W, pady=(10,0))
        self.ocr_thresh_var = tk.DoubleVar(value=self.settings['ocr_threshold'])
        ocr_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, 
                             variable=self.ocr_thresh_var, orient=tk.HORIZONTAL)
        ocr_scale.pack(fill=tk.X)
        ocr_label = ttk.Label(settings_frame, textvariable=self.ocr_thresh_var)
        ocr_label.pack()
        
        # Bind scale changes
        plate_scale.configure(command=lambda v: self.update_plate_conf())
        ocr_scale.configure(command=lambda v: self.update_ocr_thresh())
        
    def create_display_panel(self, parent):
        """Tạo panel hiển thị"""
        # Notebook for tabs
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Image tab
        image_frame = ttk.Frame(notebook)
        notebook.add(image_frame, text="Hình ảnh")
        
        # Image display
        self.image_canvas = tk.Canvas(image_frame, bg='white', relief=tk.SUNKEN, bd=2)
        self.image_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results tab
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Kết quả")
        
        # Results tree
        self.create_results_tree(results_frame)
        
        # Log tab
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="Nhật ký")
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(log_frame, height=20, width=60)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def create_results_tree(self, parent):
        """Tạo bảng kết quả"""
        # Frame for treeview and scrollbars
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Treeview
        columns = ('ID', 'Biển số', 'Tin cậy phát hiện', 'Text OCR', 'Tin cậy OCR')
        self.results_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=15)
        
        # Define headings and column widths
        widths = [50, 100, 120, 150, 100]
        for col, width in zip(columns, widths):
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=width, anchor=tk.CENTER)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Export button
        export_frame = ttk.Frame(parent)
        export_frame.pack(fill=tk.X, padx=5)
        
        ttk.Button(export_frame, text="Xuất CSV", command=self.export_csv).pack(side=tk.LEFT, padx=5)
        
    def create_status_bar(self):
        """Tạo thanh trạng thái"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_text = ttk.Label(self.status_bar, text="Sẵn sàng")
        self.status_text.pack(side=tk.LEFT, padx=5)
        
        # Processing time label
        self.time_label = ttk.Label(self.status_bar, text="")
        self.time_label.pack(side=tk.RIGHT, padx=5)
        
    def log_message(self, message):
        """Ghi log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.status_text.config(text=message)
        
    def initialize_system(self):
        """Khởi tạo hệ thống ALPR"""
        def init_thread():
            try:
                self.progress.start()
                self.log_message("Đang khởi tạo hệ thống ALPR...")
                
                self.alpr_system = StandaloneALPR(
                    plate_weight_path=self.settings['plate_model'],
                    device=self.settings['device'],
                    plate_conf=self.settings['plate_conf'],
                    ocr_threshold=self.settings['ocr_threshold']
                )
                
                self.log_message("Hệ thống ALPR đã được khởi tạo thành công!")
                self.process_button.config(state=tk.NORMAL)
                
            except Exception as e:
                self.log_message(f"Lỗi khởi tạo: {str(e)}")
                messagebox.showerror("Lỗi", f"Không thể khởi tạo hệ thống:\n{str(e)}")
                
            finally:
                self.progress.stop()
                
        threading.Thread(target=init_thread, daemon=True).start()
        
    def load_image(self):
        """Tải ảnh từ file"""
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[
                ("Tất cả ảnh", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.current_image_path = file_path
                self.current_image = cv2.imread(file_path)
                
                if self.current_image is None:
                    raise ValueError("Không thể đọc file ảnh")
                
                self.display_image(self.current_image)
                self.log_message(f"Đã tải ảnh: {os.path.basename(file_path)}")
                
                if self.alpr_system:
                    self.process_button.config(state=tk.NORMAL)
                    
            except Exception as e:
                self.log_message(f"Lỗi tải ảnh: {str(e)}")
                messagebox.showerror("Lỗi", f"Không thể tải ảnh:\n{str(e)}")
                
    def display_image(self, image, title="Ảnh gốc"):
        """Hiển thị ảnh trên canvas"""
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
                
            # Resize image to fit canvas
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                self.root.after(100, lambda: self.display_image(image, title))
                return
                
            h, w = image_rgb.shape[:2]
            
            # Calculate scale to fit
            scale_w = canvas_width / w
            scale_h = canvas_height / h
            scale = min(scale_w, scale_h, 1.0)  # Don't upscale
            
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize image
            image_resized = cv2.resize(image_rgb, (new_w, new_h))
            
            # Convert to PIL Image and then to PhotoImage
            pil_image = Image.fromarray(image_resized)
            self.photo = ImageTk.PhotoImage(pil_image)
            
            # Clear canvas and display image
            self.image_canvas.delete("all")
            
            # Center the image
            x = (canvas_width - new_w) // 2
            y = (canvas_height - new_h) // 2
            
            self.image_canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
            
        except Exception as e:
            self.log_message(f"Lỗi hiển thị ảnh: {str(e)}")
            
    def process_image(self):
        """Xử lý ảnh với ALPR"""
        if not self.alpr_system or self.current_image is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng khởi tạo hệ thống và tải ảnh trước!")
            return
            
        def process_thread():
            try:
                self.is_processing = True
                self.process_button.config(state=tk.DISABLED)
                self.progress.start()
                
                # Update settings
                self.alpr_system.plate_conf = self.plate_conf_var.get()
                self.alpr_system.ocr_threshold = self.ocr_thresh_var.get()
                
                self.log_message("Đang phân tích ảnh...")
                
                # Process image
                self.results = self.alpr_system.process_image(self.current_image, draw_results=True)
                
                # Update display
                self.display_image(self.results['image'], "Kết quả phân tích")
                self.update_results_table()
                
                # Update status
                processing_time = self.results['processing_time']
                total_plates = len(self.results['plates'])
                
                self.time_label.config(text=f"Thời gian: {processing_time:.3f}s")
                self.log_message(f"Hoàn thành! Phát hiện {total_plates} biển số")
                
            except Exception as e:
                self.log_message(f"Lỗi xử lý: {str(e)}")
                messagebox.showerror("Lỗi", f"Lỗi xử lý ảnh:\n{str(e)}")
                
            finally:
                self.is_processing = False
                self.progress.stop()
                if self.alpr_system:
                    self.process_button.config(state=tk.NORMAL)
                    
        threading.Thread(target=process_thread, daemon=True).start()
        
    def update_results_table(self):
        """Cập nhật bảng kết quả"""
        # Clear existing data
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
            
        if not self.results:
            return
            
        # Add results to table
        row_id = 1
        for i, plate in enumerate(self.results['plates']):
            plate_conf = f"{plate['confidence']:.3f}"
            plate_text = plate.get('text', 'Không đọc được')
            text_conf = f"{plate.get('text_confidence', 0):.3f}"
            
            self.results_tree.insert('', 'end', values=(
                row_id, 
                f"Biển số {i+1}",
                plate_conf, 
                plate_text, 
                text_conf
            ))
            row_id += 1
                
    def update_plate_conf(self):
        """Cập nhật ngưỡng confidence biển số"""
        self.settings['plate_conf'] = self.plate_conf_var.get()
        
    def update_ocr_thresh(self):
        """Cập nhật ngưỡng OCR"""
        self.settings['ocr_threshold'] = self.ocr_thresh_var.get()
        
    def clear_results(self):
        """Xóa kết quả"""
        self.results = None
        self.current_image = None
        self.current_image_path = None
        
        # Clear displays
        self.image_canvas.delete("all")
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
            
        self.time_label.config(text="")
        self.process_button.config(state=tk.DISABLED)
        self.log_message("Đã xóa kết quả")
        
    def save_result(self):
        """Lưu kết quả"""
        if not self.results or not self.results.get('image') is not None:
            messagebox.showwarning("Cảnh báo", "Không có kết quả để lưu!")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Lưu kết quả",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG files", "*.jpg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.results['image'])
                self.log_message(f"Đã lưu kết quả: {os.path.basename(file_path)}")
                messagebox.showinfo("Thành công", "Đã lưu kết quả thành công!")
            except Exception as e:
                self.log_message(f"Lỗi lưu file: {str(e)}")
                messagebox.showerror("Lỗi", f"Không thể lưu file:\n{str(e)}")
                
    def export_csv(self):
        """Xuất kết quả ra CSV"""
        if not self.results:
            messagebox.showwarning("Cảnh báo", "Không có kết quả để xuất!")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Xuất CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['ID', 'Biển số', 'Tin cậy phát hiện', 'Text OCR', 'Tin cậy OCR'])
                    
                    row_id = 1
                    for i, plate in enumerate(self.results['plates']):
                        writer.writerow([
                            row_id, f"Biển số {i+1}", f"{plate['confidence']:.3f}",
                            plate.get('text', 'Không đọc được'), f"{plate.get('text_confidence', 0):.3f}"
                        ])
                        row_id += 1
                            
                self.log_message(f"Đã xuất CSV: {os.path.basename(file_path)}")
                messagebox.showinfo("Thành công", "Đã xuất CSV thành công!")
            except Exception as e:
                self.log_message(f"Lỗi xuất CSV: {str(e)}")
                messagebox.showerror("Lỗi", f"Không thể xuất CSV:\n{str(e)}")
        
    def load_video(self):
        """Tải và xử lý video"""
        file_path = filedialog.askopenfilename(
            title="Chọn video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            if not self.alpr_system:
                messagebox.showwarning("Cảnh báo", "Vui lòng khởi tạo hệ thống ALPR trước!")
                return
                
            # Tạo cửa sổ video processing
            self.create_video_window(file_path)
    
    def create_video_window(self, video_path):
        """Tạo cửa sổ xử lý video"""
        self.video_window = tk.Toplevel(self.root)
        self.video_window.title("Xử lý Video - ALPR")
        self.video_window.geometry("900x700")
        self.video_window.resizable(True, True)
        
        # Make window modal
        self.video_window.transient(self.root)
        self.video_window.grab_set()
        
        # Variables
        self.video_cap = None
        self.video_playing = False
        self.video_results = []
        self.current_frame = None
        self.frame_count = 0
        self.total_frames = 0
        
        # Main frame
        main_frame = ttk.Frame(self.video_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control frame
        control_frame = ttk.LabelFrame(main_frame, text="Điều khiển video", padding=5)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Video info
        self.video_info_label = ttk.Label(control_frame, text=f"Video: {os.path.basename(video_path)}")
        self.video_info_label.pack(anchor=tk.W)
        
        # Progress bar
        self.video_progress = ttk.Progressbar(control_frame, mode='determinate')
        self.video_progress.pack(fill=tk.X, pady=5)
        
        # Frame info
        self.frame_info_label = ttk.Label(control_frame, text="Frame: 0/0")
        self.frame_info_label.pack(anchor=tk.W)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.play_button = ttk.Button(button_frame, text="▶️ Phát", command=self.play_video)
        self.play_button.pack(side=tk.LEFT, padx=2)
        
        self.pause_button = ttk.Button(button_frame, text="⏸️ Tạm dừng", command=self.pause_video, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=2)
        
        self.stop_button = ttk.Button(button_frame, text="⏹️ Dừng", command=self.stop_video, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(button_frame, text="💾 Xuất kết quả", command=self.export_video_results).pack(side=tk.RIGHT, padx=2)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Cài đặt xử lý", padding=5)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Skip frames
        ttk.Label(settings_frame, text="Bỏ qua frames:").pack(anchor=tk.W)
        self.skip_frames_var = tk.IntVar(value=5)
        skip_scale = ttk.Scale(settings_frame, from_=1, to=30, variable=self.skip_frames_var, orient=tk.HORIZONTAL)
        skip_scale.pack(fill=tk.X)
        ttk.Label(settings_frame, textvariable=self.skip_frames_var).pack()
        
        # Display frame
        display_frame = ttk.LabelFrame(main_frame, text="Hiển thị video", padding=5)
        display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Video canvas
        self.video_canvas = tk.Canvas(display_frame, bg='black', relief=tk.SUNKEN, bd=2)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Kết quả phát hiện", padding=5)
        results_frame.pack(fill=tk.X)
        
        # Results text
        self.video_results_text = scrolledtext.ScrolledText(results_frame, height=8, width=80)
        self.video_results_text.pack(fill=tk.X)
        
        # Load video
        self.load_video_file(video_path)
        
        # Handle window close
        self.video_window.protocol("WM_DELETE_WINDOW", self.close_video_window)
    
    def load_video_file(self, video_path):
        """Tải file video"""
        try:
            self.video_cap = cv2.VideoCapture(video_path)
            if not self.video_cap.isOpened():
                raise ValueError("Không thể mở file video")
            
            # Get video properties
            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = self.total_frames / fps if fps > 0 else 0
            
            # Update UI
            self.video_progress.config(maximum=self.total_frames)
            info_text = f"Video: {os.path.basename(video_path)} | {width}x{height} | {fps:.1f}fps | {duration:.1f}s | {self.total_frames} frames"
            self.video_info_label.config(text=info_text)
            self.frame_info_label.config(text=f"Frame: 0/{self.total_frames}")
            
            # Show first frame
            ret, frame = self.video_cap.read()
            if ret:
                self.current_frame = frame
                self.display_video_frame(frame)
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            
            self.log_message(f"Đã tải video: {os.path.basename(video_path)}")
            self.play_button.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải video:\n{str(e)}")
            self.close_video_window()
    
    def display_video_frame(self, frame):
        """Hiển thị frame video"""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to fit canvas
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                self.video_window.after(50, lambda: self.display_video_frame(frame))
                return
            
            h, w = frame_rgb.shape[:2]
            scale_w = canvas_width / w
            scale_h = canvas_height / h
            scale = min(scale_w, scale_h, 1.0)
            
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
            
            # Convert to PhotoImage
            pil_image = Image.fromarray(frame_resized)
            self.video_photo = ImageTk.PhotoImage(pil_image)
            
            # Display on canvas
            self.video_canvas.delete("all")
            x = (canvas_width - new_w) // 2
            y = (canvas_height - new_h) // 2
            self.video_canvas.create_image(x, y, anchor=tk.NW, image=self.video_photo)
            
        except Exception as e:
            print(f"Error displaying frame: {e}")
    
    def play_video(self):
        """Phát video và xử lý ALPR"""
        if not self.video_cap:
            return
        
        self.video_playing = True
        self.play_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL)
        
        self.process_video_frames()
    
    def process_video_frames(self):
        """Xử lý từng frame của video"""
        if not self.video_playing or not self.video_cap:
            return
        
        try:
            ret, frame = self.video_cap.read()
            if not ret:
                # End of video
                self.stop_video()
                return
            
            self.frame_count += 1
            self.current_frame = frame
            
            # Update progress
            self.video_progress.config(value=self.frame_count)
            self.frame_info_label.config(text=f"Frame: {self.frame_count}/{self.total_frames}")
            
            # Process frame if it's time (skip frames for performance)
            skip_frames = self.skip_frames_var.get()
            if self.frame_count % skip_frames == 0:
                # Update settings
                self.alpr_system.plate_conf = self.plate_conf_var.get()
                self.alpr_system.ocr_threshold = self.ocr_thresh_var.get()
                
                # Process frame for ALPR
                results = self.alpr_system.process_image(frame, draw_results=True)
                processed_frame = results['image']
                
                # Display processed frame
                self.display_video_frame(processed_frame)
                
                # Log results
                if results['plates']:
                    for i, plate in enumerate(results['plates']):
                        if plate['text']:
                            result_text = f"Frame {self.frame_count}: {plate['text']} (conf: {plate['text_confidence']:.3f})\n"
                            self.video_results_text.insert(tk.END, result_text)
                            self.video_results_text.see(tk.END)
                            
                            # Store result
                            self.video_results.append({
                                'frame': self.frame_count,
                                'time': self.frame_count / self.video_cap.get(cv2.CAP_PROP_FPS),
                                'plate_text': plate['text'],
                                'confidence': plate['text_confidence'],
                                'bbox': plate['bbox']
                            })
            else:
                # Just display frame without processing
                self.display_video_frame(frame)
            
            # Continue to next frame
            if self.video_playing:
                self.video_window.after(30, self.process_video_frames)  # ~33fps display
                
        except Exception as e:
            self.log_message(f"Lỗi xử lý video: {str(e)}")
            self.stop_video()
    
    def pause_video(self):
        """Tạm dừng video"""
        self.video_playing = False
        self.play_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
    
    def stop_video(self):
        """Dừng video"""
        self.video_playing = False
        self.play_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        
        # Reset to beginning
        if self.video_cap:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_count = 0
            self.video_progress.config(value=0)
            self.frame_info_label.config(text=f"Frame: 0/{self.total_frames}")
    
    def export_video_results(self):
        """Xuất kết quả video ra CSV"""
        if not self.video_results:
            messagebox.showwarning("Cảnh báo", "Chưa có kết quả để xuất!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Xuất kết quả video",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Frame', 'Thời gian (s)', 'Biển số', 'Độ tin cậy', 'Tọa độ'])
                    
                    for result in self.video_results:
                        writer.writerow([
                            result['frame'],
                            f"{result['time']:.2f}",
                            result['plate_text'],
                            f"{result['confidence']:.3f}",
                            f"{result['bbox']}"
                        ])
                
                messagebox.showinfo("Thành công", f"Đã xuất {len(self.video_results)} kết quả!")
                self.log_message(f"Đã xuất kết quả video: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể xuất kết quả:\n{str(e)}")
    
    def close_video_window(self):
        """Đóng cửa sổ video"""
        self.video_playing = False
        if self.video_cap:
            self.video_cap.release()
        self.video_window.destroy()
        
    def open_settings(self):
        """Mở cửa sổ cài đặt"""
        SettingsWindow(self.root, self.settings, self.on_settings_changed)
        
    def on_settings_changed(self, new_settings):
        """Callback khi cài đặt thay đổi"""
        self.settings.update(new_settings)
        self.save_settings()
        self.log_message("Cài đặt đã được cập nhật")
    
    def save_settings(self):
        """Lưu cài đặt ra file"""
        try:
            import json
            with open('settings.json', 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            self.log_message(f"Lỗi lưu cài đặt: {str(e)}")
        
    def reinitialize_system(self):
        """Khởi tạo lại hệ thống"""
        self.alpr_system = None
        self.process_button.config(state=tk.DISABLED)
        self.log_message("Hệ thống đã được reset")


class SettingsWindow:
    def __init__(self, parent, settings, callback):
        self.parent = parent
        self.settings = settings.copy()
        self.callback = callback
        
        self.window = tk.Toplevel(parent)
        self.window.title("Cài đặt hệ thống")
        self.window.geometry("500x400")
        self.window.resizable(False, False)
        
        # Make window modal
        self.window.transient(parent)
        self.window.grab_set()
        
        self.setup_ui()
        
    def setup_ui(self):
        """Thiết lập giao diện cài đặt"""
        # Main frame
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Device settings
        device_frame = ttk.LabelFrame(main_frame, text="Cài đặt thiết bị", padding=10)
        device_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(device_frame, text="Thiết bị:").pack(anchor=tk.W)
        self.device_var = tk.StringVar(value=self.settings['device'])
        device_combo = ttk.Combobox(device_frame, textvariable=self.device_var, 
                                   values=['auto', 'cpu', 'cuda:0'], state='readonly')
        device_combo.pack(fill=tk.X, pady=(0, 5))
        
        # Model paths
        model_frame = ttk.LabelFrame(main_frame, text="Đường dẫn mô hình", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Plate model
        ttk.Label(model_frame, text="Mô hình phát hiện biển số:").pack(anchor=tk.W)
        plate_frame = ttk.Frame(model_frame)
        plate_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.plate_model_var = tk.StringVar(value=self.settings['plate_model'])
        plate_entry = ttk.Entry(plate_frame, textvariable=self.plate_model_var)
        plate_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(plate_frame, text="...", width=3,
                  command=lambda: self.browse_file(self.plate_model_var, "Chọn mô hình biển số")).pack(side=tk.RIGHT)
        
        # Confidence settings
        conf_frame = ttk.LabelFrame(main_frame, text="Ngưỡng confidence", padding=10)
        conf_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Plate confidence  
        ttk.Label(conf_frame, text="Ngưỡng phát hiện biển số:").pack(anchor=tk.W)
        self.plate_conf_var = tk.DoubleVar(value=self.settings['plate_conf'])
        plate_conf_scale = ttk.Scale(conf_frame, from_=0.1, to=1.0, 
                                    variable=self.plate_conf_var, orient=tk.HORIZONTAL)
        plate_conf_scale.pack(fill=tk.X)
        plate_conf_label = ttk.Label(conf_frame, textvariable=self.plate_conf_var)
        plate_conf_label.pack()
        
        # OCR threshold
        ttk.Label(conf_frame, text="Ngưỡng OCR:").pack(anchor=tk.W, pady=(10, 0))
        self.ocr_thresh_var = tk.DoubleVar(value=self.settings['ocr_threshold'])
        ocr_thresh_scale = ttk.Scale(conf_frame, from_=0.1, to=1.0, 
                                    variable=self.ocr_thresh_var, orient=tk.HORIZONTAL)
        ocr_thresh_scale.pack(fill=tk.X)
        ocr_thresh_label = ttk.Label(conf_frame, textvariable=self.ocr_thresh_var)
        ocr_thresh_label.pack()
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Hủy", command=self.window.destroy).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Lưu", command=self.save_settings).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="Mặc định", command=self.reset_defaults).pack(side=tk.LEFT)
        
    def browse_file(self, var, title):
        """Duyệt file mô hình"""
        file_path = filedialog.askopenfilename(
            title=title,
            filetypes=[
                ("PyTorch models", "*.pt"),
                ("ONNX models", "*.onnx"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            var.set(file_path)
            
    def save_settings(self):
        """Lưu cài đặt"""
        self.settings.update({
            'device': self.device_var.get(),
            'plate_model': self.plate_model_var.get(),
            'plate_conf': self.plate_conf_var.get(),
            'ocr_threshold': self.ocr_thresh_var.get()
        })
        
        self.callback(self.settings)
        self.window.destroy()
        
    def reset_defaults(self):
        """Reset về cài đặt mặc định"""
        # Get automatic paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(current_dir, "weights")
        
        defaults = {
            'device': 'auto',
            'plate_model': os.path.join(weights_dir, "plate_yolov8n_320_2024.pt"),
            'plate_conf': 0.25,
            'ocr_threshold': 0.9
        }
        
        self.device_var.set(defaults['device'])
        self.plate_model_var.set(defaults['plate_model'])
        self.plate_conf_var.set(defaults['plate_conf'])
        self.ocr_thresh_var.set(defaults['ocr_threshold'])


def main():
    """Hàm main"""
    root = tk.Tk()
    app = ALPRApp(root)
    
    # Set window icon (if available)
    try:
        root.iconbitmap('icon.ico')
    except:
        pass
        
    # Handle window close
    def on_closing():
        if messagebox.askokcancel("Thoát", "Bạn có chắc muốn thoát?"):
            root.destroy()
            
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start main loop
    root.mainloop()


if __name__ == "__main__":
    main()