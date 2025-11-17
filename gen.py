import requests
import time
import os
import math
import re
import threading
import queue
import json  # Import JSON for config saving
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox

# ==============================================================================
# 1. API & LOGIC FUNCTIONS
# ==============================================================================

def log_message(log_queue, message):
    """Puts a message into the queue for the GUI to display."""
    log_queue.put(message)

def sanitize_foldername(name):
    """Removes characters that are invalid for a folder name."""
    name = name.lower()
    name = re.sub(r'[^a-z0-9_\- ]', '', name) # Allow letters, numbers, underscore, hyphen, space
    name = re.sub(r'\s+', '_', name) # Replace spaces with underscores
    return name

def segment_script(log_queue, script_text, num_segments):
    """Splits the full script into a specified number of roughly equal chunks."""
    words = script_text.strip().split()
    if not words:
        return []

    total_words = len(words)
    words_per_segment = math.ceil(total_words / num_segments)
    
    segments = []
    for i in range(num_segments):
        start_index = i * words_per_segment
        end_index = min((i + 1) * words_per_segment, total_words)
        segment_words = words[start_index:end_index]
        if segment_words:
            segments.append(" ".join(segment_words))
            
    log_message(log_queue, f"Script split into {len(segments)} segments.")
    return segments

def generate_prompt_from_chunk(log_queue, script_chunk, style, deepseek_key):
    """Uses DeepSeek to convert a script chunk into a descriptive image prompt."""
    log_message(log_queue, f"  > Generating prompt for chunk: '{script_chunk[:50]}...'")
    
    # --- BUG FIX: Added missing '://' ---
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {deepseek_key}"
    }
    
    system_message = f"""
    You are an expert AI prompt engineer. Your job is to convert a segment of a
    script into a single, concise, visually descriptive image prompt.
    The prompt MUST adhere to the following style: {style}
    Do NOT include any other text, just the final prompt.
    """
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": script_chunk}
        ],
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        image_prompt = data['choices'][0]['message']['content']
        return image_prompt.strip()
    except requests.RequestException as e:
        log_message(log_queue, f"  > ERROR (DeepSeek): {e}")
        return None
    except (KeyError, IndexError) as e:
        log_message(log_queue, f"  > ERROR (DeepSeek Response): Could not parse. {e}")
        return None

def start_image_generation(log_queue, prompt, freepik_key, aspect_ratio):
    """Calls the Freepik POST endpoint to start the image generation task."""
    log_message(log_queue, f"  > Starting image generation for prompt: '{prompt[:50]}...'")
    
    # --- BUG FIX: Added missing '://' ---
    url = "https://api.freepik.com/v1/ai/text-to-image/seedream"
    headers = {
        'Content-Type': 'application/json',
        'x-freepik-api-key': freepik_key
    }
    payload = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "guidance_scale": 7.5
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        
        data_obj = data.get('data', {})
        task_id = data_obj.get('task_id')
        
        if task_id:
            log_message(log_queue, f"  > Task started. Task ID: {task_id}")
            return task_id
        else:
            log_message(log_queue, f"  > ERROR (Freepik): No task_id in response.")
            log_message(log_queue, f"  > Freepik Full Response: {data}")
            return None
    except requests.RequestException as e:
        log_message(log_queue, f"  > ERROR (Freepik): {e}")
        return None

def poll_for_image_url(log_queue, task_id, freepik_key):
    """Calls the Freepik GET endpoint repeatedly until the image is ready."""
    log_message(log_queue, "  > Polling for image status (please wait)...")
    
    # --- BUG FIX: Added missing '://' ---
    url = f"https://api.freepik.com/v1/ai/text-to-image/seedream/{task_id}"
    headers = {'x-freepik-api-key': freepik_key}
    
    POLL_INTERVAL = 5
    MAX_ATTEMPTS = 20
    
    for attempt in range(MAX_ATTEMPTS):
        try:
            time.sleep(POLL_INTERVAL) # Wait *before* checking
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            data_obj = data.get('data', {})
            status = data_obj.get('status')
            generated_images = data_obj.get('generated')
            
            if generated_images and isinstance(generated_images, list) and len(generated_images) > 0:
                image_url = next((item for item in generated_images if isinstance(item, str) and item.startswith('http')), None)
                if image_url:
                    log_message(log_queue, "  > Image generation successful!")
                    return image_url
            
            log_message(log_queue, f"  > Status (Attempt {attempt+1}/{MAX_ATTEMPTS}): {status}. Waiting...")
        except requests.RequestException as e:
            log_message(log_queue, f"  > ERROR (Freepik Poll): {e}. Retrying...")
            
    log_message(log_queue, "  > ERROR: Max polling attempts reached. Job failed or timed out.")
    return None

def download_and_save_image(log_queue, image_url, file_path):
    """Downloads the final image from the URL and saves it to the disk."""
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            f.write(response.content)
        log_message(log_queue, f"  > SUCCESS: Image saved to {file_path}")
    except requests.RequestException as e:
        log_message(log_queue, f"  > ERROR (Download): Could not download image. {e}")

# ==============================================================================
# 2. MAIN PIPELINE (Unchanged)
# ==============================================================================

def main_pipeline(log_queue, config):
    """
    Main function to run the entire pipeline, now taking config as an argument.
    """
    log_message(log_queue, "Starting Mass Image Generation Pipeline...")
    
    # Unpack config
    freepik_key = config["freepik_key"]
    deepseek_key = config["deepseek_key"]
    full_script = config["full_script"]
    num_images = config["num_images"]
    style_guide = config["style_guide"]
    output_folder = config["output_folder"]
    aspect_ratio = config["aspect_ratio"] 

    os.makedirs(output_folder, exist_ok=True)
    log_message(log_queue, f"Images will be saved as 1.png, 2.png, etc. in '{output_folder}'.")

    # 1. Segment Script
    script_segments = segment_script(log_queue, full_script, num_images)
    
    if not script_segments:
        log_message(log_queue, "Error: Script is empty or could not be segmented. Exiting.")
        return
        
    image_counter = 1
    
    for i, segment in enumerate(script_segments):
        log_message(log_queue, f"\n--- Processing Segment {i+1} of {len(script_segments)} ---")
        
        # 2. Generate Prompt
        image_prompt = generate_prompt_from_chunk(log_queue, segment, style_guide, deepseek_key)
        
        if not image_prompt:
            log_message(log_queue, "  > Skipping segment due to prompt generation error.")
            continue
        log_message(log_queue, f"  > Generated Prompt: {image_prompt}")
            
        # 3. Start Image Generation
        task_id = start_image_generation(log_queue, image_prompt, freepik_key, aspect_ratio)
        
        if not task_id:
            log_message(log_queue, "  > Skipping segment due to image generation start error.")
            continue
            
        # 4. Poll for URL
        image_url = poll_for_image_url(log_queue, task_id, freepik_key)
        
        if not image_url:
            log_message(log_queue, "  > Skipping segment due to image polling error.")
            continue
        log_message(log_queue, f"  > Image URL found: {image_url}")
        
        # 5. Download and Save
        file_name = f"{image_counter}.png"
        file_path = os.path.join(output_folder, file_name)
        
        download_and_save_image(log_queue, image_url, file_path)
        image_counter += 1
        
    log_message(log_queue, "\n--- Pipeline Complete ---")
    log_message(log_queue, f"Successfully generated {image_counter - 1} images.")

# ==============================================================================
# 3. TKINTER GUI APPLICATION (New 2-Column "Best" Design)
# ==============================================================================

class App(tk.Tk):
    
    CONFIG_FILE = "config.json"
    
    BG_COLOR = "#2B2B2B"
    FG_COLOR = "#E0E0E0"
    WIDGET_BG = "#3C3F41"
    ACCENT_COLOR = "#007ACC"
    TEXT_COLOR = "#FFFFFF"
    BORDER_COLOR = "#555555"

    def __init__(self):
        super().__init__()
        
        self.title("Mass Image Generator")
        # --- NEW: Wider geometry for 2-column layout ---
        self.geometry("1400x900") 
        self.configure(bg=self.BG_COLOR)
        
        # --- Configure Styles ---
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        
        # Base style
        self.style.configure(
            '.', 
            background=self.BG_COLOR,
            foreground=self.FG_COLOR,
            fieldbackground=self.WIDGET_BG,
            bordercolor=self.BORDER_COLOR,
            lightcolor=self.WIDGET_BG,
            darkcolor=self.WIDGET_BG,
            insertcolor=self.TEXT_COLOR
        )
        self.style.map('.', 
            background=[('active', self.WIDGET_BG)], 
            foreground=[('active', self.TEXT_COLOR)]
        )
        
        # Header Label
        self.style.configure('Header.TLabel', 
            font=('Helvetica', 18, 'bold'),
            foreground=self.TEXT_COLOR
        )
        
        # Label style
        self.style.configure('TLabel', font=('Helvetica', 10))
        
        # Entry style
        self.style.configure('TEntry', font=('Helvetica', 10))
        self.style.map('TEntry', foreground=[('readonly', self.FG_COLOR)])
        
        # Combobox style
        self.style.configure('TCombobox', font=('Helvetica', 10))
        self.style.map('TCombobox', 
            fieldbackground=[('readonly', self.WIDGET_BG)],
            selectbackground=[('readonly', self.WIDGET_BG)],
            selectforeground=[('readonly', self.FG_COLOR)]
        )
        
        # Button style
        self.style.configure('TButton', 
            font=('Helvetica', 10, 'bold'),
            bordercolor=self.ACCENT_COLOR
        )
        self.style.map('TButton',
            background=[('active', '#005f9e'), ('disabled', self.WIDGET_BG)],
            foreground=[('disabled', self.FG_COLOR)]
        )
        
        # "Start" Button Style
        self.style.configure('Start.TButton', 
            background=self.ACCENT_COLOR, 
            foreground=self.TEXT_COLOR, 
            font=('Helvetica', 12, 'bold'),
            padding=(20, 10)
        )
        
        # Labelframe style
        self.style.configure('TLabelframe', 
            labelmargins=(10, 5, 10, 5),
            padding=(10, 10),
            relief=tk.RIDGE,
            bordercolor=self.BORDER_COLOR
        )
        self.style.configure('TLabelframe.Label', 
            font=('Helvetica', 11, 'bold'),
            foreground=self.TEXT_COLOR,
            background=self.BG_COLOR
        )
        
        # --- Main Layout Frame ---
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        # --- NEW: Configure 2 columns (left for settings, right for logs) ---
        main_frame.columnconfigure(0, weight=1, minsize=500) # Settings column
        main_frame.columnconfigure(1, weight=2) # Log column
        main_frame.rowconfigure(1, weight=1)    # Make content row resizable
        
        # --- Header ---
        header_label = ttk.Label(main_frame, text="🎬 Script-to-Image Mass Generator", style="Header.TLabel")
        # --- NEW: Span across 2 columns ---
        header_label.grid(row=0, column=0, columnspan=2, pady=(0, 20), sticky="w")
        
        # --- NEW: Left Frame for all settings ---
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        left_frame.columnconfigure(0, weight=1)
        
        # --- Section 1: Configuration ---
        config_frame = ttk.Labelframe(left_frame, text="Configuration")
        config_frame.grid(row=0, column=0, pady=10, sticky="ew")
        config_frame.columnconfigure(1, weight=1)
        
        ttk.Label(config_frame, text="Freepik API Key:").grid(row=0, column=0, padx=10, pady=8, sticky="w")
        self.freepik_key_entry = ttk.Entry(config_frame, width=80, show="*")
        self.freepik_key_entry.grid(row=0, column=1, columnspan=2, padx=10, pady=8, sticky="ew")
        
        ttk.Label(config_frame, text="DeepSeek API Key:").grid(row=1, column=0, padx=10, pady=8, sticky="w")
        self.deepseek_key_entry = ttk.Entry(config_frame, width=80, show="*")
        self.deepseek_key_entry.grid(row=1, column=1, columnspan=2, padx=10, pady=8, sticky="ew")
        
        ttk.Label(config_frame, text="Output Folder:").grid(row=2, column=0, padx=10, pady=8, sticky="w")
        self.output_folder_entry = ttk.Entry(config_frame, width=60, state="readonly")
        self.output_folder_entry.grid(row=2, column=1, padx=10, pady=8, sticky="ew")
        self.browse_button = ttk.Button(config_frame, text="Browse...", command=self.select_output_folder)
        self.browse_button.grid(row=2, column=2, padx=(5, 10), pady=8, sticky="w")
        
        self.output_folder_entry.configure(state="normal")
        self.output_folder_entry.insert(0, os.path.join(os.getcwd(), "generated_images"))
        self.output_folder_entry.configure(state="readonly")
        
        # --- Section 2: Generation Settings ---
        settings_frame = ttk.Labelframe(left_frame, text="Generation Settings")
        settings_frame.grid(row=1, column=0, pady=10, sticky="ew")
        settings_frame.columnconfigure(1, weight=1)
        
        ttk.Label(settings_frame, text="Number of Images:").grid(row=0, column=0, padx=10, pady=8, sticky="w")
        self.num_images_entry = ttk.Entry(settings_frame, width=10)
        self.num_images_entry.grid(row=0, column=1, padx=10, pady=8, sticky="w")
        
        ttk.Label(settings_frame, text="Aspect Ratio:").grid(row=1, column=0, padx=10, pady=8, sticky="w")
        self.aspect_ratio_options = ["square_1_1", "widescreen_16_9", "social_story_9_16"]
        self.aspect_ratio_combo = ttk.Combobox(settings_frame, values=self.aspect_ratio_options, state="readonly", width=18)
        self.aspect_ratio_combo.current(0)
        self.aspect_ratio_combo.grid(row=1, column=1, padx=10, pady=8, sticky="w")
        
        ttk.Label(settings_frame, text="Prompt Style Guide:").grid(row=2, column=0, padx=10, pady=8, sticky="nw")
        self.style_guide_text = self.create_scrolled_text(settings_frame, height=4)
        self.style_guide_text.grid(row=2, column=1, columnspan=2, padx=10, pady=8, sticky="ew")
        self.style_guide_text.insert(tk.END, "A cinematic, photorealistic style. Highly detailed, 8K, dramatic lighting, shallow depth of field. Epic and grand.")
        
        # --- Section 3: Script Input ---
        script_frame = ttk.Labelframe(left_frame, text="Your Script")
        script_frame.grid(row=2, column=0, pady=10, sticky="ew")
        script_frame.columnconfigure(0, weight=1)
        
        self.full_script_text = self.create_scrolled_text(script_frame, height=15)
        self.full_script_text.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.full_script_text.insert(tk.END, "Paste your full script here...")
        
        # --- Section 4: Start Button ---
        self.start_button = ttk.Button(left_frame, text="Start Generation", command=self.start_pipeline, style="Start.TButton")
        self.start_button.grid(row=3, column=0, pady=20)
        
        # --- NEW: Right Frame for Log ---
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, sticky="nsew", padx=(10, 0))
        right_frame.rowconfigure(0, weight=1)
        right_frame.columnconfigure(0, weight=1)

        # --- Section 5: Log Console (now in right_frame) ---
        log_frame = ttk.Labelframe(right_frame, text="Live Log")
        log_frame.grid(row=0, column=0, sticky="nsew") # Fills the entire right frame
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = self.create_scrolled_text(log_frame, height=20, font_name="Consolas", font_size=9)
        self.log_text.configure(state="disabled")
        self.log_text.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # --- Threading & Queue Setup ---
        self.log_queue = queue.Queue()
        self.is_running = False
        
        self.load_config()
        self.process_log_queue()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_scrolled_text(self, parent, height, font_name='Helvetica', font_size=10):
        """Helper to create a styled ScrolledText widget."""
        return scrolledtext.ScrolledText(
            parent, height=height, 
            bg=self.WIDGET_BG, fg=self.TEXT_COLOR, 
            insertbackground=self.TEXT_COLOR, 
            font=(font_name, font_size),
            relief=tk.FLAT, 
            borderwidth=2,
            padx=5, pady=5
        )

    def load_config(self):
        """Loads API keys from config.json if it exists."""
        try:
            if os.path.exists(self.CONFIG_FILE):
                with open(self.CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    if config.get("freepik_key"):
                        self.freepik_key_entry.delete(0, tk.END)
                        self.freepik_key_entry.insert(0, config["freepik_key"])
                    if config.get("deepseek_key"):
                        self.deepseek_key_entry.delete(0, tk.END)
                        self.deepseek_key_entry.insert(0, config["deepseek_key"])
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading config: {e}")

    def save_config(self):
        """Saves the current API keys to config.json."""
        config = {
            "freepik_key": self.freepik_key_entry.get(),
            "deepseek_key": self.deepseek_key_entry.get()
        }
        try:
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
        except IOError as e:
            print(f"Error saving config: {e}")

    def on_closing(self):
        """Called when the window is closed."""
        self.save_config()
        self.destroy()

    def select_output_folder(self):
        """Opens a dialog to select an output folder."""
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.output_folder_entry.configure(state="normal")
            self.output_folder_entry.delete(0, tk.END)
            self.output_folder_entry.insert(0, folder_selected)
            self.output_folder_entry.configure(state="readonly")
            
    def start_pipeline(self):
        """Validates inputs and starts the main_pipeline in a new thread."""
        if self.is_running:
            messagebox.showwarning("In Progress", "A generation task is already running.")
            return

        # 1. Get and Validate Inputs
        config = {
            "freepik_key": self.freepik_key_entry.get(),
            "deepseek_key": self.deepseek_key_entry.get(),
            "full_script": self.full_script_text.get("1.0", tk.END),
            "style_guide": self.style_guide_text.get("1.0", tk.END),
            "output_folder": self.output_folder_entry.get(),
            "aspect_ratio": self.aspect_ratio_combo.get()
        }
        
        if not config["freepik_key"] or not config["deepseek_key"]:
            messagebox.showerror("Error", "API keys are required.")
            return
            
        try:
            config["num_images"] = int(self.num_images_entry.get())
            if config["num_images"] <= 0: raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Number of images must be a positive integer.")
            return
            
        self.save_config()

        self.start_button.config(text="Generating... Please Wait", state="disabled")
        self.is_running = True
        
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state="disabled")

        self.pipeline_thread = threading.Thread(
            target=self.run_pipeline_thread, 
            args=(config,), 
            daemon=True
        )
        self.pipeline_thread.start()

    def run_pipeline_thread(self, config):
        """Wrapper function to run in the thread, allowing for post-processing."""
        try:
            main_pipeline(self.log_queue, config)
        except Exception as e:
            self.log_queue.put(f"\n--- A CRITICAL ERROR OCCURRED ---\n{e}")
        finally:
            self.log_queue.put("---PIPELINE_COMPLETE---")

    def process_log_queue(self):
        """Checks the log queue for messages and updates the GUI."""
        try:
            while True:
                message = self.log_queue.get_nowait()
                
                if message == "---PIPELINE_COMPLETE---":
                    self.start_button.config(text="Start Generation", state="normal")
                    self.is_running = False
                else:
                    self.log_text.configure(state="normal")
                    self.log_text.insert(tk.END, message + "\n")
                    self.log_text.see(tk.END) # Auto-scroll
                    self.log_text.configure(state="disabled")
                    
        except queue.Empty:
            pass # No messages, just check again later
        finally:
            self.after(100, self.process_log_queue)

# ==============================================================================
# 4. RUN THE APPLICATION
# ==============================================================================

if __name__ == "__main__":
    app = App()
    app.mainloop()