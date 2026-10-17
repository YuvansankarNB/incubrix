import os
import hashlib
import json
import math
import random
import urllib.parse
import urllib.request
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.config import settings
from src.logging_config import logger
import imageio_ffmpeg

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

class FootageClip:
    def __init__(
        self,
        clip_id: str,
        local_path: Path,
        source: str,
        license_name: str,
        attribution: str,
        original_url: str,
        is_ai_generated: bool,
        selection_reason: str,
        relevance_score: float = 4.2
    ):
        self.clip_id = clip_id
        self.local_path = local_path
        self.source = source
        self.license_name = license_name
        self.attribution = attribution
        self.original_url = original_url
        self.is_ai_generated = is_ai_generated
        self.selection_reason = selection_reason
        self.relevance_score = relevance_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "source": self.source,
            "license": self.license_name,
            "attribution": self.attribution,
            "url": self.original_url,
            "is_ai_generated": self.is_ai_generated,
            "selection_reason": self.selection_reason,
            "relevance_score": self.relevance_score
        }

class FootageManager:
    def __init__(self):
        self.cache_dir = Path("./data/footage_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.used_clip_ids: set = set()

    def reset_used_clips(self):
        self.used_clip_ids.clear()

    def generate_procedural_broll(self, query: str, duration: float, output_path: Path, style: str) -> FootageClip:
        """
        Synthesizes high-definition, dynamic motion graphics B-roll video clips.
        Tailored themes:
        1. Financial & Analytics (Upward line chart, ascending candlesticks, growth indicators)
        2. High-Tech Cloud & Network (3D perspective grid, server nodes, pulse packets)
        3. Partnership & Trust (Constellation particle network, golden geometric rings)
        4. Dynamic Creative Waves (Layered light streaks, orbiting kinetic particles)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        width, height = 1920, 1080
        fps = 30
        total_frames = int(max(duration, 2.0) * fps)
        
        q_lower = query.lower()
        q_hash = int(hashlib.md5(query.encode('utf-8')).hexdigest(), 16)
        random.seed(q_hash % 100000)

        # Detect semantic theme
        is_finance = any(w in q_lower for w in ["finance", "financial", "growth", "sale", "sales", "40%", "bank", "money", "invest", "yield", "profit", "market", "revenue"])
        is_tech = any(w in q_lower for w in ["cloud", "server", "data", "compute", "code", "python", "software", "cluster", "backup", "cyber", "network", "devops"])
        is_trust = any(w in q_lower for w in ["trust", "partner", "team", "relation", "slowly", "faith", "collab", "together"])

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        raw_temp = output_path.with_suffix(".raw.mp4")
        writer = cv2.VideoWriter(str(raw_temp), fourcc, fps, (width, height))

        # Pre-generate particles for constellation/network
        num_particles = 45
        particles = [
            {
                "x": random.uniform(100, width - 100),
                "y": random.uniform(100, height - 100),
                "vx": random.uniform(-1.5, 1.5),
                "vy": random.uniform(-1.5, 1.5),
                "radius": random.randint(3, 7)
            }
            for _ in range(num_particles)
        ]

        # Financial curve keypoints
        num_points = 14
        chart_points = []
        base_y = height * 0.75
        for i in range(num_points):
            px = int(width * 0.12 + (i / (num_points - 1)) * (width * 0.76))
            # Upward trending with minor fluctuations
            fluctuation = random.uniform(-40, 30)
            py = int(base_y - (i / (num_points - 1)) * (height * 0.45) + fluctuation)
            chart_points.append((px, py))

        for frame_idx in range(total_frames):
            progress = frame_idx / total_frames
            t = frame_idx / fps
            canvas = np.zeros((height, width, 3), dtype=np.uint8)

            if is_finance:
                # ----------------- THEME 1: FINANCIAL GROWTH & ANALYTICS -----------------
                # Sleek dark navy slate background
                canvas[:] = (18, 14, 10)
                
                # Subtle background coordinate grid
                for gx in range(0, width, 120):
                    cv2.line(canvas, (gx, 0), (gx, height), (35, 28, 20), 1)
                for gy in range(0, height, 90):
                    cv2.line(canvas, (0, gy), (width, gy), (35, 28, 20), 1)

                # Animated candlestick pillars in background
                for ci in range(12):
                    bar_x = int(width * 0.15 + ci * (width * 0.65 / 11))
                    bar_h = int(60 + ci * 25 + math.sin(t * 3 + ci) * 20)
                    top_y = int(height * 0.8 - bar_h)
                    # Green candle with opacity
                    color = (40, 180, 80) if ci % 3 != 1 else (60, 60, 200)
                    cv2.line(canvas, (bar_x, top_y - 15), (bar_x, int(height * 0.8) + 15), (70, 70, 70), 1)
                    cv2.rectangle(canvas, (bar_x - 12, top_y), (bar_x + 12, int(height * 0.8)), color, -1)

                # Animate upward trend curve progressively
                visible_points_count = max(2, int(progress * len(chart_points)) + 1)
                visible_pts = chart_points[:visible_points_count]
                
                if len(visible_pts) > 1:
                    # Glowing area fill under curve
                    poly_pts = [np.array([visible_pts[0][0], int(height * 0.82)])]
                    poly_pts.extend([np.array(p) for p in visible_pts])
                    poly_pts.append(np.array([visible_pts[-1][0], int(height * 0.82)]))
                    
                    fill_mask = np.zeros_like(canvas)
                    cv2.fillPoly(fill_mask, [np.array(poly_pts, dtype=np.int32)], (20, 80, 40))
                    canvas = cv2.addWeighted(canvas, 1.0, fill_mask, 0.45, 0)

                    # Draw bright glowing line
                    for i in range(len(visible_pts) - 1):
                        cv2.line(canvas, visible_pts[i], visible_pts[i+1], (60, 240, 120), 4, cv2.LINE_AA)

                    # Head point with pulsing concentric circle
                    head = visible_pts[-1]
                    pulse_r = int(12 + math.sin(t * 8) * 4)
                    cv2.circle(canvas, head, pulse_r, (100, 255, 180), 2, cv2.LINE_AA)
                    cv2.circle(canvas, head, 6, (255, 255, 255), -1, cv2.LINE_AA)

                # Modern HUD metric overlays
                cv2.putText(canvas, "+40.2% ANNUAL GROWTH", (int(width * 0.12), int(height * 0.18)), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (60, 255, 140), 2, cv2.LINE_AA)
                cv2.putText(canvas, "METRICS: COMPOUND INTEREST // CAPITAL ASSETS // YIELD", 
                            (int(width * 0.12), int(height * 0.23)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 1, cv2.LINE_AA)

            elif is_tech:
                # ----------------- THEME 2: CLOUD & INFRASTRUCTURE -----------------
                # Deep cybernetic obsidian canvas
                canvas[:] = (24, 16, 8)

                # 3D Perspective Grid receding to horizon
                horizon_y = int(height * 0.42)
                vp_x = int(width * 0.5)
                # Perspective lines
                for px in range(-width, width * 2, 160):
                    cv2.line(canvas, (vp_x, horizon_y), (px, height), (90, 60, 20), 1, cv2.LINE_AA)
                # Horizontal lines moving forward
                offset = (t * 80) % 60
                for ly in range(horizon_y + 10, height, 45):
                    actual_y = ly + int(offset)
                    if actual_y < height:
                        alpha = (actual_y - horizon_y) / (height - horizon_y)
                        line_color = (int(140 * alpha), int(90 * alpha), int(30 * alpha))
                        cv2.line(canvas, (0, actual_y), (width, actual_y), line_color, 1, cv2.LINE_AA)

                # Server rack floating nodes with glowing pulses
                for ri in range(5):
                    node_x = int(width * 0.2 + ri * (width * 0.6 / 4))
                    node_y = int(height * 0.25 + math.sin(t * 2 + ri) * 15)
                    # Server module box
                    cv2.rectangle(canvas, (node_x - 70, node_y - 45), (node_x + 70, node_y + 45), (60, 45, 25), -1)
                    cv2.rectangle(canvas, (node_x - 70, node_y - 45), (node_x + 70, node_y + 45), (220, 160, 60), 2, cv2.LINE_AA)
                    
                    # Status LED blinkers
                    led_color = (80, 255, 120) if (frame_idx + ri * 6) % 20 > 5 else (50, 80, 255)
                    cv2.circle(canvas, (node_x + 50, node_y - 25), 5, led_color, -1, cv2.LINE_AA)
                    cv2.putText(canvas, f"SRV-0{ri+1}", (node_x - 55, node_y + 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)
                    cv2.putText(canvas, "CLOUD CLUSTER", (node_x - 55, node_y + 28), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

                # Circuit packet pulses traveling across
                pulse_x = int((t * 400) % width)
                cv2.line(canvas, (0, int(height * 0.38)), (width, int(height * 0.38)), (80, 50, 20), 2)
                cv2.circle(canvas, (pulse_x, int(height * 0.38)), 8, (255, 220, 100), -1, cv2.LINE_AA)

                # Overlay HUD
                cv2.putText(canvas, "DISTRIBUTED CLUSTER // UPTIME 99.99%", (int(width * 0.1), int(height * 0.12)), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.1, (240, 180, 80), 2, cv2.LINE_AA)

            elif is_trust:
                # ----------------- THEME 3: PARTNERSHIP & TRUST -----------------
                # Warm obsidian & deep bronze
                canvas[:] = (16, 18, 22)

                # Expanding rotating geometric harmonic rings
                center = (int(width * 0.5), int(height * 0.5))
                for ri in range(4):
                    ring_r = int(120 + ri * 80 + math.sin(t * 1.5 + ri) * 30)
                    cv2.circle(canvas, center, ring_r, (40, 70, 95), 1, cv2.LINE_AA)
                    # Orbiting node on ring
                    angle = t * (0.8 - ri * 0.15) + ri * (math.pi / 2)
                    ox = int(center[0] + ring_r * math.cos(angle))
                    oy = int(center[1] + ring_r * math.sin(angle))
                    cv2.circle(canvas, (ox, oy), 7, (70, 180, 230), -1, cv2.LINE_AA)
                    cv2.circle(canvas, (ox, oy), 14, (70, 180, 230), 1, cv2.LINE_AA)

                # Particle constellation with proximity connections
                for p in particles:
                    p["x"] += p["vx"]
                    p["y"] += p["vy"]
                    if p["x"] < 50 or p["x"] > width - 50: p["vx"] *= -1
                    if p["y"] < 50 or p["y"] > height - 50: p["vy"] *= -1

                # Draw connecting lines between close particles
                for i in range(num_particles):
                    for j in range(i + 1, num_particles):
                        dx = particles[i]["x"] - particles[j]["x"]
                        dy = particles[i]["y"] - particles[j]["y"]
                        dist = math.hypot(dx, dy)
                        if dist < 140:
                            alpha = 1.0 - (dist / 140)
                            color = (int(50 * alpha), int(120 * alpha), int(180 * alpha))
                            cv2.line(canvas, (int(particles[i]["x"]), int(particles[i]["y"])), 
                                     (int(particles[j]["x"]), int(particles[j]["y"])), color, 1, cv2.LINE_AA)

                for p in particles:
                    cv2.circle(canvas, (int(p["x"]), int(p["y"])), p["radius"], (120, 210, 255), -1, cv2.LINE_AA)

                cv2.putText(canvas, "COLLABORATION // TRUST & RESILIENCE", (int(width * 0.1), int(height * 0.15)), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.1, (160, 215, 255), 2, cv2.LINE_AA)

            else:
                # ----------------- THEME 4: DYNAMIC CREATIVE MOTION GRAPHICS -----------------
                # Modern indigo-slate background
                canvas[:] = (28, 18, 20)

                # Multi-layered flowing sine wave curves
                for layer in range(5):
                    pts = []
                    layer_color = (int(80 + layer * 35), int(120 + layer * 20), int(220 - layer * 20))
                    for x in range(0, width, 20):
                        freq = 0.003 + layer * 0.001
                        y = int(height * 0.5 + math.sin(x * freq + t * 2 + layer) * (90 + layer * 20))
                        pts.append((x, y))
                    for i in range(len(pts) - 1):
                        cv2.line(canvas, pts[i], pts[i+1], layer_color, 2, cv2.LINE_AA)

                # Kinetic particle vortex
                for pi, p in enumerate(particles[:25]):
                    p_ang = t * 1.2 + pi * 0.3
                    p_rad = 180 + math.sin(t + pi) * 80
                    cx = int(width * 0.5 + p_rad * math.cos(p_ang))
                    cy = int(height * 0.5 + p_rad * math.sin(p_ang))
                    cv2.circle(canvas, (cx, cy), 6, (220, 180, 255), -1, cv2.LINE_AA)
                    cv2.line(canvas, (int(width * 0.5), int(height * 0.5)), (cx, cy), (60, 40, 80), 1, cv2.LINE_AA)

                cv2.putText(canvas, "INCUBRIX CREATIVE STUDIO", (int(width * 0.1), int(height * 0.15)), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.1, (240, 200, 255), 2, cv2.LINE_AA)

            # Subtle vignette and timestamp footer
            cv2.rectangle(canvas, (0, height - 40), (width, height), (10, 10, 10), -1)
            cv2.putText(canvas, f"INCUBRIX B-ROLL // SCENE TIMECODE: {t:05.2f}s // 30 FPS", 
                        (30, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1, cv2.LINE_AA)

            writer.write(canvas)

        writer.release()

        # Re-encode to standard H.264 using FFmpeg
        cmd = [
            FFMPEG_EXE, "-y", "-i", str(raw_temp),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
            str(output_path)
        ]
        import subprocess
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if raw_temp.exists():
            try:
                raw_temp.unlink()
            except Exception:
                pass

        clip_id = f"motion_{abs(q_hash) % 100000:05d}"
        theme_name = "Financial Growth Analytics" if is_finance else ("Cloud Infrastructure Grid" if is_tech else ("Partnership Constellation" if is_trust else "Kinetic Motion Waves"))
        
        return FootageClip(
            clip_id=clip_id,
            local_path=output_path,
            source=f"IncuBrix Kinetic Synthesis ({theme_name})",
            license_name="Commercial Royalty-Free Grant (AI-Generated)",
            attribution="IncuBrix Generative Motion Engine",
            original_url=f"synthetic://render/{clip_id}",
            is_ai_generated=True,
            selection_reason=f"Synthesized dynamic {theme_name} motion graphics matched to query: '{query}'",
            relevance_score=4.7
        )

    async def fetch_pexels_clip(self, query: str, duration: float) -> Optional[FootageClip]:
        """Queries Pexels stock video API with commercial license tracking."""
        if not settings.PEXELS_API_KEY:
            return None
        
        try:
            url = f"https://api.pexels.com/videos/search?query={urllib.parse.quote(query)}&per_page=5&orientation=landscape"
            req = urllib.request.Request(url, headers={"Authorization": settings.PEXELS_API_KEY, "User-Agent": "IncuBrixEngine/1.0"})
            with urllib.request.urlopen(req, timeout=8) as response:
                data = json.loads(response.read().decode("utf-8"))
                videos = data.get("videos", [])
                
                for v in videos:
                    v_id = f"pexels_{v['id']}"
                    if v_id in self.used_clip_ids:
                        continue # Strict zero duplicates rule
                    
                    files = v.get("video_files", [])
                    best_file = None
                    for f in files:
                        if f.get("quality") == "hd" or f.get("width", 0) >= 1280:
                            best_file = f
                            break
                    if not best_file and files:
                        best_file = files[0]

                    if best_file and best_file.get("link"):
                        download_url = best_file["link"]
                        out_file = self.cache_dir / f"{v_id}.mp4"
                        if not out_file.exists():
                            logger.info(f"Downloading Pexels clip: {v_id}")
                            urllib.request.urlretrieve(download_url, str(out_file))

                        self.used_clip_ids.add(v_id)
                        return FootageClip(
                            clip_id=v_id,
                            local_path=out_file,
                            source="Pexels Commercial Stock API",
                            license_name="Pexels Commercial License (Free to use, commercial, no attribution required)",
                            attribution=f"Video by {v.get('user', {}).get('name', 'Pexels Contributor')} on Pexels",
                            original_url=v.get("url", download_url),
                            is_ai_generated=False,
                            selection_reason=f"High-confidence stock match from Pexels API for subject '{query}'",
                            relevance_score=4.6
                        )
        except Exception as e:
            logger.warning(f"Pexels fetch failed: {str(e)}. Proceeding to fallback source.")
        return None

    async def fetch_pixabay_clip(self, query: str, duration: float) -> Optional[FootageClip]:
        """Queries Pixabay video API with commercial license tracking."""
        if not settings.PIXABAY_API_KEY:
            return None
        
        try:
            url = f"https://pixabay.com/api/videos/?key={settings.PIXABAY_API_KEY}&q={urllib.parse.quote(query)}&per_page=5"
            req = urllib.request.Request(url, headers={"User-Agent": "IncuBrixEngine/1.0"})
            with urllib.request.urlopen(req, timeout=8) as response:
                data = json.loads(response.read().decode("utf-8"))
                hits = data.get("hits", [])
                
                for h in hits:
                    v_id = f"pixabay_{h['id']}"
                    if v_id in self.used_clip_ids:
                        continue
                    
                    v_videos = h.get("videos", {})
                    chosen = v_videos.get("large") or v_videos.get("medium") or v_videos.get("small")
                    if chosen and chosen.get("url"):
                        download_url = chosen["url"]
                        out_file = self.cache_dir / f"{v_id}.mp4"
                        if not out_file.exists():
                            urllib.request.urlretrieve(download_url, str(out_file))

                        self.used_clip_ids.add(v_id)
                        return FootageClip(
                            clip_id=v_id,
                            local_path=out_file,
                            source="Pixabay Stock Video API",
                            license_name="Pixabay Content License (Free for commercial use)",
                            attribution=f"Video by {h.get('user', 'Pixabay Contributor')} on Pixabay",
                            original_url=h.get("pageURL", download_url),
                            is_ai_generated=False,
                            selection_reason=f"Pixabay library match for '{query}'",
                            relevance_score=4.5
                        )
        except Exception as e:
            logger.warning(f"Pixabay fetch failed: {str(e)}. Proceeding to fallback source.")
        return None

    async def get_clip_for_scene(
        self,
        scene_index: int,
        search_query: str,
        duration: float,
        style: str,
        preferred_source: Optional[str] = None
    ) -> FootageClip:
        """
        Dual-source selector implementing the Advanced Level Requirement:
        Rule: If external stock keys present and query describes physical entities, query Stock Library first.
        If abstract/surreal, preferred_source == 'ai_gen', or external provider down/empty, use Generative/Procedural model.
        """
        is_abstract = any(w in search_query.lower() for w in ["abstract", "future", "concept", "harmonic", "growth", "slowly", "data stream"])
        
        if preferred_source != "ai_gen" and not is_abstract:
            clip = await self.fetch_pexels_clip(search_query, duration)
            if clip:
                return clip
            
            clip = await self.fetch_pixabay_clip(search_query, duration)
            if clip:
                return clip

        # Fallback / Dual source: Generative Kinetic Motion Engine
        gen_path = self.cache_dir / f"scene_{scene_index}_{hashlib.md5(search_query.encode()).hexdigest()[:8]}.mp4"
        clip = self.generate_procedural_broll(search_query, duration, gen_path, style)
        self.used_clip_ids.add(clip.clip_id)
        return clip

footage_manager = FootageManager()
