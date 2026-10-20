import asyncio
import time
import hmac
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from src.config import settings
from src.logging_config import logger, job_id_ctx
from src.storage.s3_storage import storage
from src.pipeline.audio_engine import audio_engine, AudioEngineException
from src.pipeline.scene_segmenter import scene_segmenter
from src.pipeline.footage_manager import footage_manager
from src.pipeline.video_renderer import video_renderer
from src.pipeline.vlm_evaluator import vlm_evaluator
from src.pipeline.manifest_builder import manifest_builder
from src.pipeline.research_service import research_service
from src.pipeline.scene_planner import scene_planner
from src.pipeline.licensing_service import licensing_service

class JobState:
    def __init__(self, job_id: str, request_data: Dict[str, Any]):
        self.job_id = job_id
        self.request_data = request_data
        self.status = "queued" # queued, running, succeeded, failed
        self.progress = 0 # 0 to 100
        self.current_step = "Initialized"
        self.error = None
        self.outputs = {}
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.completed_at = None
        
        # Internal cache for fast clip swap
        self.scene_segments = {} # ratio -> list of segment paths
        self.audio_path = None
        self.scenes_meta = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "current_step": self.current_step,
            "error": self.error,
            "outputs": self.outputs,
            "created_at": self.created_at,
            "completed_at": self.completed_at
        }

class JobManager:
    def __init__(self):
        self.jobs: Dict[str, JobState] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self._worker_task = None

    def start_worker(self):
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._process_queue())

    def create_job(self, job_id: str, request_data: Dict[str, Any]) -> JobState:
        state = JobState(job_id, request_data)
        self.jobs[job_id] = state
        self.start_worker()
        self.queue.put_nowait(job_id)
        return state

    def get_job(self, job_id: str) -> Optional[JobState]:
        return self.jobs.get(job_id)

    async def _process_queue(self):
        while True:
            job_id = await self.queue.get()
            state = self.jobs.get(job_id)
            if state:
                try:
                    await self._execute_job(state)
                except Exception as e:
                    logger.exception(f"Unexpected fatal error executing job {job_id}: {str(e)}")
                    state.status = "failed"
                    state.error = {
                        "code": "INTERNAL_PIPELINE_ERROR",
                        "message": str(e),
                        "retryable": True,
                        "details": {}
                    }
                finally:
                    self.queue.task_done()
                    # Trigger optional callback webhook
                    await self._trigger_callback(state)

    async def _execute_job(self, state: JobState):
        job_id_ctx.set(state.job_id)
        state.status = "running"
        state.progress = 5
        state.current_step = "Processing speech & narration"
        t0 = time.time()
        timings = {}
        warnings = []

        req = state.request_data
        job_dir = storage.get_job_dir(state.job_id)
        footage_manager.reset_used_clips()

        # Step 0: Intelligent Web Research (Verified facts & authoritative citations)
        research_mode = req.get("research_mode", "script_plus_research")
        raw_text_for_research = req.get("script") or req.get("topic") or ""
        if not raw_text_for_research and req.get("scenes"):
            raw_text_for_research = " ".join(s.get("narration") or s.get("text", "") for s in req["scenes"])
        
        research_bundle = research_service.research_topic(
            topic_or_script=raw_text_for_research,
            mode=research_mode
        )
        if (research_mode == "research_first" or req.get("topic")) and not req.get("script"):
            req["script"] = research_bundle.get("generated_script") or f"Exploring key principles and verified insights on {req.get('topic', 'the key insights')}."

        report_md = research_service.generate_research_report_markdown(research_bundle)
        research_report_file = job_dir / "research_report.md"
        research_report_file.write_text(report_md, encoding="utf-8")

        # Step 1: Narration Ingestion (TTS or Audio Inspection)
        audio_file: Path = None
        dialogue_turns = None
        if req.get("script"):
            audio_file = job_dir / "narration.mp3"
            voice_choice = req.get("voice")
            v_gender = req.get("voice_gender")
            ct = req.get("character_type") or "human"

            if voice_choice == "auto":
                voice_choice = None

            await audio_engine.synthesize_speech(
                text=req["script"],
                voice_pref=voice_choice,
                output_path=audio_file,
                character_type=ct,
                voice_gender=v_gender
            )
            dialogue_turns = getattr(audio_engine, 'last_dialogue_turns', None)
            if dialogue_turns and len(dialogue_turns) >= 2:
                current_ct = (req.get("character_type") or "human").lower()
                if current_ct in ["human", "default", ""]:
                    spk1 = dialogue_turns[0]["speaker"].lower()
                    spk2 = dialogue_turns[1]["speaker"].lower()
                    req["character_type"] = f"{spk1}, {spk2}"
                    logger.info(f"Auto-configured character_type for dialogue: {req['character_type']}")
        elif req.get("audio_url"):
            audio_file = job_dir / "input_audio.mp3"
            logger.info(f"Downloading voiceover audio from {req['audio_url']}")
            urllib.request.urlretrieve(req["audio_url"], str(audio_file))
        elif req.get("local_audio_path"):
            audio_file = Path(req["local_audio_path"])
        elif req.get("scenes") and len(req["scenes"]) > 0:
            # Synthesize narration speech directly from the structured scene list
            combined_script = " ".join(
                s.get("narration") or s.get("text", "") 
                for s in req["scenes"] 
                if (s.get("narration") or s.get("text"))
            ).strip()
            if not combined_script:
                raise AudioEngineException(
                    code="EMPTY_SCENE_NARRATIONS",
                    message="Structured scenes must contain at least one non-empty 'narration' or 'text' field.",
                    retryable=False
                )
            audio_file = job_dir / "narration.mp3"
            ct = req.get("character_type") or "human"
            v_gender = req.get("voice_gender")
            v_pref = req.get("voice")
            if v_pref == "auto":
                v_pref = None
            await audio_engine.synthesize_speech(
                text=combined_script,
                voice_pref=v_pref,
                output_path=audio_file,
                character_type=ct,
                voice_gender=v_gender
            )
        else:
            raise AudioEngineException(
                code="MISSING_INPUT_CONTENT",
                message="Either 'script', 'audio_url', uploaded audio file, or 'scenes' with narration must be provided.",
                retryable=False
            )

        state.audio_path = audio_file
        timings["audio_prep"] = round(time.time() - t0, 2)
        state.progress = 20
        state.current_step = "Transcribing & speech alignment"

        # Step 2: Speech Alignment
        t1 = time.time()
        words, transcript, duration = audio_engine.transcribe_and_align(audio_file)
        timings["speech_alignment"] = round(time.time() - t1, 2)
        state.progress = 35
        state.current_step = "Semantic scene chunking & disambiguation"

        # Step 3: Semantic Scene Segmentation & Visual Queries
        t2 = time.time()
        scenes = scene_segmenter.segment_words_into_scenes(
            words=words,
            total_audio_duration=duration,
            explicit_scenes=req.get("scenes"),
            avoid_list=req.get("avoid"),
            style=req.get("style", "bright, modern, real-world footage"),
            dialogue_turns=dialogue_turns
        )
        timings["scene_segmentation"] = round(time.time() - t2, 2)
        state.progress = 50
        state.current_step = "Curating licence-clean B-roll footage"

        # Step 4: Sourcing Clips per Scene
        t3 = time.time()
        timeline_items = []
        raw_clip_paths = []
        verified_facts = research_bundle.get("verified_facts", [])
        total_scenes = len(scenes)
        is_vo = bool(req.get("is_voiceover", False))
        chosen_ratio = (req.get("aspect_ratios") or ["16:9"])[0]
        show_labels = bool(req.get("show_character_labels", False))

        for idx, scene in enumerate(scenes):
            scene_fact = None
            if verified_facts:
                scene_fact = verified_facts[idx % len(verified_facts)]

            clip = await footage_manager.get_clip_for_scene(
                scene_index=idx,
                search_query=scene["search_query"],
                duration=scene["duration"],
                style=req.get("style", "bright, modern"),
                audio_path=audio_file,
                start_time=scene["start_time"],
                scene_text=scene["narration"],
                character_type=req.get("character_type", "human"),
                scene_speaker=scene.get("speaker"),
                visual_hint=scene.get("visual_hint"),
                verified_fact=scene_fact,
                is_voiceover=is_vo,
                total_scenes=total_scenes,
                aspect_ratio=chosen_ratio,
                show_character_labels=show_labels,
                visual_source_preference=req.get("visual_source_preference", "hybrid")
            )
            raw_clip_paths.append(clip.local_path)
            
            # VLM Relevance Scoring
            vlm_score = vlm_evaluator.evaluate_clip_relevance(
                clip.local_path,
                scene["narration"],
                scene["search_query"]
            )
            clip.relevance_score = vlm_score

            timeline_items.append({
                "scene_index": idx,
                "narration": scene["narration"],
                "start_time": scene["start_time"],
                "end_time": scene["end_time"],
                "duration": scene["duration"],
                "visual_hint": scene.get("visual_hint"),
                "search_query": scene["search_query"],
                "clip_used": clip.to_dict()
            })

        timings["footage_retrieval"] = round(time.time() - t3, 2)
        state.scenes_meta = timeline_items
        state.progress = 70
        state.current_step = "Rendering and multiplexing video outputs"

        # Step 5: Subtitle & Video Assembly
        t4 = time.time()
        srt_file = job_dir / "captions.srt"
        vtt_file = job_dir / "captions.vtt"
        video_renderer.generate_srt(scenes, srt_file)
        video_renderer.generate_vtt(scenes, vtt_file)

        aspect_ratios = req.get("aspect_ratios") or ["16:9"]
        video_urls = {}

        for ratio in aspect_ratios:
            ratio_clean = ratio.replace(":", "-")
            # Step 5a: Pre-normalize segments
            segments = []
            for s_idx, scene in enumerate(scenes):
                seg_path = job_dir / f"seg_{ratio_clean}_{s_idx}.mp4"
                video_renderer.prepare_scene_segment(
                    clip_path=raw_clip_paths[s_idx],
                    duration=scene["duration"],
                    aspect_ratio=ratio,
                    output_path=seg_path
                )
                segments.append(seg_path)

            state.scene_segments[ratio] = segments

            # Step 5b: Assemble final video
            final_video_file = job_dir / f"video_{ratio_clean}.mp4"
            burn_in = req.get("captions", {}).get("burn_in", True)
            video_renderer.assemble_video(
                scene_segments=segments,
                audio_path=audio_file,
                aspect_ratio=ratio,
                output_path=final_video_file,
                srt_path=srt_file,
                burn_in_captions=burn_in
            )
            video_urls[ratio] = storage.get_download_url(final_video_file)

        timings["video_render"] = round(time.time() - t4, 2)

        # Step 6: Output Manifest & Timeline Files
        timeline_file = job_dir / "timeline.json"
        with open(timeline_file, "w", encoding="utf-8") as f:
            json.dump(timeline_items, f, indent=2)

        manifest_data = manifest_builder.build_manifest(
            job_id=state.job_id,
            duration=duration,
            scenes=timeline_items,
            timings=timings,
            aspect_ratios=aspect_ratios,
            warnings=warnings,
            research_bundle=research_bundle
        )
        manifest_file = job_dir / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        # Generate human-readable SOURCES.md report
        sources_md = licensing_service.build_sources_markdown(
            job_id=state.job_id,
            scenes_timeline=timeline_items,
            research_bundle=research_bundle,
            timings=timings,
            warnings=warnings
        )
        sources_file = job_dir / "SOURCES.md"
        sources_file.write_text(sources_md, encoding="utf-8")

        # Store outputs
        state.outputs = {
            "videos": video_urls,
            "captions_srt": storage.get_download_url(srt_file),
            "captions_vtt": storage.get_download_url(vtt_file),
            "timeline": storage.get_download_url(timeline_file),
            "manifest": storage.get_download_url(manifest_file),
            "sources_md": storage.get_download_url(sources_file),
            "sources_markdown": f"/api/v1/broll/jobs/{state.job_id}/sources-markdown",
            "research_report": f"/api/v1/broll/jobs/{state.job_id}/research-report"
        }
        state.status = "succeeded"
        state.progress = 100
        state.current_step = "Completed"
        state.completed_at = datetime.now(timezone.utc).isoformat()
        logger.info(f"Job {state.job_id} successfully completed in {sum(timings.values()):.2f}s")

    async def swap_clip(
        self,
        job_id: str,
        scene_index: int,
        new_query: Optional[str] = None,
        preferred_source: Optional[str] = None,
        pixabay_video_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Fast-path clip swap: re-renders ONLY the specified scene segment,
        concatenates existing segments, and returns the updated video in clearly less time.
        """
        state = self.jobs.get(job_id)
        if not state or state.status != "succeeded":
            raise AudioEngineException(
                code="JOB_NOT_READY_FOR_SWAP",
                message="Cannot swap clip on a job that is not in 'succeeded' state.",
                retryable=False
            )

        if scene_index < 0 or scene_index >= len(state.scenes_meta):
            raise AudioEngineException(
                code="INVALID_SCENE_INDEX",
                message=f"Scene index {scene_index} is out of bounds (0 to {len(state.scenes_meta)-1}).",
                retryable=False
            )

        scene = state.scenes_meta[scene_index]
        query = new_query or scene["search_query"]
        duration = scene["duration"]
        job_dir = storage.get_job_dir(job_id)

        # Sourcing replacement clip
        new_clip = await footage_manager.get_clip_for_scene(
            scene_index=scene_index,
            search_query=query,
            duration=duration,
            style=state.request_data.get("style", "bright, modern"),
            preferred_source=preferred_source,
            audio_path=state.audio_path,
            start_time=scene["start_time"],
            scene_text=scene["narration"],
            character_type=state.request_data.get("character_type", "human"),
            visual_hint=scene.get("visual_hint"),
            show_character_labels=state.request_data.get("show_character_labels", False),
            pixabay_video_id=pixabay_video_id
        )
        scene["clip_used"] = new_clip.to_dict()
        scene["search_query"] = query

        # Re-render ONLY this segment across ratios
        for ratio, seg_list in state.scene_segments.items():
            ratio_clean = ratio.replace(":", "-")
            seg_path = job_dir / f"seg_{ratio_clean}_{scene_index}.mp4"
            video_renderer.prepare_scene_segment(
                clip_path=new_clip.local_path,
                duration=duration,
                aspect_ratio=ratio,
                output_path=seg_path
            )
            seg_list[scene_index] = seg_path
            
            # Fast stitch
            final_video_file = job_dir / f"video_{ratio_clean}.mp4"
            srt_file = job_dir / "captions.srt"
            burn_in = state.request_data.get("captions", {}).get("burn_in", True)
            video_renderer.assemble_video(
                scene_segments=seg_list,
                audio_path=state.audio_path,
                aspect_ratio=ratio,
                output_path=final_video_file,
                srt_path=srt_file,
                burn_in_captions=burn_in
            )

        # Update timeline.json
        timeline_file = job_dir / "timeline.json"
        with open(timeline_file, "w", encoding="utf-8") as f:
            json.dump(state.scenes_meta, f, indent=2)

        # Update manifest.json & SOURCES.md
        manifest_file = job_dir / "manifest.json"
        if manifest_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    mf = json.load(f)
                mf["scene_timeline"] = state.scenes_meta
                with open(manifest_file, "w", encoding="utf-8") as f:
                    json.dump(mf, f, indent=2)
            except Exception:
                pass

        sources_file = job_dir / "SOURCES.md"
        try:
            from src.pipeline.licensing_service import licensing_service
            sources_md = licensing_service.build_sources_markdown(
                job_id=job_id,
                scenes_timeline=state.scenes_meta
            )
            with open(sources_file, "w", encoding="utf-8") as f:
                f.write(sources_md)
        except Exception:
            pass

        return state.outputs

    async def _trigger_callback(self, state: JobState):
        callback_url = state.request_data.get("callback_url")
        if not callback_url:
            return

        try:
            payload = json.dumps(state.to_dict()).encode("utf-8")
            # HMAC-SHA256 signature
            signature = hmac.new(settings.SECRET_KEY.encode(), payload, hashlib.sha256).hexdigest()
            req = urllib.request.Request(
                callback_url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-IncuBrix-Signature": signature
                }
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                logger.info(f"Callback dispatched to {callback_url}, response status {resp.status}")
        except Exception as e:
            logger.warning(f"Callback dispatch failed to {callback_url}: {str(e)}")

job_manager = JobManager()
