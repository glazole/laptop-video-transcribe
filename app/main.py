"""Gradio UI for local video-to-text + summarization."""
import logging
import gradio as gr
from .config import Settings
from .pipeline import process, rerun_summary
from .prompts import PROMPTS
from .media import SUPPORTED

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
CONFIG = Settings()
ASR_MODELS = ["large-v3", "turbo", "medium", "small"]

def handle_upload(file_path: str, model: str, kind: str, progress=gr.Progress()):
    try:
        progress(0, desc="Создание задания")
        result = process(file_path, model, kind, CONFIG,
                         progress=lambda text: progress(None, desc=text))
        progress(1, desc="Готово")
        return result
    except Exception as exc:
        logging.exception("Processing failed")
        raise gr.Error(str(exc)) from exc

def handle_summary(job_id: str, kind: str, progress=gr.Progress()):
    try:
        return rerun_summary((job_id or "").strip(), kind, CONFIG,
                             progress=lambda text: progress(None, desc=text))
    except Exception as exc:
        logging.exception("Summarization failed")
        raise gr.Error(str(exc)) from exc

def build_ui():
    with gr.Blocks(title="Laptop Video Transcribe") as demo:
        gr.Markdown("# Laptop Video Transcribe v1.0\nЛокальная транскрибация аудио/видео и саммаризация Qwen. После первоначальной загрузки моделей интернет не нужен.")
        with gr.Row():
            upload = gr.File(label="Видео или аудио", type="filepath", file_types=sorted(SUPPORTED))
            with gr.Column():
                model = gr.Dropdown(ASR_MODELS,
                    value=CONFIG.asr_model if CONFIG.asr_model in ASR_MODELS else "large-v3",
                    label="Модель ASR")
                kind = gr.Dropdown([(p.label, key) for key, p in PROMPTS.items()],
                                    value="general", label="Тип саммаризации")
                launch = gr.Button("Транскрибировать и сделать резюме", variant="primary")
        status = gr.Markdown()
        job_id = gr.Textbox(label="ID задания для повторной саммаризации", interactive=True)
        with gr.Tab("Транскрипция"):
            transcript = gr.Textbox(lines=22, label="Текст с таймкодами")
        with gr.Tab("Саммаризация"):
            summary = gr.Markdown()
        outputs = gr.File(label="Скачать результаты", file_count="multiple")
        gr.Markdown("### Повторная саммаризация\nВыберите другой тип и нажмите кнопку; ASR повторно не запускается.")
        retry = gr.Button("Повторить только саммаризацию")
        launch.click(handle_upload, [upload, model, kind],
                     [job_id, transcript, summary, outputs, status], concurrency_limit=1)
        retry.click(handle_summary, [job_id, kind], [summary, outputs, status], concurrency_limit=1)
    return demo

def main():
    CONFIG.validate()
    CONFIG.data_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.cache_dir.mkdir(parents=True, exist_ok=True)
    build_ui().queue(default_concurrency_limit=1, max_size=10).launch(
        server_name=CONFIG.app_host, server_port=CONFIG.app_port,
        share=False, max_file_size=f"{CONFIG.max_upload_mb}mb")

if __name__ == "__main__":
    main()
