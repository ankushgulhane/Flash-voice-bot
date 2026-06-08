"""Gradio UI for the Voice RAG Demo: speak -> transcribe -> RAG -> speak."""
import gradio as gr

from rag import ask
from voice import speak, transcribe


LANGS = {"English": "en", "हिंदी (Hindi)": "hi"}


def pipeline(audio_path, language_label):
    lang = LANGS.get(language_label, "en")
    if not audio_path:
        return None, "**No audio received.** Please record a question with the microphone.", ""

    transcript = transcribe(audio_path, language=lang)
    if not transcript:
        return None, "**Could not understand the audio.** Please try again.", ""

    answer, sources = ask(transcript, lang=lang)
    reply_audio = speak(answer, lang=lang)

    md = f"**You said:** {transcript}\n\n**Answer:** {answer}"
    if sources:
        src_text = "\n\n".join(
            f"[{s['id']}] ({s['source']}, distance={s['distance']})\n{s['text']}" for s in sources
        )
    else:
        src_text = "No sources used (out of scope)."
    return reply_audio, md, src_text


with gr.Blocks(title="Voice RAG Demo") as demo:
    gr.Markdown("# Voice RAG Demo\nAsk a question by voice. Answers come only from the local knowledge base.")
    with gr.Row():
        language = gr.Radio(choices=list(LANGS.keys()), value="English", label="Language")
    with gr.Row():
        mic = gr.Audio(sources=["microphone"], type="filepath", label="Ask your question")
    submit = gr.Button("Ask", variant="primary")
    reply = gr.Audio(label="Answer (audio)", autoplay=True)
    answer_md = gr.Markdown(label="Conversation")
    sources_box = gr.Textbox(label="Retrieved sources", lines=8)

    submit.click(pipeline, inputs=[mic, language], outputs=[reply, answer_md, sources_box])
    mic.stop_recording(pipeline, inputs=[mic, language], outputs=[reply, answer_md, sources_box])


if __name__ == "__main__":
    demo.launch()
