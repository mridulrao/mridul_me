import argparse
import sys
from pathlib import Path

import gradio as gr

from backend.load_model import Backend, ModelConfig, ModelRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch a chat UI for a local model."
    )
    parser.add_argument(
        "--model",
        default="backend/models/LFM2.5-350M",
        help="Model id or local model path.",
    )
    parser.add_argument(
        "--backend",
        choices=["mlx", "cuda", "cpu"],
        default="mlx",
        help="Hardware/backend to run the model on.",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional local LoRA adapter path from model/tunings/.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for the frontend server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for the frontend server.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of generated tokens per reply.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling value.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Penalty to reduce repetitive text.",
    )
    return parser.parse_args()


def build_runner(args) -> ModelRunner:
    config = ModelConfig(
        model_id=args.model,
        backend=args.backend,
        adapter_path=args.adapter,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    return ModelRunner(config)


def build_messages(history, message: str):
    messages = []

    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role and content:
                messages.append({"role": role, "content": str(content)})
            continue

        if isinstance(item, (list, tuple)) and len(item) == 2:
            user_message, assistant_message = item
            if user_message:
                messages.append({"role": "user", "content": str(user_message)})
            if assistant_message:
                messages.append({"role": "assistant", "content": str(assistant_message)})

    messages.append({"role": "user", "content": message})
    return messages


def create_app(runner: ModelRunner, backend: Backend, model_name: str, adapter_path: str | None):
    def normalize_reply(reply):
        if isinstance(reply, str):
            return reply

        if isinstance(reply, dict):
            for key in ("content", "text", "response", "answer"):
                value = reply.get(key)
                if value:
                    return str(value)
            return str(reply)

        if isinstance(reply, (list, tuple)):
            return "\n".join(str(part) for part in reply if part is not None)

        return str(reply)

    def respond(message, history):
        history = history or []

        if not message or not message.strip():
            return history, ""

        messages = build_messages(history, message)
        reply = normalize_reply(runner.chat(messages))
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ]
        return history, ""

    description = f"Model: {model_name} | Backend: {backend}"
    if adapter_path:
        description += f" | Adapter: {adapter_path}"

    with gr.Blocks(title="Mridul ME") as app:
        gr.Markdown(description)

        chatbot = gr.Chatbot()
        message_box = gr.Textbox(
            placeholder="Type a message...",
            lines=3,
            label="Message",
        )

        with gr.Row():
            send_button = gr.Button("Send", variant="primary")
            clear_button = gr.Button("Clear")

        send_button.click(
            fn=respond,
            inputs=[message_box, chatbot],
            outputs=[chatbot, message_box],
        )
        message_box.submit(
            fn=respond,
            inputs=[message_box, chatbot],
            outputs=[chatbot, message_box],
        )
        clear_button.click(
            fn=lambda: ([], ""),
            inputs=None,
            outputs=[chatbot, message_box],
        )

    return app


def main():
    args = parse_args()
    runner = build_runner(args)
    app = create_app(runner, args.backend, args.model, args.adapter)
    app.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
