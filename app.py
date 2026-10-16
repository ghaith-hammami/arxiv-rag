"""
Gradio app for asking questions about ArXiv papers.
"""

# 1. standard library
import json

# 2. third-party
import gradio as gr

# 3. local
from src.generate import warm_generator_resources
from src.pipeline import run
from src.retrieve import warm_retrieval_resources


def answer_question(query: str) -> tuple[str, float, str]:
    """
    Runs the RAG pipeline for one user question.

    Args:
        query: user question

    Returns:
        answer text, confidence score, and formatted sources
    """
    if not query.strip():
        return "Please enter a question.", 0.0, ""

    result = run(query)
    sources_text = json.dumps(result["sources"], indent=2)
    return result["answer"], result["confidence"], sources_text


def build_chatbot_display(
    chat_history: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    Converts internal chat history into the format expected by Gradio Chatbot.

    Args:
        chat_history: stored chat history

    Returns:
        chatbot display messages
    """
    return list(chat_history)


def chat_with_documents(
    query: str,
    chat_history: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], float, str]:
    """
    Runs one chatbot turn with conversation history.

    Args:
        query: latest user question
        chat_history: stored chat history

    Returns:
        updated chatbot messages, updated history, confidence, and sources
    """
    if not query.strip():
        return build_chatbot_display(chat_history), chat_history, 0.0, ""

    result = run(query, chat_history=chat_history)
    updated_history = list(chat_history)
    updated_history.append({"role": "user", "content": query})
    updated_history.append({"role": "assistant", "content": result["answer"]})

    sources_text = json.dumps(result["sources"], indent=2)
    return (
        build_chatbot_display(updated_history),
        updated_history,
        result["confidence"],
        sources_text,
    )


def clear_chat() -> tuple[list[dict[str, str]], list[dict[str, str]], float, str]:
    """
    Clears the chatbot state.

    Returns:
        empty chat state, empty history, zero confidence, and empty sources
    """
    return [], [], 0.0, ""


def build_app() -> gr.Blocks:
    """
    Builds the Gradio interface.

    Returns:
        the configured Gradio app
    """
    with gr.Blocks(title="ArXiv RAG") as app:
        gr.Markdown("# ArXiv RAG")
        gr.Markdown("Chat with the papers currently indexed from ArXiv.")

        chat_history_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Chat", height=500)
                query_input = gr.Textbox(
                    label="Question",
                    placeholder="Ask a follow-up question about the indexed papers...",
                    lines=3,
                )
                with gr.Row():
                    ask_button = gr.Button("Send", variant="primary")
                    clear_button = gr.Button("Clear chat")

            with gr.Column(scale=2):
                confidence_output = gr.Number(label="Retrieval relevance (0-1)")
                sources_output = gr.Code(label="Sources", language="json")

        ask_button.click(
            fn=chat_with_documents,
            inputs=[query_input, chat_history_state],
            outputs=[chatbot, chat_history_state, confidence_output, sources_output],
        ).then(
            fn=lambda: "",
            inputs=None,
            outputs=query_input,
        )

        query_input.submit(
            fn=chat_with_documents,
            inputs=[query_input, chat_history_state],
            outputs=[chatbot, chat_history_state, confidence_output, sources_output],
        ).then(
            fn=lambda: "",
            inputs=None,
            outputs=query_input,
        )

        clear_button.click(
            fn=clear_chat,
            inputs=None,
            outputs=[chatbot, chat_history_state, confidence_output, sources_output],
        )

    return app


def main() -> None:
    """
    Launches the Gradio app.
    """
    warm_retrieval_resources()
    warm_generator_resources()
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()
